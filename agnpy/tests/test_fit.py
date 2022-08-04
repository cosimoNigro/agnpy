# test the fitting with both the gammapy and sherpa wrappers
import numpy as np
import astropy.units as u
from astropy.constants import c, G, M_sun
from astropy.table import Table
import pytest

# sherpa
from sherpa import data
import sherpa.fit
from sherpa.stats import Chi2
from sherpa.optmethods import LevMar

# gammapy
from gammapy.estimators import FluxPoints
from gammapy.datasets import FluxPointsDataset, Datasets
from gammapy.modeling.models import SkyModel
import gammapy.modeling

# agnpy
from agnpy.spectra import BrokenPowerLaw, ExpCutoffPowerLaw
from agnpy.fit import (
    add_systematic_errors_flux_points,
    SynchrotronSelfComptonModel,
    ExternalComptonModel,
)


def load_gammapy_flux_points(sed_path):
    """Load the MWL SED at `sed_path` in a list of
    `~gammapy.datasets.FluxPointsDataset`. Add the systematic errors."""
    datasets = Datasets()

    table = Table.read(sed_path)
    table = table.group_by("instrument")

    # do not use frequency point below 1e11 Hz, affected by non-blazar emission
    E_min_fit = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())

    for group in table.groups:
        name = group["instrument"][0]
        data = FluxPoints.from_table(group, sed_type="e2dnde", format="gadf-sed")

        # add systematic errors
        if name == "MAGIC":
            add_systematic_errors_flux_points(data, 0.30)
        elif name in [
            "Fermi",
            "Fermi-LAT",
            "RXTE/PCA",
            "Swift/BAT",
            "Swift/XRT",
            "Swift-XRT",
        ]:
            add_systematic_errors_flux_points(data, 0.10)
        else:
            add_systematic_errors_flux_points(data, 0.05)

        # load the flux points in a dataset
        dataset = FluxPointsDataset(data=data, name=name)

        # set the minimum energy to be used for the fit
        dataset.mask_fit = dataset.data.energy_ref > E_min_fit

        datasets.append(dataset)

    return datasets


def load_sherpa_flux_points(sed_path):
    """Load the MWL SED at `sed_path` in a `~sherpa.data.Data1D` object."""
    table = Table.read(sed_path)
    table = table.group_by("instrument")

    x = []
    y = []
    y_err_stat = []
    y_err_syst = []

    for tab in table.groups:
        name = tab["instrument"][0]

        nu = tab["e_ref"].quantity.to_value("Hz", equivalencies=u.spectral())
        e2dnde = tab["e2dnde"]
        e2dnde_err_stat = tab["e2dnde_errn"]

        # add systematic error
        if name == "MAGIC":
            e2dnde_err_syst = 0.30 * e2dnde
        elif name in [
            "Fermi",
            "Fermi-LAT",
            "RXTE/PCA",
            "Swift/BAT",
            "Swift/XRT",
            "Swift-XRT",
        ]:
            e2dnde_err_syst = 0.10 * e2dnde
        else:
            e2dnde_err_syst = 0.05 * e2dnde

        x.extend(nu)
        y.extend(e2dnde)
        y_err_stat.extend(e2dnde_err_stat)
        y_err_syst.extend(e2dnde_err_syst)

    sed = data.Data1D(
        "sed",
        np.asarray(x),
        np.asarray(y),
        staterror=np.asarray(y_err_stat),
        syserror=np.asarray(y_err_syst),
    )

    # set the minimum energy to be used for the fit
    min_x = 1e11 * u.Hz
    max_x = 1e30 * u.Hz
    sed.notice(min_x, max_x)

    return sed


class TestFit:
    """Test the fitting with the gammapy and sherpa models"""

    @pytest.mark.parametrize(
        "sed_path",
        [
            "agnpy/data/mwl_seds/Mrk421_2011.ecsv",
            "agnpy/data/mwl_seds/PKS1510-089_2015b.ecsv",
        ],
    )
    def test_sed_loading(self, sed_path):
        """Test that the MWL SED data are correctly loaded by Gammapy and sherpa."""

        # load the flux points
        datasets_gammapy = load_gammapy_flux_points(sed_path)
        data_1d_sherpa = load_sherpa_flux_points(sed_path)

        # assert that the SED points are the same
        # gammapy data are divided in dataset, let us group them together again
        energy_gammapy = []
        e2dnde_gammappy = []
        e2dnde_err_gammapy = []
        for dataset in datasets_gammapy:
            energy_gammapy.extend(dataset.data.energy_ref.data)
            e2dnde_gammappy.extend(
                dataset.data.e2dnde.quantity.flatten().to_value("erg cm-2 s-1")
            )
            e2dnde_err_gammapy.extend(
                dataset.data.e2dnde_errn.quantity.flatten().to_value("erg cm-2 s-1")
            )

        energy_sherpa = (data_1d_sherpa.x * u.Hz).to_value(
            "eV", equivalencies=u.spectral()
        )

        assert np.allclose(energy_gammapy, energy_sherpa, atol=0, rtol=1e-3)
        assert np.allclose(e2dnde_gammappy, data_1d_sherpa.y, atol=0, rtol=1e-3)
        assert np.allclose(
            e2dnde_err_gammapy, data_1d_sherpa.get_error(), atol=0, rtol=1e-3
        )

    def test_mrk_421_fit(self):
        """test the fit of Mrk421 MWL SED. Check that the sherpa and gammapy
        wrappers return the same results."""
        # electron energy distribution
        n_e = BrokenPowerLaw(
            k_e=1e-8 * u.Unit("cm-3"),
            p1=2.02,
            p2=3.43,
            gamma_b=1e5,
            gamma_min=500,
            gamma_max=1e6,
        )

        # initialise the wrappers
        ssc_model_sherpa = SynchrotronSelfComptonModel(n_e, backend="sherpa")
        ssc_model_gammapy = SynchrotronSelfComptonModel(n_e, backend="gammapy")

        # set the emission region parameters
        z = 0.0308
        delta_D = 18
        t_var = 1 * u.d
        log10_B = -1.3
        # for the gammapy wrapper
        ssc_model_gammapy.z.value = z
        ssc_model_gammapy.delta_D.value = delta_D
        ssc_model_gammapy.t_var.value = t_var.to_value("s")
        ssc_model_gammapy.t_var.frozen = True
        ssc_model_gammapy.log10_B.value = log10_B
        # for the sherpa wrapper
        ssc_model_sherpa.z = z
        ssc_model_sherpa.delta_D = delta_D
        ssc_model_sherpa.t_var = t_var.to_value("s")
        ssc_model_sherpa.t_var.freeze()
        ssc_model_sherpa.log10_B = log10_B

        # load the gammapy dataset
        datasets_gammapy = load_gammapy_flux_points(
            "agnpy/data/mwl_seds/Mrk421_2011.ecsv"
        )
        sky_model = SkyModel(spectral_model=ssc_model_gammapy, name="Mrk421")
        datasets_gammapy.models = [sky_model]
        gammapy_fitter = gammapy.modeling.Fit()

        # load the sherpa dataset
        data_1d_sherpa = load_sherpa_flux_points("agnpy/data/mwl_seds/Mrk421_2011.ecsv")
        sherpa_fitter = sherpa.fit.Fit(
            data_1d_sherpa, ssc_model_sherpa, stat=Chi2(), method=LevMar()
        )

        # assert that the inital statistics are within 1%
        sherpa_stat = sherpa_fitter.calc_stat()
        gammapy_stat = datasets_gammapy.stat_sum()
        assert np.isclose(sherpa_stat, gammapy_stat, atol=0, rtol=0.01)

        # run the fit!
        gammapy_result = gammapy_fitter.run(datasets_gammapy)
        sherpa_result = sherpa_fitter.fit()

        # assert both fits converged
        assert gammapy_result.success == sherpa_result.succeeded

        # assert that the starting statistics are within 1%
        sherpa_stat = sherpa_fitter.calc_stat()
        gammapy_stat = datasets_gammapy.stat_sum()
        assert np.isclose(sherpa_stat, gammapy_stat, atol=0, rtol=0.01)

        # assert that the final parameters are compatible as well
        assert np.isclose(
            ssc_model_gammapy.log10_k_e.value,
            ssc_model_sherpa.log10_k_e.val,
            atol=0,
            rtol=0.01,
        )
        assert np.isclose(
            ssc_model_gammapy.p1.value, ssc_model_sherpa.p1.val, atol=0, rtol=0.01
        )
        assert np.isclose(
            ssc_model_gammapy.p2.value, ssc_model_sherpa.p2.val, atol=0, rtol=0.01
        )
        assert np.isclose(
            ssc_model_gammapy.log10_gamma_b.value,
            ssc_model_sherpa.log10_gamma_b.val,
            atol=0,
            rtol=0.01,
        )
        assert np.isclose(
            ssc_model_gammapy.delta_D.value,
            ssc_model_sherpa.delta_D.val,
            atol=0,
            rtol=0.01,
        )
        assert np.isclose(
            ssc_model_gammapy.log10_B.value,
            ssc_model_sherpa.log10_B.val,
            atol=0,
            rtol=0.01,
        )

    def test_pks_1510_fit(self):
        """test the fit of Mrk421 MWL SED. Check that the sherpa and gammapy
        wrappers return the same results."""
        # electron energy distribution
        n_e = ExpCutoffPowerLaw(
            k_e=1e4 * u.Unit("cm-3"), p=2.0, gamma_c=5e3, gamma_min=1, gamma_max=5e4
        )

        # initialise the wrappers
        ec_model_sherpa = ExternalComptonModel(n_e, ["dt"], ssa=True, backend="sherpa")
        ec_model_gammapy = ExternalComptonModel(
            n_e, ["dt"], ssa=True, backend="gammapy"
        )

        # emission region parameters
        z = 0.361
        Gamma = 20
        delta_D = 25
        Beta = np.sqrt(1 - 1 / np.power(Gamma, 2))  # jet relativistic speed
        mu_s = (1 - 1 / (Gamma * delta_D)) / Beta  # viewing angle
        B = 0.35 * u.G
        t_var = 0.5 * u.d
        r = 6e17 * u.cm
        # targets parameters
        L_disk = 6.7e45 * u.Unit("erg s-1")  # disk luminosity
        M_BH = 5.71 * 1e7 * M_sun
        eta = 1 / 12
        m_dot = (L_disk / (eta * c ** 2)).to("g s-1")
        R_g = ((G * M_BH) / c ** 2).to("cm")
        R_in = 6 * R_g
        R_out = 3e4 * R_g
        xi_dt = 0.6
        T_dt = 2e3 * u.K
        R_dt = 2.5 * 1e18 * np.sqrt(L_disk.to_value("erg s-1") / 1e45) * u.cm

        # for the gammapy wrapper
        ec_model_gammapy.z.value = z
        ec_model_gammapy.delta_D.value = delta_D
        ec_model_gammapy.t_var.value = t_var.to_value("s")
        ec_model_gammapy.t_var.frozen = True
        ec_model_gammapy.log10_B.value = np.log10(B.to_value("G"))
        ec_model_gammapy.mu_s.value = mu_s
        ec_model_gammapy.log10_r.value = np.log10(r.to_value("cm"))
        ec_model_gammapy.log10_L_disk.value = np.log10(L_disk.to_value("erg s-1"))
        ec_model_gammapy.M_BH.value = M_BH.to_value("g")
        ec_model_gammapy.m_dot.value = m_dot.to_value("g s-1")
        ec_model_gammapy.R_in.value = R_in.to_value("cm")
        ec_model_gammapy.R_out.value = R_out.to_value("cm")
        ec_model_gammapy.xi_dt.value = xi_dt
        ec_model_gammapy.T_dt.value = T_dt.to_value("K")
        ec_model_gammapy.R_dt.value = R_dt.to_value("cm")
        # for the sherpa wrapper
        ec_model_sherpa.z = z
        ec_model_sherpa.delta_D = delta_D
        ec_model_sherpa.t_var = t_var.to_value("s")
        ec_model_sherpa.t_var.freeze()
        ec_model_sherpa.log10_B = np.log10(B.to_value("G"))
        ec_model_sherpa.mu_s = mu_s
        ec_model_sherpa.log10_r = np.log10(r.to_value("cm"))
        ec_model_sherpa.log10_L_disk = np.log10(L_disk.to_value("erg s-1"))
        ec_model_sherpa.M_BH = M_BH.to_value("g")
        ec_model_sherpa.m_dot = m_dot.to_value("g s-1")
        ec_model_sherpa.R_in = R_in.to_value("cm")
        ec_model_sherpa.R_out = R_out.to_value("cm")
        ec_model_sherpa.xi_dt = xi_dt
        ec_model_sherpa.T_dt = T_dt.to_value("K")
        ec_model_sherpa.R_dt = R_dt.to_value("cm")

        # load the gammapy dataset
        datasets_gammapy = load_gammapy_flux_points(
            "agnpy/data/mwl_seds/PKS1510-089_2015b.ecsv"
        )
        sky_model = SkyModel(spectral_model=ec_model_gammapy, name="Mrk421")
        datasets_gammapy.models = [sky_model]
        gammapy_fitter = gammapy.modeling.Fit()

        # load the sherpa dataset
        data_1d_sherpa = load_sherpa_flux_points(
            "agnpy/data/mwl_seds/PKS1510-089_2015b.ecsv"
        )
        sherpa_fitter = sherpa.fit.Fit(
            data_1d_sherpa, ec_model_sherpa, stat=Chi2(), method=LevMar()
        )

        # assert that the inital statistics are within 1%
        sherpa_stat = sherpa_fitter.calc_stat()
        gammapy_stat = datasets_gammapy.stat_sum()
        assert np.isclose(sherpa_stat, gammapy_stat, atol=0, rtol=0.01)

        # run the fit!
        gammapy_result = gammapy_fitter.run(datasets_gammapy)
        sherpa_result = sherpa_fitter.fit()

        # assert both fits converged
        assert gammapy_result.success == sherpa_result.succeeded

        # assert that the starting statistics are within 1%
        sherpa_stat = sherpa_fitter.calc_stat()
        gammapy_stat = datasets_gammapy.stat_sum()
        assert np.isclose(sherpa_stat, gammapy_stat, atol=0, rtol=0.01)

        # assert that the final parameters are compatible as well
        assert np.isclose(
            ec_model_gammapy.log10_k_e.value,
            ec_model_sherpa.log10_k_e.val,
            atol=0,
            rtol=0.01,
        )
        assert np.isclose(
            ec_model_gammapy.p.value, ec_model_sherpa.p.val, atol=0, rtol=0.01
        )
        assert np.isclose(
            ec_model_gammapy.log10_gamma_c.value,
            ec_model_sherpa.log10_gamma_c.val,
            atol=0,
            rtol=0.01,
        )
        assert np.isclose(
            ec_model_gammapy.delta_D.value,
            ec_model_sherpa.delta_D.val,
            atol=0,
            rtol=0.01,
        )
        assert np.isclose(
            ec_model_gammapy.log10_B.value,
            ec_model_sherpa.log10_B.val,
            atol=0,
            rtol=0.01,
        )
