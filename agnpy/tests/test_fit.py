# test the fitting with both the gammapy and sherpa wrappers
import numpy as np
import astropy.units as u
from astropy.constants import c, G, M_sun
import pytest

# sherpa
import sherpa.fit
from sherpa.stats import Chi2
from sherpa.optmethods import LevMar

# gammapy
from gammapy.modeling.models import SkyModel
import gammapy.modeling

# agnpy
from agnpy.spectra import BrokenPowerLaw, LogParabola
from agnpy.fit import (
    load_sherpa_flux_points,
    load_gammapy_flux_points,
    SynchrotronSelfComptonModel,
    ExternalComptonModel,
)

# dictionaries with the systematics for the Mrk421 and PKS1510-089 data
# let us assume 30% for VHE, 10% for HE and X-ray and 5% for the rest
systemtics_dict_mrk421 = {
    "Fermi": 0.10,
    "GASP": 0.05,
    "GRT": 0.05,
    "MAGIC": 0.30,
    "MITSuME": 0.05,
    "Medicina": 0.05,
    "Metsahovi": 0.05,
    "NewMexicoSkies": 0.05,
    "Noto": 0.05,
    "OAGH": 0.05,
    "OVRO": 0.05,
    "RATAN": 0.05,
    "ROVOR": 0.05,
    "RXTE/PCA": 0.10,
    "SMA": 0.05,
    "Swift/BAT": 0.10,
    "Swift/UVOT": 0.05,
    "Swift/XRT": 0.10,
    "VLBA(BK150)": 0.05,
    "VLBA(BP143)": 0.05,
    "VLBA(MOJAVE)": 0.05,
    "VLBA_core(BP143)": 0.05,
    "VLBA_core(MOJAVE)": 0.05,
    "WIRO": 0.05,
}

systematics_dict_pks1510 = {
    "Fermi-LAT": 0.10,
    "KVA1": 0.05,
    "KVA2": 0.05,
    "MAGIC": 0.30,
    "Metsahovi": 0.05,
    "NICS1": 0.05,
    "NICS2": 0.05,
    "SMART1": 0.05,
    "Swift-XRT": 0.10,
    "TCS1": 0.05,
    "TCS2": 0.05,
    "TCS3": 0.05,
    "TCS4": 0.05,
    "TCS5": 0.05,
    "TCS6": 0.05,
    "UVOT": 0.05,
}


class TestFit:
    """Test the fitting with the gammapy and sherpa models"""

    @pytest.mark.parametrize(
        "sed_path, systematics_dict",
        [
            ("agnpy/data/mwl_seds/Mrk421_2011.ecsv", systemtics_dict_mrk421),
            ("agnpy/data/mwl_seds/PKS1510-089_2015b.ecsv", systematics_dict_pks1510),
        ],
    )
    def test_sed_loading(self, sed_path, systematics_dict):
        """Test that the same values are loaded by gammapy and sherpa from the
        MWL SED files."""

        # min and max energy to be considered in the fit
        E_min = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())
        E_max = 100 * u.TeV

        # load the flux points
        datasets_gammapy = load_gammapy_flux_points(
            sed_path, E_min, E_max, systematics_dict
        )
        data_1d_sherpa = load_sherpa_flux_points(
            sed_path, E_min, E_max, systematics_dict
        )

        # assert that the SED points are the same
        # gammapy data are divided in dataset, let us group them together again
        energy_gammapy = np.asarray([])
        e2dnde_gammapy = np.asarray([])
        e2dnde_errn_gammapy = np.asarray([])
        e2dnde_errp_gammapy = np.asarray([])

        for dataset in datasets_gammapy:
            energy_gammapy = np.append(energy_gammapy, dataset.data.energy_ref.data)
            e2dnde_gammapy = np.append(
                e2dnde_gammapy,
                dataset.data.e2dnde.quantity.flatten().to_value("erg cm-2 s-1"),
            )
            e2dnde_errn_gammapy = np.append(
                e2dnde_errn_gammapy,
                dataset.data.e2dnde_errn.quantity.flatten().to_value("erg cm-2 s-1"),
            )
            e2dnde_errp_gammapy = np.append(
                e2dnde_errp_gammapy,
                dataset.data.e2dnde_errp.quantity.flatten().to_value("erg cm-2 s-1"),
            )

        # this is the error used in the Chi2 computation by gammapy
        e2dnde_err_gammapy = (e2dnde_errn_gammapy + e2dnde_errp_gammapy) / 2

        assert np.allclose(energy_gammapy, data_1d_sherpa.x, atol=0, rtol=1e-5)
        assert np.allclose(e2dnde_gammapy, data_1d_sherpa.y, atol=0, rtol=1e-5)
        assert np.allclose(
            e2dnde_err_gammapy, data_1d_sherpa.get_error(), atol=0, rtol=1e-3
        )

    def test_mrk_421_fit(self):
        """Test the fit of Mrk421 MWL SED. Validate that the sherpa and gammapy
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
        # min and max energy to be considered in the fit
        E_min = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())
        E_max = 100 * u.TeV
        datasets_gammapy = load_gammapy_flux_points(
            "agnpy/data/mwl_seds/Mrk421_2011.ecsv", E_min, E_max, systemtics_dict_mrk421
        )
        sky_model = SkyModel(spectral_model=ssc_model_gammapy, name="Mrk421")
        datasets_gammapy.models = [sky_model]
        gammapy_fitter = gammapy.modeling.Fit()

        # load the sherpa dataset
        data_1d_sherpa = load_sherpa_flux_points(
            "agnpy/data/mwl_seds/Mrk421_2011.ecsv", E_min, E_max, systemtics_dict_mrk421
        )
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

        # assert that the final statistics are within 1%
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
        n_e = LogParabola(
            k_e=1 * u.Unit("cm-3"),
            p=2.0,
            q=0.2,
            gamma_0=1e2,
            gamma_min=1,
            gamma_max=3e4,
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

        # set the gammapy wrapper parameters
        # freeze the log parabola reference energy
        ec_model_gammapy.log10_gamma_0.frozen = True
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
        # set the sherpa wrapper parameters
        # freeze the log parabola reference energy
        ec_model_sherpa.log10_gamma_0.freeze()
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
        # min and max energy to be considered in the fit
        E_min = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())
        E_max = 100 * u.TeV
        datasets_gammapy = load_gammapy_flux_points(
            "agnpy/data/mwl_seds/PKS1510-089_2015b.ecsv",
            E_min,
            E_max,
            systematics_dict_pks1510,
        )
        sky_model = SkyModel(spectral_model=ec_model_gammapy, name="Mrk421")
        datasets_gammapy.models = [sky_model]
        gammapy_fitter = gammapy.modeling.Fit()

        # load the sherpa dataset
        data_1d_sherpa = load_sherpa_flux_points(
            "agnpy/data/mwl_seds/PKS1510-089_2015b.ecsv",
            E_min,
            E_max,
            systematics_dict_pks1510,
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

        # assert that the final statistics are within 1%
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
            ec_model_gammapy.q.value, ec_model_sherpa.q.val, atol=0, rtol=0.01
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
