# test the gammapy and sherpa wrappers
import numpy as np
import astropy.units as u
from astropy.constants import m_e, M_sun
from astropy.coordinates import Distance
import pytest
import shutil
from pathlib import Path
from agnpy.spectra import PowerLaw, BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.targets import SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.fit import SynchrotronSelfComptonModel, ExternalComptonModel
from .utils import make_comparison_plot, check_deviation, clean_and_make_dir


agnpy_dir = Path(__file__).parent.parent
# where to read sampled files
data_dir = agnpy_dir / "data"
# where to save figures, clean-up before making the new
figures_dir = clean_and_make_dir(agnpy_dir.parent, "crosschecks/figures/fit")

# Blobs to be used by SSC and EC models
R_b = 1e16 * u.cm
V_b = 4 / 3 * np.pi * R_b ** 3
W_e_ssc = 1e48 * u.Unit("erg")
W_e_ec = 6e42 * u.Unit("erg")
z_ssc = Distance(1e27, unit=u.cm).z
z_ec = 1

n_e_ssc = PowerLaw.from_total_energy(
    W_e_ssc, V_b, m_e, p=2.8, gamma_min=1e2, gamma_max=1e7
)
blob_ssc = Blob(R_b=R_b, z=z_ssc, delta_D=10, Gamma=10, B=1 * u.G, n_e=n_e_ssc)

n_e_ec = BrokenPowerLaw.from_total_energy(
    W_e_ec, V_b, m_e, p1=2.0, p2=3.5, gamma_b=1e4, gamma_min=20, gamma_max=5e7
)
blob_ec = Blob(
    R_b=R_b, z=z_ec, delta_D=40, Gamma=40, B=1 * u.G, n_e=n_e_ec, gamma_e_size=300
)

# targets definition
# - disk
L_disk = 2 * 1e46 * u.Unit("erg s-1")
M_BH = 1.2 * 1e9 * M_sun
eta = 1 / 12
R_g = 1.77 * 1e14 * u.cm
R_in = 6 * R_g
R_out = 200 * R_g
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out)

# - BLR definition
xi_line = 0.024
R_line = 1e17 * u.cm
blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
# - dust torus definition
T_dt = 1e3 * u.K
csi_dt = 0.1
dt = RingDustTorus(L_disk, csi_dt, T_dt)

# distance
r = 1e16 * u.cm

# frequency / energies for SED computation
nu = np.logspace(9, 29, 100) * u.Hz
E = nu.to("eV", equivalencies=u.spectral())

ec_blr_parameters_names = [
    "log10_L_disk",
    "M_BH",
    "m_dot",
    "R_in",
    "R_out",
    "xi_line",
    "lambda_line",
    "R_line",
]
ec_dt_parameters_names = [
    "log10_L_disk",
    "M_BH",
    "m_dot",
    "R_in",
    "R_out",
    "xi_dt",
    "T_dt",
    "R_dt",
]
ec_blr_dt_parameters_names = [
    "log10_L_disk",
    "M_BH",
    "m_dot",
    "R_in",
    "R_out",
    "xi_line",
    "lambda_line",
    "R_line",
    "xi_dt",
    "T_dt",
    "R_dt",
]


class TestWrappers:
    """Test the wrappers for fitting."""

    def test_gammapy_spectral_emission_region_parameters_names(self):
        """Test that the parameters of the spectrum and of the emission region
        can be correctly fetched."""
        ssc_model = SynchrotronSelfComptonModel(blob_ssc.n_e, backend="gammapy")
        assert ssc_model.spectral_parameters.names == [
            "log10_k",
            "p",
            "log10_gamma_min",
            "log10_gamma_max",
        ]
        assert ssc_model.emission_region_parameters.names == [
            "z",
            "delta_D",
            "log10_B",
            "t_var",
        ]

        # check that, when changed, parameters are updated accordingly
        # in both lists
        log10_k = -2
        ssc_model.parameters["log10_k"].value = log10_k
        assert u.isclose(ssc_model.spectral_parameters["log10_k"].value, log10_k)

    @pytest.mark.parametrize(
        "targets, targets_parameters_names",
        [
            (["blr"], ec_blr_parameters_names),
            (["dt"], ec_dt_parameters_names),
            (["blr", "dt"], ec_blr_dt_parameters_names),
        ],
    )
    def test_gamampy_targets_parameters_names(self, targets, targets_parameters_names):
        """In this function we just test that the targets are correctly loaded
        and that the list of parameters is what we expect."""
        ec_model = ExternalComptonModel(blob_ec.n_e, targets, backend="gammapy")
        assert ec_model.targets_parameters.names == targets_parameters_names

    def test_sherpa_ssc_model_parameters_names(self):
        """Test the names of the SSC model parameters for the `sherpa` wrapper."""
        ssc_model = SynchrotronSelfComptonModel(blob_ssc.n_e, backend="sherpa")
        pars_names = [par.name for par in ssc_model.pars]
        assert pars_names == [
            "log10_k",
            "p",
            "log10_gamma_min",
            "log10_gamma_max",
            "z",
            "delta_D",
            "log10_B",
            "t_var",
        ]

    @pytest.mark.parametrize(
        "targets, targets_parameters_names",
        [
            (["blr"], ec_blr_parameters_names),
            (["dt"], ec_dt_parameters_names),
            (["blr", "dt"], ec_blr_dt_parameters_names),
        ],
    )
    def test_sherpa_ec_model_parameters_names(self, targets, targets_parameters_names):
        """Test the names of the EC model parameters for the `sherpa` wrapper."""
        ec_model = ExternalComptonModel(blob_ec.n_e, targets, backend="sherpa")
        pars_names = [par.name for par in ec_model.pars]
        blob_pars = [
            "log10_k",
            "p1",
            "p2",
            "log10_gamma_b",
            "log10_gamma_min",
            "log10_gamma_max",
            "z",
            "delta_D",
            "log10_B",
            "t_var",
            "mu_s",
            "log10_r",
        ]
        assert pars_names == blob_pars + targets_parameters_names

    @pytest.mark.parametrize("backend", ["gammapy", "sherpa"])
    def test_synchrotron_self_compton_model(self, backend):
        """Test the SSC model SED computation using agnpy classes against that
        obtained with the Gammapy and sherpa wrappers."""
        # agnpy radiative processes
        synch = Synchrotron(blob_ssc, ssa=True)
        ssc = SynchrotronSelfCompton(blob_ssc, ssa=True)

        # model
        ssc_model = SynchrotronSelfComptonModel(blob_ssc.n_e, ssa=True, backend=backend)
        ssc_model.set_emission_region_parameters_from_blob(blob_ssc)

        # compute SEDs from agnpy and the wrappers
        if backend == "sherpa":
            sed_wrapper = ssc_model(E.to_value("eV"))
        if backend == "gammapy":
            sed_wrapper = (E ** 2 * ssc_model(E)).to("erg cm-2 s-1")

        sed_agnpy = synch.sed_flux(nu) + ssc.sed_flux(nu)

        nu_range = [1e9, 1e27] * u.Hz
        make_comparison_plot(
            nu,
            sed_wrapper,
            sed_agnpy,
            f"{backend} wrapper",
            "agnpy",
            "SSC scenario",
            figures_dir / f"ssc_scenario_{backend}_wrapper.png",
            "sed",
            y_range=[1e-13, 1e-9],
            comparison_range=nu_range.to_value("Hz"),
        )

        # requires that the SED points deviate less than 1% from the figure
        assert check_deviation(nu, sed_wrapper, sed_agnpy, 0.1, nu_range)

    @pytest.mark.parametrize("backend", ["gammapy", "sherpa"])
    @pytest.mark.parametrize(
        "targets, r",
        [(["blr"], 1e16 * u.cm), (["dt"], 1e18 * u.cm), (["blr", "dt"], 2e17 * u.cm)],
    )
    def test_external_compton_model(self, backend, targets, r):
        """Check that the Synch + SSC + EC SED computed with the wrappers
        corresponds to the one computed with agnpy."""
        # agnpy radiative processes
        ec_blr = ExternalCompton(blob_ec, blr, r)
        ec_dt = ExternalCompton(blob_ec, dt, r)
        synch = Synchrotron(blob_ec)
        ssc = SynchrotronSelfCompton(blob_ec)
        sed_agnpy = synch.sed_flux(nu) + ssc.sed_flux(nu) + disk.sed_flux(nu, blob_ec.z)

        # Gammapy wrapper
        ec_model = ExternalComptonModel(blob_ec.n_e, targets, backend=backend)
        ec_model.set_emission_region_parameters_from_blob(blob_ec, r)

        # now set the parameters of the target and compute the SEDs
        if targets == ["blr"]:
            ec_model.set_targets_parameters_from_targets(disk=disk, blr=blr)
            sed_agnpy += ec_blr.sed_flux(nu)
            title = "EC on BLR scenario"
            fig_name = f"ec_blr_scenario_{backend}_wrapper.png"

        if targets == ["dt"]:
            ec_model.set_targets_parameters_from_targets(disk=disk, dt=dt)
            sed_agnpy += dt.sed_flux(nu, blob_ec.z)
            sed_agnpy += ec_dt.sed_flux(nu)
            title = "EC on DT scenario"
            fig_name = f"ec_dt_scenario_{backend}_wrapper.png"

        if targets == ["blr", "dt"]:
            ec_model.set_targets_parameters_from_targets(disk=disk, blr=blr, dt=dt)
            sed_agnpy += dt.sed_flux(nu, blob_ec.z)
            sed_agnpy += ec_blr.sed_flux(nu)
            sed_agnpy += ec_dt.sed_flux(nu)
            title = "EC on BLR and DT scenario"
            fig_name = f"ec_blr_dt_scenario_{backend}_wrapper.png"

        if backend == "sherpa":
            sed_wrapper = ec_model(E.to_value("eV"))
        if backend == "gammapy":
            sed_wrapper = (E ** 2 * ec_model(E)).to("erg cm-2 s-1")

        make_comparison_plot(
            nu,
            sed_wrapper,
            sed_agnpy,
            f"{backend} wrapper",
            "agnpy",
            title,
            figures_dir / fig_name,
            "sed",
        )

        # requires that the SED points deviate less than 1% from the figure
        assert check_deviation(nu, sed_wrapper, sed_agnpy, 0.1)
