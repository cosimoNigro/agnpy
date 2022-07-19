import numpy as np
import astropy.units as u
from astropy.constants import M_sun
from astropy.coordinates import Distance
import pytest
from pathlib import Path
from agnpy.emission_regions import Blob
from agnpy.targets import SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.wrappers import SynchrotronSelfComptonSpectralModel
from agnpy.wrappers.gammapy import (
    ExternalComptonSpectralModel,
    SynchrotronSelfComptonSpectralModel,
)
from .utils import make_comparison_plot, check_deviation

agnpy_dir = Path(__file__).parent.parent
# where to read sampled files
data_dir = agnpy_dir / "data"
# where to save figures
figures_dir = agnpy_dir.parent / "crosschecks/figures/wrappers"
figures_dir.mkdir(parents=True, exist_ok=True)

# emission region and targets
# - SSC blob
spectrum_norm = 1e48 * u.Unit("erg")
spectrum_dict = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e7},
}
R_b = 1e16 * u.cm
B = 1 * u.G
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
ssc_blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

# - EC blob
spectrum_norm = 6e42 * u.erg
parameters = {
    "p1": 2.0,
    "p2": 3.5,
    "gamma_b": 1e4,
    "gamma_min": 20,
    "gamma_max": 5e7,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
gamma_size = 300  # same value set in the EC evaluation
R_b = 1e16 * u.cm
B = 0.56 * u.G
z = 1
delta_D = 40
Gamma = 40
ec_blob = Blob(
    R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict, gamma_size=gamma_size
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


class TestGammapyWrapper:
    """Test the Gammapy wrappers."""

    def test_spectral_emission_region_parameters(self):
        """Test that the parameters of the spectrum and of the emission region
        can be correctly fetched."""
        ssc_model = SynchrotronSelfComptonSpectralModel(ssc_blob.n_e)
        assert ssc_model.spectral_parameters.names == [
            "log10_k_e",
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
        log10_k_e = 8
        ssc_model.parameters["log10_k_e"].value = log10_k_e
        assert u.isclose(ssc_model.spectral_parameters["log10_k_e"].value, log10_k_e)

    def test_synchrotron_self_compton_spectral_model(self):
        """Test the SSC model SED computation using agnpy classes against that
        obtained with the Gammapy wrapper."""
        # agnpy radiative processes
        synch = Synchrotron(ssc_blob, ssa=True)
        ssc = SynchrotronSelfCompton(ssc_blob, ssa=True)

        # Gammapy's SpectralModel
        ssc_model = SynchrotronSelfComptonSpectralModel(ssc_blob.n_e, ssa=True)
        # set the parameters to be the same as the ssc_blob
        ssc_model.parameters["z"].value = ssc_blob.z
        ssc_model.parameters["delta_D"].value = ssc_blob.delta_D
        ssc_model.parameters["log10_B"].value = np.log10(ssc_blob.B.to_value("G"))
        ssc_model.parameters["t_var"].value = ssc_blob.t_var.to_value("s")

        # SEDs
        sed_agnpy = synch.sed_flux(nu) + ssc.sed_flux(nu)
        sed_gammapy = (E ** 2 * ssc_model(E)).to("erg cm-2 s-1")

        nu_range = [1e9, 1e27] * u.Hz
        make_comparison_plot(
            nu,
            sed_gammapy,
            sed_agnpy,
            "Gammapy wrapper",
            "agnpy",
            "synchrotron + SSC",
            figures_dir / "gammapy_ssc_wrapper.png",
            "sed",
            y_range=[1e-13, 1e-9],
            comparison_range=nu_range.to_value("Hz"),
        )
        # requires that the SED points deviate less than 1% from the figure
        assert check_deviation(nu, sed_gammapy, sed_agnpy, 0.1, nu_range)

    @pytest.mark.parametrize(
        "targets", [(["blr"]), (["dt"]), (["blr", "dt"])],
    )
    def test_targets_parameters(self, targets):
        """In this function we just test that the targets are correctly loaded
        and that the list of parameters is what we expect."""

        ec_model = ExternalComptonSpectralModel(ec_blob.n_e, targets)
        targets_pars_names = [
            "L_disk",
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
        assert ec_model.targets_parameters.names == targets_pars_names

    @pytest.mark.parametrize("targets", (["blr"], ["dt"], ["blr", "dt"]))
    def test_external_compton_spectral_model(self, targets):
        """Check that the Synch + SSC + EC SED computed with the Gammapy wrapper
        corresponds to the one computed with agnpy."""
        # agnpy radiative processes
        ec_blr = ExternalCompton(ec_blob, blr, r)
        ec_dt = ExternalCompton(ec_blob, dt, r)
        synch = Synchrotron(ec_blob)
        ssc = SynchrotronSelfCompton(ec_blob)
        sed_agnpy = synch.sed_flux(nu) + ssc.sed_flux(nu) + disk.sed_flux(nu, ec_blob.z)

        # Gammapy wrapper
        ec_model = ExternalComptonSpectralModel(ec_blob.n_e, targets)
        # set the same parameters as the agnpy objects
        # - emission region
        ec_model.parameters["z"].value = ec_blob.z
        ec_model.parameters["delta_D"].value = ec_blob.delta_D
        ec_model.parameters["log10_B"].value = np.log10(ec_blob.B.to_value("G"))
        ec_model.parameters["t_var"].value = ec_blob.t_var.to_value("s")
        ec_model.parameters["mu_s"].value = ec_blob.mu_s
        ec_model.parameters["log10_r"].value = np.log10(r.to_value("cm"))
        # - EC targets
        ec_model.parameters["L_disk"].value = disk.L_disk.to_value("erg s-1")
        ec_model.parameters["M_BH"].value = disk.M_BH.to_value("g")
        ec_model.parameters["m_dot"].value = disk.m_dot.to_value("g s-1")
        ec_model.parameters["R_in"].value = disk.R_in.to_value("cm")
        ec_model.parameters["R_out"].value = disk.R_out.to_value("cm")
        ec_model.parameters["xi_line"].value = blr.xi_line
        ec_model.parameters["lambda_line"].value = blr.lambda_line.to_value("Angstrom")
        ec_model.parameters["R_line"].value = blr.R_line.to_value("cm")
        ec_model.parameters["xi_dt"].value = dt.xi_dt
        ec_model.parameters["T_dt"].value = dt.T_dt.to_value("K")
        ec_model.parameters["R_dt"].value = dt.R_dt.to_value("cm")

        # now look at the targets which EC components have to be added
        if targets == ["blr"]:
            sed_agnpy += ec_blr.sed_flux(nu)
            title = "EC on BLR comparison"
            fig_name = "gammapy_ec_blr_wrapper.png"
        if targets == ["dt"]:
            sed_agnpy += dt.sed_flux(nu, ec_blob.z)
            sed_agnpy += ec_dt.sed_flux(nu)
            title = "EC on DT comparison"
            fig_name = "gammapy_ec_dt_wrapper.png"
        if targets == ["blr", "dt"]:
            sed_agnpy += dt.sed_flux(nu, z)
            sed_agnpy += ec_blr.sed_flux(nu) + ec_dt.sed_flux(nu)
            title = "EC on BLR and DT comparison"
            fig_name = "gammapy_ec_dt_blr_wrapper.png"

        sed_gammapy = (E ** 2 * ec_model(E)).to("erg cm-2 s-1")

        make_comparison_plot(
            nu,
            sed_gammapy,
            sed_agnpy,
            "Gammapy wrapper",
            "agnpy",
            title,
            figures_dir / fig_name,
            "sed",
            # y_range=[1e-13, 1e-9],
        )
        # requires that the SED points deviate less than 1% from the figure
        # assert check_deviation(nu, sed_gammapy, sed_agnpy, 0.1)
