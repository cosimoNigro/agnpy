import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
import pytest
from pathlib import Path
from agnpy.emission_regions import Blob
from agnpy.targets import SphericalShellBLR, RingDustTorus
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
R_b = 1e16 * u.cm
B = 0.56 * u.G
z = 1
delta_D = 40
Gamma = 40
ec_blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

# - BLR definition
xi_line = 0.024
L_disk = 1e46 * u.Unit("erg s-1")
R_line = 1e17 * u.cm
blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
# - dust torus definition
T_dt = 1e3 * u.K
csi_dt = 0.1
dt = RingDustTorus(L_disk, csi_dt, T_dt)

# distance
r = 1e18 * u.cm

# frequency / energies for SED computation
nu = np.logspace(9, 29, 100) * u.Hz
E = nu.to("eV", equivalencies=u.spectral())


class TestGammapyWrapper:
    """Test the Gammapy wrappers."""

    def test_spectral_emission_region_parameters(self):
        """Test that the parameters of the spectrum and of the emission region
        can be correctly fetched."""
        ssc_model = SynchrotronSelfComptonSpectralModel(ssc_blob)
        assert ssc_model.spectral_parameters.names == [
            "k_e",
            "p",
            "gamma_min",
            "gamma_max",
        ]
        assert ssc_model.emission_region_parameters.names == [
            "z",
            "d_L",
            "delta_D",
            "B",
            "R_b",
        ]

        # check that, when changed, parameters are updated accordingly
        # in both lists
        k_e = 1e8
        ssc_model.parameters["k_e"].value = 1e8
        assert u.isclose(ssc_model.spectral_parameters["k_e"].value, k_e)

    def test_synchrotron_self_compton_spectral_model(self):
        """Test the SSC model SED computation using agnpy classes against that
        obtained with the Gammapy wrapper."""
        # agnpy
        synch = Synchrotron(ssc_blob, ssa=True)
        ssc = SynchrotronSelfCompton(ssc_blob, ssa=True)

        # Gammapy's SpectralModel
        ssc_model = SynchrotronSelfComptonSpectralModel(ssc_blob, ssa=True)

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
            comparison_range=nu_range.to_value("Hz")
        )
        # requires that the SED points deviate less than 1% from the figure
        assert check_deviation(nu, sed_gammapy, sed_agnpy, 0.1, nu_range)

    @pytest.mark.parametrize(
        "targets, targets_pars_names",
        [
            ([blr], ["L_disk", "xi_line", "epsilon_line", "R_line"]),
            ([dt], ["L_disk", "xi_dt", "epsilon_dt", "R_dt"]),
            (
                [blr, dt],
                [
                    "L_disk",
                    "xi_line",
                    "epsilon_line",
                    "R_line",
                    "xi_dt",
                    "epsilon_dt",
                    "R_dt",
                ],
            ),
        ],
    )
    def test_targets_parameters(self, targets, targets_pars_names):
        """In this function we just test that the targets are correctly loaded
        and that the list of parameters is what we expect."""

        ec_model = ExternalComptonSpectralModel(ec_blob, r, targets)
        assert ec_model.targets_parameters.names == targets_pars_names

    @pytest.mark.parametrize("targets", ([blr], [dt], [blr, dt]))
    def test_external_compton_spectral_model(self, targets):
        """Check that the Synch + SSC + EC SED computed with the Gammapy wrapper
        corresponds to the one computed with agnpy."""
        # Gammapy wrapper
        ec_model = ExternalComptonSpectralModel(ec_blob, r, targets)
        # agnpy radiative processes
        ec_blr = ExternalCompton(ec_blob, blr, r)
        ec_dt = ExternalCompton(ec_blob, dt, r)
        synch = Synchrotron(ec_blob)
        ssc = SynchrotronSelfCompton(ec_blob)
        sed_agnpy = synch.sed_flux(nu) + ssc.sed_flux(nu)

        # now look at the targets which EC components have to be added
        if targets == [blr]:
            sed_agnpy += ec_blr.sed_flux(nu)
            title = "EC on BLR comparison"
            fig_name = "gammapy_ec_blr_wrapper.png"
        if targets == [dt]:
            sed_agnpy += ec_dt.sed_flux(nu)
            title = "EC on DT comparison"
            fig_name = "gammapy_ec_dt_wrapper.png"
        if targets == [blr, dt]:
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
        assert check_deviation(nu, sed_gammapy, sed_agnpy, 0.1)