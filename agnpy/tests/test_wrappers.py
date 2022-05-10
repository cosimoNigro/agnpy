import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
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
ec_blob.set_gamma_size(1000)

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

    def test_synchrotron_self_compton_spectral_model(self):
        """Test the SSC model SED computation using agnpy classes against that
        obtained with the Gammapy wrapper."""
        # agnpy
        synch = Synchrotron(ssc_blob, ssa=True)
        ssc = SynchrotronSelfCompton(ssc_blob, ssa=True)

        # Gammapy's SpectralModel
        ssc_model = SynchrotronSelfComptonSpectralModel(ssc_blob, ssa=True)

        # SEDs
        sed_ssc_agnpy = synch.sed_flux(nu) + ssc.sed_flux(nu)
        sed_ssc_gammapy = (E**2 * ssc_model(E)).to("erg cm-2 s-1")

        make_comparison_plot(
            nu,
            sed_ssc_gammapy,
            sed_ssc_agnpy,
            "Gammapy wrapper",
            "agnpy",
            "synchrotron + SSC",
            figures_dir / "gammapy_ssc_wrapper.png",
            "sed",
            y_range=[1e-13, 1e-9],
        )
        # requires that the SED points deviate less than 1% from the figure
        # assert check_deviation(nu, sed_ssc_gammapy, sed_ssc_agnpy, 0.1)

    def test_external_compton_spectral_model(self):
        """Test the EC model SED computation using agnpy classes against that
        obtained with the Gammapy wrapper."""
        # agnpy
        synch = Synchrotron(ec_blob)
        ssc = SynchrotronSelfCompton(ec_blob)
        ec_blr = ExternalCompton(ec_blob, blr, r)
        ec_dt = ExternalCompton(ec_blob, dt, r)

        # Gammapy's SpectralModel
        ec_model = ExternalComptonSpectralModel(ec_blob, blr, dt, r, ec_blr=False)

        # SEDs
        sed_ec_agnpy = synch.sed_flux(nu) + ssc.sed_flux(nu) + ec_dt.sed_flux(nu)
        sed_ec_gammapy = (E**2 * ec_model(E)).to("erg cm-2 s-1")

        make_comparison_plot(
            nu,
            sed_ec_gammapy,
            sed_ec_agnpy,
            "Gammapy wrapper",
            "agnpy",
            "synchrotron + SSC + EC on DT",
            figures_dir / "gammapy_ec_wrapper.png",
            "sed",
            # y_range=[1e-13, 1e-9]
        )
        # requires that the SED points deviate less than 1% from the figure
        # assert check_deviation(nu, sed_ec_gammapy, sed_ec_agnpy, 0.1)
