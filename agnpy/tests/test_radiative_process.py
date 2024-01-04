import numpy as np
import astropy.units as u
from astropy.constants import m_e
from astropy.coordinates import Distance
from agnpy.spectra import PowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.radiative_process import RadiativeProcess


def estimate_powerlaw_integral(x, y):
    """Fit a power law to the data, then estimate its integral."""
    index, log10_norm = np.polyfit(np.log10(x), np.log10(y), 1)
    norm = 10**log10_norm
    integral_index = index + 1
    integral = (
        norm / (integral_index) * (x[-1] ** integral_index - x[0] ** integral_index)
    )
    return integral


def create_synchrotron_model():
    """A simple synchrotron model to be used for test purpoises.
    Based on the synchrotron example from the documentation."""
    R_b = 1e16 * u.cm
    V_b = 4 / 3 * np.pi * R_b**3
    z = Distance(1e27, unit=u.cm).z
    B = 1 * u.G
    delta_D = 10
    n_e = PowerLaw.from_total_energy(
        1e48 * u.erg, V_b, p=2.8, gamma_min=1e2, gamma_max=1e7, mass=m_e
    )
    blob = Blob(R_b=R_b, z=z, delta_D=delta_D, Gamma=delta_D, B=B, n_e=n_e)
    synch = Synchrotron(blob)
    return synch


class FlatFNuSedGenerator(RadiativeProcess):
    """A dummy generator returning a flat (constant) SED in F_nu."""

    def __init__(self, F_nu):
        # the constant F_nu value
        self.F_nu = F_nu

    def sed_flux(self, nu):
        F_nu = np.ones_like(nu).value * self.F_nu
        return (nu * F_nu).to("erg cm-2 s-1")


class TestSedIntegration:
    def test_flat_energy_flux_integral(self):
        """Integrate over flat SED (Fnu equal to 1.0 for all frequencies)."""
        nu_min = 10 * u.Hz
        nu_max = 1e9 * u.Hz
        F_nu = 1 * u.Unit("erg cm-2 s-1 Hz-1")
        energy_flux_integral = FlatFNuSedGenerator(F_nu).energy_flux_integral(
            nu_min, nu_max
        )
        assert u.isclose(energy_flux_integral, 1e9 * u.Unit("erg cm-2 s-1"))

    def test_synchrotron_integrals(self):
        """Create an example synchrotron SED, fit the flux with a power-law part
        and compare its integral."""
        synch = create_synchrotron_model()
        nu = np.logspace(13, 20, 50) * u.Hz
        energy = nu.to("eV", equivalencies=u.spectral())

        energy_flux_integral = synch.energy_flux_integral(nu[0], nu[-1]).value
        flux_integral = synch.flux_integral(nu[0], nu[-1]).value

        F_nu = synch.sed_flux(nu) / nu
        dphidE = synch.diff_energy_flux(nu)

        energy_flux_integral_from_fit = estimate_powerlaw_integral(nu.value, F_nu.value)
        flux_integral_from_fit = estimate_powerlaw_integral(energy.value, dphidE.value)

        assert u.isclose(energy_flux_integral, energy_flux_integral_from_fit, rtol=5e-2)
        assert u.isclose(flux_integral, flux_integral_from_fit, rtol=5e-2)
