import numpy as np
import math
import astropy.units as u
from astropy.constants import m_e, h
from astropy.coordinates import Distance
from agnpy.spectra import PowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.radiation.radiative_process import RadiativeProcess


def _estimate_powerlaw_integral(x_values, y_values):
    slope, intercept = np.polyfit(np.log10(x_values.value), np.log10(y_values.value), 1)
    power = slope + 1
    estimated_integral = (10 ** intercept) * (
            x_values[-1].value ** power / power - x_values[0].value ** power / power)
    return estimated_integral


def _sample_synchrotron_model():
    """The model based on the synchrotron example from the agnpy documentation."""
    R_b = 1e16 * u.cm
    n_e = PowerLaw.from_total_energy(1e48 * u.erg, 4 / 3 * np.pi * R_b ** 3, p=2.8, gamma_min=1e2,
                                     gamma_max=1e7, mass=m_e)
    blob = Blob(R_b, z=Distance(1e27, unit=u.cm).z, delta_D=10, Gamma=10, B=1 * u.G, n_e=n_e)
    synch = Synchrotron(blob)
    return synch


class FlatSedGenerator(RadiativeProcess):
    """A dummy generator returning flat (constant) flux, of the same value across the whole spectrum."""
    def __init__(self, flat_value):
        self.flat_value = flat_value

    def sed_flux(self, nu):
        return np.full(fill_value=self.flat_value, shape=nu.shape, dtype=np.float64) * (
                    u.erg / (u.cm ** 2 * u.s * u.Hz)) * nu


class TestSedIntegration:

    def test_flat_integral(self):
        """Integrate over flat SED (Fnu equal to 1.0 for all frequencies)."""
        nu_min = 10 * u.Hz
        nu_max = 10 ** 20 * u.Hz
        flux = FlatSedGenerator(1.0).integrate_flux(nu_min, nu_max)
        assert flux == 1e+20 * u.erg / (u.cm ** 2 * u.s)


    def test_synchrotron_energy_flux_integral(self):
        """Integrate over sample synchrotron flux and compare with the value estimated using the power law integral."""
        synch = _sample_synchrotron_model()

        nu_min_log = 13
        nu_max_log = 20
        nu_min = 10 ** nu_min_log * u.Hz
        nu_max = 10 ** nu_max_log * u.Hz

        nu = np.logspace(nu_min_log, nu_max_log) * u.Hz
        Fnu = synch.sed_flux(nu) / nu
        estimated_integral = _estimate_powerlaw_integral(nu, Fnu)

        actual_integral = synch.integrate_flux(nu_min, nu_max).value

        assert math.isclose(actual_integral, estimated_integral, rel_tol=0.05)

    def test_synchrotron_photon_flux_integral(self):
        """Integrate over sample synchrotron photon flux and compare with the value estimated using the power law integral."""
        synch = _sample_synchrotron_model()

        nu_min_log = 13
        nu_max_log = 20
        nu_min = 10 ** nu_min_log * u.Hz
        nu_max = 10 ** nu_max_log * u.Hz

        nu = np.logspace(nu_min_log, nu_max_log) * u.Hz
        Fnu = synch.sed_flux(nu) / nu
        # convert the energy flux to photon flux in photons / s / cm2 / eV
        photons_per_Hz = Fnu / nu.to("erg", equivalencies=u.spectral())  # Fnu is in ergs so must use the same unit
        h_in_eV = h.to("eV/Hz")
        energies_eV = nu * h_in_eV
        photons_per_eV = photons_per_Hz / h_in_eV

        estimated_integral = _estimate_powerlaw_integral(energies_eV, photons_per_eV)

        actual_integral = synch.integrate_photon_flux(nu_min, nu_max).value

        assert math.isclose(actual_integral, estimated_integral, rel_tol=0.05)
