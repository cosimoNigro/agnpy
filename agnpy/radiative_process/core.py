# core class describing all radiative processes
import numpy as np
import astropy.units as u


def _nu_logspace(nu_min, nu_max, nu_points=50):
    """A thin wrapper around numpy.logspace() function, capable of
    spectral-equivalency conversion of input parameters."""
    nu_min = nu_min.to(u.Hz, equivalencies=u.spectral())
    nu_max = nu_max.to(u.Hz, equivalencies=u.spectral())
    nu = np.logspace(np.log10(nu_min.value), np.log10(nu_max.value), nu_points) * u.Hz
    return nu


class RadiativeProcess:
    """Base class for all the radiative processes.
    It contains functionalities to handle all the basic flux calculations, e.g.
    conversion to differential energy flux, integration, etc.
    """

    def diff_energy_flux(self, nu):
        """Similar to sed_flux(), but returns the differential energy flux [eV-1 cm-2 s-1]."""
        energy = nu.to("eV", equivalencies=u.spectral())
        nuF_nu = self.sed_flux(nu)
        dphidE = nuF_nu / energy**2
        return dphidE.to("eV-1 cm-2 s-1")

    def energy_flux_integral(self, nu_min, nu_max, nu_points=50):
        """Evaluates the integral energy flux [erg cm-2 s-1] over a range of frequencies.

        Parameters
        ----------
        nu_min : `~astropy.units.Quantity`
            start frequency (in Hz or equivalent)
        nu_max : `~astropy.units.Quantity`
            end frequency (in Hz or equivalent)
        nu_points: int
            number of points (between nu_min and nu_max) in log scale
        """
        nu = _nu_logspace(nu_min, nu_max, nu_points)
        Fnu = self.sed_flux(nu) / nu
        return np.trapz(Fnu, nu).to("erg cm-2 s-1")

    def flux_integral(self, nu_min, nu_max, nu_points=50):
        """Evaluates the integral flux [cm-2 s-1] over a range of frequencies.

        Parameters
        ----------
        nu_min : `~astropy.units.Quantity`
            start frequency (in Hz or equivalent)
        nu_max : `~astropy.units.Quantity`
            end frequency (in Hz or equivalent)
        nu_points: int
            number of points (between nu_min and nu_max) in log scale
        """
        nu = _nu_logspace(nu_min, nu_max, nu_points)
        energy = nu.to("eV", equivalencies=u.spectral())
        dphidE = self.diff_energy_flux(nu)
        return np.trapz(dphidE, energy).to("cm-2 s-1")
