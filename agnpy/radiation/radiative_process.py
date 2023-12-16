import numpy as np
import astropy.units as u


class RadiativeProcess:
    """Base class for radiative processes. Contains common logic, e.g. calculating SED flux integrals."""
    def integrate_flux(self, nu_min, nu_max, nu_points=50):
        r""" Evaluates the SED flux integral over the span of provided frequencies

        Parameters
        ----------
        nu_min : start frequency (in Hz or equivalent)
        nu_max : end frequency (in Hz or equivalent)
        nu_points: number of points (between nu_min and nu_max) on the log scale
        """
        nu = self._nu_logspace(nu_min, nu_max, nu_points)
        sed_nufnu = self.sed_flux(nu)  # erg / s / cm2 / log(nu)
        sed_fnu = sed_nufnu / nu       # erg / s / cm2 / Hz
        return np.trapz(sed_fnu, nu)   # erg / s / cm2

    def integrate_photon_flux(self, nu_min, nu_max, nu_points=50):
        r""" Evaluates the photon flux integral from SED over the span of provided frequencies

        Parameters
        ----------
        nu_min : start frequency (in Hz or equivalent)
        nu_max : end frequency (in Hz or equivalent)
        nu_points: number of points (between nu_min and nu_max) on the log scale
        """
        nu = self._nu_logspace(nu_min, nu_max, nu_points)
        photon_energy = nu.to("erg", equivalencies=u.spectral())
        sed_nufnu = self.sed_flux(nu)   # erg / s / cm2 / log(nu)
        sed_fnu = sed_nufnu / nu        # erg / s / cm2 / Hz
        n = sed_fnu / photon_energy     # photons / s / cm2 / Hz
        return np.trapz(n, nu)          # photons / s / cm2

    def _nu_logspace(self, nu_min, nu_max, nu_points=50):
        """A thin wrapper around numpy.logspace() function, capable of spectral-equivalency conversion of input parameters."""
        nu_min_hz = nu_min.to(u.Hz, equivalencies=u.spectral())
        nu_max_hz = nu_max.to(u.Hz, equivalencies=u.spectral())
        nu = np.logspace(start=np.log10(nu_min_hz.value), stop=np.log10(nu_max_hz.value), num=nu_points) * u.Hz
        return nu
