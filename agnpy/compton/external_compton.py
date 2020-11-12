# module containing the External Compton radiative process
import numpy as np
from astropy.constants import c, sigma_T
from ..utils.math import trapz_loglog, log, axes_reshaper
from ..utils.conversion import nu_to_epsilon_prime
from ..targets import (
    CMB,
    PointSourceBehindJet,
    SSDisk,
    SphericalShellBLR,
    RingDustTorus,
)
from .kernels import isotropic_kernel, compton_kernel

__all__ = ["ExternalCompton"]

# default arrays to be used for integration
gamma_to_integrate = np.logspace(1, 9, 200)
mu_to_integrate = np.linspace(-1, 1, 100)
phi_to_integrate = np.linspace(0, 2 * np.pi, 50)


class ExternalCompton:
    """class for External Compton radiation computation
    Parameters
    ----------
    blob : :class:`~agnpy.emission_region.Blob`
        emission region and electron distribution hitting the photon target
    target : :class:`~agnpy.targets`
        class describing the target photon field    
    r : :class:`~astropy.units.Quantity`
        distance of the blob from the Black Hole (i.e. from the target photons)
    """

    def __init__(self, blob, target, r=None):
        self.blob = blob
        # we integrate on a larger grid to account for the transformation
        # of the electron density in the reference frame of the BH
        self.gamma = self.blob.gamma_to_integrate
        self.target = target
        self.r = r
        self.set_mu()
        self.set_phi()

    def set_mu(self, mu_size=100):
        self.mu_size = mu_size
        if isinstance(self.target, SSDisk):
            # in case of hte disk the mu interval does not go from -1 to 1
            r_tilde = (self.r / self.target.R_g).to_value("")
            self.mu = self.target.mu_from_r_tilde(r_tilde)
        else:
            self.mu = np.linspace(-1, 1, self.mu_size)

    def set_phi(self, phi_size=50):
        self.phi_size = phi_size
        self.phi = np.linspace(0, 2 * np.pi, self.phi_size)

    @staticmethod
    def evaluate_sed_flux_iso_mono(
        nu,
        z,
        d_L,
        delta_D,
        mu_s,
        R_b,
        epsilon_0,
        u_0,
        n_e,
        *args,
        integrator=np.trapz,
        gamma=gamma_to_integrate,
        mu=mu_to_integrate,
        phi=phi_to_integrate
    ):
        r"""Evaluates the flux SED 
        :math:`\nu F_{\nu} \, [\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]`
        for External Compton on a monochromatic isotropic target photon field
        for a general set of model parameters

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed 
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        d_L : :class:`~astropy.units.Quantity` 
            luminosity distance of the source
        delta_D: float
            Doppler factor of the relativistic outflow
        mu_s : float
            cosine of the angle between the blob motion and the jet axis
        R_b : :class:`~astropy.units.Quantity`
            size of the emitting region (spherical blob assumed)
        epsilon_0 : float
            dimensionless energy (in electron rest mass energy units) of the
            target photon field
        u_0 : :class:`~astropy.units.Quantity`
            energy density [erg cm-3] of the target photon field
        n_e : :class:`~agnpy.spectra.ElectronDistribution`
            electron energy distribution
        *args
            parameters of the electron energy distribution (k_e, p, ...)
        ssa : bool
            whether to consider or not the self-absorption, default false
        integrator : func
            which function to use for integration, default `numpy.trapz`
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to integrate the electron 
            distribution
        mu, phi : :class:`~numpy.ndarray`
            arrays of cosine of zenith and azimuth angles to integrate over

        **Note** arguments after *args are keyword-only arguments

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        # conversion
        epsilon_s = nu_to_epsilon_prime(nu, z, delta_D)
        # multi-dimensional integration
        _gamma, _mu, _phi, _epsilon_s = axes_reshaper(gamma, mu, phi, epsilon_s)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_e.evaluate(_gamma / delta_D, *args)
        kernel = compton_kernel(_gamma, _epsilon_s, epsilon_0, mu_s, _mu, _phi)
        integrand = N_e / np.power(_gamma, 2) * kernel
        integral_gamma = integrator(integrand, gamma, axis=0)
        integral_mu = np.trapz(integral_gamma, mu, axis=0)
        integral = np.trapz(integral_mu, phi, axis=0)
        prefactor_num = (
            3 * c * sigma_T * u_0 * np.power(epsilon_s, 2) * np.power(delta_D, 3)
        )
        prefactor_denom = (
            np.power(2, 7)
            * np.power(np.pi, 2)
            * np.power(d_L, 2)
            * np.power(epsilon_0, 2)
        )
        sed = (prefactor_num / prefactor_denom * integral).to("erg cm-2 s-1")
        return sed

    def sed_flux_cmb(self, nu):
        return self.evaluate_sed_flux_iso_mono(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.mu_s,
            self.blob.R_b,
            self.target.epsilon_0,
            self.target.u_0,
            self.blob.n_e,
            *self.blob.n_e.parameters,
            gamma=gamma_to_integrate,
            mu=mu_to_integrate,
            phi=phi_to_integrate
        )

    @staticmethod
    def evaluate_sed_flux_ps_behind_jet(
        nu,
        z,
        d_L,
        delta_D,
        mu_s,
        R_b,
        epsilon_0,
        L_0,
        r,
        n_e,
        *args,
        integrator=np.trapz,
        gamma=gamma_to_integrate,
        mu=mu_to_integrate,
        phi=phi_to_integrate
    ):
        r"""Evaluates the flux SED 
        :math:`\nu F_{\nu} \, [\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]`
        for External Compton on a monochromatic point-like source behind the jet
        for a general set of model parameters.

        same parameters as in :func:`~agnpy.ExternalCompton.evaluate_sed_flux_iso_mono`
        """
        # conversion
        epsilon_s = nu_to_epsilon_prime(nu, z, delta_D)
        # multi-dimensional integration
        _gamma, _epsilon_s = axes_reshaper(gamma, epsilon_s)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_e.evaluate(_gamma / delta_D, *args)
        kernel = compton_kernel(_gamma, _epsilon_s, epsilon_0, mu_s, 1, 0)
        integrand = N_e / np.power(_gamma, 2) * kernel
        integral = integrator(integrand, gamma, axis=0)
        prefactor_num = (
            3 * sigma_T * L_0 * np.power(epsilon_s, 2) * np.power(delta_D, 3)
        )
        prefactor_denom = (
            np.power(2, 7)
            * np.power(np.pi, 2)
            * np.power(d_L, 2)
            * np.power(r, 2)
            * np.power(epsilon_0, 2)
        )
        sed = (prefactor_num / prefactor_denom * integral).to("erg cm-2 s-1")
        return sed

    def sed_flux_ps_behind_jet(self, nu):
        return self.evaluate_sed_flux_ps_behind_jet(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.mu_s,
            self.blob.R_b,
            self.target.epsilon_0,
            self.target.L_0,
            self.r,
            self.blob.n_e,
            *self.blob.n_e.parameters,
            integrator=np.trapz,
            gamma=gamma_to_integrate,
            mu=mu_to_integrate,
            phi=phi_to_integrate
        )

    def sed_flux(self, nu):
        """EC flux SED"""
        if isinstance(self.target, CMB):
            return self.sed_flux_cmb(nu)
        if isinstance(self.target, PointSourceBehindJet):
            return self.sed_flux_ps_behind_jet(nu)
