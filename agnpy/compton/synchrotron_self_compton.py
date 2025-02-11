# module containing the synchrotron self Compton radiative process
import numpy as np
from astropy.constants import c, sigma_T, m_e

from .kernels import isotropic_kernel
from .. import Blob
from ..synchrotron import Synchrotron
from ..utils.math import (
    axes_reshaper,
    gamma_e_to_integrate,
    nu_to_integrate, log
)
from ..utils.conversion import nu_to_epsilon_prime
from ..radiative_process import RadiativeProcess

__all__ = ["SynchrotronSelfCompton"]


class SynchrotronSelfCompton(RadiativeProcess):
    """class for Synchrotron Self Compton radiation computation

    Parameters
    ----------
    blob : :class:`~agnpy.emission_region.Blob`
        emission region and electron distribution hitting the photon target
    synchrotron : :class:`~agnpy.synchrotron.Synchrotron`
        class describing the synchrotron photons target
    integrator : func
        function to be used for integration (default = `np.trapz`)
    """

    def __init__(self, blob, ssa=False, integrator=np.trapz):
        self.blob = blob
        self.ssa = ssa
        self.integrator = integrator

    def electron_energy_loss_rate_thomson(self, gamma):
        return self._electron_energy_loss_thomson_formula_prefactor() * gamma ** 2

    def _electron_energy_loss_thomson_formula_prefactor(self):
        return ((4 / 3) * sigma_T * c * self.blob.u_ph_synch).to("erg/s")

    def electron_energy_loss_rate(self, gamma):
        epsilon = nu_to_epsilon_prime(nu_to_integrate, self.blob.z, self.blob.delta_D)
        sed_flux_synch = Synchrotron.evaluate_sed_flux(
            nu_to_integrate,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.B,
            self.blob.R_b,
            self.blob.n_e,
            *self.blob.n_e.parameters,
            gamma=gamma_e_to_integrate
        )

        # differential energy density of photons
        u_epsilon = self.blob.u_ph_synch_diff(epsilon, sed_flux_synch)

        # reshape for multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)

        # formulas from paragraph 2.1, [Moderski2005]
        b = 4 * _gamma * _epsilon # matrix [gamma x epsilon] - b < 1 means Thomson regime, b > 1 means K-N regime
        # convert b value to the K-N correction parameter:
        f_KN = np.where(b < 1e4, 1 / (1 + b) ** 1.5, (9 / (2 * b**2)) * (log(b) - 11/6))
        # fold K-N correction parameters with the corresponding photon energy densities:
        F_KN = np.trapz(f_KN * u_epsilon, epsilon)
        # F_KN is basically total energy density corrected to KN regime for each gamma value
        energyLossRate = (4 / 3) * sigma_T * c * F_KN * gamma ** 2
        return energyLossRate.to("erg/s")

    @staticmethod
    def evaluate_sed_flux(
        nu,
        z,
        d_L,
        delta_D,
        B,
        R_b,
        n_e,
        *args,
        ssa=False,
        integrator=np.trapz,
        gamma=gamma_e_to_integrate,
    ):
        r"""Evaluates the flux SED (:math:`\nu F_{\nu}`) for synchrotron self-Compton,
        for a general set of model parameters. Eq. 9 in [Finke2008]_.

        **Note** parameters after \*args need to be passed with a keyword

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed 
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        d_L : :class:`~astropy.units.Quantity` 
            luminosity distance of the source
        delta_D : float
            Doppler factor of the relativistic outflow
        B : :class:`~astropy.units.Quantity`
            magnetic field in the blob 
        R_b : :class:`~astropy.units.Quantity`
            size of the emitting region (spherical blob assumed)
        n_e : :class:`~agnpy.spectra.ElectronDistribution`
            electron energy distribution
        \*args
            parameters of the electron energy distribution (k_e, p, ...)
        ssa : bool
            whether to consider or not the self-absorption, default false
        integrator : func
            which function to use for integration, default `numpy.trapz`
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to integrate the electron 
            distribution

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        # conversions
        # synchrotron frequencies to be integrated over
        epsilon = nu_to_epsilon_prime(nu_to_integrate, z, delta_D)
        # frequencies of the final sed
        epsilon_s = nu_to_epsilon_prime(nu, z, delta_D)
        sed_synch = Synchrotron.evaluate_sed_flux(
            nu_to_integrate,
            z,
            d_L,
            delta_D,
            B,
            R_b,
            n_e,
            *args,
            ssa=ssa,
            integrator=integrator,
            gamma=gamma,
        )
        u_synch = Blob.evaluate_u_ph_synch_diff(R_b, d_L, delta_D, epsilon, sed_synch)
        # multidimensional integration
        _gamma, _epsilon, _epsilon_s = axes_reshaper(gamma, epsilon, epsilon_s)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_e.evaluate(_gamma, *args)
        # reshape u as epsilon
        _u_synch = np.reshape(u_synch, (1, u_synch.size, 1))
        # integrate
        kernel = isotropic_kernel(_gamma, _epsilon, _epsilon_s)
        integrand = (
            _u_synch / np.power(_epsilon, 2) * N_e / np.power(_gamma, 2) * kernel
        )
        integral_gamma = integrator(integrand, gamma, axis=0)
        integral_epsilon = integrator(integral_gamma, epsilon, axis=0).reshape(epsilon_s.shape)
        emissivity = 3 / 4 * c * sigma_T * np.power(epsilon_s, 2) * integral_epsilon
        prefactor = np.power(delta_D, 4) / (4 * np.pi * np.power(d_L, 2))
        return (prefactor * emissivity).to("erg cm-2 s-1")

    def sed_flux(self, nu):
        """Evaluates the SSC flux SED for a SynchrotronSelfComtpon 
        object built from a Blob."""
        return self.evaluate_sed_flux(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.B,
            self.blob.R_b,
            self.blob.n_e,
            *self.blob.n_e.parameters,
            ssa=self.ssa,
            integrator=self.integrator,
            gamma=self.blob.gamma_e,
        )

    def sed_luminosity(self, nu):
        r"""Evaluates the SSC luminosity SED
        :math:`\nu L_{\nu} \, [\mathrm{erg}\,\mathrm{s}^{-1}]`
        for a a SynchrotronSelfCompton object built from a blob."""
        sphere = 4 * np.pi * np.power(self.blob.d_L, 2)
        return (sphere * self.sed_flux(nu)).to("erg s-1")

    def sed_peak_flux(self, nu):
        """provided a grid of frequencies nu, returns the peak flux of the SED
        """
        return self.sed_flux(nu).max()

    def sed_peak_nu(self, nu):
        """provided a grid of frequencies nu, returns the frequency at which the SED peaks
        """
        idx_max = self.sed_flux(nu).argmax()
        return nu[idx_max]
