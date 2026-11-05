# module containing the synchrotron self Compton radiative process from two interacting nested blobs
import numpy as np
import astropy.units as u
from astropy.constants import c, sigma_T

from .kernels import isotropic_kernel
from ..synchrotron import Synchrotron
from ..utils.math import (
    axes_reshaper,
    gamma_e_to_integrate,
    nu_to_integrate,
)
from ..utils.conversion import nu_to_epsilon_prime
from ..radiative_process import RadiativeProcess

__all__ = ["InteractingBlobs"]


class InteractingBlobs(RadiativeProcess):
    """class for the computation of the inverse Compton radiation from the interaction of two nested blobs,
    where the target low-energy synchrotron photons are produced by one blob and up-scattered by the electrons of the other blob

    Parameters
    ----------
    blob_e : :class:`~agnpy.emission_region.Blob`
        emission region and electron distribution upscattering the low-energy synchrotron photon density
    blob_t : :class:`~agnpy.emission_region.Blob`
        emission region and electron distribution producing the synchrotron photon targets for the inverse Compton scattering
    integrator : func
        function to be used for integration (default = `np.trapz`)
    """

    def __init__(self, blob_e, blob_t, integrator=np.trapz):
        self.blob_e = blob_e
        self.blob_t = blob_t
        self.integrator = integrator
        assert np.isclose(
            self.blob_e.z, self.blob_t.z, atol=0.0001, rtol=0
        ), "The two blobs must be at the same redshift"

    def relative_Gamma_factor(self):
        """Relative Gamma factor between the two blobs, Eq. 1 in [Ghisellini2005]_"""
        Gamma_e = self.blob_e.Gamma
        Gamma_t = self.blob_t.Gamma
        Beta_e = self.blob_e.Beta
        Beta_t = self.blob_t.Beta
        return Gamma_e * Gamma_t * (1 - Beta_e * Beta_t)

    def relative_Beta_factor(self):
        """Relative Beta factor between the two blobs"""
        Beta_e = self.blob_e.Beta
        Beta_t = self.blob_t.Beta
        return (Beta_e - Beta_t) * (1 - Beta_e * Beta_t)

    def sed_flux(self, nu):
        """Compute the SED flux of the inverse Compton radiation from the interaction of two nested blobs

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed
            **note** these are observed frequencies (observer frame)

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """

        R_min = min([self.blob_t.R_b, self.blob_e.R_b])
        z = self.blob_t.z
        d_L = self.blob_t.d_L

        # energy density of synchrotron photons target, in the comoving frame of blob_t
        synch = Synchrotron(self.blob_t)
        sed_synch = synch.sed_flux(nu_to_integrate)
        R_b = self.blob_t.R_b
        delta_D_t = self.blob_t.delta_D
        epsilon = nu_to_epsilon_prime(nu_to_integrate, z, delta_D_t)
        epsilon_s = nu_to_epsilon_prime(nu, z, delta_D_t)
        u_synch = (3 * np.power(d_L, 2) * sed_synch) / (
            c * np.power(R_b, 2) * np.power(delta_D_t, 4) * epsilon
        )
        u_synch *= 3 / 4

        # transformation from blob_t to blob_e frame
        Gamma_r = self.relative_Gamma_factor()
        Beta_r = self.relative_Beta_factor()
        delta_D_r = 1.0 / (Gamma_r * (1 - Beta_r))
        epsilon_s /= delta_D_r

        # compute the SED of the inverse Compton radiation by the electrons of blob_e
        gamma = gamma_e_to_integrate
        _gamma, _epsilon, _epsilon_s = axes_reshaper(gamma, epsilon, epsilon_s)
        n_e = self.blob_e.n_e
        delta_D_e = self.blob_e.delta_D
        V_b = 4 / 3 * np.pi * np.power(R_min, 3)
        N_e = V_b * n_e(_gamma)
        _u_synch = np.reshape(u_synch, (1, u_synch.size, 1))
        kernel = isotropic_kernel(_gamma, _epsilon, _epsilon_s)
        integrand = (
            _u_synch / np.power(_epsilon, 2) * N_e / np.power(_gamma, 2) * kernel
        )
        integral_gamma = self.integrator(integrand, gamma, axis=0)
        integral_epsilon = self.integrator(integral_gamma, epsilon, axis=0)
        emissivity = 3 / 4 * c * sigma_T * np.power(epsilon_s, 2) * integral_epsilon
        prefactor = np.power(delta_D_e, 4) / (4 * np.pi * np.power(d_L, 2))
        interaction = (prefactor * emissivity).to("erg cm-2 s-1")
        return interaction
