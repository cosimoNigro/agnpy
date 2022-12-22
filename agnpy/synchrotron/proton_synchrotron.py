# module containing the synchrotron radiative process
import numpy as np
import astropy.units as u
from astropy.constants import e, h, c, m_e, m_p, sigma_T, G
from ..utils.math import axes_reshaper, gamma_e_to_integrate
from ..utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c_p
from agnpy.synchrotron import Synchrotron as syn

__all__ = ["ProtonSynchrotron"]

e = e.gauss
B_cr = 4.414e13 * u.G  # critical magnetic field

class ProtonSynchrotron:
    """Class for synchrotron radiation computation

    Parameters
    ----------
    blob : :class:`~agnpy.emission_region.Blob`
        emitting region and electron distribution
    ssa : bool
        whether or not to consider synchrotron self absorption (SSA).
        The absorption factor will be taken into account in
        :func:`~agnpy.synchrotron.Synchrotron.com_sed_emissivity`, in order to be
        propagated to :func:`~agnpy.synchrotron.Synchrotron.sed_luminosity` and
        :func:`~agnpy.synchrotron.Synchrotron.sed_flux`.
    integrator : func
        function to be used for integration (default = `np.trapz`)
	"""

    def __init__(self, blob, ssa=False, integrator=np.trapz):
        self.blob = blob
        self.ssa = ssa
        self.integrator = integrator

    @staticmethod
    def evaluate_tau_ssa(
        nu,
        z,
        d_L,
        delta_D,
        B,
        R_b,
        n_p,
        *args,
        integrator=np.trapz,
        gamma=gamma_e_to_integrate,
    ):
        """Computes the syncrotron self-absorption opacity for a general set
        of model parameters, see :func:`~agnpy:sycnhrotron.Synchrotron.evaluate_sed_flux`
        for parameters defintion. Eq. before 7.122 in [DermerMenon2009]_."""
        # conversions
        epsilon = nu_to_epsilon_prime(nu, z, delta_D, m = m_p)
        B_cgs = B_to_cgs(B)
        # multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        SSA_integrand = n_p.evaluate_SSA_integrand(_gamma, *args)
        integrand = SSA_integrand * syn.single_particle_synch_power(B_cgs, _epsilon, _gamma, mass = m_p)
        integral = integrator(integrand, gamma, axis=0)
        prefactor_k_epsilon = (
            -1 / (8 * np.pi * m_p * np.power(epsilon, 2)) * np.power(lambda_c_p / c, 3)
        )
        k_epsilon = (prefactor_k_epsilon * integral).to("cm-1")
        return (2 * k_epsilon * R_b).to_value("")

    @staticmethod
    def evaluate_sed_flux(
        nu,
        z,
        d_L,
        delta_D,
        B,
        R_b,
        n_p,
        *args,
        ssa=False,
        integrator=np.trapz,
        gamma=gamma_e_to_integrate,
    ):
        r"""Evaluates the flux SED (:math:`\nu F_{\nu}`) due to synchrotron radiation,
        for a general set of model parameters. Eq. 21 in [Finke2008]_.

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
        epsilon = nu_to_epsilon_prime(nu, z, delta_D, m = m_p)
        B_cgs = B_to_cgs(B)
        # reshape for multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_p.evaluate(_gamma, *args)
        # fold the electron distribution with the synchrotron power
        integrand = N_p * syn.single_particle_synch_power(B_cgs, _epsilon, _gamma, mass=m_p)
        emissivity = integrator(integrand, gamma, axis=0)
        prefactor = np.power(delta_D, 4) / (4 * np.pi * np.power(d_L, 2))
        sed = (prefactor * epsilon * emissivity).to("erg cm-2 s-1")

        if ssa:
            tau = ProtonSynchrotron.evaluate_tau_ssa(
                nu,
                z,
                d_L,
                delta_D,
                B,
                R_b,
                n_e,
                *args,
                integrator=integrator,
                gamma=gamma,
            )
            attenuation = syn.tau_to_attenuation(tau)
            sed *= attenuation

        return sed

    def sed_flux(self, nu):
        r"""Evaluates the synchrotron flux SED for a Synchrotron object built
        from a Blob."""
        return self.evaluate_sed_flux(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.B,
            self.blob.R_b,
            self.blob.n_p,
            *self.blob.n_p.parameters,
            ssa=self.ssa,
            integrator=self.integrator,
            gamma=self.blob.gamma_p,
        )

    def sed_luminosity(self, nu):
        r"""Evaluates the synchrotron luminosity SED
        :math:`\nu L_{\nu} \, [\mathrm{erg}\,\mathrm{s}^{-1}]`
        for a a Synchrotron object built from a blob."""
        sphere = 4 * np.pi * np.power(self.blob.d_L, 2)
        return (sphere * self.sed_flux(nu)).to("erg s-1")

    def sed_peak_flux(self, nu):
        """provided a grid of frequencies nu, returns the peak flux of the SED
        """
        return self.sed_flux(nu).max()

    def sed_peak_nu(self, nu):
        """provided a grid of frequencies nu, returns the frequency at which the
        SED peaks
        """
        idx_max = self.sed_flux(nu).argmax()
        return nu[idx_max]
