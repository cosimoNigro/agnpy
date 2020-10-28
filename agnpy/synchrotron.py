# module containing the synchrotron radiative processes
import numpy as np
import astropy.units as u
from astropy.constants import e, h, c, m_e, sigma_T
from .spectra import PowerLaw
from .utils.math import trapz_loglog, axes_reshaper
from .utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c

e = e.gauss
B_cr = 4.414e13 * u.G  # critical magnetic field
# default gamma grid to be used for integration
gamma_to_integrate = np.logspace(1, 9, 200)

__all__ = ["R", "nu_synch_peak", "Synchrotron"]


def R(x):
    """Eq. 7.45 in [Dermer2009]_, angle-averaged integrand of the radiated power, the 
    approximation of this function, given in Eq. D7 of [Aharonian2010]_, is used.
    """
    term_1_num = 1.808 * np.power(x, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(x, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(x, 2 / 3) + 0.347 * np.power(x, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(x, 2 / 3) + 0.217 * np.power(x, 4 / 3)
    return term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-x)


def nu_synch_peak(B, gamma):
    """observed peak frequency for monoenergetic electrons 
    Eq. 7.19 in [DermerMenon2009]_"""
    B = B_to_cgs(B)
    nu_peak = (e * B / (2 * np.pi * m_e * c)) * np.power(gamma, 2)
    return nu_peak.to("Hz")


def calc_x(B_cgs, epsilon, gamma):
    """ratio of the frequency to the critical synchrotron frequency from 
    Eq. 7.34 in [DermerMenon2009]_, argument of R(x),
    note B has to be in cgs Gauss units"""
    x = (
        4
        * np.pi
        * epsilon
        * np.power(m_e, 2)
        * np.power(c, 3)
        / (3 * e * B_cgs * np.power(gamma, 2))
    )
    return x.to_value("")


def epsilon_B(B):
    r""":math:`\epsilon_B`, Eq. 7.21 [DermerMenon2009]_"""
    return (B / B_cr).to_value("")


def single_electron_synch_power(B_cgs, epsilon, gamma):
    """angle-averaged synchrotron power for a single electron, 
    to be folded with the electron distribution    
    """
    x = calc_x(B_cgs, epsilon, gamma)
    prefactor = np.sqrt(3) * np.power(e, 3) * B_cgs / h
    return prefactor * R(x)


class Synchrotron:
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
    integrator : (`~agnpy.math.utils.trapz_loglog`, `~numpy.trapz`)
        function to be used for the integration
	"""

    def __init__(self, blob=None, ssa=False, integrator=np.trapz):
        self.blob = blob
        self.epsilon_B = (self.blob.B / B_cr).to_value("")
        self.ssa = ssa
        self.integrator = integrator

    def _evaluate_sed_flux(nu, z, d_L, delta_D, B, R_b, gamma, integrator, n_e, *args):
        """evaluate the synchrotron SED for a general set of model parameters
        """
        # conversions
        epsilon = nu_to_epsilon_prime(nu, z, delta_D)
        B_cgs = B_to_cgs(B)
        # multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_e.evaluate(_gamma, *args)
        # fold the electron distribution with the synchrotron power
        integrand = N_e * single_electron_synch_power(B_cgs, _epsilon, _gamma)
        emissivity = integrator(integrand, _gamma, axis=0)
        sed = np.power(delta_D, 4) / (4 * np.pi * np.power(delta_L, 2)) * emissivity
        return sed.to("erg cm-2 s-1")

    def _evaluate_tau_ssa(nu, z, d_L, delta_D, B, R_b, gamma, integrator, n_e, *args):
        pass

    def tau_ssa(self, epsilon):
        """SSA opacity, Eq. before 7.122 in [DermerMenon2009]_
        since we will have formulas dividing by tau, avoid 0 or very small 
        float values, replacing them with 1e-99"""
        tau = (2 * self.k_epsilon(epsilon) * self.blob.R_b).to_value("")
        tau[tau < 1e-99] = 1e-99
        return tau

    def attenuation_ssa(self, epsilon):
        """SSA attenuation, Eq. 7.122 in [DermerMenon2009]_"""
        tau = self.tau_ssa(epsilon)
        u = 1 / 2 + np.exp(-tau) / tau - (1 - np.exp(-tau)) / np.power(tau, 2)
        attenuation = 3 * u / tau
        condition = tau < 1e-3
        attenuation[condition] = 1
        return attenuation

    def com_sed_emissivity(self, epsilon):
        r"""Synchrotron  emissivity:

        .. math::
            \epsilon'\,J'_{\mathrm{syn}}(\epsilon')\,[\mathrm{erg}\,\mathrm{s}^{-1}]

        Eq. 7.116 in [DermerMenon2009]_ or Eq. 18 in [Finke2008]_.

        The **SSA** is taken into account by this function and propagated
        to the other ones computing SEDs by invoking this one. 

        **Note:** This emissivity is computed in the co-moving frame of the blob.
        When calling this function from another, these energies
        have to be transformed in the co-moving frame of the plasmoid.
        
        Parameters
        ----------
        epsilon : :class:`~numpy.ndarray`
            array of dimensionless energies (in electron rest mass units) 
            to compute the sed, :math:`\epsilon = h \nu / (m_e c^2)`

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the emissivity corresponding to each dimensionless energy
        """
        gamma = self.blob.gamma
        N_e = self.blob.N_e(gamma)
        prefactor = np.sqrt(3) * epsilon * np.power(e, 3) * self.blob.B_cgs / h
        # for multidimensional integration
        # axis 0: electrons gamma
        # axis 1: photons epsilon
        # arrays starting with _ are multidimensional and used for integration
        _gamma = np.reshape(gamma, (gamma.size, 1))
        _N_e = np.reshape(N_e, (N_e.size, 1))
        _epsilon = np.reshape(epsilon, (1, epsilon.size))
        print("_gamma shape: ", _gamma.shape)
        print("_epsilon shape: ", _epsilon.shape)
        print("_N_e shape: ", _N_e.shape)
        x_num = 4 * np.pi * _epsilon * np.power(m_e, 2) * np.power(c, 3)
        x_denom = 3 * e * self.blob.B_cgs * h * np.power(_gamma, 2)
        x = (x_num / x_denom).to_value("")
        integrand = _N_e * R(x)
        print("R(x) shape: ", R(x).shape)
        integral = trapz_loglog(integrand, gamma, axis=0)
        emissivity = (prefactor * integral).to("erg s-1")
        if self.ssa:
            emissivity *= self.attenuation_ssa(epsilon)
        return emissivity.to("erg s-1")

    def sed_luminosity(self, nu):
        r"""Synchrotron luminosity SED: 

        .. math::
            \nu L_{\nu} \, [\mathrm{erg}\,\mathrm{s}^{-1}]

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed, **note** these are 
            observed frequencies (observer frame).

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        epsilon_prime = nu_to_epsilon_prime(nu, self.blob.z, self.blob.delta_D)
        prefactor = np.power(self.blob.delta_D, 4)
        return prefactor * self.com_sed_emissivity(epsilon_prime)

    def sed_flux(self, nu):
        r"""Synchrotron flux SED:

        .. math::
            \nu F_{\nu} \, [\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]

        Eq. 21 in [Finke2008]_.

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed, **note** these are 
            observed frequencies (observer frame).

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        epsilon_prime = nu_to_epsilon_prime(nu, self.blob.z, self.blob.delta_D)
        prefactor = np.power(self.blob.delta_D, 4) / (
            4 * np.pi * np.power(self.blob.d_L, 2)
        )
        sed = prefactor * self.com_sed_emissivity(epsilon_prime)
        return sed.to("erg cm-2 s-1")

    def sed_flux_delta_approx(self, nu):
        """synchrotron flux SED using the delta approximation for the synchrotron
        radiation Eq. 7.70 [DermerMenon2009]_"""
        epsilon_prime = nu_to_epsilon_prime(nu, self.blob.z, self.blob.delta_D)
        gamma_s = np.sqrt(epsilon_prime / epsilon_B(self.blob.B))
        prefactor = (
            np.power(self.blob.delta_D, 4)
            / (6 * np.pi * np.power(self.blob.d_L, 2))
            * c
            * sigma_T
            * self.blob.U_B
        )
        value = prefactor * np.power(gamma_s, 3) * self.blob.N_e(gamma_s)
        return value.to("erg cm-2 s-1")

    def sed_peak_flux(self, nu):
        """provided a grid of frequencies nu, returns the peak flux of the SED
        """
        return self.sed_flux(nu).max()

    def sed_peak_nu(self, nu):
        """provided a grid of frequencies nu, returns the frequency at which the SED peaks
        """
        idx_max = self.sed_flux(nu).argmax()
        return nu[idx_max]

    @staticmethod
    def evaluate_sed_flux(
        nu,
        z,
        d_L,
        delta_D,
        B,
        R_b,
        gamma=gamma_to_integrate,
        integrator=np.trapz,
        electron_distribution=PowerLaw,
        *args
    ):
        """evaluate the synchrotron SED

        Parameters
        ----------
        epsilon: :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed, **note** these are 
            observed frequencies (observer frame).
        B: :class:`~astropy.units.Quantity`
            magnetic field
        integrator: function
            integrating function to be used
        electron_distribution: :class:`~agnpy.spectra`
            electron distribution function
        args: :class:`~astropy.units.Quantity`
            arguments of the electron distribution function
        """
        # conversions
        B = B_to_cgs(B)
        epsilon = nu_to_epsilon_prime(nu, z, delta_D)
        # multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        V_b = 4 / 3 * np.pi * R_b ** 3
        N_e = V_b * electron_distribution.evaluate(_gamma, *args)
        x = (
            4
            * np.pi
            * _epsilon
            * m_e ** 2
            * c ** 3
            / (3 * e * B * h * np.power(_gamma, 2))
        ).to_value("")
        integrand = N_e * R(x)
        integral = integrator(integrand, _gamma, axis=0)
        emissivity = np.sqrt(3) * epsilon * e ** 3 * B / h * integral
        sed = delta_D ** 4 / (4 * np.pi * d_L ** 2) * emissivity
        return sed.to("erg cm-2 s-1")
