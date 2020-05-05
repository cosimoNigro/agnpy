import numpy as np
from astropy.constants import h, e, c, m_e, sigma_T
import astropy.units as u


e = e.gauss
mec2 = m_e.to("erg", equivalencies=u.mass_energy())
B_cr = 4.414e13 * u.G  # critical magnetic field
lambda_c = (h / (m_e * c)).to("cm")  # Compton wavelength
# equivalency to transform frequencies to energies in electron rest mass units
epsilon_equivalency = [
    (u.Hz, u.Unit(""), lambda x: h.cgs * x / mec2, lambda x: x * mec2 / h.cgs)
]


__all__ = ["R", "U_B", "Synchrotron"]


def R(x):
    """Eq. 7.45 in [Dermer2009]_, angle-averaged integrand of the radiated power, the 
    approximation of this function, given in Eq. D7 of [Aharonian2010]_, is used.
    """
    term_1_num = 1.808 * np.power(x, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(x, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(x, 2 / 3) + 0.347 * np.power(x, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(x, 2 / 3) + 0.217 * np.power(x, 4 / 3)
    value = term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-x)
    return value


def U_B(B):
    """:math:`U_B` Eq. 7.14 in [DermerMenon2009]_"""
    return (np.power(B, 2) / (8 * np.pi)).to("erg cm-3")


def nu_B(B):
    """:math:`\\nu_B` Eq. 7.19 in [DermerMenon2009]_"""
    return (e * B / (2 * np.pi * m_e * c)).to("Hz")


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
	"""

    def __init__(self, blob, ssa=False):
        self.blob = blob
        self.U_B = U_B(self.blob.B_cgs)
        self.nu_B = nu_B(self.blob.B_cgs)
        self.epsilon_B = (self.blob.B / B_cr).to_value("")
        self.ssa = ssa

    def k_epsilon(self, epsilon):
        """SSA absorption factor Eq. 7.142 in [DermerMenon2009]_.
        The part of the integrand that is dependent on :math:`\gamma` is 
        computed analytically in each of the :class:`~agnpy.spectra` classes."""
        gamma = self.blob.gamma
        SSA_integrand = self.blob.n_e.SSA_integrand(gamma)
        # for multidimensional integration
        # axis 0: electrons gamma
        # axis 1: photons epsilon
        # arrays starting with _ are multidimensional and used for integration
        _gamma = np.reshape(gamma, (gamma.size, 1))
        _SSA_integrand = np.reshape(SSA_integrand, (SSA_integrand.size, 1))
        _epsilon = np.reshape(epsilon, (1, epsilon.size))
        prefactor_P_syn = np.sqrt(3) * np.power(e, 3) * self.blob.B_cgs / h
        prefactor_k_epsilon = (
            -1 / (8 * np.pi * m_e * np.power(epsilon, 2)) * np.power(lambda_c / c, 3)
        )
        x_num = 4 * np.pi * _epsilon * np.power(m_e, 2) * np.power(c, 3)
        x_denom = 3 * e * self.blob.B_cgs * h * np.power(_gamma, 2)
        x = (x_num / x_denom).to_value("")
        integrand = R(x) * _SSA_integrand
        integral = np.trapz(integrand, gamma, axis=0)
        return (prefactor_P_syn * prefactor_k_epsilon * integral).to("cm-1")

    def tau_ssa(self, epsilon):
        """SSA opacity, Eq. before 7.122 in [DermerMenon2009]_"""
        return (2 * self.k_epsilon(epsilon) * self.blob.R_b).to_value("")

    def attenuation_ssa(self, epsilon):
        """SSA attenuation, Eq. 7.122 in [DermerMenon2009]_"""
        tau = self.tau_ssa(epsilon)
        u = 1 / 2 + np.exp(-tau) / tau - (1 - np.exp(-tau)) / np.power(tau, 2)
        attenuation = 3 * u / tau
        condition = tau < 1e-3
        attenuation[condition] = 1
        return attenuation

    def com_sed_emissivity(self, epsilon):
        """Synchrotron  emissivity:

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
            to compute the sed, :math:`\epsilon = h\\nu / (m_e c^2)`

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
        x_num = 4 * np.pi * _epsilon * np.power(m_e, 2) * np.power(c, 3)
        x_denom = 3 * e * self.blob.B_cgs * h * np.power(_gamma, 2)
        x = (x_num / x_denom).to_value("")
        integrand = _N_e * R(x)
        integral = np.trapz(integrand, gamma, axis=0)
        emissivity = (prefactor * integral).to("erg s-1")
        if self.ssa:
            emissivity *= self.attenuation_ssa(epsilon)
        return emissivity.to("erg s-1")

    def sed_luminosity(self, nu):
        """Synchrotron luminosity SED: 

        .. math::
            \\nu L_{\\nu} \, [\mathrm{erg}\,\mathrm{s}^{-1}]

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
        epsilon = nu.to("", equivalencies=epsilon_equivalency)
        # correct epsilon to the jet comoving frame
        epsilon_prime = (1 + self.blob.z) * epsilon / self.blob.delta_D
        prefactor = np.power(self.blob.delta_D, 4)
        return prefactor * self.com_sed_emissivity(epsilon_prime)

    def sed_flux(self, nu):
        """Synchrotron flux SED:

        .. math::
            \\nu F_{\\nu} \, [\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]
		
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
        epsilon = nu.to("", equivalencies=epsilon_equivalency)
        # correct epsilon to the jet comoving frame
        epsilon_prime = (1 + self.blob.z) * epsilon / self.blob.delta_D
        prefactor = np.power(self.blob.delta_D, 4) / (
            4 * np.pi * np.power(self.blob.d_L, 2)
        )
        sed = prefactor * self.com_sed_emissivity(epsilon_prime)
        return sed.to("erg cm-2 s-1")
