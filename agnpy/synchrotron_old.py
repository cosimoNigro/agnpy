import numpy as np
import astropy.constants as const
import astropy.units as u
from .compton import simple_kernel

SIGMA_T = const.sigma_T.cgs.value
C = const.c.cgs.value
ME = const.m_e.cgs.value
MEC = (const.m_e * const.c).cgs.value
MEC2 = (const.m_e * const.c * const.c).cgs.value
H = const.h.cgs.value
E = 4.80320425e-10  # electron charge in cgs (statcoulomb)
B_cr = 4.414e13  # critical magnetic field Gauss
U_cr = np.power(B_cr, 2) / (8 * np.pi)
LAMBDA_C = 2.4263e-10  # Compton Wavelength of electron
unit_emissivity = u.Unit("erg s-1")
unit_sed = u.Unit("erg cm-2 s-1")

__all__ = ["R", "Synchrotron"]


def R(x):
    """Eq. 7.45 in Dermer (see class for reference). Angle-averaged integrand
    of the radiated power, the approximation of this formula given in
    https://ui.adsabs.harvard.edu//#abs/2010PhRvD..82d3002A/abstract
    is used, Eq. (D7)
    """
    term_1_num = 1.808 * np.power(x, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(x, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(x, 2 / 3) + 0.347 * np.power(x, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(x, 2 / 3) + 0.217 * np.power(x, 4 / 3)
    value = term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-x)
    return value


class Synchrotron:
    """class for synchrotron radiation computation

    Parameters
    ----------
    blob : `~agnpy.particles.Blob`
        emitting region and electron distribution hitting the photon target

    Reference
    ---------
    [1] : Dermer, Menon; High Energy Radiation From Black Holes;
    Princeton Series in Astrophysics
    [2] : http://adsabs.harvard.edu/abs/2008ApJ...686..181F
    """

    def __init__(self, blob):
        self.blob = blob
        self.B = self.blob.B
        # U_B Eq. 7.14 in [1]
        self.U_B = np.power(self.B, 2) / (8 * np.pi)
        # nu_B Eq. 7.19 in [1]
        self.nu_B = E * self.B / MEC
        # epsilon_B Eq. 7.21 in [1]
        self.epsilon_B = self.B / B_cr
        # grid of target photons to compute SSC
        self.epsilon_syn = np.logspace(-13, 10, 500)

    def com_sed_emissivity(self, epsilon):
        """Synchrotron  emissivity  SED (epsilon * J(epsilon))
        Eq. 116 in [1] or Eq. 18 in [2]
        This emissivity is computed in the co-moving frame of the blob.

        Parameters
        ----------
        epsilon : array_like
            array of the dimensionless energy to compute the sed.

        Note: when calling this function from another these energies
        have to be transformed in the co-moving frame of the plasmoid.
        """
        prefactor = np.sqrt(3) * epsilon * np.power(E, 3) * self.B / H
        gamma = self.blob.gamma
        N_e = self.blob.N_e(gamma)
        _gamma = gamma.reshape(gamma.size, 1)
        _N_e = N_e.reshape(N_e.size, 1)
        _epsilon = epsilon.reshape(1, epsilon.size)
        x_num = 4 * np.pi * _epsilon * np.power(ME, 2) * np.power(C, 3)
        x_denom = 3 * E * self.B * H * np.power(_gamma, 2)
        x = x_num / x_denom
        integrand = _N_e * R(x)
        integral = np.trapz(integrand, gamma, axis=0)
        return prefactor * integral * unit_emissivity

    def k_epsilon(self, epsilon):
        """SSA absorption factor Eq. 7.142 in [1]"""
        gamma = self.blob.gamma
        SSA_integrand = self.blob.n_e.SSA_integrand(gamma)
        _gamma = gamma.reshape(gamma.size, 1)
        _SSA_integrand = SSA_integrand.reshape(SSA_integrand.size, 1)
        _epsilon = epsilon.reshape(1, epsilon.size)

        prefactor_P_syn = np.sqrt(3) * np.power(E, 3) * self.B / H
        prefactor_k_epsilon = (
            -1 / (8 * np.pi * ME * np.power(epsilon, 2)) * np.power(LAMBDA_C / C, 3)
        )

        x_num = 4 * np.pi * _epsilon * np.power(ME, 2) * np.power(C, 3)
        x_denom = 3 * E * self.B * H * np.power(_gamma, 2)
        x = x_num / x_denom
        integrand = R(x) * _SSA_integrand
        integral = np.trapz(integrand, gamma, axis=0)
        return prefactor_k_epsilon * prefactor_P_syn * integral

    def tau_SSA(self, epsilon):
        """SSA opacity, Eq. before 7.122 in [1]"""
        return 2 * self.k_epsilon(epsilon) * self.blob.R_b

    def attenuation_SSA(self, epsilon):
        """SSA attenuation, Eq. 7.122 in [1]"""
        tau = self.tau_SSA(epsilon)
        u = 1 / 2 + np.exp(-tau) / tau - (1 - np.exp(-tau)) / np.power(tau, 2)
        attenuation = 3 * u / tau
        condition = tau < 1e-3
        attenuation[condition] = 1
        return attenuation

    def sed_flux(self, nu, SSA=False):
        """Synchrotron flux SED in erg cm-2 s-1

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of the frequencies, in Hz, to compute the sed.

        Note: these are observed frequencies (lab frame).
        """
        epsilon = H * nu.to("Hz").value / MEC2
        # correct epsilon to the comoving frame
        epsilon_prime = (1 + self.blob.z) * epsilon / self.blob.delta_D
        prefactor_num = np.power(self.blob.delta_D, 4)
        prefactor_denom = 4 * np.pi * np.power(self.blob.d_L, 2)
        prefactor = prefactor_num / prefactor_denom
        emissivity = self.com_sed_emissivity(epsilon_prime).value
        if SSA:
            emissivity *= self.attenuation_SSA(epsilon_prime)
        return prefactor * emissivity * unit_sed

    def ssc_sed_emissivity(self, epsilon):
        """Synchrotron self compton flux SED.
        Eq. 8 and 9 of [2].
        Although it is a Compton process I put it in this class and leave
        the compton.py class only for external photon fields.

        Parameters
        ----------
        epsilon : array_like
            dimensionless energy of the scattered photons

        Note: when calling this function from another these energies
        have to be transformed in the co-moving frame of the plasmoid.
        """
        # Eq. 22 of [2], transformed expanding f^syn_epsilon
        # the factor 3 / 4 accounts for averaging in a sphere
        # not included in Dermer and Finke's paper
        # we use a fixed grid of epsilon_syn to compute the SSC
        J_epsilon_syn = (
            3 / 4 * self.com_sed_emissivity(self.epsilon_syn) / self.epsilon_syn
        )
        gamma = self.blob.gamma
        N_e = self.blob.N_e(gamma)
        # variables that have to be integrated will start their names with "_"
        # in order to preserve original arrays shapes without reshaping again.
        # Quantities will be computed as matrices with:
        # axis = 0 : electrons Lorentz factors, dimension: N_gamma
        # axis = 1 : target photons energies, dimension: N_epsilon
        # axis = 2 : scattered photons energies, dimension : N_epsilon_s
        _gamma = gamma.reshape(gamma.size, 1, 1)
        _N_e = N_e.reshape(N_e.size, 1, 1)
        _epsilon_syn = self.epsilon_syn.reshape(1, self.epsilon_syn.size, 1)
        _J_epsilon_syn = J_epsilon_syn.reshape(1, J_epsilon_syn.size, 1)
        _epsilon = epsilon.reshape(1, 1, epsilon.size)
        _kernel = simple_kernel(_gamma, _epsilon_syn, _epsilon)

        integrand_epsilon = _J_epsilon_syn / np.power(_epsilon_syn, 2)
        integrand_gamma = _N_e / np.power(_gamma, 2) * _kernel
        integrand = integrand_epsilon * integrand_gamma

        integral_gamma = np.trapz(integrand, gamma, axis=0)
        integral_epsilon = np.trapz(integral_gamma, self.epsilon_syn, axis=0)

        prefactor_num = 9 * SIGMA_T * np.power(epsilon, 2)
        prefactor_denom = 16 * np.pi * np.power(self.blob.R_b, 2)
        return prefactor_num / prefactor_denom * integral_epsilon * unit_emissivity

    def ssc_sed_flux(self, nu):
        """SSC flux SED in erg cm-2 s-1

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of the scattered photons, in Hz

        Note: these are observed frequencies (lab frame).
        """
        epsilon = H * nu.to("Hz").value / MEC2
        # correct epsilon to the blob comoving frame
        epsilon *= (1 + self.blob.z) / self.blob.delta_D
        prefactor_num = np.power(self.blob.delta_D, 4)
        prefactor_denom = 4 * np.pi * np.power(self.blob.d_L, 2)
        prefactor = prefactor_num / prefactor_denom
        return prefactor * self.ssc_sed_emissivity(epsilon).value * unit_sed
