"""classes and functions for inverse Compton radiation"""
import numpy as np
import astropy.constants as const
import astropy.units as u
from numba import jit

# cgs is not well-handled by astropy.units
# every variable indicated with capital letters is dimensionsless
E = 4.80320425e-10  # statC (not handled by astropy units)
H = const.h.cgs.value
C = const.c.cgs.value
ME = const.m_e.cgs.value
MEC = (const.m_e * const.c).cgs.value
MEC2 = (const.m_e * const.c * const.c).cgs.value
SIGMA_T = const.sigma_T.cgs.value
EMISSIVITY_UNIT = "erg s-1"
SED_UNIT = "erg cm-2 s-1"


__all__ = ["SynchrotronSelfCompton"]


@jit(nopython=True, cache=True)
def F_c(q, gamma_e):
    """isotropic Compton kernel, Eq. 6.75 in [1], Eq. 10 in [2]"""
    term_1 = 2 * q * np.log(q)
    term_2 = (1 + 2 * q) * (1 - q)
    term_3 = 1 / 2 * np.power(gamma_e * q, 2) / (1 + gamma_e * q) * (1 - q)
    return term_1 + term_2 + term_3


def isotropic_kernel(gamma, epsilon, epsilon_s):
    """Compton kernel for isotropic nonthermal electrons scattering photons of 
    an isotropic external radiation field.
    Integrand of Eq. 6.74 in [1].

    Parameters
    ----------
    gamma : array_like
        Lorentz factors of the electrons distribution
    epsilon : array_like
        dimesnionless energies of the target photon field
    epsilon_s : array_like
        dimensionless energies of the scattered photons
    """
    gamma_e = 4 * gamma * epsilon
    q = (epsilon_s / gamma) / (gamma_e * (1 - epsilon_s / gamma))
    q_min = 1 / (4 * np.power(gamma, 2))
    values = np.where((q_min <= q) * (q <= 1), F_c(q, gamma_e), 0)
    return values


class SynchrotronSelfCompton:
    """class for SSC radiation computation

    Parameters
    ----------
    blob : `~agnpy.emission_region.Blob`
        emitting region and electron distribution hitting the photon target
    synchrotron : `~agnpy.synchrotron.Synchrotron`
        class describing the synchrotron photons target
    """

    def __init__(self, blob, synchrotron):
        self.blob = blob
        self.synchrotron = synchrotron
        # default grid of epsilon values over which to integrate
        self.epsilon_syn = np.logspace(-13, 10, 300)
        self.synch_sed_emissivity = self.synchrotron.com_sed_emissivity(
            self.epsilon_syn
        )

    def com_sed_emissivity(self, epsilon):
        """SSC  emissivity  (\epsilon' * J'_{SSC}(\epsilon')) [erg s-1]
        Eq. 8 and 9 of [2].

        Parameters
        ----------
        epsilon : array_like
            dimensionless energies of the scattered photons

        Note: when calling this function from another these energies
        have to be transformed in the co-moving frame of the plasmoid.
        """
        gamma = self.blob.gamma
        N_e = self.blob.N_e(gamma).value
        # Eq. 22 of [2], the factor 3 / 4 accounts for averaging in a sphere
        # not included in Dermer and Finke's papers
        J_epsilon_syn = 3 / 4 * self.synch_sed_emissivity.value / self.epsilon_syn
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
        _kernel = isotropic_kernel(_gamma, _epsilon_syn, _epsilon)
        # build the integrands of Eq. 9 in [2], using the reshaped arrays
        integrand_epsilon = _J_epsilon_syn / np.power(_epsilon_syn, 2)
        integrand_gamma = _N_e / np.power(_gamma, 2) * _kernel
        integrand = integrand_epsilon * integrand_gamma
        # integrate the Lorentz factor and the target synchrotron energies axes
        integral_gamma = np.trapz(integrand, gamma, axis=0)
        integral_epsilon = np.trapz(integral_gamma, self.epsilon_syn, axis=0)

        prefactor = (
            9
            * SIGMA_T
            * np.power(epsilon, 2)
            / (16 * np.pi * np.power(self.blob.R_b.value, 2))
        )
        return prefactor * integral_epsilon * u.Unit(EMISSIVITY_UNIT)

    def sed_luminosity(self, nu):
        """SSC luminosity SED (\nu L_{\nu}) [erg s-1]

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
          array of the frequencies, in Hz, to compute the sed

        Note: these are observed frequencies (observer frame).
        """
        epsilon = H * nu.to("Hz").value / MEC2
        # correct epsilon to the jet comoving frame
        epsilon_prime = (1 + self.blob.z) * epsilon / self.blob.delta_D
        return prefactor * self.com_sed_emissivity(epsilon_prime)

    def sed_flux(self, nu):
        """SSC flux SED (\nu F_{\nu})) [erg cm-2 s-1]
        Eq. 15 in [2]

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of the scattered photons, in Hz

        Note: these are observed frequencies (lab frame).
        """
        epsilon = H * nu.to("Hz").value / MEC2
        # correct epsilon to the jet comoving frame
        epsilon_prime = (1 + self.blob.z) * epsilon / self.blob.delta_D
        prefactor = np.power(self.blob.delta_D, 4) / (
            4 * np.pi * np.power(self.blob.d_L, 2)
        )
        return (prefactor * self.com_sed_emissivity(epsilon_prime)).to(SED_UNIT)
