"""classes and functions describing the inverse Compton radiative processes"""
import numpy as np
import astropy.constants as const
import astropy.units as u
from numba import jit, float64


# electromagnetic cgs units are not well-handled by astropy.units
# every variable indicated with capital letters is dimensionsless
# will be used in SED computations for speed-up
E = 4.80320425e-10  # statC (not handled by astropy units)
H = const.h.cgs.value
C = const.c.cgs.value
ME = const.m_e.cgs.value
MEC = (const.m_e * const.c).cgs.value
MEC2 = (const.m_e * const.c * const.c).cgs.value
SIGMA_T = const.sigma_T.cgs.value
EMISSIVITY_UNIT = "erg s-1"
SED_UNIT = "erg cm-2 s-1"


__all__ = ["SynchrotronSelfCompton", "ExternalCompton"]


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


def cos_psi(mu_s, mu, phi):
    """Compute the angle between the blob (with zenith mu_s) and a photon with
    zenith and azimuth (mu, phi). The system is symmetric in azimuth for the
    electron phi_s = 0, Eq. 8 in [5]."""
    term_1 = mu * mu_s
    term_2 = np.sqrt(1 - np.power(mu, 2)) * np.sqrt(1 - np.power(mu_s, 2))
    term_3 = np.cos(phi)
    return term_1 + term_2 * term_3


def get_gamma_min(epsilon_s, epsilon, mu_s, mu, phi):
    """minimum Lorentz factor for Compton integration, 
    Eq. 29 in [4], Eq. 38 in [5]."""
    sqrt_term = np.sqrt(1 + 2 / (epsilon * epsilon_s * (1 - cos_psi(mu_s, mu, phi))))
    return epsilon_s / 2 * (1 + sqrt_term)


def compton_kernel(gamma, epsilon_s, epsilon, mu_s, mu, phi):
    """Angle dependent Compton kernel:
    Eq. 26-27 in [4].

    Parameters
    ----------
    gamma : array_like
        Lorentz factors of the electrons distribution
    epsilon : array_like
        energies of the target photon field
    epsilon_s : array_like
        energies of the scattered photon field
    mu_s : float
        cosine of the zenith angle of the blob
    mu : `array_like`
        cosine of the zenith angle of the target
    phi : `array_like`
        azimuth angle of the target
    """
    epsilon_bar = gamma * epsilon * (1 - cos_psi(mu_s, mu, phi))
    y = 1 - epsilon_s / gamma
    y_1 = -(2 * epsilon_s) / (gamma * epsilon_bar * y)
    y_2 = np.power(epsilon_s, 2) / np.power(gamma * epsilon_bar * y, 2)
    values = y + 1 / y + y_1 + y_2
    gamma_min = get_gamma_min(epsilon_s, epsilon, mu_s, mu, phi)
    values = np.where(gamma >= gamma_min, y + 1 / y + y_1 + y_2, 0)
    return values


def x_re(mu, R_re, r):
    """Distance between the blob and the reprocessing material,
    see Fig. 9 and Eq. (76) in [5].
    
    Parameters
    ----------
    mu :  array_like
        cosine of the zenith from the central BH to the reprocessing material
    R_re : float 
        distance (in cm) from the BH to the reprocessing material
    r : float
        height of the emission region in the jet
    """
    value = np.sqrt(np.power(R_re, 2) + np.power(r, 2) - 2 * r * R_re * mu)
    return value


def mu_star(mu, R_re, r):
    """cosine of the angle between the blob and the reprocessing material,
    see Fig. 9 and Eq. (76) in [5].

    Parameters
    ----------
    mu :  array_like
        cosine of the zenith from the central BH to the reprocessing material
    R_re : float 
        distance (in cm) from the BH to the reprocessing material
    r : float
        height of the emission region in the jet
    """
    addend = np.power(R_re / x_re(mu, R_re, r), 2) * (1 - np.power(mu, 2))
    return np.sqrt(1 - addend)


class SynchrotronSelfCompton:
    """class for Synchrotron Self Compton radiation computation

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
        # axis = 0 : electrons Lorentz factors
        # axis = 1 : target photons energies
        # axis = 2 : scattered photons energies
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
            array of the frequencies, in Hz, to compute the sed

        Note: these are observed frequencies (lab frame).
        """
        epsilon = H * nu.to("Hz").value / MEC2
        # correct epsilon to the jet comoving frame
        epsilon_prime = (1 + self.blob.z) * epsilon / self.blob.delta_D
        prefactor = np.power(self.blob.delta_D, 4) / (
            4 * np.pi * np.power(self.blob.d_L, 2)
        )
        return (prefactor * self.com_sed_emissivity(epsilon_prime)).to(SED_UNIT)


class ExternalCompton:
    """class for External Compton radiation computation

    Parameters
    ----------
    blob : `~agnpy.emission_region.Blob`
        emitting region and electron distribution hitting the photon target
    target : `~agnpy.targets.Target`
        class describing the target photon field    
    r : `~astropy.units.Quantity`
        distance of the blob from the Black Hole (i.e. from the target photons)
    mu_min : float
        minimum zenith (cosine theta) subtended by the target photon field
    mu_max : float
        maximum zenith (cosine theta) subtended by the target photon field
    mu_size : int
        size of the array of the zenith dependence of the target field
    phi : array_like
        size of the array of the azimuth dependence of the target field
    """

    def __init__(self, blob, target, r):
        self.blob = blob
        # we integrate on a larger grid to account for the transf.
        # of the electron density in the reference frame of the BH
        self.gamma = self.blob.gamma_to_integrate
        self.transformed_N_e = self.blob.N_e(self.gamma / self.blob.delta_D).value
        self.target = target
        self.r = r.cgs
        self._r = self.r.value
        self.set_mu()
        self.set_phi()

    def set_mu(self, mu_size=50):
        self.mu_size = mu_size
        if self.target.type == "SSDisk":
            self.mu_min = 1 / np.sqrt(1 + np.power(self.target._R_out / self._r, 2))
            self.mu_max = 1 / np.sqrt(1 + np.power(self.target._R_in / self._r, 2))
        else:
            self.mu_min = -1
            self.mu_max = 1
        self.mu = np.linspace(self.mu_min, self.mu_max, self.mu_size)

    def set_phi(self, phi_size=50):
        self.phi_size = phi_size
        self.phi = np.linspace(0, 2 * np.pi, self.phi_size)

    def _sed_flux_disk(self, nu):
        """Flux SED (\nu F_{\nu})) [erg cm-2 s-1] for EC on SS Disk
        Eq. 70 in [4].

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of the frequencies, in Hz, to compute the sed

        Note: frequencies are observed, i.e. in the lab frame
        """
        # define the dimensionless energy
        epsilon_s_obs = H * nu.to("Hz").value / MEC2
        # transform to BH frame
        epsilon_s = epsilon_s_obs * (1 + self.blob.z)
        # variables that have to be integrated will start their names with "_"
        # in order to preserve original arrays shapes without reshaping again.
        # Quantities will be computed as matrices with:
        # axis 0: gamma
        # axis 1: mu
        # axis 2: phi
        # axis 3: epsilon_s
        _mu_s = self.blob.mu_s
        _gamma = self.gamma.reshape(self.gamma.size, 1, 1, 1)
        _N_e = self.transformed_N_e.reshape(self.transformed_N_e.size, 1, 1, 1)
        _mu = self.mu.reshape(1, self.mu.size, 1, 1)
        _phi = self.phi.reshape(1, 1, self.phi.size, 1)
        _epsilon_s = epsilon_s.reshape(1, 1, 1, epsilon_s.size)
        _epsilon = self.target._epsilon_mu(_mu, self._r)
        # define integrating function
        _kernel = compton_kernel(_gamma, _epsilon_s, _epsilon, _mu_s, _mu, _phi)
        # set to zero everything below gamma_low
        _integrand_mu_num = self.target._phi_disk_mu(_mu, self._r)
        _integrand_mu_denom = np.power(np.power(_mu, -2) - 1, 3 / 2) * np.power(
            _epsilon, 2
        )
        _integrand = (
            _integrand_mu_num
            / _integrand_mu_denom
            * np.power(_gamma, -2)
            * _N_e
            * _kernel
        )
        integral_gamma = np.trapz(_integrand, self.gamma, axis=0)
        integral_mu = np.trapz(integral_gamma, self.mu, axis=0)
        integral_phi = np.trapz(integral_mu, self.phi, axis=0)
        prefactor_num = (
            9
            * SIGMA_T
            * np.power(epsilon_s, 2)
            * self.target.l_Edd
            * self.target.L_Edd.value
            * np.power(self.blob.delta_D, 3)
        )
        prefactor_denom = (
            np.power(2, 9)
            * np.power(np.pi, 3)
            * np.power(self.blob.d_L.value, 2)
            * np.power(self.target.R_g.value, 2)
            * self.target.eta
            * np.power(self._r / self.target._R_g, 3)
        )
        return prefactor_num / prefactor_denom * integral_phi * u.Unit(SED_UNIT)

    def _sed_flux_shell_blr(self, nu):
        """SED flux for Compton Scattering over the blr

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of the frequencies, in Hz, to compute the sed

        Note: frequencies are observed, i.e. in the lab frame
        """
        # define the dimensionless energy
        epsilon_s_obs = H * nu.to("Hz").value / MEC2
        # transform to BH frame
        epsilon_s = epsilon_s_obs * (1 + self.blob.z)
        # variables that have to be integrated will start their names with "_"
        # in order to preserve original arrays shapes without reshaping again.
        # Quantities will be computed as matrices with:
        # axis 0: gamma
        # axis 1: mu_re (cosine zenith from BH to re-processing material)
        # axis 2: phi
        # axis 3: epsilon_s
        _mu_s = self.blob.mu_s
        _gamma = self.gamma.reshape(self.gamma.size, 1, 1, 1)
        _N_e = self.transformed_N_e.reshape(self.transformed_N_e.size, 1, 1, 1)
        _mu = self.mu.reshape(1, self.mu.size, 1, 1)
        _phi = self.phi.reshape(1, 1, self.phi.size, 1)
        _epsilon_s = epsilon_s.reshape(1, 1, 1, epsilon_s.size)
        # define integrating function
        _x = x_re(_mu, self.target._R_line, self._r)
        _mu_star = mu_star(_mu, self.target._R_line, self._r)
        _kernel = compton_kernel(
            _gamma, _epsilon_s, self.target.epsilon_line, _mu_s, _mu_star, _phi
        )
        _integrand = np.power(_x, -2) * np.power(_gamma, -2) * _N_e * _kernel
        integral_gamma = np.trapz(_integrand, self.gamma, axis=0)
        integral_mu = np.trapz(integral_gamma, self.mu, axis=0)
        integral_phi = np.trapz(integral_mu, self.phi, axis=0)
        prefactor_num = (
            3
            * SIGMA_T
            * np.power(epsilon_s, 2)
            * np.power(self.blob.delta_D, 3)
            * self.target.parent_disk.L_disk.value
            * self.target.csi_line
        )
        prefactor_denom = (
            8
            * np.power(4 * np.pi, 3)
            * np.power(self.blob.d_L.value, 2)
            * np.power(self.target.epsilon_line, 2)
        )
        return prefactor_num / prefactor_denom * integral_phi * u.Unit(SED_UNIT)

    def sed_flux(self, nu):
        if self.target.type == "SSDisk":
            return self._sed_flux_disk(nu)
        if self.target.type == "SphericalShellBLR":
            return self._sed_flux_shell_blr(nu)