"""class and functions for Compton radiation"""
import numpy as np
import astropy.units as u
import astropy.constants as const
from .utils import cos_psi

SIGMA_T = const.sigma_T.cgs.value
H = const.h.cgs.value
C = const.c.cgs.value
MEC2 = (const.m_e * const.c * const.c).cgs.value
unit_emissivity = u.Unit("erg cm-3 s-1")
unit_sed = u.Unit("erg cm-2 s-1")

__all__ = ["F_c", "simple_kernel", "compton_kernel", "Compton"]


def F_c(q, gamma_e):
    """Compton kernel for scattering calculation
    Eq. 6.75 in [1], see class for reference [1]
    """
    term_1 = 2 * q * np.log(q)
    term_2 = (1 + 2 * q) * (1 - q)
    term_3 = 1 / 2 * np.power(gamma_e * q, 2) / (1 + gamma_e * q) * (1 - q)
    return term_1 + term_2 + term_3


def simple_kernel(gamma, epsilon, epsilon_s):
    """this integration kernel (angle independent) will implements the
    last two factors of the gamma-dependent integrand in (6.74) in [1]

    Parameters
    ----------
    gamma : array_like
        Lorentz factors of the electrons distribution
    epsilon : array_like
        energies of the target photon field
    epsilon_s : array_like
        energies of the scattered photon field
    """
    gamma_e = 4 * gamma * epsilon
    q = (epsilon_s / gamma) / (gamma_e * (1 - epsilon_s / gamma))
    q_min = 1 / (4 * np.power(gamma, 2))
    values = F_c(q, gamma_e)
    # apply the Heaviside function for q in (6.74)
    condition = (q_min <= q) * (q <= 1)
    values[~condition] = 0
    return values


def _get_gamma_min(epsilon_s, epsilon, mu_s, mu, phi):
    """Solve Eq. (38) in [3], same parameter as _get_gamma
    """
    _cos_psi = cos_psi(mu_s, mu, phi)
    _sqrt_term = np.sqrt(1 + 2 / (epsilon * epsilon_s * (1 - _cos_psi)))
    return epsilon_s / 2 * (1 + _sqrt_term)


def compton_kernel(gamma, epsilon_s, epsilon, mu_s, mu, phi):
    """full Compton kernel (angle dependent):
    Eq. (26) and (27) in [4] or
    Eq. (6.30) in [1]

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
    _cos_psi = cos_psi(mu_s, mu, phi)
    epsilon_bar = gamma * epsilon * (1 - _cos_psi)
    y = 1 - epsilon_s / gamma
    _term1 = -(2 * epsilon_s) / (gamma * epsilon_bar * y)
    _term2 = np.power(epsilon_s, 2) / np.power(gamma * epsilon_bar * y, 2)
    values = y + 1 / y + _term1 + _term2
    gamma_min = _get_gamma_min(epsilon_s, epsilon, mu_s, mu, phi)
    values[gamma < gamma_min] = 0
    return values


class Compton:
    """class for Compton scattering computations

    Parameters
    ----------
    blob : `~agnpy.particles.Blob`
        emitting region and electron distribution hitting the photon target
        
    References
    ----------
    [1] : Dermer, Menon; High Energy Radiation From Black Holes;
    Princeton Series in Astrophysics
    [2] : http://adsabs.harvard.edu/abs/2008ApJ...686..181F
    [3] : https://ui.adsabs.harvard.edu/#abs/2016ApJ...830...94F/abstract
    [4] : https://ui.adsabs.harvard.edu/#abs/2009ApJ...692...32D/abstract
    """

    def __init__(self, blob):
        self.blob = blob

    def sed_emissivity_iso_mono_ph(self, epsilon_s, target):
        """sed emissivity for electrons scattering over an isotropic
        monochromatic photon field.
        Eq. 6.74 of [1]

        Parameters
        ----------
        epsilon_s : `array_like`
            dimensionless energy of the scattered photons
        target : `~agnpy.targets.Monochromatic`
            isotropic target photon field
        """
        gamma = self.blob.gamma
        n_e = self.blob.n_e(gamma)
        # variables that have to be integrated will start their names with "_"
        # in order to preserve their original sizes without reshaping them again
        # quantities will be computed as matrices with:
        # axis = 0 : electrons Lorentz factors
        # axis = 1 : photons scattered energies
        _gamma = gamma.reshape(gamma.size, 1)
        _n_e = n_e.reshape(n_e.size, 1)
        _epsilon_s = epsilon_s.reshape(1, epsilon_s.size)
        _kernel = simple_kernel(_gamma, target.epsilon, _epsilon_s)

        _integrand = _n_e / np.power(_gamma, 2) * _kernel
        # the axis with the electron Lorentz factors is collapsed, an array
        # with the same dimension as epsilon_s remains
        integral = np.trapz(_integrand, _gamma, axis=0)

        prefactor = (
            3 / 4 * C * SIGMA_T * target.u_0 * np.power(epsilon_s / target.epsilon, 2)
        )
        return prefactor * integral * unit_emissivity

    def sed_emissivity_iso_pwl_ph(self, epsilon_s, target):
        """sed emissivity for electrons scattering over an isotropic
        monochromatic power-law photon field.
        Eq. 6.89 of [1]

        Parameters
        ----------
        epsilon_s : `array_like`
            dimensionless energy of the scattered photons
        target : `~agnpy.targets.PowerLaw`
            isotropic target photon field
        """
        gamma = self.blob.gamma
        n_e = self.blob.n_e(gamma)
        epsilon = target.epsilon
        u = target.u
        # Variables that have to be integrated will start their names with "_"
        # in order to preserve original arrays shapes without reshaping again.
        # Quantities will be computed as matrices with:
        # axis = 0 : electrons Lorentz factors, dimension: N_gamma
        # axis = 1 : target photons energies, dimension: N_epsilon
        # axis = 2 : scattered photons energies, dimension : N_epsilon_s
        _gamma = gamma.reshape(gamma.size, 1, 1)
        _n_e = n_e.reshape(n_e.size, 1, 1)
        _epsilon = epsilon.reshape(1, epsilon.size, 1)
        _u = u.reshape(1, u.size, 1)
        _epsilon_s = epsilon_s.reshape(1, 1, epsilon_s.size)
        _kernel = simple_kernel(_gamma, _epsilon, _epsilon_s)

        integrand_epsilon = _u / np.power(_epsilon, 2)
        integrand_gamma = _n_e / np.power(_gamma, 2) * _kernel
        integrand = integrand_epsilon * integrand_gamma

        integral_gamma = np.trapz(integrand, gamma, axis=0)
        integral_epsilon = np.trapz(integral_gamma, epsilon, axis=0)

        prefactor = 3 / 4 * C * SIGMA_T * np.power(epsilon_s, 2)
        return prefactor * integral_epsilon * unit_emissivity

    def sed_flux_disk(self, nu, target):
        """SED flux for Compton Scattering over the disk
        Eq. (70) in [4]

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of the scattered photons, in Hz
        target : `~agnpy.targets.Disk`
            target disk field

        Note: frequencies are observed, i.e. in the lab frame
        """
        # define the dimensionless energy
        epsilon_s_obs = H * nu.to("Hz").value / MEC2
        # transform to BH frame
        epsilon_s = epsilon_s_obs * (1 + self.blob.z)
        gamma = self.blob.gamma_to_integrate
        # axis order
        # axis 0: gamma
        # axis 1: mu
        # axis 2: phi
        # axis 3: epsilon_s
        _mu_s = self.blob.mu_s
        _gamma = gamma.reshape(gamma.size, 1, 1, 1)
        _mu = target.mu.reshape(1, target.mu.size, 1, 1)
        _phi = target.phi.reshape(1, 1, target.phi.size, 1)
        _epsilon_s = epsilon_s.reshape(1, 1, 1, epsilon_s.size)
        _epsilon = target.epsilon_disk_mu(_mu)
        # define integrating function
        _kernel = compton_kernel(_gamma, _epsilon_s, _epsilon, _mu_s, _mu, _phi)
        _N_e = self.blob.N_e(_gamma / self.blob.delta_D)
        # set to zero everything below gamma_low
        _integrand_mu_num = target.phi_disk_mu(_mu)
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
        integral_gamma = np.trapz(_integrand, gamma, axis=0)
        integral_mu = np.trapz(integral_gamma, target.mu, axis=0)
        integral_phi = np.trapz(integral_mu, target.phi, axis=0)
        prefactor_num = (
            9
            * SIGMA_T
            * np.power(epsilon_s, 2)
            * target.l_Edd
            * target.L_Edd
            * np.power(self.blob.delta_D, 3)
        )
        prefactor_denom = (
            np.power(2, 9)
            * np.power(np.pi, 3)
            * np.power(self.blob.d_L, 2)
            * np.power(target.R_g, 2)
            * target.eta
            * np.power(target.r_tilde, 3)
        )
        import IPython

        IPython.embed()
        return prefactor_num / prefactor_denom * integral_phi * unit_sed

    def sed_flux_shell_blr(self, nu, target):
        """SED flux for Compton Scattering over the blr

        Eq. (70) and (71) in [3]

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of the scattered photons, in Hz
        target : `~agnpy.targets.SphericalShellBLR`
            target blr field

        Note: frequencies are observed, i.e. in the lab frame
        """
        gamma = self.blob.gamma_to_integrate
        # define the dimensionless energy
        epsilon_s_obs = H * nu.to("Hz").value / MEC2
        # transform to BH frame
        epsilon_s = epsilon_s_obs * (1 + self.blob.z)
        # axis 0: gamma
        # axis 1: mu_re (cosine zenith from BH to re-processing material)
        # axis 2: phi
        # axis 3: epsilon_s
        _mu_s = self.blob.mu_s
        _gamma = gamma.reshape(gamma.size, 1, 1, 1)
        _mu = target.mu.reshape(1, target.mu.size, 1, 1)
        _phi = target.phi.reshape(1, 1, target.phi.size, 1)
        _epsilon_s = epsilon_s.reshape(1, 1, 1, epsilon_s.size)
        # define integrating function
        _x = target.x(_mu)
        _mu_star = target.mu_star(_mu)
        _kernel = compton_kernel(
            _gamma, _epsilon_s, target.epsilon_line, _mu_s, _mu_star, _phi
        )
        _N_e = self.blob.N_e(_gamma / self.blob.delta_D)
        _integrand = np.power(_x, -2) * np.power(_gamma, -2) * _N_e * _kernel
        integral_gamma = np.trapz(_integrand, gamma, axis=0)
        integral_mu = np.trapz(integral_gamma, target.mu, axis=0)
        integral_phi = np.trapz(integral_mu, target.phi, axis=0)
        prefactor_num = (
            3
            * SIGMA_T
            * np.power(epsilon_s, 2)
            * np.power(self.blob.delta_D, 3)
            * target.L_disk
            * target.csi_line
        )
        prefactor_denom = (
            8
            * np.power(4 * np.pi, 3)
            * np.power(self.blob.d_L, 2)
            * np.power(target.epsilon_line, 2)
        )
        return prefactor_num / prefactor_denom * integral_phi * unit_sed

    def sed_flux_ring_torus(self, nu, target):
        """SED flux for Compton Scattering over the blr

        Eq. (70) and (71) in [3]

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of the scattered photons, in Hz
        target : `~agnpy.targets.RingDustTorus`
            target blr field

        Note: frequencies are observed, i.e. in the lab frame
        """
        gamma = self.blob.gamma_to_integrate
        # define the dimensionless energy
        epsilon_s_obs = H * nu.to("Hz").value / MEC2
        # transform to BH frame
        epsilon_s = epsilon_s_obs * (1 + self.blob.z)
        # axis 0: gamma
        # axis 1: phi
        # axis 2: epsilon_s
        _mu_s = self.blob.mu_s
        # here we plug mu =  r / x. Delta function in Eq. (91) of [3]
        _mu = target.r / target.x
        _gamma = gamma.reshape(gamma.size, 1, 1)
        _phi = target.phi.reshape(1, target.phi.size, 1)
        _epsilon_s = epsilon_s.reshape(1, 1, epsilon_s.size)
        # define integrating function
        _kernel = compton_kernel(
            _gamma, _epsilon_s, target.epsilon_dt, _mu_s, _mu, _phi
        )
        _N_e = self.blob.N_e(_gamma / self.blob.delta_D)
        _integrand = np.power(_gamma, -2) * _N_e * _kernel
        integral_gamma = np.trapz(_integrand, gamma, axis=0)
        integral_phi = np.trapz(integral_gamma, target.phi, axis=0)
        prefactor_num = (
            3
            * SIGMA_T
            * np.power(epsilon_s, 2)
            * np.power(self.blob.delta_D, 3)
            * target.L_disk
            * target.csi_dt
        )
        prefactor_denom = (
            8
            * np.power(4 * np.pi, 3)
            * np.power(self.blob.d_L, 2)
            * np.power(target.epsilon_dt, 2)
            * np.power(target.x, 2)
        )
        return prefactor_num / prefactor_denom * integral_phi * unit_sed
