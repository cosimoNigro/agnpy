import numpy as np
import astropy.constants as const

SIGMA_T = const.sigma_T.cgs.value
ME = const.m_e.cgs.value
C = const.c.cgs.value
MEC2 = (const.m_e * const.c * const.c).cgs.value
H = const.h.cgs.value

__all__ = ["sigma", "Tau"]


def sigma(s):
    """photon-photon annihilation cross section, Eq. (119) of [3]"""
    beta_cm = np.sqrt(1 - np.power(s, -1))
    _prefactor = 3 / 8 * SIGMA_T * (1 - np.power(beta_cm, 2))
    _term1 = (3 - np.power(beta_cm, 4)) * np.log((1 + beta_cm) / (1 - beta_cm))
    _term2 = -2 * beta_cm * (2 - np.power(beta_cm, 2))
    values = _prefactor * (_term1 + _term2)
    values[s < 1] = 0
    return values


class Tau:
    def __init__(self):
        """class containing the optical depth calculation

        References
        ----------
        [1] : Dermer, Menon; High Energy Radiation From Black Holes;
        Princeton Series in Astrophysics
        [2] : http://adsabs.harvard.edu/abs/2008ApJ...686..181F
        [3] : https://ui.adsabs.harvard.edu/#abs/2016ApJ...830...94F/abstract
        [4] : https://ui.adsabs.harvard.edu/#abs/2009ApJ...692...32D/abstract
        """
        pass

    def disk(self, nu, blob, target):
        # Eq. (121) of [3], Eq. (80) of [4]
        # replacing t_tilde = 1 / l_tilde
        _prefactor = (
            -1e7
            * np.power(target.l_Edd, 3 / 4)
            * np.power(target.M_8, 1 / 4)
            / np.power(target.eta, 3 / 4)
        )
        epsilon_1 = H * nu.to("Hz").value / MEC2
        epsilon_1 *= 1 + blob.z
        # l_tilde = np.logspace(np.log10(target.r_tilde), 10 * np.log10(target.r_tilde))
        t_tilde = np.linspace(1 / target.r_tilde, 0, 100)
        # axis 0: R_tilde
        # axis 1: l_tilde
        # axis 2: epsilon_1
        _R = target.R.reshape(target.R.size, 1, 1)
        _R_tilde = target.R_tilde.reshape(target.R_tilde.size, 1, 1)
        _t_tilde = t_tilde.reshape(1, t_tilde.size, 1)
        _epsilon_1 = epsilon_1.reshape(1, 1, epsilon_1.size)
        # here _mu changes with the distance
        _mu = np.power(1 + np.power(_t_tilde * _R_tilde, 2), -1 / 2)
        _epsilon = target.epsilon_disk(_R)
        _s = _epsilon * _epsilon_1 * (1 - _mu) / 2
        _sigma = sigma(_s)
        _integrand_num = target.phi_disk(_R) * _sigma * (1 - _mu)
        _integrand_denom = (
            #            np.power(_l_tilde, 2)
            np.power(_R_tilde, 5 / 4)
            * np.power(1 + np.power(_t_tilde * _R_tilde, 2), 3 / 2)
            * SIGMA_T
        )
        _integrand = _integrand_num / _integrand_denom
        _integral_R_tilde = np.trapz(_integrand, target.R_tilde, axis=0)
        _integral_t_tilde = np.trapz(_integral_R_tilde, t_tilde, axis=0)
        return _prefactor * _integral_t_tilde

    def shell_blr(self, nu, blob, target):
        # Eq. (121) of [3], Eq. (80) of [4]
        _prefactor = 900 * target.csi_line * target.l_Edd / target.epsilon_line
        epsilon_1 = H * nu.to("Hz").value / MEC2
        epsilon_1 *= 1 + blob.z
        l_tilde = np.logspace(np.log10(target.r_tilde), 6 * np.log10(target.r_tilde))
        # axis 0: mu
        # axis 1: l_tilde
        # axis 2: epsilon_1
        _mu = target.mu.reshape(target.mu.size, 1, 1)
        _l_tilde = l_tilde.reshape(1, l_tilde.size, 1)
        _epsilon_1 = epsilon_1.reshape(1, 1, epsilon_1.size)
        _l = _l_tilde * target.R_g
        _x_tilde = (
            np.sqrt(
                np.power(target.R_line, 2)
                + np.power(_l, 2)
                - 2 * _l * target.R_line * _mu
            )
            / target.R_g
        )
        _mu_star = np.sqrt(
            1
            - np.power(target.R_line / (_x_tilde * target.R_g), 2)
            * (1 - np.power(_mu, 2))
        )
        _s = target.epsilon_line * _epsilon_1 * (1 - _mu_star) / 2
        _sigma = sigma(_s)
        _integrand = np.power(_x_tilde, -2) * _sigma / SIGMA_T * (1 - _mu_star)
        _integral_mu = np.trapz(_integrand, target.mu, axis=0)
        _integral_l_tilde = np.trapz(_integral_mu, l_tilde, axis=0)
        return _prefactor * _integral_l_tilde

    def dust_torus(self, nu, blob, target):
        _prefactor = 900 * target.csi_dt * target.l_Edd / target.epsilon_dt
        epsilon_1 = H * nu.to("Hz").value / MEC2
        epsilon_1 *= 1 + blob.z
        l_tilde = np.logspace(np.log10(target.r_tilde), 6 * np.log10(target.r_tilde))
        # axis 0: l_tilde
        # axis 1: epsilon_1
        _l_tilde = l_tilde.reshape(l_tilde.size, 1)
        _epsilon_1 = epsilon_1.reshape(1, epsilon_1.size)
        _l = _l_tilde * target.R_g
        _x_tilde = np.sqrt(
            np.power(target.R_dt / target.R_g, 2) + np.power(_l_tilde, 2)
        )
        _s = target.epsilon_dt * _epsilon_1 * (1 - _l_tilde / _x_tilde) / 2
        _sigma = sigma(_s)
        _integrand = (
            np.power(_x_tilde, -2) * _sigma / SIGMA_T * (1 - _l_tilde / _x_tilde)
        )
        _integral = np.trapz(_integrand, l_tilde, axis=0)
        return _prefactor * _integral
