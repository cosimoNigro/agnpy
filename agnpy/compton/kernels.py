# integration kernels for inverse Compton evaluation
import numpy as np
from ..utils.math import log
from ..utils.geometry import cos_psi

__all__ = ["F_c", "isotropic_kernel", "get_gamma_min", "compton_kernel"]


def F_c(q, gamma_e):
    """isotropic Compton kernel, Eq. 6.75 in [DermerMenon2009]_, Eq. 10 in [Finke2008]_"""
    term_1 = 2 * q * log(q)
    term_2 = (1 + 2 * q) * (1 - q)
    term_3 = 1 / 2 * np.power(gamma_e * q, 2) / (1 + gamma_e * q) * (1 - q)
    return term_1 + term_2 + term_3


def isotropic_kernel(gamma, epsilon, epsilon_s):
    """Compton kernel for isotropic nonthermal electrons scattering photons of 
    an isotropic external radiation field.
    Integrand of Eq. 6.74 in [DermerMenon2009]_.
    Parameters
    ----------
    gamma : :class:`~numpy.ndarray`
        Lorentz factors of the electrons distribution
    epsilon : :class:`~numpy.ndarray`
        dimensionless energies (in electron rest mass units) of the target photons
    epsilon_s : :class:`~numpy.ndarray`
        dimensionless energies (in electron rest mass units) of the scattered photons
    """
    gamma_e = 4 * gamma * epsilon
    q = (epsilon_s / gamma) / (gamma_e * (1 - epsilon_s / gamma))
    q_min = 1 / (4 * np.power(gamma, 2))
    return np.where((q_min <= q) * (q <= 1), F_c(q, gamma_e), 0)


def get_gamma_min(epsilon_s, epsilon, mu_s, mu, phi):
    """minimum Lorentz factor for Compton integration, 
    Eq. 29 in [Dermer2009]_, Eq. 38 in [Finke2016]_."""
    sqrt_term = np.sqrt(1 + 2 / (epsilon * epsilon_s * (1 - cos_psi(mu_s, mu, phi))))
    return epsilon_s / 2 * (1 + sqrt_term)


def compton_kernel(gamma, epsilon_s, epsilon, mu_s, mu, phi):
    """angle dependent Compton kernel:
    Eq. 26-27 in [Dermer2009]_.
    Parameters
    ----------
    gamma : :class:`~numpy.ndarray`
        Lorentz factors of the electrons distribution
    epsilon : :class:`~numpy.ndarray`
        dimesnionless energies (in electron rest mass units) of the target photons
    epsilon_s : :class:`~numpy.ndarray`
        dimensionless energies (in electron rest mass units) of the scattered photons
    mu_s : float
        cosine of the zenith angle of the blob w.r.t the jet
    mu : :class:`~numpy.ndarray` or float
        (array of) cosine of the zenith angle subtended by the target
    phi : :class:`~numpy.ndarray` or float
        (array of) of the azimuth angle subtended by the target
    """
    epsilon_bar = gamma * epsilon * (1 - cos_psi(mu_s, mu, phi))
    y = 1 - epsilon_s / gamma
    y_1 = -(2 * epsilon_s) / (gamma * epsilon_bar * y)
    y_2 = np.power(epsilon_s, 2) / np.power(gamma * epsilon_bar * y, 2)
    values = y + 1 / y + y_1 + y_2
    gamma_min = get_gamma_min(epsilon_s, epsilon, mu_s, mu, phi)
    values = np.where(gamma >= gamma_min, y + 1 / y + y_1 + y_2, 0)
    return values
