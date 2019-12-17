import numpy as np
import astropy.constants as const
import astropy.units as u

ME = const.m_e.cgs.value
C = const.c.cgs.value
LAMBDA_C = 2.4263e-10  # Compton Wavelength of electron


def _power_law(gamma, k_e, p, gamma_min, gamma_max):
    """simple power law"""
    pwl = np.power(gamma, -p)
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return k_e * pwl


def _power_law_ssa_integrand(gamma, k_e, p, gamma_min, gamma_max):
    """\gamma^2 d / d \gamma (n_e / \gamma^2)"""
    pwl = np.power(gamma, -p - 1)
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return (-p - 2) * k_e * pwl


def _broken_power_law(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
    """power law with two spectral indices"""
    pwl = np.power(gamma / gamma_b, -p1)
    # compute power law with the second spectral index
    p2_condition = gamma > gamma_b
    pwl[p2_condition] = np.power(gamma[p2_condition] / gamma_b, -p2)
    # return zero outside minimum and maximum Lorentz factor values
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return k_e * pwl


def _broken_power_law_ssa_integrand(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
    """\gamma^2 d / d \gamma (n_e / \gamma^2)"""
    pwl = np.power(gamma / gamma_b, -p1 - 1)
    pwl_prefactor = (-p1 - 2) / gamma
    # compute power law with the second spectral index
    p2_condition = gamma > gamma_b
    pwl[p2_condition] = np.power(gamma[p2_condition] / gamma_b, -p2 - 1)
    pwl_prefactor[p2_condition] = (-p2 - 2) / gamma[p2_condition]
    # return zero outside minimum and maximum Lorentz factor values
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return k_e * pwl_prefactor * pwl


def _broken_power_law2(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
    """Tavecchio's Broken Power Law
    http://adsabs.harvard.edu/abs/1998ApJ...509..608T"""
    pwl = np.power(gamma, -p1)
    p2_condition = gamma > gamma_b
    pwl[p2_condition] = np.power(gamma_b, p2 - p1) * np.power(gamma[p2_condition], -p2)
    # return zero outside minimum and maximum Lorentz factor values
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return k_e * pwl


def _broken_power_law2_ssa_integrand(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
    """\gamma^2 d / d \gamma (n_e / \gamma^2)"""
    pwl = (-p1 - 2) * np.power(gamma, -p1 - 1)
    # compute power law with the second spectral index
    p2_condition = gamma > gamma_b
    pwl[p2_condition] = (
        (-p2 - 2) * np.power(gamma_b, p2 - p1) * np.power(gamma[p2_condition], -p2 - 1)
    )
    # return zero outside minimum and maximum Lorentz factor values
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return k_e * pwl


def cos_psi(mu_s, mu, phi):
    """Compute the angle between the blob (with zenith mu_s) and a photon with
    zenith and azimuth (mu, phi). The system is symmetric in azimuth for the
    electron phi_s = 0, Eq. (8) in [1], see Finke class."""
    _term_1 = mu * mu_s
    _term_2 = np.sqrt(1 - np.power(mu, 2)) * np.sqrt(1 - np.power(mu_s, 2))
    _term_3 = np.cos(phi)
    return _term_1 + _term_2 * _term_3


def get_R_g(M_BH):
    """get the gravitational radius given the BH mass"""
    R_g = (const.G * M_BH) / (const.c * const.c)
    return R_g.to("cm")


def get_L_Edd(M_BH):
    """get the Eddington luminosity"""
    M_8 = 1e8 * const.M_sun.cgs
    L_Edd = 1.26 * 1e46 * (M_BH / M_8).decompose() * u.Unit("erg / s")
    return L_Edd


def I_bb_epsilon(epsilon, Theta):
    """Black-Body intensity, as in Eq. 5.15 of [1]"""
    prefactor = (2 * ME * np.power(C, 3) * np.power(epsilon, 3)) / np.power(LAMBDA_C, 3)
    return prefactor * 1 / (np.exp(epsilon / Theta) - 1)
