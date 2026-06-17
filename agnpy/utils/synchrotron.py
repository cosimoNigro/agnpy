import numpy as np


def Z(eta):
    """Eq. 7.45 in [Dermer2009]_, angle-averaged integrand of the radiated power, the
    approximation of this function, given in Eq. D7 of [Aharonian2010]_, is used.
    """
    term_1_num = 1.808 * np.power(eta, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(eta, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(eta, 2 / 3) + 0.347 * np.power(eta, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(eta, 2 / 3) + 0.217 * np.power(eta, 4 / 3)
    return term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-eta)


def tau_to_attenuation(tau):
    """Convert SSA optical depth → attenuation factor.
    Eq. 7.122 in [DermerMenon2009]_."""
    u = 0.5 + np.exp(-tau) / tau - (1 - np.exp(-tau)) / np.power(tau, 2)
    return np.where(tau < 1e-3, 1, 3 * u / tau)
