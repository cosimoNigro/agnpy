import numpy as np
from scipy.special import gammaincc, gamma

def inc_gamma(a,z):
    return gammaincc(a,z)*gamma(a)

def exp_cutoff_broken_power_law_integral(k_e, p1, p2, gamma_c, gamma_b, gamma_min, gamma_max):
    """Analytical integral of the exponential cutoff broken power law."""
    if np.allclose(p1, 1.0):
        term_1 = gamma_b * ( inc_gamma(0,gamma_min/gamma_c) - inc_gamma(0,gamma_b/gamma_c) )
    else:
        term_1 = np.power(gamma_b,p1) * np.power(gamma_c,1-p1) * ( inc_gamma(1-p1,gamma_min/gamma_c) - inc_gamma(1-p1,gamma_b/gamma_c) )

    if np.allclose(p1, 1.0):
        term_2 = gamma_b * ( inc_gamma(0,gamma_b/gamma_c) - inc_gamma(0,gamma_max/gamma_c) )
    else:
        term_2 = np.power(gamma_b,p2) * np.power(gamma_c,1-p2) * ( inc_gamma(1-p2,gamma_b/gamma_c) - inc_gamma(1-p2,gamma_max/gamma_c) )

    return k_e * (term_1 + term_2)

def exp_cutoff_broken_power_law_times_gamma_integral(k_e, p1, p2, gamma_c, gamma_b, gamma_min, gamma_max):
    """Analytical integral of the exponential cutoff broken power law multiplied by gamma."""
    if np.allclose(p1, 1.0):
        term_1 = gamma_b * ( inc_gamma(0,gamma_min/gamma_c) - inc_gamma(0,gamma_b/gamma_c) )
    else:
        term_1 = np.power(gamma_b,p1) * np.power(gamma_c,2-p1) * ( inc_gamma(2-p1,gamma_min/gamma_c) - inc_gamma(2-p1,gamma_b/gamma_c) )

    if np.allclose(p1, 1.0):
        term_2 = gamma_b * ( inc_gamma(0,gamma_b/gamma_c) - inc_gamma(0,gamma_max/gamma_c) )
    else:
        term_2 = np.power(gamma_b,p2) * np.power(gamma_c,2-p2) * ( inc_gamma(2-p2,gamma_b/gamma_c) - inc_gamma(2-p2,gamma_max/gamma_c) )

    return k_e * (term_1 + term_2)
