import numpy as np
import astropy.units as u
from scipy.special import gammaincc, gamma

def inc_gamma(a,z):
    return gammaincc(a,z)*gamma(a)

def exp_cutoff_broken_power_law_integral(k_e, p1, p2, gamma_c, gamma_b, gamma_min, gamma_max):
    """Analytical integral of the exponential cutoff broken power law."""
    if np.allclose(p1, 1.0):
        term_1 = gamma_b * ( inc_gamma(0,gamma_min/gamma_c) - inc_gamma(0,gamma_b/gamma_c) )
    else:
        term_1 = np.power(gamma_b,p1) * np.power(gamma_c,1-p1) * ( inc_gamma(1-p1,gamma_min/gamma_c) - inc_gamma(1-p1,gamma_b/gamma_c) )

    if np.allclose(p2, 1.0):
        term_2 = gamma_b * ( inc_gamma(0,gamma_b/gamma_c) - inc_gamma(0,gamma_max/gamma_c) )
    else:
        term_2 = np.power(gamma_b,p2) * np.power(gamma_c,1-p2) * ( inc_gamma(1-p2,gamma_b/gamma_c) - inc_gamma(1-p2,gamma_max/gamma_c) )

    return k_e * (term_1 + term_2)

def simpler(k_e, p1, p2, gamma_c, gamma_b, gamma_min, gamma_max):
    term =  inc_gamma(1-p1,gamma_min/gamma_c)
    return term

k=3.75e-5 * u.Unit("cm-3")
p1=2.0
p2=4.32
gamma_b=4e3
gamma_min=1
gamma_max=6e5
gamma_c  =6e4

print (exp_cutoff_broken_power_law_integral(k,p1,p2,gamma_c,gamma_b,gamma_min,gamma_max))
print (simpler(k,p1,p2,gamma_c,gamma_b,gamma_min,gamma_max))

print (inc_gamma(-1,gamma_b/gamma_c))
