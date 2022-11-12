# tests on agnpy.spectra module
import numpy as np
import astropy.units as u
from astropy.constants import m_e, m_p
from agnpy.spectra import (
    PowerLaw,
    BrokenPowerLaw,
    LogParabola,
    ExpCutoffPowerLaw,
    ExpCutoffBrokenPowerLaw,
    InterpolatedDistribution,
)
from agnpy.utils.math import trapz_loglog, inc_gamma
from agnpy.utils.conversion import mec2, mpc2
from astropy.coordinates import Distance
from matplotlib import pyplot as plt
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron


def broken_power_law_integral(k_e, p1, p2, gamma_b, gamma_min, gamma_max):
    """Analytical integral of the broken power law."""
    if np.allclose(p1, 1.0):
        term_1 = gamma_b * np.log(gamma_b / gamma_min)
    else:
        term_1 = gamma_b * (1 - np.power(gamma_min / gamma_b, 1 - p1)) / (1 - p1)
    if np.allclose(p2, 1.0):
        term_2 = gamma_b * np.log(gamma_max / gamma_b)
    else:
        term_2 = gamma_b * (np.power(gamma_max / gamma_b, 1 - p2) - 1) / (1 - p2)
    return k_e * (term_1 + term_2)

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

gamma_power = 0.
k=5e-5 * u.Unit("cm-3")
p1=2.0
p2=4.32
gamma_b=4e3
gamma_min=1
gamma_max=6e5
gamma_c = 6e4
integrator = np.trapz

ebpwl = ExpCutoffBrokenPowerLaw(
    k, p1, p2, gamma_b, gamma_min, gamma_max,gamma_c, integrator=integrator
)
bpwl  = BrokenPowerLaw(
    k, p1, p2, gamma_b, gamma_min, gamma_max
)


numerical_integral = ebpwl.integrate(
    gamma_min, gamma_max, gamma_power=gamma_power
)

analytical_integral = exp_cutoff_broken_power_law_integral(
    k, p1, p2, gamma_b, gamma_min, gamma_max, gamma_c
)

print (analytical_integral)
print (numerical_integral)

numerical_integral = bpwl.integrate(
    gamma_min, gamma_max, gamma_power=gamma_power
)

analytical_integral = broken_power_law_integral(
    k, p1, p2, gamma_b, gamma_min, gamma_max
)


print (analytical_integral)
print (numerical_integral)
