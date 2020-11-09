import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from agnpy.spectra import PowerLaw
from agnpy.synchrotron import Synchrotron

import matplotlib.pyplot as plt

pwl = PowerLaw()

R_b = 1e16 * u.cm
B = 1 * u.G
d_L = 1e27 * u.cm
z = Distance(d_L).z
delta_D = 10
Gamma = 10

nu = np.logspace(10, 20) * u.Hz

sed = Synchrotron.evaluate_sed_flux(
    nu, z, d_L, delta_D, B, R_b, PowerLaw, *pwl.parameters, ssa=False
)

plt.loglog(nu, sed)
plt.show()
