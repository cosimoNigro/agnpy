# check absorption for disk
import numpy as np
import astropy.units as u
import astropy.constants as const
from agnpy.targets import SSDisk
from agnpy.absorption import Absorption
import matplotlib.pyplot as plt

# disk parameters
M_BH = 1.2 * 1e9 * const.M_sun.cgs
L_disk = 2 * 1e46 * u.Unit("erg s-1")
eta = 1 / 12
R_in = 6
R_out = 200
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)

# consider a fixed distance of the blob from the target fields
z = 0.859
R_Ly_alpha = 1.1e17 * u.cm
r = 1e-1 * R_Ly_alpha

absorption_disk = Absorption(disk, r=r, z=z)

E = np.logspace(0, 5) * u.GeV
nu = E.to("Hz", equivalencies=u.spectral())
tau_disk = absorption_disk.tau(nu)

import IPython

IPython.embed()

fig, ax = plt.subplots()
ax.loglog(E, tau_disk, lw=2, ls="-", label="SS disk")
ax.legend()
ax.set_xlabel("E / GeV")
ax.set_ylabel(r"$\tau_{\gamma \gamma}$")
ax.set_xlim([1, 1e5])
ax.set_ylim([1e-3, 1e5])
plt.show()
