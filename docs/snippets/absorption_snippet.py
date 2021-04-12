import numpy as np
import astropy.units as u
import astropy.constants as const
from agnpy.targets import SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.absorption import Absorption
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc


# matplotlib adjustments
load_mpl_rc()


# disk parameters
M_BH = 1.2 * 1e9 * const.M_sun.cgs
L_disk = 2 * 1e46 * u.Unit("erg s-1")
eta = 1 / 12
R_in = 6
R_out = 200
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)

# blr definition
csi_line = 0.024
R_line = 1e17 * u.cm
blr = SphericalShellBLR(L_disk, csi_line, "Lyalpha", R_line)

# dust torus definition
T_dt = 1e3 * u.K
csi_dt = 0.1
dt = RingDustTorus(L_disk, csi_dt, T_dt)

# consider a fixed distance of the blob from the target fields
r = 1.1e16 * u.cm
# let us consider 3C 454.3 as source
z = 0.859

absorption_disk = Absorption(disk, r=r, z=z)
absorption_blr = Absorption(blr, r=r, z=z)
absorption_dt = Absorption(dt, r=r, z=z)

# plot using the same energy range as in Finke 2016
E = np.logspace(0, 5) * u.GeV
nu = E.to("Hz", equivalencies=u.spectral())

tau_disk = absorption_disk.tau(nu)
tau_blr = absorption_blr.tau(nu)
tau_dt = absorption_dt.tau(nu)

# plot it
plt.loglog(E, tau_disk, lw=2, ls="-", label="SS disk")
plt.loglog(E, tau_blr, lw=2, ls="--", label="spherical shell BLR")
plt.loglog(E, tau_dt, lw=2, ls="-.", label="ring dust torus")
plt.legend()
plt.xlabel(r"$E\,/\,GeV$")
plt.ylabel(r"$\tau_{\gamma \gamma}$")
plt.xlim([1, 1e5])
plt.ylim([1e-3, 1e5])
plt.show()
