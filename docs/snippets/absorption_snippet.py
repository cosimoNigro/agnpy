import numpy as np
import astropy.units as u
from agnpy.targets import SphericalShellBLR, RingDustTorus
from agnpy.absorption import Absorption
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc


# blr definition
L_disk = 2 * 1e46 * u.Unit("erg s-1")
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

absorption_blr = Absorption(blr, r=r, z=z)
absorption_dt = Absorption(dt, r=r, z=z)

# plot using the same energy range as in Finke 2016
E = np.logspace(0, 5) * u.GeV
nu = E.to("Hz", equivalencies=u.spectral())

tau_blr = absorption_blr.tau(nu)
tau_dt = absorption_dt.tau(nu)

# plot it
load_mpl_rc()
plt.loglog(E, tau_blr, lw=2, ls="--", label="spherical shell BLR")
plt.loglog(E, tau_dt, lw=2, ls="-.", label="ring dust torus")
plt.legend()
plt.xlabel(r"$E\,/\,GeV$")
plt.ylabel(r"$\tau_{\gamma \gamma}$")
plt.xlim([1, 1e5])
plt.ylim([1e-3, 1e5])
plt.show()
