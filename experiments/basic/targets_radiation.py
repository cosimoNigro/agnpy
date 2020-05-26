import sys
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

sys.path.append("../../")
from agnpy.targets import SphericalShellBLR, RingDustTorus

blr = SphericalShellBLR(1e46 * u.Unit("erg s-1"), 0.1, "Lyalpha", 1e17 * u.cm)
dt = RingDustTorus(1e46 * u.Unit("erg s-1"), 0.6, 1000 * u.K)
print(blr)
print(dt)

r = np.logspace(14, 21, 200) * u.cm

plt.loglog(r, blr.u(r), label="BLR")
plt.loglog(r, dt.u(r), label="Torus")
plt.xlabel(r"$r\,/\,{\rm cm}$")
plt.ylabel(r"$u\,/\,({\rm erg}\,{\rm cm}^{-3})$")
plt.legend()
plt.show()
