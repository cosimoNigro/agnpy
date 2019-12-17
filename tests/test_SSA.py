import sys

sys.path.append("../")
import numpy as np
import astropy.units as u
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from agnpy.particles import Blob
from agnpy.synchrotron import Synchrotron
from astropy.coordinates import Distance
import time

spectrum_norm = 1e48 * u.Unit("erg")
parameters = {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e7}
spectrum_dict = {"type": "PowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 1 * u.G
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
print(f"total density {blob.norm:.2e}")
print(f"total energy {blob.W_e:.2e}")

nu = np.logspace(8, 25, 200) * u.Hz

synchrotron = Synchrotron(blob)
sed_synch = synchrotron.sed_flux(nu, SSA=False)
sed_synch_SSA = synchrotron.sed_flux(nu, SSA=True)

fig, ax = plt.subplots()
plt.loglog(nu, sed_synch, lw=2.5, color="k", label="synchrotron")
plt.loglog(
    nu, sed_synch_SSA, lw=2.5, color="k", ls="--", label="self-absorbed synchrotron"
)

plt.legend(fontsize=13, loc=2)

plt.xlabel(r"$\nu\,/\,Hz$")
plt.ylabel(r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$")
# plt.grid(which="both")
# plt.xlim([1e9, 1e30])
plt.ylim([1e-12, 1e-9])
plt.show()
fig.savefig("results/SSA.png")
fig.savefig("results/SSA.pdf")
