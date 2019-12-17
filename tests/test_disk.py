import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from agnpy.particles import Blob
from agnpy.targets import Disk
from agnpy.compton import Compton
import astropy.constants as const
import astropy.units as u
import time

print("reproduce Figure 8 of Finke")
spectrum_norm = 5e42 * u.erg
parameters = {
    "p1": 2.0001,
    "p2": 3.5,
    "gamma_b": 1e4,
    "gamma_min": 20,
    "gamma_max": 5e7,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 0.56 * u.G
z = 1
delta_D = 40
Gamma = 40
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
blob.set_gamma_size(600)
gamma = np.logspace(1, 8, 200)
plt.loglog(gamma, np.power(gamma, 2) * blob.n_e(gamma), lw=2)
plt.xlabel(r"$\gamma'$")
plt.ylabel(r"${\gamma'}^2\,n_{e}(\gamma')$")
plt.show()
print(f"total density {blob.norm:.2e}")
print(f"total energy {blob.W_e:.2e}")


# disk parameters
M_sun = const.M_sun.cgs
M_BH = 1.2 * 1e9 * M_sun
R_g = ((const.G * M_BH) / (const.c * const.c)).cgs
L_disk = 2 * 1e46 * u.Unit("erg s-1")
eta = 1 / 12
R_in = 6 * R_g
R_out = 200 * R_g

disk1 = Disk(M_BH, L_disk, eta, R_in, R_out, r=1e17 * u.cm)
disk2 = Disk(M_BH, L_disk, eta, R_in, R_out, r=1e18 * u.cm)
disk3 = Disk(M_BH, L_disk, eta, R_in, R_out, r=1e19 * u.cm)


print("disk SED")
compton = Compton(blob)
nu = np.logspace(17, 30, 50) * u.Hz

tstart = time.time()
sed1 = compton.sed_flux_disk(nu, disk1)
tstop = time.time()
elapsed = tstop - tstart
print(f"elapsed time Dermer disk {elapsed:.2e} s")

sed2 = compton.sed_flux_disk(nu, disk2)
sed3 = compton.sed_flux_disk(nu, disk3)

fig, ax = plt.subplots()
plt.loglog(nu, sed1, lw=2.5, ls="-", color="k", label=r"$r = 10^{17}\,\mathrm{cm}$")
plt.loglog(
    nu, sed2, lw=2.5, ls="-", color="crimson", label="$r = 10^{18}\,\mathrm{cm}$"
)
plt.loglog(
    nu, sed3, lw=2.5, ls="-", color="dodgerblue", label="$r = 10^{19}\,\mathrm{cm}$"
)
plt.xlabel(r"$\nu\,/\,Hz$")
plt.ylabel(r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$")
# plt.grid(which="both")
plt.ylim(1e-26, 1e-12)
plt.xlim(1e18, 1e29)
plt.legend(fontsize=13)
plt.show()
fig.savefig("results/Figure_8_Finke_2016.png")
fig.savefig("results/Figure_8_Finke_2016.pdf")
