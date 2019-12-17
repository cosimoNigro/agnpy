import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from agnpy.particles import Blob
from agnpy.targets import RingDustTorus
from agnpy.compton import Compton
import astropy.units as u
import astropy.constants as const
import time

print("reproduce Figure 10 of Finke")
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
blob.set_gamma_size(800)
gamma = np.logspace(1, 8, 200)
plt.loglog(gamma, np.power(gamma, 2) * blob.n_e(gamma), lw=2)
plt.xlabel(r"$\gamma'$")
plt.ylabel(r"${\gamma'}^2\,n_{e}(\gamma')$")
plt.show()
print(f"total density {blob.norm:.2e}")
print(f"total energy {blob.W_e:.2e}")

# BLR parameters
M_sun = const.M_sun.cgs
M_BH = 1.2 * 1e9 * M_sun
L_disk = 2 * 1e46 * u.Unit("erg s-1")
T_dt = 1e3 * u.K
epsilon_dt = 2.7 * ((const.k_B * T_dt) / (const.m_e * const.c * const.c)).decompose()
csi_dt = 0.1

print("BLR SED")
compton = Compton(blob)
nu = np.logspace(15, 30, 50) * u.Hz

dt1 = RingDustTorus(M_BH, L_disk, csi_dt, epsilon_dt, r=1e18 * u.cm)
dt2 = RingDustTorus(M_BH, L_disk, csi_dt, epsilon_dt, r=1e19 * u.cm)
dt3 = RingDustTorus(M_BH, L_disk, csi_dt, epsilon_dt, r=1e20 * u.cm)
dt4 = RingDustTorus(M_BH, L_disk, csi_dt, epsilon_dt, r=1e21 * u.cm)

dt1.set_phi_size(500)

tstart = time.time()
sed1 = compton.sed_flux_ring_torus(nu, dt1)
tstop = time.time()
elapsed = tstop - tstart
print(f"elapsed time Dermer dust torus {elapsed:.2e} s")
sed2 = compton.sed_flux_ring_torus(nu, dt2)
sed3 = compton.sed_flux_ring_torus(nu, dt3)
sed4 = compton.sed_flux_ring_torus(nu, dt4)

fig, ax = plt.subplots()
# plt.loglog(nu, sed1, lw=2.5, color="k", label="$r = 10^{18}\,\mathrm{cm}$")
plt.loglog(nu, sed2, lw=2.5, color="k", label="$r = 10^{19}\,\mathrm{cm}$")
plt.loglog(nu, sed3, lw=2.5, color="crimson", label="$r = 10^{20}\,\mathrm{cm}$")
plt.loglog(nu, sed4, lw=2.5, color="dodgerblue", label="$r = 10^{21}\,\mathrm{cm}$")
plt.xlabel(r"$\nu\,/\,Hz$")
plt.ylabel(r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$")
# plt.grid(which="both")
plt.ylim(1e-28, 1e-14)
plt.xlim(1e16, 1e29)
plt.legend(fontsize=12)
plt.show()
fig.savefig("results/Figure_11_Finke_2016.png")
fig.savefig("results/Figure_11_Finke_2016.pdf")
