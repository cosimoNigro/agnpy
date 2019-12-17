import sys

sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
from agnpy.particles import Blob
from agnpy.targets import SphericalShellBLR
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
blob.set_gamma_size(500)
gamma = np.logspace(1, 8, 300)
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
epsilon_line = 2e-5
csi_line = 0.024
R_line = 1e17 * u.cm
r = 1e16 * u.cm

blr = SphericalShellBLR(M_BH, L_disk, csi_line, epsilon_line, R_line, r)

check_mu_and_x = False
if check_mu_and_x:
    mu_re = np.linspace(-1, 1, 200)
    plt.plot(mu_re, blr.mu_star(mu_re), lw=2)
    plt.xlabel(r"$\mu_{\mathrm{re}}$")
    plt.ylabel(r"$\mu$")
    plt.show()

    plt.semilogy(mu_re, blr.x(mu_re), lw=2)
    plt.xlabel(r"$\mu_{\mathrm{re}}$")
    plt.ylabel(r"$x$")
    plt.show()

print("BLR SED")
compton = Compton(blob)
nu = np.logspace(15, 30, 50) * u.Hz

blr1 = SphericalShellBLR(M_BH, L_disk, csi_line, epsilon_line, R_line, r=1e16 * u.cm)
blr2 = SphericalShellBLR(M_BH, L_disk, csi_line, epsilon_line, R_line, r=1e18 * u.cm)
blr3 = SphericalShellBLR(M_BH, L_disk, csi_line, epsilon_line, R_line, r=1e19 * u.cm)
blr3.set_mu_size(80)

tstart = time.time()
sed1 = compton.sed_flux_shell_blr(nu, blr1)
tstop = time.time()
elapsed = tstop - tstart
print(f"elapsed time Dermer blr {elapsed:.2e} s")
sed2 = compton.sed_flux_shell_blr(nu, blr2)
sed3 = compton.sed_flux_shell_blr(nu, blr3)

fig, ax = plt.subplots()
plt.loglog(nu[:-6], sed1[:-6], lw=2.5, color="k", label="$r = 10^{16}\,\mathrm{cm}$")
plt.loglog(
    nu[:-6], sed2[:-6], lw=2.5, color="crimson", label="$r = 10^{18}\,\mathrm{cm}$"
)
plt.loglog(
    nu[:-6], sed3[:-6], lw=2.5, color="dodgerblue", label="$r = 10^{19}\,\mathrm{cm}$"
)
plt.xlabel(r"$\nu\,/\,Hz$")
plt.ylabel(r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}$)")
# plt.grid(which="both")
plt.ylim(1e-25, 1e-12)
plt.xlim(1e15, 1e29)
plt.legend(fontsize=13)
plt.show()
fig.savefig("results/Figure_10_Finke_2016.png")
fig.savefig("results/Figure_10_Finke_2016.pdf")
