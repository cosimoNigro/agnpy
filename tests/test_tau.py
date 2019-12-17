import sys

sys.path.append("../")
import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from agnpy.particles import Blob
from agnpy.targets import Disk, SphericalShellBLR, RingDustTorus
from agnpy.tau import sigma, Tau

r_e = 2.817_940_322_7e-13  # cm

s = np.logspace(-1, 4, 1000)

fig, ax = plt.subplots()
plt.loglog(s, sigma(s) / (2 * np.pi * np.power(r_e, 2)), lw=2.5, color="crimson")
ax.set_xlim([0.5, 1e3])
ax.set_ylim([0.01, 1])
ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$\sigma_{\gamma \gamma}(s)\,/\,(4/3\,\sigma_T)$")
plt.show()
fig.savefig("results/sigma_YY.pdf")

# blob
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

# disk parameters
M_sun = const.M_sun.cgs
M_BH = 1.2 * 1e9 * M_sun
R_g = ((const.G * M_BH) / (const.c * const.c)).cgs
L_disk = 2e46 * u.Unit("erg s-1")
eta = 1 / 12
R_in = 6 * R_g
R_out = 200 * R_g
R_Ly_alpha = 1.1 * 1e17 * u.cm

# blr parameters
epsilon_line = 2e-5
csi_line = 0.024
R_line = 1e17 * u.cm

# torus parameters
T_dt = 1e3 * u.K
epsilon_dt = 2.7 * ((const.k_B * T_dt) / (const.m_e * const.c * const.c)).decompose()
csi_dt = 0.1


disk1 = Disk(M_BH, L_disk, eta, R_in, R_out, r=0.1 * R_Ly_alpha)
blr1 = SphericalShellBLR(
    M_BH, L_disk, csi_line, epsilon_line, R_line, r=0.1 * R_Ly_alpha
)
torus1 = RingDustTorus(M_BH, L_disk, csi_dt, epsilon_dt, r=0.1 * R_Ly_alpha)

disk2 = Disk(M_BH, L_disk, eta, R_in, R_out, r=R_Ly_alpha)
blr2 = SphericalShellBLR(M_BH, L_disk, csi_line, epsilon_line, R_line, r=R_Ly_alpha)
torus2 = RingDustTorus(M_BH, L_disk, csi_dt, epsilon_dt, r=R_Ly_alpha)


disk3 = Disk(M_BH, L_disk, eta, R_in, R_out, r=10 * R_Ly_alpha)
blr3 = SphericalShellBLR(
    M_BH, L_disk, csi_line, epsilon_line, R_line, r=10 * R_Ly_alpha
)
torus3 = RingDustTorus(M_BH, L_disk, csi_dt, epsilon_dt, r=10 * R_Ly_alpha)

tau = Tau()
energy = np.logspace(-1, 6, 100) * u.GeV
nu = energy.to(u.Hz, equivalencies=u.spectral())

# import IPython
# IPython.embed()

tauYY_disk1 = tau.disk(nu, blob, disk1)
tauYY_blr1 = tau.shell_blr(nu, blob, blr1)
tauYY_torus1 = tau.dust_torus(nu, blob, torus1)

tauYY_disk2 = tau.disk(nu, blob, disk2)
tauYY_blr2 = tau.shell_blr(nu, blob, blr2)
tauYY_torus2 = tau.dust_torus(nu, blob, torus2)

tauYY_disk3 = tau.disk(nu, blob, disk3)
tauYY_blr3 = tau.shell_blr(nu, blob, blr3)
tauYY_torus3 = tau.dust_torus(nu, blob, torus3)

fig, ax = plt.subplots()
plt.plot(energy.to("GeV").value, tauYY_disk1, color="k", lw=2)
plt.plot(energy.to("GeV").value, tauYY_blr1, color="k", lw=2, ls="--")
plt.plot(energy.to("GeV").value, tauYY_torus1, color="k", lw=2, ls="-.")

plt.plot(energy.to("GeV").value, tauYY_disk2, color="crimson", lw=2)
plt.plot(energy.to("GeV").value, tauYY_blr2, color="crimson", lw=2, ls="--")
plt.plot(energy.to("GeV").value, tauYY_torus2, color="crimson", lw=2, ls="-.")

plt.plot(energy.to("GeV").value, tauYY_disk3, color="dodgerblue", lw=2)
plt.plot(energy.to("GeV").value, tauYY_blr3, color="dodgerblue", lw=2, ls="--")
plt.plot(energy.to("GeV").value, tauYY_torus3, color="dodgerblue", lw=2, ls="-.")

plt.axhline(1, lw=2, ls=":", color="dimgray")

dist1 = mlines.Line2D(
    [], [], color="k", marker="", ls="-", lw=2, label=r"$r=0.1\,R(Ly\alpha)$"
)
dist2 = mlines.Line2D(
    [], [], color="crimson", marker="", ls="-", lw=2, label=r"$r=R(Ly\alpha)$"
)
dist3 = mlines.Line2D(
    [], [], color="dodgerblue", marker="", ls="-", lw=2, label=r"$r=10\,R(Ly\alpha)$"
)
disk = mlines.Line2D([], [], color="dimgray", marker="", ls="-", lw=2, label="disk")
blr = mlines.Line2D(
    [], [], color="dimgray", marker="", ls="--", lw=2, label="shell BLR"
)
torus = mlines.Line2D(
    [], [], color="dimgray", marker="", ls="-.", lw=2, label="ring torus"
)
handles = [dist1, dist2, dist3, disk, blr, torus]
plt.legend(handles=handles, fontsize=11, loc=2)

plt.xlim([1, 1e5])
plt.ylim([1e-3, 1e5])
plt.ylabel(r"$\tau_{\gamma \gamma}$")
plt.xlabel(r"$E\,/\,GeV$")
plt.xscale("log")
plt.yscale("log")
plt.show()
fig.savefig("results/figure_14_finke_2016.pdf")

# Protheroe Donea Check
fig, ax = plt.subplots()
M_8 = 10
M_BH = M_8 * 1e8 * M_sun
R_g = ((const.G * M_BH) / (const.c * const.c)).cgs

energy = np.logspace(0, 7) * u.GeV
nu = energy.to(u.Hz, equivalencies=u.spectral())

for (m_dot, M_8) in zip([1, 0.1, 1, 0.1, 0.01], [10, 10, 1, 1, 1]):

    disk_prot = Disk(
        M_BH,
        L_disk=m_dot * 1.26 * 1e46 * M_8 * u.Unit("erg s-1"),
        eta=0.1,
        R_in=10 * R_g,
        R_out=200 * R_g,
        r=0.01 * u.pc,
    )

    tauYY_disk_prot = tau.disk(nu, blob, disk_prot)
    plt.loglog(
        energy.to("GeV").value,
        tauYY_disk_prot,
        lw=2,
        label=r"$\dot{m}=$" + f"{m_dot} ," + r"$M_8=$" + f"{M_8}",
    )

plt.ylabel(r"$\tau_{\gamma \gamma}$")
plt.xlabel("E / GeV")
plt.xlim([1e2, 1e6])
plt.ylim([1e-1, 1e5])
plt.legend(fontsize=12)
plt.show()
