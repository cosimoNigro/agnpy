import sys

sys.path.append("../../")
import numpy as np
import astropy.units as u
import astropy.constants as const
from agnpy.emission_regions import Blob
from agnpy.targets import SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.absorption import Absorption
import matplotlib.pyplot as plt

MEC2 = const.m_e * const.c * const.c

# define the blob
spectrum_norm = 1e47 * u.erg
parameters = {"p": 2.8, "gamma_min": 10, "gamma_max": 1e6}
spectrum_dict = {"type": "PowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 0.56 * u.G
z = 0
delta_D = 40
Gamma = 40
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
print("blob definition:")
print(blob)

# disk parameters
M_sun = const.M_sun.cgs
M_BH = 1.2 * 1e9 * M_sun
R_g = ((const.G * M_BH) / (const.c * const.c)).cgs
L_disk = 2 * 1e46 * u.Unit("erg s-1")
eta = 1 / 12
R_in = 6 * R_g
R_out = 200 * R_g
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out)
print("disk definition:")
print(disk)

# blr definition
epsilon_line = 2e-5
csi_line = 0.024
R_line = 1e17 * u.cm
blr = SphericalShellBLR(disk, csi_line, epsilon_line, R_line)
print("blr definition:")
print(blr)

# dust torus definition
T_dt = 1e3 * u.K
epsilon_dt = 2.7 * ((const.k_B * T_dt) / (const.m_e * const.c * const.c)).decompose()
csi_dt = 0.1
dt = RingDustTorus(disk, csi_dt, epsilon_dt)
print("torus definition:")
print(dt)

# let us make a 2D plot of where s will be bigger than 1
r = 1.1e16 * u.cm

absorption_disk = Absorption(blob, disk, r=r)
absorption_blr = Absorption(blob, blr, r=r)
absorption_dt = Absorption(blob, dt, r=r)

# a check on the values for which s > 1 in the case of the disk
E = np.logspace(0, 5) * u.GeV
epsilon_1 = (E / MEC2).decompose().value
epsilon_disk = disk._epsilon_mu(absorption_disk.mu, r.value)
E_disk = (epsilon_disk * MEC2).to("eV")


def where_s_1(mu, r):
    s = epsilon_1 * disk._epsilon_mu(mu, r) * (1 - mu) / 2
    return E[s > 1][0]


for _r in [1e15, 1e16, 1e17]:
    E_thr = [where_s_1(mu, _r).value for mu in absorption_disk.mu]
    plt.semilogy(absorption_disk.mu, E_thr, label=f"r = {_r:.0e}")

plt.xlabel(r"$\mu$")
plt.ylabel("E (s>1) / GeV")
plt.legend()
plt.show()


# let's plot the opacities

nu = E.to("Hz", equivalencies=u.spectral())

tau_disk = absorption_disk.tau(nu)
tau_blr = absorption_blr.tau(nu)
tau_dt = absorption_dt.tau(nu)

fig, ax = plt.subplots()
ax.loglog(E, tau_disk, lw=2, ls="-", label="SS disk")
ax.loglog(E, tau_blr, lw=2, ls="--", label="spherical shell BLR")
ax.loglog(E, tau_dt, lw=2, ls="-.", label="ring dust torus")
ax.legend()
ax.set_xlabel("E / GeV")
ax.set_ylabel(r"$\tau_{\gamma \gamma}$")
ax.set_xlim([1, 1e5])
ax.set_ylim([1e-3, 1e5])
plt.show()
