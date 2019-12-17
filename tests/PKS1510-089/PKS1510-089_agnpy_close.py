import sys

sys.path.append("../../")
import numpy as np
import matplotlib.pyplot as plt
from agnpy.particles import Blob
from agnpy.targets import RingDustTorus, SphericalShellBLR, Disk
from agnpy.compton import Compton
from agnpy.synchrotron import Synchrotron
from agnpy.tau import Tau
import astropy.units as u
import astropy.constants as const
from astropy.table import Table

MEC2 = (const.m_e * const.c * const.c).cgs.value

spectrum_norm = 8e47 * u.Unit("erg")
parameters = {"p1": 1.9, "p2": 3.5, "gamma_b": 130, "gamma_min": 2.0, "gamma_max": 3e5}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
R_b = 2e16 * u.cm
B = 0.35 * u.G
z = 0.36
delta_D = 35.3
Gamma = 22.5

blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
# plt.loglog(blob.gamma, blob.n_e(blob.gamma))
# plt.show()
print(f"normalization {blob.norm:.2e}")

# disk parameters
M_BH = np.power(10, 8.20) * const.M_sun
L_disk = 1.13 * 1e46 * u.Unit("erg s-1")
R_g = ((const.G * M_BH) / (const.c * const.c)).cgs
eta = 0.1
R_in = 6 * R_g
R_out = 1000 * R_g
# BLR parameters
lambda_H_beta = 486.13615 * u.nm
# epsilon_line = lambda_H_beta.to("erg", equivalencies=u.spectral()).value / MEC2
epsilon_line = 2e-5
csi_line = 0.1
R_line = 2.6 * 1e17 * u.cm
# torus parameters
T_dt = 1e3 * u.K
epsilon_dt = 2.7 * ((const.k_B * T_dt) / (const.m_e * const.c * const.c)).decompose()
R_dt = 6.5 * 1e18 * u.cm
csi_dt = 0.6

r = 7e17 * u.cm

compton = Compton(blob)
synchro = Synchrotron(blob)
nu = np.logspace(14, 28, 50) * u.Hz

disk = Disk(M_BH, L_disk, eta, R_in, R_out, r)
print(f"gravitaitonal radius -> {disk.R_g:.2e}")
blr = SphericalShellBLR(M_BH, L_disk, csi_line, epsilon_line, R_line, r)
dust_torus = RingDustTorus(M_BH, L_disk, csi_dt, epsilon_dt, r, R_dt)


nu = np.logspace(8, 30, 110) * u.Hz
# set sizes
blob.set_gamma_size(500)
sed_syn = synchro.sed_flux(nu, SSA=True)
sed_ssc = synchro.ssc_sed_flux(nu)
sed_torus = dust_torus.sed(nu, blob)
sed_disk = disk.sed(nu, blob)
sed_ec_disk = compton.sed_flux_disk(nu, disk)
sed_ec_blr = compton.sed_flux_shell_blr(nu, blr)
dust_torus.set_phi_size(600)
sed_ec_torus = compton.sed_flux_ring_torus(nu, dust_torus)

total_sed = (
    sed_syn + sed_torus + sed_disk + sed_ssc + sed_ec_disk + sed_ec_blr + sed_ec_torus
)

tau = Tau()

tauYY_blr = tau.shell_blr(nu, blob, blr)
tauYY_torus = tau.dust_torus(nu, blob, dust_torus)

attenuation = np.exp(-(tauYY_blr + tauYY_torus))

fig, ax = plt.subplots()
ax.loglog(nu, total_sed * attenuation, lw=2, color="crimson", label="total")
ax.loglog(nu, sed_syn, lw=2, label="Synchrotron", color="dodgerblue", ls="--")
ax.loglog(nu, sed_disk, lw=2, label="Disk", color="dimgray", ls="-.")
ax.loglog(nu, sed_torus, lw=2, label="Torus", color="dimgray", ls=":")
ax.loglog(nu, sed_ssc, lw=2, label="SSC", color="darkorange", ls="--")
ax.loglog(nu, sed_ec_disk, lw=2, label="EC Disk", color="forestgreen", ls="--")
ax.loglog(nu, sed_ec_blr, lw=2, label="EC BLR", color="darkviolet", ls="--")
ax.loglog(nu, sed_ec_torus, lw=2, label="EC Torus", color="rosybrown", ls="--")

ax.set_xlabel(r"$\nu\;/\;Hz$")
ax.set_ylabel(r"$\nu F_{\nu}\;/\;\mathrm{erg}\;(\mathrm{cm}^{-2}\;\mathrm{s}^{-1})$")

t = Table.read("astropy_sed.ecsv")
ax.errorbar(
    t["x"], t["y"], yerr=t["dy"], ls="", marker="o", color="k", label="PKS1510-089"
)

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel(r"$\tau_{\gamma \gamma}$")  # we already handled the x-label with ax1
ax2.loglog(
    nu, tauYY_blr, color="k", ls="--", label=r"$\tau_{\gamma \gamma,\;{\rm BLR}}$"
)
ax2.loglog(
    nu, tauYY_torus, color="k", ls=":", label=r"$\tau_{\gamma \gamma,\;{\rm torus}}$"
)

# plt.grid(which="both")
ax.set_xlim([1e9, 1e30])
ax.set_ylim([1e-16, 1e-6])
# Put a legend to the right of the current axis
ax.legend(loc=2, ncol=2)
ax2.legend(loc=1)
plt.show()
fig.savefig("../PKS1510-089_agnpy.png")

# save everything in an astropy table to re-plot easily
names = [
    "nu",
    "nuFnu_syn",
    "nuFnu_torus",
    "nuFnu_disk",
    "nuFnu_ssc",
    "nuFnu_ec_disk",
    "nuFnu_ec_blr",
    "nuFnu_ec_torus",
    "tauYY_blr",
    "tauYY_torus",
]

data = [
    nu,
    sed_syn,
    sed_torus,
    sed_disk,
    sed_ssc,
    sed_ec_disk,
    sed_ec_blr,
    sed_ec_torus,
    tauYY_blr,
    tauYY_torus,
]

model_table = Table(data, names=names)
model_table.meta = {"obj_name": "PKS 1510-089", "z": 0.361}
model_table.write("PKS1510-089_model.ecsv", format="ascii.ecsv")
