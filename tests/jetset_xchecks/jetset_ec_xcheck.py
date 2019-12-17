import sys

sys.path.append("../../../agnpy")
import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from agnpy.particles import Blob
from agnpy.compton import Compton
from agnpy.targets import RingDustTorus, SphericalShellBLR
from jetset.jet_model import Jet

spectrum_norm = 5e42 * u.Unit("erg")
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

jet = Jet(name="test_torus", electron_distribution="bkn")
jet.set_par("B", val=blob.B)
jet.set_par("N", val=blob.norm.value)
jet.set_par("R", val=np.log10(blob.R_b))
jet.set_par("beam_obj", val=blob.delta_D)
jet.set_par("gmin", val=blob.gamma_min)
jet.set_par("gmax", val=blob.gamma_max)
jet.set_par("gamma_break", val=blob.n_e.gamma_b)
jet.set_par("p", val=blob.n_e.p1)
jet.set_par("p_1", val=blob.n_e.p2)
jet.set_par("z_cosm", val=blob.z)
jet.set_gamma_grid_size(1000)
jet.show_pars()

plt.loglog(
    jet.electron_distribution.gamma,
    jet.electron_distribution.n_gamma,
    lw=2,
    label="jetset",
)
plt.loglog(blob.gamma, blob.n_e(blob.gamma), lw=2, ls="--", label="agnpy")
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$n_e(\gamma)$")
plt.legend()
plt.show()
print(f"total density {blob.norm:.2e}")
print(f"total energy {blob.W_e:.2e}")

distances = [1e18, 1e20, 1e21] * u.cm

# torus parameters
M_sun = const.M_sun.cgs
M_BH = 1.2 * 1e9 * M_sun
L_disk = 2 * 1e46 * u.Unit("erg s-1")
T_dt = 1e3 * u.K
epsilon_dt = 2.7 * ((const.k_B * T_dt) / (const.m_e * const.c * const.c)).decompose()
csi_dt = 0.1

dt1 = RingDustTorus(M_BH, L_disk, csi_dt, epsilon_dt, r=distances[0])
dt2 = RingDustTorus(M_BH, L_disk, csi_dt, epsilon_dt, r=distances[1])
dt3 = RingDustTorus(M_BH, L_disk, csi_dt, epsilon_dt, r=distances[2])

dt1.set_phi_size(800)
compton = Compton(blob)

nu = np.logspace(15, 30, 50) * u.Hz
sed1 = compton.sed_flux_ring_torus(nu, dt1)
sed2 = compton.sed_flux_ring_torus(nu, dt2)
sed3 = compton.sed_flux_ring_torus(nu, dt3)

agnpy_seds = [sed1, sed2, sed3]

jet.add_EC_component("EC_DT")
jet.spectral_components.Sync.state = "off"
jet.show_pars()

jet.set_par("tau_DT", dt1.csi_dt)
jet.set_par("T_DT", dt1.T_dt)
jet.set_par("R_DT", dt1.R_dt)

nu_seds = []
jetset_seds = []

for distance in distances:
    jet._blob.R_H = distance.value
    jet.eval()
    x, y = jet.get_SED_points(name="EC_DT")
    nu_seds.append(x)
    jetset_seds.append(y)

for freq, sed, r, ls in zip(nu_seds, jetset_seds, distances, ("-", "--", ":")):
    plt.loglog(freq, sed, lw=2, ls=ls, color="k", label=f"jetset, r = {r:.2e}")

for sed, dt, ls in zip(agnpy_seds, (dt1, dt2, dt3), ("-", "--", ":")):
    plt.loglog(nu, sed, lw=2, color="crimson", ls=ls, label=f"agnpy, r = {dt.r:.2e} cm")

plt.legend(bbox_to_anchor=(0.01, 0.5), fontsize=10)
plt.title("Dust Torus")
plt.ylim([1e-28, 1e-14])
plt.xlim([1e16, 1e29])
plt.xlabel(r"$\nu / Hz$")
plt.ylabel(r"$\nu F_{\nu} / (erg\,cm^{-2}\,s^{-1})$")
plt.show()
