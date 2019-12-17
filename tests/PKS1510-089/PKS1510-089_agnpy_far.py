# coding: utf-8

# In[1]:


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


# In[2]:


spectrum_norm = 9e47 * u.Unit("erg")
parameters = {"p1": 1.9, "p2": 3.7, "gamma_b": 300, "gamma_min": 1.5, "gamma_max": 3e5}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
R_b = 2e16 * u.cm
B = 0.35 * u.G
z = 0.361
delta_D = 35.5
Gamma = 22.5

blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
plt.loglog(blob.gamma, blob.n_e(blob.gamma))


# In[3]:


print(f"normalization {blob.norm:.2e}")


# In[4]:


# disk parameters
M_BH = np.power(10, 8.20) * const.M_sun
L_disk = 1.13 * 1e46 * u.Unit("erg s-1")
R_g = ((const.G * M_BH) / (const.c * const.c)).cgs
eta = 0.1
R_in = 3 * R_g
R_out = 200 * R_g
# BLR parameters
lambda_H_beta = 486.13615 * u.nm
epsilon_line = lambda_H_beta.to("erg", equivalencies=u.spectral()).value / MEC2
L_BLR = 6.7 * 1e45 * u.Unit("erg s-1")
csi_line = 0.1
R_line = 2.6 * 1e17 * u.cm
# torus parameters
T_dt = 1e3 * u.K
epsilon_dt = 2.7 * ((const.k_B * T_dt) / (const.m_e * const.c * const.c)).decompose()
R_dt = 6.5 * 1e18 * u.cm
csi_dt = 0.6

r = 3e18 * u.cm

compton = Compton(blob)
synchro = Synchrotron(blob)
nu = np.logspace(14, 28, 50) * u.Hz

disk = Disk(M_BH, L_disk, eta, R_in, R_out, r)
blr = SphericalShellBLR(M_BH, L_disk, csi_line, epsilon_line, R_line, r)
dt = RingDustTorus(M_BH, L_disk, csi_dt, epsilon_dt, r, R_dt)


# In[5]:


nu = np.logspace(8, 30, 80) * u.Hz
# set sizes
blob.set_gamma_size(500)
sed_syn = synchro.sed_flux(nu, SSA=True)
sed_ssc = synchro.ssc_sed_flux(nu)
sed_ec_disk = compton.sed_flux_disk(nu, disk)
sed_ec_blr = compton.sed_flux_shell_blr(nu, blr)
dt.set_phi_size(600)
sed_ec_torus = compton.sed_flux_ring_torus(nu, dt)


# In[6]:


fig, ax = plt.subplots()
plt.loglog(
    nu,
    sed_syn + sed_ssc + sed_ec_disk + sed_ec_blr + sed_ec_torus,
    lw=2,
    color="crimson",
    label="total",
)
plt.loglog(nu, sed_syn, lw=2, label="Synchrotron", color="dodgerblue", ls="--")
plt.loglog(nu, sed_ssc, lw=2, label="SSC", color="darkorange", ls="--")
plt.loglog(nu, sed_ec_disk, lw=2, label="EC Disk", color="forestgreen", ls="--")
plt.loglog(nu, sed_ec_blr, lw=2, label="EC BLR", color="darkviolet", ls="--")
plt.loglog(nu, sed_ec_torus, lw=2, label="EC Torus", color="rosybrown", ls="--")

t = Table.read("astropy_sed.ecsv")
plt.errorbar(
    t["x"], t["y"], yerr=t["dy"], ls="", marker="o", color="k", label="PKS1510-089"
)
plt.xlabel(r"$\nu\;/\;Hz$")
plt.ylabel(r"$\nu F_{\nu}\;/\;\mathrm{erg}\;\mathrm{cm}^{-2}\;\mathrm{s}^{-1}$")
# plt.grid(which="both")
plt.xlim([1e9, 1e27])
plt.ylim([1e-15, 1e-8])
plt.legend(fontsize=10, bbox_to_anchor=[1.02, 0.8])
plt.show()
fig.savefig("../results/PKS1510-089.png")


# In[7]:


tau = Tau()
# import IPython
# IPython.embed()

tauYY_disk = tau.disk(nu, blob, disk)
tauYY_blr = tau.shell_blr(nu, blob, blr)
tauYY_dt = tau.dust_torus(nu, blob, dt)


# In[8]:


plt.loglog(nu, tauYY_disk, color="k", lw=2, ls="-")
plt.loglog(nu, tauYY_blr, color="k", lw=2, ls="--")
plt.loglog(nu, tauYY_dt, color="k", lw=2, ls="-.")
