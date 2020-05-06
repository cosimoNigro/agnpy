# macro for testing various limits on the gamma factors of electrons

import numpy as np
import math
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Distance
import matplotlib.pyplot as plt

plt.ion()
import sys

sys.path.append("../../")
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import ExternalCompton, SynchrotronSelfCompton
from agnpy.targets import SSDisk, SphericalShellBLR, RingDustTorus

# parameters of the blob
B0 = 0.1 * u.G
gmin0 = 10.0
gmax0 = 3.0e3
gbreak = 300.0
z = 0.94
delta_D = 20
Gamma = 17
# r0=1.e16 * u.cm
r0 = 1.0e14 * u.m
dist = 3.0e16 * u.cm
xi = 1.0e-4
nu = np.logspace(8, 26,200) * u.Hz
norm = 15000.0 * u.Unit("cm-3")

parameters = {
    "p1": 2.0,
    "p2": 3.,
    "gamma_b": gbreak,
    "gamma_min": gmin0,
    "gamma_max": gmax0,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}

blob = Blob(r0, z, delta_D, Gamma, B0, norm, spectrum_dict, xi=xi)

# plt.loglog(blob.gamma, blob.n_e (blob.gamma))

#############################################
# limits from confinement of particles inside the blob:
gmaxconf = blob.gamma_max_larmor
# computing larmor radius of this electron, should be of the size of the blob
# R_L = 33.36 km * (p/(GeV/c)) * (G/B) * Z^-1
# https://w3.iihe.ac.be/~aguilar/PHYS-467/PA3.pdf
rlarmor = (33.36 * u.km * gmaxconf * 511.0e3 / 1.0e9 / (blob.B / u.G)).to("cm")

# both values are similar
print("R_L (gmaxconf)=", rlarmor, "R_b=", blob.R_b)

#############################################
# now maximum from balistic time
gmaxbal = blob.gamma_max_ballistic

# compute acceleration time for those electrons
# eq 2 from https://arxiv.org/abs/1208.6200a, note that this is rough scaling accurate to ~10%
tau_acc = 1.0 * gmaxbal * 511.0e3 / 1.0e9 / (blob.xi / 1.0e-4 * blob.B / u.G) * u.s
# during this time side of R_b of the jet should pass through the blob (in the blob frame!)
dist_cross = (tau_acc * const.c).to("cm")

# again both values are similar
print(f"dist_cross (tau_acc(gmaxbal))={dist_cross:.2e}, R_b={blob.R_b:.2e}")


#############################################
# now maximum from synchrotron losses
gmaxsyn = blob.gamma_max_synch

# calculate t_acc
tau_acc = 1.0 * gmaxsyn * 511.0e3 / 1.0e9 / (blob.xi / 1.0e-4 * blob.B / u.G) * u.s
# calculate synchrotron energy loss from the well known formula:
# dE/dt = 4/3 * gamma^2 *U_b * c  * sigma_T
Ub = (blob.B / u.G) ** 2 / (8 * np.pi) * u.Unit("erg cm-3")
dEdt = 4.0 / 3.0 * (gmaxsyn) ** 2 * Ub * (const.c * const.sigma_T).to("cm3 s-1")
Elost = (dEdt * tau_acc).to("GeV")
Emax = (gmaxsyn * const.m_e * (const.c) ** 2).to("GeV")

# both values are similar
print(f"E(gmaxsyn) = {Emax:.2e}, Elost = {Elost:.2e}")

# print(gmaxconf, gmaxbal, gmaxsyn)

#############################################
# check of synchrotron cooling break

# eq F.1 from https://ui.adsabs.harvard.edu/abs/2020arXiv200107729M/abstract
# gammab = 3pi me c^2 / sigma_T B^2 R
# here we use 6 instead of 3 because we only have synchrotron losses and compare then
# with dynamical time scale of crossing R

# now compare the value from the class (computed using SI units) with the formula using CGS units
gamma_b = blob.gamma_break_synch
gamma_break_check = (
    6
    * np.pi
    * 511.0e3
    * u.eV.to("erg")
    / (0.665e-24 * (blob.B / u.G) ** 2 * (blob.R_b / u.cm))
)

print(f"gamma_break = {gamma_b:.5e}, gamma_break_check = {gamma_break_check:.5e}")

#############################################
# limits for SSC 
#print(blob.u_e)
#print(blob.u_dens_synchr)

#redo blob without beaming
Gamma=1.01
delta_D=1.1
z=0.01
blob1 = Blob(r0, z, delta_D, Gamma, B0*10., norm, spectrum_dict, xi=xi)

u_dens_synchr=blob1.u_dens_synchr # energy density of synchr photons
energy_flux_predicted=(blob1.u_dens_synchr * blob1.V_b * const.c.cgs/blob1.R_b *np.power(blob1.d_L,-2)/(4*np.pi)).to("erg cm-2 s-1")

synch1 = Synchrotron(blob1, ssa=False)
synch1_sed = synch1.sed_flux(nu)

energy_flux_sim = np.trapz(synch1_sed/(nu*const.h.cgs), nu*const.h.cgs)
print(f"predicted energy flux: {energy_flux_predicted:.5e}, simulated energy flux: {energy_flux_sim:.5e}")
# similar values, but not very precise

ssc1 = SynchrotronSelfCompton(blob1, synch1)
ssc1_sed = ssc1.sed_flux(nu)

print("UB/Usynch = ",blob1.u_B/u_dens_synchr)
print("SED_synch/SED_SSC=",energy_flux_sim/ np.trapz(ssc1_sed/(nu*const.h.cgs), nu*const.h.cgs))
# also here there is a factor ~1.5 of difference  
print("break_synchr/break_SSC = ",blob1.gamma_break_synch/blob1.gamma_break_SSC)

# SSC is at the same level as Synchr. so the cooling breaks should be also similar:


plt.rc("figure", figsize=(7.5, 5.5))
plt.rc("font", size=12)
plt.rc("axes", grid=True)
plt.rc("grid", ls=":")
sed_x_label = r"$\nu\,/\,Hz$"
sed_y_label = r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$"

plt.loglog(nu, synch1_sed, color="k", ls="-", lw=1, label="Synchr.")  #
plt.loglog(nu, ssc1_sed, color="r", ls="-", lw=1, label="SSC")  #
plt.ylim(1e-15, 1e-10)
plt.xlim(1e8, 1e27)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(sed_x_label)
plt.ylabel(sed_y_label)
plt.legend()
plt.show()

