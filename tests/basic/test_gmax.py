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
gmax0 = 3.0e4
gbreak = 300.0
z = 0.94
delta_D = 20
Gamma = 17
# r0=1.e16 * u.cm
r0 = 1.0e14 * u.m
dist = 3.0e16 * u.cm
xi = 1.0e-4
nu = np.logspace(8, 23) * u.Hz
norm = 10.0 * u.Unit("cm-3")

parameters = {
    "p1": 2.0,
    "p2": 3.9,
    "gamma_b": gbreak,
    "gamma_min": gmin0,
    "gamma_max": gmax0,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}

blob = Blob(r0, z, delta_D, Gamma, B0, norm, spectrum_dict, xi=xi)

# plt.loglog(blob.gamma, blob.n_e (blob.gamma))

# gmaxconf=blob.R_b/(const.m_e*const.c / (blob.B.si * const.e.si)).to("cm")
# confinement of particles inside the blob:
gmaxconf = blob.gamma_max_larmor
# computing larmor radius of this electron, should be of the size of the blob
# R_L = 33.36 km * (p/(GeV/c)) * (G/B) * Z^-1
# https://w3.iihe.ac.be/~aguilar/PHYS-467/PA3.pdf
rlarmor = (33.36 * u.km * gmaxconf * 511.0e3 / 1.0e9 / (blob.B / u.G)).to("cm")

# both values are similar
print("R_L (gmaxconf)=", rlarmor, "R_b=", blob.R_b)

# now maximum from balistic time
gmaxbal = blob.gamma_max_ballistic

# compute acceleration time for those electrons
# eq 2 from https://arxiv.org/abs/1208.6200a, note that this is rough scaling accurate to ~10%
tau_acc = 1.0 * gmaxbal * 511.0e3 / 1.0e9 / (blob.xi / 1.0e-4 * blob.B / u.G) * u.s
# during this time side of R_b of the jet should pass through the blob (in the blob frame!)
dist_cross = (tau_acc * const.c).to("cm")

# again both values are similar
print(f"dist_cross (tau_acc(gmaxbal))={dist_cross:.2e}, R_b={blob.R_b:.2e}")


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
