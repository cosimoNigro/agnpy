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
from agnpy.spectral_constraints import SpectralConstraints

# parameters of the blob with a narrow EED
B0 = 10.1 * u.G
gmin0 = 500.0 * 2
gmax0 = 800.0 * 2
gbreak = gmin0
z = 0.01
Gamma = 17
delta_D = 1.99 * Gamma
r0 = 1.0e15 * u.cm
xi = 1.0e-4
nu = np.logspace(8, 26, 200) * u.Hz
norm = 0.1 * u.Unit("erg cm-3")

parameters = {
    "p1": 2.0,
    "p2": 3.0,
    "gamma_b": gbreak,
    "gamma_min": gmin0,
    "gamma_max": gmax0,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}


#####################
# test one with emission region in the center of the DT
L_disk = 0.91e45 * u.Unit("erg s-1")
xi_dt = 0.6
T_dt = 100 * u.K
R_dt = 1.0e18 * u.cm
h = 0.01 * R_dt

## test with lower numbers, gives virtually the same
# B0/=10
# T_dt/=10
# L_disk/=100

blob1 = Blob(r0, z, delta_D, Gamma, B0, norm, spectrum_dict, xi=xi)
dt1 = RingDustTorus(L_disk, xi_dt, T_dt, R_dt=R_dt)

# energy density of DT radiation field in the blob
u_dt1 = dt1.u(h, blob1)
u_synch1 = blob1.u_ph_synch
print(
    "energy density in the blob, DT radiation: ",
    u_dt1,
    "synchrotron photons: ",
    u_synch1,
)
dt1_sed = dt1.sed_flux(nu, z)
# energy density was set to be the same

synch1 = Synchrotron(blob1, ssa=False)
synch1_sed = synch1.sed_flux(nu)

ssc1 = SynchrotronSelfCompton(blob1, synch1)
ssc1_sed = ssc1.sed_flux(nu)

ec_dt1 = ExternalCompton(blob1, dt1, h)
ec_dt1_sed = ec_dt1.sed_flux(nu)

ssc1_total = (np.trapz(ssc1_sed / nu, nu)).to("erg cm-2 s-1")
ec_dt1_total = (np.trapz(ec_dt1_sed / nu, nu)).to("erg cm-2 s-1")
print("SSC total=", ssc1_total, ", EC DT total=", ec_dt1_total)
# similar density of radiation but there is a factor of ~1.4 difference in the integrated flux
# this probably comes from different angular distribution of the radiation

sc1 = SpectralConstraints(blob1)
gbreakssc = sc1.gamma_break_SSC
gbreakdt = sc1.gamma_break_EC_DT(dt1, h)
print(
    "break SSC=", gbreakssc, ", break EC=", gbreakdt, ", ratio: ", gbreakssc / gbreakdt
)
# values of the break scale with energy density so they are the same

gmaxssc = sc1.gamma_max_SSC
gmaxdt = sc1.gamma_max_EC_DT(dt1, h)
print("max SSC=", gmaxssc, ", max EC=", gmaxdt, ", ratio: ", gmaxssc / gmaxdt)
# the same with values of the maximum gamma factor

plt.rc("figure", figsize=(7.5, 5.5))
plt.rc("font", size=12)
plt.rc("axes", grid=True)
plt.rc("grid", ls=":")
sed_x_label = r"$\nu\,/\,Hz$"
sed_y_label = r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$"

plt.loglog(
    nu / Gamma, synch1_sed, color="k", ls="-", lw=1, label="Synchr. (shifted)"
)  # /np.power(delta_D,4)*np.power(R_dt/r0,2)
plt.loglog(nu, ssc1_sed, color="r", ls="-", lw=1, label="SSC")  #
plt.loglog(nu, ec_dt1_sed, color="b", ls="-", lw=1, label="EC DT")  #
plt.loglog(
    nu * Gamma, dt1_sed, color="g", ls="-", lw=1, label="DT1 (shifted)"
)  # *np.power(Gamma,2)
plt.ylim(1e-19, 1e-6)
plt.xlim(1e8, 1e27)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(sed_x_label)
plt.ylabel(sed_y_label)

######################
# test with emission region further along the jet

h = 0.9 * R_dt
dist = np.sqrt(h * h + R_dt * R_dt)
factor = blob1.Gamma * (1 - blob1.Beta * h / dist)

# before we had boosting by Gamma, now by factor
T_dt2 = T_dt / (factor / blob1.Gamma)  # shift the energies
L_disk2 = (
    L_disk * np.power(factor / blob1.Gamma, -2) * np.power(dist / R_dt, 2)
)  # shift the luminosity for the beaming and larger distance
dt2 = RingDustTorus(L_disk2, xi_dt, T_dt2, R_dt=R_dt)
ec_dt2 = ExternalCompton(blob1, dt2, h)
ec_dt2_sed = ec_dt2.sed_flux(nu)
dt2_sed = dt2.sed_flux(nu, z)

u_dt2 = dt2.u(h, blob1)
print("DT energy density (test 1)=", u_dt1, ", (test2)=", u_dt2)
# same energy densities

ec_dt2_total = (np.trapz(ec_dt2_sed / nu, nu)).to("erg cm-2 s-1")
print(
    "SSC total=",
    ssc1_total,
    ", EC DT total (test1)=",
    ec_dt1_total,
    ", (test2)=",
    ec_dt2_total,
)
# the obtained spectrum from EC in test 1 and 2 is the same, but even while the radiation density is the same, it has different angular distribution. Still, both test1 and test2 are nearly head-on in blob's frame, so the resulting EC is very similar

angle2 = np.arccos((h / dist - blob1.Beta) / (1 - blob1.Beta * h / dist)).to("deg")
print("test2, photons in blob at angle", angle2)

plt.loglog(nu * factor, dt2_sed, color="g", ls=":", lw=1, label="DT2 (shifted)")
plt.loglog(nu, ec_dt2_sed, color="b", ls=":", lw=1, label="EC DT2 (over EC DT1)")


######################
# test with emission region far enough along the jet that beaming disappears

h = 17.0 * R_dt
dist = np.sqrt(h * h + R_dt * R_dt)
factor = blob1.Gamma * (1 - blob1.Beta * h / dist)

print(factor)
# before we had boosting by Gamma, now by factor
T_dt3 = T_dt / (factor / blob1.Gamma)  # shift the energies
L_disk3 = (
    L_disk * np.power(factor / blob1.Gamma, -2) * np.power(dist / R_dt, 2)
)  # shift the luminosity for the beaming and larger distance
dt3 = RingDustTorus(L_disk3, xi_dt, T_dt3, R_dt=R_dt)
ec_dt3 = ExternalCompton(blob1, dt3, h)
ec_dt3_sed = ec_dt3.sed_flux(nu)
dt3_sed = dt3.sed_flux(nu, z)

u_dt3 = dt3.u(h, blob1)
print("DT energy density (test 1)=", u_dt1, ", (test3)=", u_dt3)
# same energy densities

ec_dt3_total = (np.trapz(ec_dt3_sed / nu, nu)).to("erg cm-2 s-1")
print(
    "SSC total=",
    ssc1_total,
    ", EC DT total (test1)=",
    ec_dt1_total,
    ", (test3)=",
    ec_dt3_total,
)
# now with the same energy density, but with photons perpendicular to the jet direction in the frame of the blob we get a factor of ~4 lower flux.
# the factor of 2 should come from the cross section,
# and another factor of 2 from transformation of energy of the photon to the electron's frame
# seems consistent
print(ec_dt1_total / ec_dt3_total)
angle3 = np.arccos((h / dist - blob1.Beta) / (1 - blob1.Beta * h / dist)).to("deg")
print("test3, photons in blob at angle", angle3)

plt.loglog(nu * factor, dt3_sed, color="g", ls="-.", lw=1, label="DT3 (shifted)")
plt.loglog(nu, ec_dt3_sed, color="b", ls="-.", lw=1, label="EC DT3")

plt.legend()
plt.show()

# sys.exit()
