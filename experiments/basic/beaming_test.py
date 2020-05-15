import numpy as np
import math
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys

sys.path.append("/home/jsitarek/zdalne/agnpy/agnpy/")
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron

nu = np.logspace(8, 27, 100) * u.Hz  # for SED calculations

spectrum_norm = 1.0 * u.Unit("erg cm-3")
parameters = {
    "p1": 1.5,
    "p2": 2.5,
    "gamma_b": 1.0e3,
    "gamma_min": 1,
    "gamma_max": 1.0e6,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
delta_D = 1.01
Gamma = 1.01
B = 1.0 * u.G
r_b = 1.0e15 * u.cm
# no beaming
blob0 = Blob(r_b, 0.01, delta_D, Gamma, B, spectrum_norm, spectrum_dict, xi=0.01)

synch0 = Synchrotron(blob0, ssa=True)
synch0_sed = synch0.sed_flux(nu)

# beaming
delta_D = 20
Gamma = 15
blob1 = Blob(r_b, 0.01, delta_D, Gamma, B, spectrum_norm, spectrum_dict, xi=0.01)

synch1 = Synchrotron(blob1, ssa=True)
synch1_sed = synch1.sed_flux(nu)

# doing beaming by hand: dN/dOmega dt depsilon scales like D^2, and E^2 in SED scales with another D^2
synch0_sed_scale = synch0_sed * delta_D ** 4
nu_scale = nu * delta_D
plt.rc("figure", figsize=(7.5, 5.5))
plt.rc("font", size=12)
plt.rc("axes", grid=True)
plt.rc("grid", ls=":")
sed_x_label = r"$\nu\,/\,Hz$"
sed_y_label = r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$"

plt.loglog(nu, synch0_sed, color="k", ls=":", lw=1, label="No beaming")  #
plt.loglog(nu, synch1_sed, color="r", ls=":", lw=1, label="Beaming")  #
plt.loglog(
    nu_scale, synch0_sed_scale * 1.1, color="b", ls=":", lw=1, label="scaled"
)  # 1.1 so both curves show up

plt.ylim(1e-15, 1e-7)
plt.xlim(1e8, 1e27)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(sed_x_label)
plt.ylabel(sed_y_label)
plt.legend()
plt.show()
