import sys

sys.path.append("../")
import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from agnpy.emission_region import Blob
from agnpy.synchrotron import Synchrotron
from astropy.coordinates import Distance
import time

spectrum_norm = 1e48 * u.Unit("erg")
parameters = {"p": 2.5, "gamma_min": 1e2, "gamma_max": 1e7}
spectrum_dict = {"type": "PowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 1 * u.G
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
print(f"total density {blob.norm:.2e}")
print(f"total energy {blob.W_e:.2e}")

nu = np.logspace(8, 22, 200) * u.Hz

synch = Synchrotron(blob)
import IPython

IPython.embed()
sed_synch = synch.sed_flux(nu)
sed_synch_SSA = synch.sed_flux(nu, SSA=True)
plt.loglog(nu, sed_synch, label="synch")
plt.loglog(nu, sed_synch_SSA, ls=":", label="synch + SSA")
plt.legend()
plt.show()

quit()
tstart = time.time()
sed_ssc = synchrotron.ssc_sed_flux(nu)
tstop = time.time()
elapsed = tstop - tstart
print(f"elapsed time SSC SED {elapsed:.2e} s")

print("producing figure 7.4 of Dermer")
fig, ax = plt.subplots()
plt.loglog(nu, sed_synch, lw=2.5, color="k")
plt.loglog(nu, sed_ssc, lw=2.5, color="k", ls="--")

# change the maximum Lorentz factor to 1e7
parameters = {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e5}
spectrum_dict = {"type": "PowerLaw", "parameters": parameters}
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
synchrotron = Synchrotron(blob)

sed_synch = synchrotron.sed_flux(nu)
sed_ssc = synchrotron.ssc_sed_flux(nu)
plt.loglog(nu, sed_synch, lw=2.5, color="crimson")
plt.loglog(nu, sed_ssc, lw=2.5, color="crimson", ls="--")

syn = mlines.Line2D([], [], color="dimgray", marker="", ls="-", lw=2, label="syn")
ssc = mlines.Line2D([], [], color="dimgray", marker="", ls="--", lw=2, label="SSC")
gamma = mlines.Line2D(
    [], [], color="k", marker="", ls="-", lw=2, label=r"$\gamma_2=10^7$"
)
gamma_2 = mlines.Line2D(
    [], [], color="crimson", marker="", ls="-", lw=2, label=r"$\gamma_2=10^5$"
)

plt.legend(handles=[syn, ssc, gamma, gamma_2], fontsize=12, loc=2)

plt.xlabel(r"$\nu\,/\,Hz$")
plt.ylabel(r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$")
# plt.grid(which="both")
plt.xlim([1e9, 1e30])
plt.ylim([1e-12, 1e-9])
plt.show()
fig.savefig("results/Figure_7_4_Dermer_2009.png")
fig.savefig("results/Figure_7_4_Dermer_2009.pdf")
