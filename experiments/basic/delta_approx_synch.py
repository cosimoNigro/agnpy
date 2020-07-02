# test delta approximation for the synchrotron SED
import sys
sys.path.append("../../")
import astropy.units as u
import numpy as np
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron, nu_synch_peak
import matplotlib.pyplot as plt

# set the spectrum normalisation (total energy in electrons in this case)
spectrum_norm = 1e48 * u.Unit("erg")
# define the spectral function through a dictionary
spectrum_dict = {
    "type": "BrokenPowerLaw",
    "parameters": {"p1": 2.5, "p2":3.5, "gamma_b":1e4, "gamma_min": 1e2, "gamma_max": 1e7}
}
R_b = 1e16 * u.cm
B = 1 * u.G
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

nu = np.logspace(8, 23) * u.Hz

synch = Synchrotron(blob)

sed = synch.sed_flux(nu)
sed_delta_approx = synch.sed_flux_delta_approx(nu)
sed_peak_nu = synch.sed_peak_nu(nu)
nu_synch_mono = nu_synch_peak(blob.B_cgs, spectrum_dict["parameters"]["gamma_b"])

plt.loglog(nu, sed, ls="-")
plt.loglog(nu, sed_delta_approx, ls=":")
plt.axvline(sed_peak_nu.to_value("Hz"), ls="--", color="k")
plt.axvline(nu_synch_mono.to_value("Hz"), ls="--", color="dimgray")
plt.show()
