# test delta approximation for the synchrotron SED
import sys

sys.path.append("../../")
import astropy.units as u
import numpy as np
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron, synch_sed_param_bpl
import matplotlib.pyplot as plt

# set the spectrum normalisation (total energy in electrons in this case)
spectrum_norm = 1e48 * u.Unit("erg")
# define the spectral function through a dictionary
spectrum_dict = {
    "type": "BrokenPowerLaw",
    "parameters": {
        "p1": 2.5,
        "p2": 3.5,
        "gamma_b": 1e4,
        "gamma_min": 1e2,
        "gamma_max": 1e7,
    },
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
# check that the synchrotron parameterisation work
y = blob.B.value * blob.delta_D
k_eq = (blob.u_e / blob.U_B).to_value("")
sed_param = synch_sed_param_bpl(
    nu.value,
    y,
    k_eq,
    blob.n_e.p1,
    blob.n_e.p2,
    blob.n_e.gamma_b,
    blob.n_e.gamma_min,
    blob.n_e.gamma_max,
    blob.d_L.cgs.value,
    blob.R_b.cgs.value,
    blob.z,
)

plt.loglog(nu, sed, ls="-", label="numerical integration")
plt.loglog(nu, sed_delta_approx, ls="--", label=r"$\delta$" + "-function approx.")
plt.loglog(nu, sed_param, ls=":", label="parametrisation in " + r"$(y,\,k_{\rm eq})$")

plt.xlabel(r"$\nu\,/\,{\rm Hz}$")
plt.ylabel(r"$\nu F_{\nu}\,/\,({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$")
plt.legend()
plt.show()
