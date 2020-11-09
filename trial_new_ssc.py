import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.utils.math import trapz_loglog
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton
from agnpy.tests.utils import extract_columns_sample_file
import matplotlib.pyplot as plt

# set the spectrum normalisation (total energy in electrons in this case)
spectrum_norm = 1e48 * u.Unit("erg")
# define the spectral function through a dictionary
spectrum_dict = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e5},
}
R_b = 1e16 * u.cm
B = 1 * u.G
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
print(blob)
synch = Synchrotron(blob)

# - synchro part
# check against Dermer's ssc
nu_ref, sed_ref = extract_columns_sample_file(
    "data/sampled_seds/synch_figure_7_4_dermer_menon_2009.txt", "Hz", "erg cm-2 s-1"
)
sed_synch = synch.sed_flux(nu_ref)
plt.loglog(nu_ref, sed_ref, ls="-", marker="o", label="Dermer")
plt.loglog(nu_ref, sed_synch, ls="--", marker=".", label="agnpy")
plt.legend()
plt.show()

# - SSC part
# check against Dermer's ssc
nu_ref, sed_ref = extract_columns_sample_file(
    "data/sampled_seds/ssc_figure_7_4_dermer_menon_2009.txt", "Hz", "erg cm-2 s-1"
)
# evaluate SSC at the same frequencies
ssc = SynchrotronSelfCompton(blob)
sed_ssc = ssc.sed_flux(nu_ref)
plt.loglog(nu_ref, sed_ref, ls="-", marker="o", label="Dermer")
plt.loglog(nu_ref, sed_ssc, ls="--", marker=".", label="agnpy")
plt.legend()
plt.show()
