import numpy as np
import astropy.units as u
from agnpy.spectra import PowerLaw, BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.compton import InteractingBlobs
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc, plot_sed

z = 0.1

# Define the first emission region and its electron distribution.
# The electrons in this blob will up-scatter the synchrotron photons produced in the second blob
k_e = 1e3 * u.Unit("cm-3")
p1_e = 2.0
p2_e = 3.0
gamma_b_e = 1e4
gamma_min_e = 1e2
gamma_max_e = 1e6
n_blob_e = BrokenPowerLaw(
    k=k_e,
    p1=p1_e,
    p2=p2_e,
    gamma_b=gamma_b_e,
    gamma_min=gamma_min_e,
    gamma_max=gamma_max_e,
)
delta_D_e = 20
R_b_e = 1e15 * u.cm
B_e = 0.1 * u.G
blob_e = Blob(R_b=R_b_e, z=z, delta_D=delta_D_e, B=B_e, n_e=n_blob_e)

# Define the second emission region and its electron distribution
# The synchrotron photons produced in this blob will be used as target for the inverse Compton scattering of the electrons from the first blob
k_t = 1e1 * u.Unit("cm-3")
p_t = 2.5
gamma_min_t = 1e2
gamma_max_t = 1e5
n_blob_t = PowerLaw(k=k_t, p=p_t, gamma_min=gamma_min_t, gamma_max=gamma_max_t)
delta_D_t = 4
R_b_t = 1e17 * u.cm
B_t = 0.05 * u.G
blob_t = Blob(R_b=R_b_t, z=z, delta_D=delta_D_t, B=B_t, n_e=n_blob_t)

# Declare the interacting blobs instance
interacting_blobs = InteractingBlobs(blob_e=blob_e, blob_t=blob_t)

# Compute the SED over an array of frequencies
nu = np.logspace(15, 28) * u.Hz
sed_interacting_blobs = interacting_blobs.sed_flux(nu)

# Plot the SED
load_mpl_rc()
plot_sed(nu, sed_interacting_blobs, label="Interaction of co-spatial blobs")

plt.show()
