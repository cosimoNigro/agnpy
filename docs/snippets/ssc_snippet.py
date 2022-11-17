import numpy as np
import astropy.units as u
from astropy.constants import m_e
from astropy.coordinates import Distance
from agnpy.spectra import PowerLaw
from agnpy.emission_regions import Blob
from agnpy.compton import SynchrotronSelfCompton
from agnpy.utils.plot import plot_sed, load_mpl_rc
import matplotlib.pyplot as plt


# set the quantities defining the blob
R_b = 1e16 * u.cm
V_b = 4 / 3 * np.pi * R_b ** 3
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
B = 1 * u.G

# electron distribution
W_e = 1e48 * u.erg # total energy in electrons

n_e = PowerLaw.from_total_energy(
    W_e,
    V_b,
    p=2.8,
    gamma_min=1e2,
    gamma_max=1e7,
    mass=m_e,
)

# define the emission region and the radiative process
blob = Blob(R_b, z, delta_D, Gamma, B, n_e=n_e)
ssc = SynchrotronSelfCompton(blob)

# compute the SED over an array of frequencies
nu = np.logspace(15, 28) * u.Hz
sed = ssc.sed_flux(nu)

# plot it
load_mpl_rc()
plot_sed(nu, sed, label="synchrotron self Compton")

plt.show()
