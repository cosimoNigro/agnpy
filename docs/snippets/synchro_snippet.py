import numpy as np
import astropy.units as u
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc


# matplotlib adjustments
load_mpl_rc()


# define the emission region and the radiative process
blob = Blob()
synch = Synchrotron(blob)

# compute the SED over an array of frequencies
nu = np.logspace(8, 23) * u.Hz
sed = synch.sed_flux(nu)

# plot it
plot_sed(nu, sed, label="Synchrotron")
plt.show()
