import numpy as np
import astropy.units as u
from agnpy.emission_regions import Blob
from agnpy.compton import SynchrotronSelfCompton
from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc


# matplotlib adjustments
load_mpl_rc()


# define the emission region and the radiative process
blob = Blob()
ssc = SynchrotronSelfCompton(blob)

# compute the SED over an array of frequencies
nu = np.logspace(15, 28) * u.Hz
sed = ssc.sed_flux(nu)

# plot it
plot_sed(nu, sed, label="Synchrotron Self Compton")
plt.show()
