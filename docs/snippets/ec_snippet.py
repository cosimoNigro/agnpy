import numpy as np
import astropy.units as u
import astropy.constants as const
from agnpy.emission_regions import Blob
from agnpy.compton import ExternalCompton
from agnpy.targets import SSDisk
from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc


# matplotlib adjustments
load_mpl_rc()


# define the emission region
blob = Blob()

# define the target
M_BH = 1.2 * 1e9 * const.M_sun
L_disk = 2 * 1e46 * u.Unit("erg s-1")
eta = 1 / 12
R_in = 6
R_out = 200
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)

# declare the external Compton process
r = 1e17 * u.cm  # distance between the blob and the target
ec = ExternalCompton(blob, disk, r)

# compute the SED over an array of frequencies
nu = np.logspace(15, 30) * u.Hz
sed = ec.sed_flux(nu)

# plot it
plot_sed(nu, sed, label="External Compton on Shakura Sunyaev Disk")
plt.ylim([1e-14, 1e-8])
plt.show()
