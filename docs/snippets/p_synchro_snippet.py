import numpy as np
import astropy.units as u
from astropy.constants import m_p
from agnpy.spectra import PowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron, ProtonSynchrotron
from agnpy.utils.plot import load_mpl_rc, plot_sed
import matplotlib.pyplot as plt


# proton distribution
n_p = PowerLaw(k=1e-9 * u.Unit("cm-3"), p=2.3, gamma_min=1e4, gamma_max=1e10, mass=m_p)

# define the emission region and the proton synchrotron radiative process
blob = Blob(n_p=n_p)
psynch = ProtonSynchrotron(blob)

# compute the proton synchrotron SED
nu = np.logspace(8, 26, 200) * u.Hz
sed_psynch = psynch.sed_flux(nu)

# compute also the electrons' synchrotron radiation
# (n_e is automatically initialised by the Blob, if not specified)
synch = Synchrotron(blob)
sed_synch = synch.sed_flux(nu)

# display both
load_mpl_rc()
plot_sed(nu, sed_synch, label="electron synchrotron")
plot_sed(nu, sed_psynch, color="crimson", label="proton synchrotron")
plt.ylim([1e-30, 1e-25])
plt.show()
