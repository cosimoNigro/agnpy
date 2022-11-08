import numpy as np
import astropy.units as u
from agnpy.spectra import PowerLaw
from agnpy.emission_regions import Blob
#from agnpy.synchrotron import Synchrotron
from synchrotron_new import Synchrotron
from proton_synchrotron import ProtonSynchrotron
from proton_synchrotron2 import ProtonSynchrotron2
from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
#from agnpy.utils.plot import load_mpl_rc
#from astropy.constants import e, h, c, m_e, m_p, sigma_T
from astropy.constants import m_p


# define the emission region and the radiative process
n_p = PowerLaw(k=1e-13 * u.Unit("cm-3"),
        p=2.1,
        gamma_min=10,
        gamma_max=1e5,
        mass=m_p,
        #integrator=np.trapz,
)

blob = Blob(n_p = n_p)
synch = Synchrotron(blob)
psynch= ProtonSynchrotron(blob)
psynch2 = ProtonSynchrotron2(blob)

# compute the SED over an array of frequencies
nu = np.logspace(8, 23) * u.Hz
sed = synch.sed_flux(nu)
psed= psynch.sed_flux(nu)
psed2 = psynch2.sed_flux(nu)

# plot it
plot_sed(nu, sed, label="Synchrotron")
plot_sed(nu, psed, label = 'ProtonSynchrotron')
plot_sed(nu, psed2, label = 'ProtonSynchrotron2')
plt.show()
