import numpy as np
import astropy.units as u
from agnpy.spectra import BrokenPowerLaw, ExpCutoffPowerLaw, ExpCutoffBrokenPowerLaw
from agnpy.emission_regions import Blob
#from agnpy.synchrotron import Synchrotron
#from synchrotron_new import Synchrotron
from agnpy.synchrotron import ProtonSynchrotron
#from proton_synchrotron import ProtonSynchrotron

from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc
#from astropy.constants import e, h, c, m_e, m_p, sigma_T
from astropy.constants import m_p, m_e
from astropy.coordinates import Distance
from agnpy.absorption import EBL
from agnpy.utils.conversion import mec2, mpc2
#import matplotlib.style
from agnpy.compton import SynchrotronSelfCompton

load_mpl_rc()  # adopt agnpy plotting style

# Extract data of Cerruti

lognu, lognuFnu= np.genfromtxt('data/Cerruti/second_email/test_pss.dat',  dtype = 'float', comments = '#', usecols = (0,4), unpack = True)
nu_data = 10**lognu
nuFnu_data = 10**lognuFnu

# Define source parameters

B = 10 * u.G
redshift = 0.32
distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 15 # viewing angle: 0.1 degrees
R = 1e16 * u.cm #radius of the blob
vol = (4. / 3) * np.pi * R ** 3

# First Comparison

n_p = ExpCutoffPowerLaw(k=120000 * u.Unit('cm-3'),
        p = 2.2,
        gamma_c= 2.5e9,
        gamma_min= 1,
        gamma_max=1e20,
        mass=m_p
)

blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_p=n_p
)

psynch = ProtonSynchrotron(blob, ssa = True)

# compute the SED over an array of frequencies
nu = np.logspace(9, 29, 100) * u.Hz
psed = psynch.sed_flux(nu)

# plot
plt.figure(figsize = (6.92, 4.29))
plt.scatter(nu_data, nuFnu_data, color = 'black')
plot_sed(nu, psed, label = 'ProtonSynchrotron')
plt.ylim(1e-22,1e-9)
plt.show()


# Comparing the two distributions:


gamma, dndg = np.genfromtxt('./data/Cerruti/second_email/test_ps.dat',  dtype = 'float', comments = '#', usecols = (2,3), unpack = True)

n_p = ExpCutoffPowerLaw(k=120000 * u.Unit('cm-3'),
        p = 2.2,
        gamma_c= 2.5e9,
        gamma_min= 1,
        gamma_max=1e20,
        mass=m_p
)

n = n_p(gamma)
# If you zoom in you can see the two seperate points
plt.loglog(gamma,n, '.')
plt.loglog(gamma,dndg, '.')
# plt.show()
