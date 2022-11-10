import numpy as np
import astropy.units as u
from agnpy.spectra import PowerLaw
from agnpy.emission_regions import Blob
#from agnpy.synchrotron import Synchrotron
from synchrotron_new import Synchrotron
from proton_synchrotron import ProtonSynchrotron

from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc
from astropy.constants import e, h, c, m_e, m_p, sigma_T
from astropy.coordinates import Distance
from agnpy.absorption import EBL
from agnpy.utils.conversion import mec2, mpc2

# data from dermer
synch = np.loadtxt('data/Dermer/synchrotron_gamma_max_1e5.txt', delimiter=',')
ssc   = np.loadtxt('data/Dermer/ssc_gamma_max_1e5.txt', delimiter=',')

nu_synch  = synch[:,0]
sed_synch = synch[:,1]
nu_ssc  = ssc[:,0]
sed_ssc = ssc[:,0]

# blob parameters: Dermer figure 7.4, page 131
R_b = 1e16 * u.cm
V_b = 4 / 3 * np.pi * R_b ** 3
z = Distance(1e27, unit=u.cm).z
B = 1 * u.G
delta_d = 10
Gamma_bulk = 10
# electron parameters from dermer
W_e = 1e48 * u.Unit("erg")
p = 2.8
gamma_min = 1e2
gamma_max = 1e5

#defining blob and n_e (Deverything from Dermer)
n_e = PowerLaw.from_total_energy(
    W_e, V_b, m_e, p = p, gamma_min = gamma_min, gamma_max= gamma_max
)

blob = Blob(
    R_b=R_b, z=z, delta_D = delta_d, Gamma= Gamma_bulk, B = B, n_e=n_e
)

synch = Synchrotron(blob)
sed1 = synch.sed_flux(nu_synch * u.Hz)

# Both data and parameter values are for Dermer. The point is that if the
# agnpy couldn't reproduce the correct diagram with these parameters,
# then synchrotron.py would have been wrong

# plot it
plt.figure(figsize = (6.92, 4.29))
plt.scatter(nu_synch, sed_synch, color = 'black')
plot_sed(nu_synch,  sed1, label = 'ElectronSynctrotron')
plt.ylim(1e-14, 1e-8)
plt.xlim(1e10, 1e28) # For frequencies
#plt.savefig('Comparison.png')
plt.show()
