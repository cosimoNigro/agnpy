import numpy as np
import astropy.units as u
from agnpy.spectra import ExpCutoffPowerLaw, ExpCutoffBrokenPowerLaw
from agnpy.emission_regions import Blob
#from agnpy.synchrotron import Synchrotron
from synchrotron_new import Synchrotron
from proton_synchrotron import ProtonSynchrotron

from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc
#from astropy.constants import e, h, c, m_e, m_p, sigma_T
from astropy.constants import m_p#, m_e
from astropy.coordinates import Distance
from agnpy.absorption import EBL
from agnpy.utils.conversion import mec2, mpc2
#import matplotlib.style

load_mpl_rc()  # adopt agnpy plotting style

# Try user-defined plot style
#matplotlib.style.use('/Users/ilaria/Desktop/Dottorato_data/Plot_style/file.mplstyle')
#matplotlib.style.use('seaborn-muted')
#print(matplotlib.style.available)

# Extract data of PKS 2155-304
pks_sed = np.loadtxt('data/PKS2155-304_data_circa.txt')
lognu = pks_sed[:,0]
lognuFnu = pks_sed[:,1]
nu_data = 10**lognu
nuFnu_data = 10**lognuFnu


# Define source parameters
B = 80 * u.G
redshift = 0.117
distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 16
R = 5.2e14 * u.cm #radius of the blob
vol = (4. / 3) * np.pi * R ** 3

# Proton distribution: Exponential cut-off power-law (ECPL)
# "power-law defined above gamma_p_min, with slope alpha_p and with an
# exponential cut-off at gamma_p_max" (from M. Cerruti 2012)

norm_p2 = 12e3 / u.Unit('cm3')
u_p = 3.7e2 * u.Unit('erg cm-3')
#k = (norm_p2 / mpc2.to('eV')) * vol
print(norm_p2 / mpc2.to('eV'))

# define the proton distribution
n_p = ExpCutoffPowerLaw(k=12e3 / u.Unit('cm3'), #12e3 / u.Unit('cm3'),
        p = 2.0 ,
        gamma_c= 1e9,
        gamma_min= 1,
        gamma_max=1e20,
        mass=m_p
)

# Define electron distribution
# k_m = k_c * gamma_b ** a
n_e = ExpCutoffBrokenPowerLaw(k=3.75e-5 * u.Unit("cm-3"), # k_m = 6e2
        p1=2.0,
        p2=4.32,
        gamma_b=4e3,
        gamma_min=1,
        gamma_max=6e5,
        gamma_c  =6e4
)

blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_e=n_e,
        n_p=n_p
)

print(blob)
print(blob.delta_D)

synch = Synchrotron(blob, ssa=True)
psynch = ProtonSynchrotron(blob)

# compute the SED over an array of frequencies
nu = np.logspace(10, 30) * u.Hz
# nu_obs = nu * blob.delta_D / (1+redshift): we don't need to do it in the end,
# since the doppler shift of the freqs has already being done from "nu_to_epsilon_prime" of utils (line 174, original synchrotron.py)

sed = synch.sed_flux(nu)
psed = psynch.sed_flux(nu)

#"franceschini", "dominguez", "finke", "saldana-lopez"
ebl = EBL("finke")
absorption = ebl.absorption(redshift, nu)

sed_abs  = sed  * absorption
psed_abs = psed * absorption # Check if it is correct

# plot it
plt.figure(figsize = (6.92, 4.29))
plt.scatter(nu_data, nuFnu_data, color = 'black')
plot_sed(nu,  sed_abs, label = 'ElectronSynctrotron')
plot_sed(nu, psed, label = 'ProtonSynchrotron')
plot_sed(nu, psed_abs, label = 'ProtonSynchrotron, EBL corrected')
plt.ylim(1e-14, 1e-8)
plt.xlim(1e10, 1e28) # For frequencies
#plt.savefig('Comparison.png')
plt.show()

plt.figure(figsize = (6.92, 4.29))
blob.plot_n_e(label = 'Electron distribution')
blob.plot_n_p(label = 'Proton distribution')
#n_e.plot()
plt.ylim(1e-44, 1e5)
plt.legend()
#plt.savefig('Particle_distr.png')
