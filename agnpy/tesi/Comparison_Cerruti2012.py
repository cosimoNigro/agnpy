import numpy as np
import astropy.units as u
from agnpy.spectra import ExpCutoffPowerLaw, BrokenPowerLaw
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
pks_sed = np.loadtxt('PKS2155-304_data_circa.txt')
lognu = pks_sed[:,0]
lognuFnu = pks_sed[:,1]
nu_data = 10**lognu
nuFnu_data = 10**lognuFnu


# Define source parameters
B = 80 * u.G
redshift = 0.117
distPKS = Distance(z=redshift)
doppler_s = 30
Gamma_bulk = 16
R = 5.2e14 * u.cm #radius of the blob
vol = (4. / 3) * np.pi * R ** 3

# Proton distribution: Exponential cut-off power-law (ECPL)
# "power-law defined above gamma_p_min, with slope alpha_p and with an
# exponential cut-off at gamma_p_max" (from M. Cerruti 2012)
#


# So I made a bit 'cleaner' the code.
# Also, I deleted my proton_synch program and I kept only yours, renaming it
# from proton_synchrotron2 to just proton_synchrotron.
#
# About the SED:
#
# The doppler factor to the power of 4 is already implemented so we dont
# have to put it (look like 183,184 from the original synchrotron file).
# I playd a bit with the normalization of the electron, there's a pretty good
# fit for a k ~ 1e-4 or something like that for the electron synch. So this k = 1e2 is strange,
# maybe the definition of the norm is different? I will check tomorrow to the documentation of their software that they did the fit.
# As for the proton, I really have no idea. So my idea is that first of all,
# we try to fit just the electron synchrotron to the data that Cosimo already has,
# just to be sure that we are doing everything correctly. Then we try to fit these data.
# As for the implementation of the broken exp, I think it can wait until we manage to
# have some reasonable fits. I think very important there's something that we are missing.



norm_p2 = 12e3 / u.Unit('cm3')
u_p = 3.7e2 * u.Unit('erg cm-3')
#k = (norm_p2 / mpc2.to('eV')) * vol
print(norm_p2 / mpc2.to('eV'))

# define the proton distribution
n_p = ExpCutoffPowerLaw(k=12e3 / u.Unit('cm3'), #k,
        p = 2.0 ,
        gamma_c= 1e9,
        gamma_min= 1,
        gamma_max=1e20,
        mass=m_p
)

# Define electron distribution
n_e = BrokenPowerLaw(k=6e-5 * u.Unit("cm-3"), # k = 6e2, kp = 12e3
        p1=2.0,
        p2=4.32,
        gamma_b=4e3,
        gamma_min=1,
        gamma_max=6e4,
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
nu_obs = nu * blob.delta_D

sed = synch.sed_flux(nu)
psed = psynch.sed_flux(nu)


ebl = EBL("saldana-lopez")
absorption = ebl.absorption(redshift, nu_obs)


sed_abs  = sed  * absorption
psed_abs = psed * absorption # Check if it is correct

# plot it
plt.figure(figsize = (6.92, 4.29))
plt.scatter(nu_data, nuFnu_data, color = 'black')
plot_sed(nu,  sed_abs, label = 'ElectronSynctrotron')
plot_sed(nu_obs, psed, label = 'ProtonSynchrotron')
plot_sed(nu_obs, psed_abs, label = 'ProtonSynchrotron, EBL corrected')
plt.ylim(1e-14, 1e-8)
plt.xlim(1e10, 1e28) # For frequencies
plt.savefig('Comparison.png')
plt.show()


plt.figure(figsize = (6.92, 4.29))
blob.plot_n_e(label = 'Electron distribution')
blob.plot_n_p(label = 'Proton distribution')
#n_e.plot()
plt.ylim(1e-44, 1e5)
plt.legend()
plt.savefig('Particle_distr.png')
