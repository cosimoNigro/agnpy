import numpy as np
import astropy.units as u
from agnpy.spectra import ExpCutoffPowerLaw, BrokenPowerLaw
from agnpy.emission_regions import Blob
#from agnpy.synchrotron import Synchrotron
from synchrotron_new import Synchrotron
#from proton_synchrotron import ProtonSynchrotron
from proton_synchrotron2 import ProtonSynchrotron2
from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
#from agnpy.utils.plot import load_mpl_rc
#from astropy.constants import e, h, c, m_e, m_p, sigma_T
from astropy.constants import m_p, m_e
from astropy.coordinates import Distance
from agnpy.absorption import EBL

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
# Proton dostribution parameters
alpha_p = 2.0
gamma_pcut = 1e9 # Cut-off Lorentz factor
gamma_pmin = 1
norm_p = 12e3 / u.Unit('cm3')
u_p = 3.7e2 * u.Unit('erg cm-3')

# define the proton distribution
n_p = ExpCutoffPowerLaw(k=norm_p,
        p=alpha_p,
        gamma_c=gamma_pcut,
        gamma_min=gamma_pmin,
        gamma_max=1e12,
        mass=m_p,
                        )

# Define electron distribution
n_e = BrokenPowerLaw(k=6e2 * u.Unit("cm-3"),
        p1=2.0,
        p2=4.32,
        gamma_b=4e3,
        gamma_min=1,
        gamma_max=6e3,
        mass=m_e,
                     )

blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_e=n_e,
        n_p=n_p,
            )

synch = Synchrotron(blob)
psynch2 = ProtonSynchrotron2(blob)

# compute the SED over an array of frequencies
nu = np.logspace(8, 28) * u.Hz
#sed = synch.sed_flux(nu)
psed2 = psynch2.sed_flux(nu)
psed2_obs = psed2 * doppler_s**4

ebl = EBL("saldana-lopez")
absorption = ebl.absorption(redshift, nu)
psed2_obs_abs = psed2_obs * absorption # Check if it is correct

# plot it
plt.figure()
plt.scatter(nu_data, nuFnu_data, color = 'black')
#plot_sed(nu, sed, label="Synchrotron")
plot_sed(nu, psed2_obs_abs, label = 'ProtonSynchrotron2')
plt.ylim(1e-14, 1e-9)
plt.xlim(1e10, 1e28) # For frequencies

plt.show()
