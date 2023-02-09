import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from astropy.table import Table, Column
from astropy.coordinates import Distance
from scipy.integrate import quad, dblquad, nquad, simps, trapz
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import timeit
import re
from agnpy.photomeson.kenurio import PhotoHadronicInteraction
from agnpy.spectra import ExpCutoffPowerLaw as ECPL
from agnpy.emission_regions import Blob

def BlackBody(epsilon):
    T = 2.7 *u.K
    kT = (k_B * T).to('eV').value
    c1 = c.to('cm s-1').value
    h1 = h.to('eV s').value
    norm = 8*np.pi/(h1**3*c1**3)
    num = (mpc2.value *epsilon) ** 2
    denom = np.exp(mpc2.value * epsilon / kT) - 1
    return norm * (num / denom) * u.Unit('cm-3')

start = timeit.default_timer()

# Define source parameters
B = 80 * u.G
redshift = 0.117
distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 16
R = 5.2e14 * u.cm #radius of the blob
vol = (4. / 3) * np.pi * R ** 3

mec2 = (m_e * c ** 2).to('eV')

Ec = 3*1e20 * u.eV
mpc2 = (m_p * c ** 2).to('eV')

p_dist = ECPL(0.265*1e11/mpc2.value**2 * u.Unit('cm-3'), 2., Ec/mpc2, 1e1, 1e13,  m_p)

blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_p= p_dist
)

nu = np.logspace(26,43,15)*u.Hz
gamma = np.logspace(3,22,15)

proton_gamma = PhotoHadronicInteraction(['photon','electron'], blob, BlackBody)

#sed = proton_gamma.sed_flux(nu)
print ('sed2: .....')
sed2= proton_gamma.sed_flux_particle(gamma, 'electron')

#plt.loglog((nu), (sed * nu), lw=2.2, ls='-', color='orange',label = 'agnpy')

energies = nu*h.to('eV s')
energies2 = gamma*mec2
plt.loglog(energies, sed)
plt.loglog(energies2, sed2)
plt.show()

stop = timeit.default_timer()
print("Elapsed time for computation = {} secs".format(stop - start))
