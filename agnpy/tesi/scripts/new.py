import numpy as np
import astropy.units as u
from agnpy.spectra import PowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron, ProtonSynchrotron
from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc
from astropy.constants import e, h, c, m_e, m_p, sigma_T
from astropy.coordinates import Distance
from agnpy.absorption import EBL
from agnpy.utils.conversion import mec2, mpc2

# define the emission region and the radiative process

n_e = PowerLaw(k=1e-13 * u.Unit("cm-3"),
        p = 2.1,
        gamma_min=10,
        gamma_max=1e5
)

n_p = PowerLaw(k=1e-13 * u.Unit("cm-3"),
        p=2.1,
        gamma_min=1e3,
        gamma_max=1e6,
        mass=m_p,
        #integrator=np.trapz,
)

# blob parameters: Dermer figure 7.4, page 131
R_b = 1e16 * u.cm
V_b = 4 / 3 * np.pi * R_b ** 3
z = Distance(1e27, unit=u.cm).z
B = 1 * u.G
delta_d = 10
Gamma_bulk = 10


blob = Blob(
    R_b=R_b, z=z, delta_D = delta_d, Gamma= Gamma_bulk, B = B, n_e=n_e, n_p=n_p
)

synch = Synchrotron(blob)
psynch= ProtonSynchrotron(blob)

# compute the SED over an array of frequencies
nu = np.logspace(8, 23) * u.Hz
sed = synch.sed_flux(nu)
psed = psynch.sed_flux(nu)

# plot it
plot_sed(nu, sed, label="Synchrotron")
plot_sed(nu, psed, label = 'ProtonSynchrotron')
plt.ylim(1e-60, 1e-20)
plt.xlim(1e10, 1e20)

plt.show()
