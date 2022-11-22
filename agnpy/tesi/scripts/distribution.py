import numpy as np
from matplotlib import pyplot as plt
from agnpy.spectra import ExpCutoffPowerLaw
from astropy.constants import m_p, m_e
import astropy.units as u


gamma, dndg = np.genfromtxt('data/Cerruti/second_email/test_ps.dat',  dtype = 'float', comments = '#', usecols = (2,3), unpack = True)

n_p = ExpCutoffPowerLaw(k=120000 * u.Unit('cm-3'),
        p = 2.2,
        gamma_c= 2.5e9,
        gamma_min= 1,
        gamma_max=1e20,
        mass=m_p
)

n = n_p(gamma)

plt.loglog(gamma,n)
plt.loglog(gamma,dndg)
plt.show()
