from agnpy.spectra import PowerLaw
from agnpy.emission_regions import Blob
import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from astropy.table import Table, Column
from astropy.coordinates import Distance
import numpy as np

B = 80 * u.G
redshift = 0.117
distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 16
R = 5.2e14 * u.cm #radius of the blob
V = (4. / 3) * np.pi * R ** 3
A = (4 * np.pi * distPKS ** 2)
# FOR THE EXAMPLE OF AHARONIAN
Ec = 3*1e20 * u.eV # characteristic energy of protons
Ecut = Ec
dist = PowerLaw(k=1e-13 * u.Unit('cm-3'), p=2., gamma_min = 1e2, gamma_max = 1e8)

blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_p =dist
)
a =[]
b= []
gamma = np.logspace(2,8,25)
for i in gamma:
    a.append(blob.n_p.evaluate(i, k=1e-13 * u.Unit('cm-3'),
            p=2., gamma_min = 1e2, gamma_max = 1e8).value)
    b.append(dist(i).value)

print (a,b)
