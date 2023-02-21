from agnpy.spectra import PowerLaw as PL
import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h, hbar, k_B
from scipy.integrate import quad, dblquad, nquad, simps, trapz
import numpy as np
import matplotlib.pyplot as plt

p_dist = PL(1 * u.Unit('cm-3'), 2., 1, 2e30)

def dist(x):
    return 1e20 * p_dist(x).value

int = []
a = np.logspace(3,20,100)
k = -1
for i in a:
    k +=1
    x_range = [1,i]
    print ('x range is: ', x_range)
    int.append(nquad(dist, [x_range])[0] / 1e20)
    print ('while the integral is: ', int[k])


plt.loglog(a, int)
plt.show()
