import numpy as np
import astropy.units as u
from agnpy.spectra import BrokenPowerLaw, ExpCutoffBrokenPowerLaw
from matplotlib import pyplot as plt

plt.style.use('one_distribution')


k = 1e-13 * u.Unit("cm-3")
p1 = 2.1
p2 = 4.1
gamma_min = 10
gamma_max = 1e6
gamma_b = 1e3
gamma_c = 1e5

n_exp = ExpCutoffBrokenPowerLaw(
    k, p1, p2,gamma_c, gamma_b, gamma_min, gamma_max
)

gamma = np.logspace(np.log10(gamma_min),np.log10(gamma_max),100) #just for exp
n_ex = n_exp(gamma)

plt.loglog(gamma,n_ex, color = 'black')
plt.xlabel('γ' )
plt.ylabel('$ n $ [{0}]'.format(n_ex.unit.to_string('latex_inline')) )

plt.show()
