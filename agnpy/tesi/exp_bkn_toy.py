import numpy as np
import astropy.units as u
from agnpy.spectra import BrokenPowerLaw, ExpCutoffBrokenPowerLaw
from agnpy.emission_regions import Blob
#from agnpy.synchrotron import Synchrotron
from synchrotron_new import Synchrotron
from proton_synchrotron import ProtonSynchrotron

from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc
#from astropy.constants import e, h, c, m_e, m_p, sigma_T
from astropy.constants import m_p, m_e
from astropy.coordinates import Distance
from agnpy.absorption import EBL
import matplotlib.style


# Exp Cut-off Broken Power Law
# global PowerLaw
k_e_test = 1e-13 * u.Unit("cm-3")
gamma_min = 10
gamma_max = 1e7
p1 = 2.1
p2 = 3.1
gamma_b = 1e3

exp_broken = ExpCutoffBrokenPowerLaw(
    k_e_test, p1, p2, gamma_b, gamma_min, gamma_max
)
broken = BrokenPowerLaw(
    k_e_test, p1, p2, gamma_b, gamma_min, gamma_max
)

gamma = np.logspace(np.log10(gamma_min),np.log10(gamma_max*10),100) #just for exp
n1 = exp_broken(gamma)
n2 = broken(gamma)

plt.loglog(gamma,n1)
plt.loglog(gamma,n2)
plt.show()

    # @staticmethod
    # def evaluate(gamma, k, p1, p2, gamma_b, gamma_min, gamma_max):
    #     index = np.where(gamma <= gamma_b, p1, p2)
    #     k_p = np.where(gamma <= gamma_b, k, k*gamma**(p2-p1))
    #     return np.where(
    #         (gamma_min <= gamma),
    #         k_p * (gamma / gamma_b) * np.exp(-gamma/gamma_max)**(-index),
    #         0,
    #     )
