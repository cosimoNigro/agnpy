# check absorption for different EBL model at redshift z=1
import numpy as np
import astropy.units as u
from agnpy.absorption import EBL
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc

# matplotlib adjustments
load_mpl_rc()

z = 10
nu = np.logspace(15, 25) * u.Hz
model = "franceschini"
ebl = EBL(model)
absorption = ebl.absorption(0.1, nu)
plt.loglog(nu, absorption, label=model)
absorption = ebl.absorption(1, nu)
plt.loglog(nu, absorption, label=model)


plt.xlabel(r"$\nu\,/\,{Hz}$")
plt.title("EBL absorption")
plt.ylabel("absorption")
plt.legend()
plt.show()
