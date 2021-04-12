# check absorption for different EBL model at redshift z=1
import numpy as np
import astropy.units as u
from agnpy.absorption import EBL
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc


# matplotlib adjustments
load_mpl_rc()


z = 1
nu = np.logspace(15, 25) * u.Hz

for model in ["franceschini", "dominguez", "finke"]:
    ebl = EBL(model)
    absorption = ebl.absorption(z, nu)
    plt.loglog(nu, absorption, label=model)
plt.xlabel(r"$\nu\,/\,{Hz}$")
plt.title("EBL absorption at z=1")
plt.ylabel("absorption")
plt.legend()
plt.show()
