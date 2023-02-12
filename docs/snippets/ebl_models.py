# check absorption for different EBL model at redshift z=1
import numpy as np
import astropy.units as u
from agnpy.absorption import EBL
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc


# redshift and array of frequencies over which to evaluate the absorption
z = 0.5
nu = np.logspace(15, 28, 200) * u.Hz
E = nu.to("eV", equivalencies=u.spectral())

load_mpl_rc()

for model in ["franceschini", "dominguez", "finke", "saldana-lopez"]:
    ebl = EBL(model)
    absorption = ebl.absorption(z, nu)
    plt.loglog(E, absorption, label=model)

plt.xlabel(r"$E\,/\,{\rm eV}$")
plt.title("EBL absorption at z=1")
plt.ylabel("absorption")
plt.ylim([1e-4, 2])
plt.legend()
plt.show()
