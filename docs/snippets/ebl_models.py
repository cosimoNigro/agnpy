# check absorption for different EBL model at redshift z=1
import numpy as np
import astropy.units as u
from agnpy.absorption import EBL
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc

load_mpl_rc()

# redshift and array of frequencies over which to evaluate the absorption
z = 1.0
nu = np.logspace(22, 28, 200) * u.Hz
E = nu.to("TeV", equivalencies=u.spectral())

for model in ["franceschini", "dominguez", "finke", "saldana-lopez"]:
    ebl = EBL(model)
    absorption = ebl.absorption(
        nu, z
    )  # warning: in agnpy <= v0.4.0 the arguments order is reversed!
    plt.loglog(E, absorption, label=model)

plt.xlabel(r"$E\,/\,{\rm TeV}$")
plt.title("EBL absorption at z=1")
plt.ylabel("absorption")
plt.ylim([1e-2, 2])
plt.legend()
plt.show()
