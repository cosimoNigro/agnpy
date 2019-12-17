import sys

sys.path.append("../")
import numpy as np
import astropy.units as u
from agnpy.particles import Blob
import matplotlib.pyplot as plt

print("power law test")
spectrum_norm = 1e48 * u.erg
parameters = {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e5}
spectrum_dict = {"type": "PowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 1 * u.G
z = 1
delta_D = 10
Gamma = 10
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
gamma = np.logspace(1, 8, 200)
plt.loglog(gamma, np.power(gamma, 2) * blob.n_e(gamma), lw=2)
plt.xlabel(r"$\gamma'$")
plt.ylabel(r"${\gamma'}^2\,n_{e}(\gamma')$")
plt.show()
print("total energy, W_e : ", blob.W_e(gamma))


print("broken power law test")
spectrum_norm = 1e48 * u.erg
parameters = {
    "p1": 2.0001,
    "p2": 3.5,
    "gamma_b": 1e4,
    "gamma_min": 20,
    "gamma_max": 5e7,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 0.56 * u.G
z = 1
delta_D = 40
Gamma = 40
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
gamma = np.logspace(1, 8, 200)
plt.loglog(gamma, np.power(gamma, 2) * blob.n_e(gamma), lw=2)
plt.xlabel(r"$\gamma'$")
plt.ylabel(r"${\gamma'}^2\,n_{e}(\gamma')$")
plt.show()
print("total energy, W_e : ", blob.W_e(gamma))
