import sys
import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
sys.path.append("../../")
from agnpy.particles import Blob
import timeit
import pstats, cProfile
import core_synch
import core_synch_cy

ME = 9.10938356e-28
C = 2.99792458e10
E = 4.80320425e-10
H = 6.62607004e-27
SIGMA_T = 6.65245872e-25
MEC2 = 8.187105649650028e-07

spectrum_norm = 1e48 * u.Unit("erg")
parameters = {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e7}
spectrum_dict = {"type": "PowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 1 * u.G
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
print(f"total density {blob.norm:.2e}")
print(f"total energy {blob.W_e:.2e}")

nu = np.logspace(8, 21)

epsilon = H * nu / MEC2
gamma = blob.gamma
N_e = blob.N_e(gamma)
R_b = blob.R_b

number = 1000
print("\n\n python testing")
py_command = "core_synch.com_sed_emissivity(epsilon, gamma, N_e, B.value)"
cy_command = "core_synch_cy.com_sed_emissivity(epsilon, gamma, N_e, B.value)"

print("...synch profiling section...")
cProfile.run(py_command, "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats(10)

print("...synch timing section...")
timer_py = timeit.timeit(py_command, globals=globals(), number=number)
print(f"{timer_py} s / {number} = {timer_py / number:.2e} s")

print("\n\n cython testing")

print("...synch profiling section...")
cProfile.run(cy_command, "Profile.prof")
s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats(10)

print("...synch timing section...")
timer_cy = timeit.timeit(cy_command, globals=globals(), number=number)
print(f"{timer_cy} s / {number} = {timer_cy / number:.2e} s")

print(f"speedup factor: {timer_py / timer_cy:.2f}")

plt.loglog(epsilon, core_synch.com_sed_emissivity(epsilon, gamma, N_e, B.value))
plt.loglog(epsilon, core_synch_cy.com_sed_emissivity(epsilon, gamma, N_e, B.value), ls=":")
plt.show()

