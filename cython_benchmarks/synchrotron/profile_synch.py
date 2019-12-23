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


def profile(command, label):
    """function to profile a given command"""
    print(f"->{label} profiling section...")
    cProfile.run(command, f"Profile_{label}.prof")
    s = pstats.Stats(f"Profile_{label}.prof")
    s.strip_dirs().sort_stats("time").print_stats(10)


def timing(command, number):
    """function to time a given command, returns time in seconds"""
    return timeit.timeit(command, globals=globals(), number=number)


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

nu_syn = np.logspace(8, 21)
nu_ssc = np.logspace(15, 30)

epsilon_syn = H * nu_syn / MEC2
epsilon_ssc = H * nu_ssc / MEC2
gamma = blob.gamma
N_e = blob.N_e(gamma)
com_sed_syn_py = core_synch.com_sed_emissivity(epsilon_syn, gamma, N_e, B.value)
com_sed_syn_cy = core_synch_cy.com_sed_emissivity(epsilon_syn, gamma, N_e, B.value)

number = 10
command_py_syn = "core_synch.com_sed_emissivity(epsilon_syn, gamma, N_e, B.value)"
command_cy_syn = "core_synch_cy.com_sed_emissivity(epsilon_syn, gamma, N_e, B.value)"
command_py_ssc = "core_synch.ssc_sed_emissivity(epsilon_syn, com_sed_syn_py, epsilon_ssc, gamma, N_e, B.value, R_b.value)"
command_cy_ssc = "core_synch_cy.ssc_sed_emissivity(epsilon_syn, com_sed_syn_cy, epsilon_ssc, gamma, N_e, B.value, R_b.value)"

profile(command_py_syn, "synch_py")
timer_py_syn = timing(command_py_syn, number)
profile(command_cy_syn, "synch_cy")
timer_cy_syn = timing(command_cy_syn, number)
print(f"numpy / cython synch speedup factor: {timer_py_syn / timer_cy_syn:.2f}")

profile(command_py_ssc, "ssc_py")
timer_py_ssc = timing(command_py_ssc, number)
profile(command_cy_ssc, "ssc_cy")
timer_cy_ssc = timing(command_cy_ssc, number)
print(f"numpy / cython ssc speedup factor: {timer_py_ssc / timer_cy_ssc:.2f}")

plt.loglog(epsilon_syn, core_synch.com_sed_emissivity(epsilon_syn, gamma, N_e, B.value))
plt.loglog(
    epsilon_syn,
    core_synch_cy.com_sed_emissivity(epsilon_syn, gamma, N_e, B.value),
    ls=":",
)
plt.loglog(
    epsilon_ssc,
    core_synch.ssc_sed_emissivity(
        epsilon_syn, com_sed_syn_py, epsilon_ssc, gamma, N_e, B.value, R_b.value
    ),
)
plt.loglog(
    epsilon_ssc,
    core_synch_cy.ssc_sed_emissivity(
        epsilon_syn, com_sed_syn_cy, epsilon_ssc, gamma, N_e, B.value, R_b.value
    ),
    ls=":",
)
plt.show()
