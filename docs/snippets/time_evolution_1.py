import numpy as np
from astropy import units as u
from astropy.constants import m_e
from agnpy.spectra import PowerLaw
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.time_evolution.time_evolution import TimeEvolution, synchrotron_loss
import matplotlib.pyplot as plt


# set the quantities defining the blob and the electron distribution
R_b = 1e16 * u.cm
V_b = 4 / 3 * np.pi * R_b ** 3
W_e = 1e48 * u.erg # total energy in electrons

# initial electron distribution
n_e_initial = PowerLaw.from_total_energy(
    W_e,
    V_b,
    p=2.8,
    gamma_min=1e2,
    gamma_max=1e7,
    mass=m_e,
)

# define the blob and the energy loss mechanism
blob = Blob(n_e=n_e_initial)
synch = Synchrotron(blob)

# perform the time evolution over 5 minutes, considering synchrotron losses
total_time = 5 * u.min
time_evolution = TimeEvolution(blob, total_time, synchrotron_loss(synch))
time_evolution_result = time_evolution.evaluate()

# let us plot both particle distributions, the initial and the evolved one
gamma = time_evolution_result.gamma
n_e_evol = time_evolution_result.density

fig, ax = plt.subplots()
n_e_initial.plot(ax=ax, gamma_power=2, label="initial distribution")
ax.plot(gamma, n_e_evol * gamma**2, label="evolved distribution")
ax.set_xlabel(r"$\gamma$")
ax.set_ylabel(r"$\gamma^2 n_e(\gamma)$")
ax.legend()
plt.show()
