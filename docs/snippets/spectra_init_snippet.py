import numpy as np
import astropy.units as u
from astropy.constants import m_e
from agnpy.spectra import LogParabola
import matplotlib.pyplot as plt

# initialise the electron distribution, let us use the same spectral parameters
# for all the distributions
p = 2.3
q = 0.2
gamma_0 = 1e3
gamma_min = 1
gamma_max = 1e6

# - from total density
n_tot = 1e-3 * u.Unit("cm-3")

n_1 = LogParabola.from_total_density(
    n_tot, p=p, q=q, gamma_0=gamma_0, gamma_min=gamma_min, gamma_max=gamma_max, mass=m_e
)

# - from total energy density
u_tot = 1e-8 * u.Unit("erg cm-3")

n_2 = LogParabola.from_total_energy_density(
    u_tot, p=p, q=q, gamma_0=gamma_0, gamma_min=gamma_min, gamma_max=gamma_max, mass=m_e
)

# - from total energy, we need also the volume to convert to an energy density
W = 1e40 * u.erg
V_b = 4 / 3 * np.pi * (1e16 * u.cm) ** 3

n_3 = LogParabola.from_total_energy(
    W,
    V_b,
    p=p,
    q=q,
    gamma_0=gamma_0,
    gamma_min=gamma_min,
    gamma_max=gamma_max,
    mass=m_e,
)

# - from the denisty at gamma = 1
n_gamma_1 = 1e-4 * u.Unit("cm-3")

n_4 = LogParabola.from_density_at_gamma_1(
    n_gamma_1,
    p=p,
    q=q,
    gamma_0=gamma_0,
    gamma_min=gamma_min,
    gamma_max=gamma_max,
    mass=m_e,
)

# let us plot all the particles distributions
fig, ax = plt.subplots()

n_1.plot(ax=ax, gamma_power=2, label="from " + r"$n_{\rm tot}$")
n_2.plot(ax=ax, gamma_power=2, label="from " + r"$u_{\rm tot}$")
n_3.plot(ax=ax, gamma_power=2, label="from " + r"$W$")
n_4.plot(ax=ax, gamma_power=2, label="from " + r"$n(\gamma=1)$")

ax.legend()
plt.show()
