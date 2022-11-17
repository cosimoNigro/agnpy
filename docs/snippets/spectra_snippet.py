import astropy.units as u
from astropy.constants import m_e, m_p
from agnpy.spectra import PowerLaw, BrokenPowerLaw
import matplotlib.pyplot as plt

# electron distribution
n_e = BrokenPowerLaw(
    k=1e-8 * u.Unit("cm-3"),
    p1=1.9,
    p2=2.6,
    gamma_b=1e4,
    gamma_min=10,
    gamma_max=1e6,
    mass=m_e,
)

# proton distribution
n_p = PowerLaw(k=0.1 * u.Unit("cm-3"), p=2.3, gamma_min=10, gamma_max=1e6, mass=m_p)

# let us plot both particle distributions
fig, ax = plt.subplots()

n_e.plot(ax=ax, gamma_power=2, label="electrons")
n_p.plot(ax=ax, gamma_power=2, label="protons")

ax.legend()
plt.show()
