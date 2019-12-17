import sys

sys.path.append("../")
import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from agnpy.particles import Blob
from agnpy.targets import Monochromatic, PowerLaw
from agnpy.compton import Compton

spectrum_norm = 1 * u.Unit("erg cm-3")
parameters = {"p": 2.2, "gamma_min": 1e2, "gamma_max": 1e7}
spectrum_dict = {"type": "PowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 1 * u.G
z = 1
delta_D = 10
Gamma = 10
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

nu = np.logspace(12, 28) * u.Hz
MEC2 = (const.m_e * const.c ** 2).to("erg")
energy = nu.to(u.erg, equivalencies=u.spectral())
epsilon_s = (energy / MEC2).decompose().value

compton = Compton(blob)

# test for isotropic monochromatic target
print("producing figure 6.7 of Dermer without approximations")
fig, ax = plt.subplots()
for epsilon in (1e-8, 1e-6, 1e-4, 1e-2):
    target_mono_iso = Monochromatic(1 * u.Unit("erg cm-3"), epsilon)
    mono_emissivity = compton.sed_emissivity_iso_mono_ph(epsilon_s, target_mono_iso)

    plt.loglog(
        nu,
        mono_emissivity,
        lw=3,
        label="$\epsilon = {:.0e}$".format(target_mono_iso.epsilon),
    )

plt.legend()
plt.grid()
plt.xlim([1e12, 1e28])
plt.ylim([1e-16, 1e-2])
plt.xlabel(r"$\nu\,/\,Hz$")
plt.ylabel(
    "$\epsilon_{s}\;j(\epsilon_{s})\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$"
)
plt.show()
fig.savefig("results/Figure_6.7_Dermer_2009.png")

# test for isotropic power-law targets
print("producing figure 6.8 of Dermer")
fig, ax = plt.subplots()
for (epsilon_1, epsilon_2) in ([1e-9, 1e-6], [1e-6, 1e-3], [1e-3, 1]):
    # test for isotropic power law target
    u_0 = 1 * u.Unit("erg cm-3")
    alpha = 0.5
    target_pwl = PowerLaw(u_0, alpha, epsilon_1, epsilon_2)

    pwl_emissivity = compton.sed_emissivity_iso_pwl_ph(epsilon_s, target_pwl)

    plt.loglog(
        nu,
        pwl_emissivity,
        lw=3,
        label="$\epsilon_1, \epsilon_2 = {:.0e}, {:.0e}$".format(epsilon_1, epsilon_2),
    )

plt.legend()
plt.grid()
plt.xlim([1e12, 1e28])
plt.ylim([1e-16, 1e-2])
plt.xlabel(r"$\nu\,/\,Hz$")
plt.ylabel(
    "$\epsilon_{s}\;j(\epsilon_{s})\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$"
)
plt.show()
fig.savefig("results/Figure_6.8_Dermer_2009.png")
