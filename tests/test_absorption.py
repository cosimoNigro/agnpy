"""profile and test external Comton radiation"""
import sys

sys.path.append("../")
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.targets import SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.absorption import Absorption
import matplotlib.pyplot as plt

# to profile
import cProfile, pstats
import timeit

# functions to profile and time
def profile(command, label):
    """function to profile a given command"""
    print(f"->{command} profiling section...")
    cProfile.run(command, f"Profile_{label}.prof")
    prof = pstats.Stats(f"Profile_{label}.prof")
    prof.strip_dirs().sort_stats("time").print_stats(10)


def timing(command, number):
    """function to time a given command, returns time in seconds"""
    return timeit.timeit(command, globals=globals(), number=number)


# produce the 3C273 example
# define the blob
spectrum_norm = 1e45 * u.erg
parameters = {
    "p": 2.5,
    "gamma_min": 10,
    "gamma_max": 1e6,
}
spectrum_dict = {"type": "PowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 1 * u.G
z = 1
delta_D = 40
Gamma = 40

blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

# disk parameters
M_sun = const.M_sun.cgs
M_BH = 7 * 1e9 * M_sun
R_g = ((const.G * M_BH) / (const.c * const.c)).cgs
L_disk = 2 * 1e46 * u.Unit("erg s-1")
eta = 1 / 2
R_in = 7.8 * 1e15 * u.cm
R_out = 1e4 * R_in
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out)
print("\ndisk definition:")
print(disk)
print("\ntemperature at the inner radius")
print(disk.T(R_in).to("K"))

absorption_1 = Absorption(blob, disk, r=10 * R_in)
absorption_2 = Absorption(blob, disk, r=30 * R_in)
absorption_3 = Absorption(blob, disk, r=100 * R_in)
absorption_4 = Absorption(blob, disk, r=300 * R_in)

energy = np.logspace(0, 5) * u.GeV
nu = energy.to("Hz", equivalencies=u.spectral())

tau_disk_1 = absorption_1._opacity_disk(nu)
tau_disk_2 = absorption_2._opacity_disk(nu)
tau_disk_3 = absorption_3._opacity_disk(nu)
tau_disk_4 = absorption_4._opacity_disk(nu)

plt.plot(np.log10(energy.to("TeV").value), np.log10(tau_disk_1), lw=2, label=r"$r = 10\,R_{\rm in}$")
plt.plot(np.log10(energy.to("TeV").value), np.log10(tau_disk_2), lw=2, label=r"$r = 30\,R_{\rm in}$")
plt.plot(np.log10(energy.to("TeV").value), np.log10(tau_disk_3), lw=2, label=r"$r = 100\,R_{\rm in}$")
plt.plot(np.log10(energy.to("TeV").value), np.log10(tau_disk_4), lw=2, label=r"$r = 300\,R_{\rm in}$")
plt.xlabel("log(E / TeV)")
plt.ylabel(r"log$\tau_{\gamma \gamma}$")
plt.ylim([-2, 3.2])
plt.xlim([-1.5, 1.5])
plt.legend()
plt.show()