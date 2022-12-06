import numpy as np
import astropy.units as u
from astropy.constants import m_e
from agnpy.spectra import BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.compton import ExternalCompton
from agnpy.targets import SphericalShellBLR
from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc



# define the emission region and the electron distribution
R_b = 1e16 * u.cm
V_b = 4 / 3 * np.pi * R_b ** 3
z = 1
delta_D = 40
Gamma = 40
B = 0.56 * u.G

W_e = 6e45 * u.erg
n_e = BrokenPowerLaw.from_total_energy(
    W=W_e, V=V_b, p1=2.0, p2=3.5, gamma_b=1e4, gamma_min=20, gamma_max=5e7, mass=m_e
)

blob = Blob(R_b, z, delta_D, Gamma, B, n_e, gamma_e_size=500)


# define the target
L_disk = 2 * 1e46 * u.Unit("erg s-1")
xi_line = 0.024
R_line = 1e17 * u.cm

blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)


# declare the external Compton process
# distance between the blob and the target
r = 2e17 * u.cm
ec = ExternalCompton(blob, blr, r)

# compute the SED over an array of frequencies
nu = np.logspace(15, 30) * u.Hz
sed = ec.sed_flux(nu)

# plot it
load_mpl_rc()
plot_sed(nu, sed, label="external Compton on broad line region")

plt.show()
