import numpy as np
import astropy.units as u
import astropy.constants as const
from agnpy.targets import SSDisk

# quantities defining the disk
M_BH = 1.2 * 1e9 * const.M_sun
L_disk = 2 * 1e46 * u.Unit("erg s-1")
eta = 1 / 12
R_g = 1.77 * 1e14 * u.cm
R_in = 6 * R_g
R_out = 200 * R_g

disk = SSDisk(M_BH, L_disk, eta, R_in, R_out)

# alternative initialisation using gravitational radius units
disk = SSDisk(M_BH, L_disk, eta, 6, 200, R_g_units=True)
print(disk)


from agnpy.targets import SphericalShellBLR

# quantities defining the BLR
xi_line = 0.024
R_line = 1e17 * u.cm

blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
print(blr)


from agnpy.targets import RingDustTorus

# quantities defining the DT
T_dt = 1e3 * u.K
xi_dt = 0.1

dt = RingDustTorus(L_disk, xi_dt, T_dt)
print(dt)


from agnpy.utils.plot import load_mpl_rc, plot_sed
import matplotlib.pyplot as plt

# redshift of the host galaxy
z = 0.1
# array of frequencies to compute the SEDs
nu = np.logspace(12, 18) * u.Hz

# compute the SEDs
disk_bb_sed = disk.sed_flux(nu, z)
dt_bb_sed = dt.sed_flux(nu, z)

# plot them
load_mpl_rc()
fig, ax = plt.subplots()

plot_sed(nu, disk_bb_sed, lw=2, label="Accretion Disk")
plot_sed(nu, dt_bb_sed, lw=2, label="Dust Torus")

plt.ylim([1e-13, 1e-9])
plt.show()

fig.savefig("../_static/disk_torus_black_bodies.png")