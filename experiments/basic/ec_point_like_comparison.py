# compare if in the limit of large distances the SED for EC on the BLR and on
# the dust torus tend to the one generated by a point like source behind the jet

# import numpy, astropy and matplotlib for basic functionalities
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

# import agnpy classes
from agnpy.emission_regions import Blob
from agnpy.compton import ExternalCompton
from agnpy.targets import PointSourceBehindJet, SphericalShellBLR, RingDustTorus

spectrum_norm = 6e42 * u.erg
parameters = {
    "p1": 2.0,
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
blob.set_gamma_size(500)


L_disk = 2 * 1e46 * u.Unit("erg s-1")
# dust torus
T_dt = 1e3 * u.K
csi_dt = 0.1
dt = RingDustTorus(L_disk, csi_dt, T_dt)
# blr
xi_line = 0.024
R_line = 1e17 * u.cm
blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)

# point source behind the jet approximating the DT
ps_dt = PointSourceBehindJet(dt.xi_dt * L_disk, dt.epsilon_dt)
# point source behind the jet approximating the BLR
ps_blr = PointSourceBehindJet(blr.xi_line * L_disk, blr.epsilon_line)

ec_dt = ExternalCompton(blob, dt, r=1e22 * u.cm)
ec_blr = ExternalCompton(blob, blr, r=1e22 * u.cm)
ec_ps_dt = ExternalCompton(blob, ps_dt, r=1e22 * u.cm)
ec_ps_blr = ExternalCompton(blob, ps_blr, r=1e22 * u.cm)

# seds
nu = np.logspace(15, 30) * u.Hz

sed_blr = ec_blr.sed_flux(nu)
sed_ps_blr = ec_ps_blr.sed_flux(nu)
sed_dt = ec_dt.sed_flux(nu)
sed_ps_dt = ec_ps_dt.sed_flux(nu)

plt.loglog(nu, sed_blr, ls="-", lw=2, color="k", label="EC on BLR")
plt.loglog(nu, sed_dt, ls="-", lw=2, color="dimgray", label="EC on DT")
plt.loglog(
    nu,
    sed_ps_blr,
    ls=":",
    lw=2,
    color="crimson",
    label="EC on point source approx. BLR",
)
plt.loglog(
    nu,
    sed_ps_dt,
    ls=":",
    lw=2,
    color="darkorange",
    label="EC on point source approx. DT",
)
plt.legend()
plt.xlabel(r"$\nu\,/\,{\rm Hz}$")
plt.ylabel(r"$\nu F_{\nu}\,/\,({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$")
plt.show()
