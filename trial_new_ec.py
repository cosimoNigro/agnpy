import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.compton import ExternalCompton
from agnpy.targets import (
    CMB,
    PointSourceBehindJet,
    SSDisk,
    SphericalShellBLR,
    RingDustTorus,
)
import matplotlib.pyplot as plt

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
delta_D = 20
Gamma = 20
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
print(f"total number: {blob.N_e_tot:.2e}")
print(f"total energy: {blob.W_e:.2e}")

# define the array of frequencies over which to calculate the SED
nu = np.logspace(15, 30) * u.Hz

# EC on CMB
cmb = CMB(z=z)
ec_cmb = ExternalCompton(blob, cmb)
ec_cmb_sed = ec_cmb.sed_flux(nu)
plt.loglog(nu, ec_cmb_sed)
plt.show()

# EC on point source behind the jet
ps = PointSourceBehindJet(L_0=1e44 * u.Unit("erg s-1"), epsilon_0=1e-5)
ec_ps = ExternalCompton(blob, ps, r=1e18 * u.cm)
ec_ps_sed = ec_ps.sed_flux(nu)
plt.loglog(nu, ec_ps_sed)
plt.show()
