import astropy.units as u
from astropy.constants import m_e, m_p
from astropy.coordinates import Distance
from agnpy.spectra import PowerLaw, BrokenPowerLaw
from agnpy.emission_regions import Blob


# set the quantities defining the blob
R_b = 1e16 * u.cm
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
B = 1 * u.G

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

blob = Blob(R_b, z, delta_D, Gamma, B, n_e=n_e)


# set the Doppler factor from the bulk Lorentz factor and the viewing angle
blob = Blob()
blob.set_delta_D(Gamma=20, theta_s=10 * u.deg)
print(f"{blob.delta_D:.2f}")


# add a proton energy distribution
n_p = PowerLaw(k=0.1 * u.Unit("cm-3"), p=2.3, gamma_min=10, gamma_max=1e6, mass=m_p)
blob = Blob(R_b, z, delta_D, Gamma, B, n_e=n_e, n_p=n_p)
print(blob)


# print energy densities
print(f"{blob.u_e:.2e}")
print(f"{blob.u_p:.2e}")
print(f"{blob.U_B:.2e}")
print(f"{blob.u_ph_synch:.2e}")
