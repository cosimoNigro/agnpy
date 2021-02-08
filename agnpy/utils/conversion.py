# conversion utilities for agnpy
from astropy.constants import m_e, h, c, G
import astropy.units as u


mec2 = m_e.to("erg", equivalencies=u.mass_energy())
lambda_c = (h / (m_e * c)).to("cm")  # Compton wavelength
# equivalency for decomposing Gauss in Gaussian-cgs units (not available in astropy)
Gauss_cgs_unit = "cm(-1/2) g(1/2) s-1"
Gauss_cgs_equivalency = [(u.G, u.Unit(Gauss_cgs_unit), lambda x: x, lambda x: x)]
# equivalency to transform frequencies to energies in electron rest mass units
epsilon_equivalency = [
    (u.Hz, u.Unit(""), lambda x: h.cgs * x / mec2, lambda x: x * mec2 / h.cgs)
]


def nu_to_epsilon_prime(nu, z=0, delta_D=1):
    """convert the frequency to a dimensionless energy in another reference
    frame with redshift z and moving with doppler factor delta_D"""
    epsilon = nu.to("", equivalencies=epsilon_equivalency)
    return (1 + z) * epsilon / delta_D


def B_to_cgs(B):
    """convert a magnetic field to CGS units"""
    return B.to(Gauss_cgs_unit, equivalencies=Gauss_cgs_equivalency)


def to_R_g_units(r, M):
    """convert a distance in graviational radii units"""
    R_g = G * M / c ** 2
    return (r / R_g).to("")
