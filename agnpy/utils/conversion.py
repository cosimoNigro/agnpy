# conversion utilities for agnpy
from astropy.constants import m_e, m_p, h, c, G
import astropy.units as u
from astropy.constants import e, h, c, m_e, m_p, sigma_T, G

mec2 = m_e.to("erg", equivalencies=u.mass_energy())
mpc2 = m_p.to("erg", equivalencies=u.mass_energy())
lambda_c_e = (h / (m_e * c)).to("cm")  # Compton wavelength
lambda_c_p = (h / (m_p * c)).to("cm")  # Compton wavelength of protons
# equivalency for decomposing Gauss in Gaussian-cgs units (not available in astropy)
Gauss_cgs_unit = "cm(-1/2) g(1/2) s-1"
Gauss_cgs_equivalency = [(u.G, u.Unit(Gauss_cgs_unit), lambda x: x, lambda x: x)]
# equivalency to transform frequencies to energies in electron rest mass units
def epsilon_equivalency(m = m_e):
    if m == m_e:
        epsilon_equivalency = [
            (u.Hz, u.Unit(""), lambda x: h.cgs * x / mec2, lambda x: x * mec2 / h.cgs)
            ]
    elif m == m_p:
        epsilon_equivalency = [
            (u.Hz, u.Unit(""), lambda x: h.cgs * x / mpc2, lambda x: x * mpc2 / h.cgs)
            ]
    else:
        raise ValueError(
            f"Provide either the electron or the proton mass. "
        )
    return epsilon_equivalency


def nu_to_epsilon_prime(nu, z=0, delta_D=1, m = m_e):
    """convert the frequency to a dimensionless energy in another reference
<<<<<<< HEAD
    frame with redshift z and moving with doppler factor delta_D"""
=======
    frame with redshift z and moving with doppler factor delta_D,
    making use of eq.2.25 in [DermerMenon2009]
    """
>>>>>>> 9f67c876189d05e5fc62c1725409bc0e38e378ca
    epsilon_eq = epsilon_equivalency(m)
    epsilon = nu.to("", equivalencies=epsilon_eq)
    return (1 + z) * epsilon / delta_D


def B_to_cgs(B):
    """convert a magnetic field to CGS units"""
    return B.to(Gauss_cgs_unit, equivalencies=Gauss_cgs_equivalency)


def to_R_g_units(r, M):
    """convert a distance in graviational radii units"""
    R_g = G * M / c ** 2
    return (r / R_g).to("")
