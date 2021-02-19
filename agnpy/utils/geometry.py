# geometry utilities for agnpy
import numpy as np


def cos_psi(mu_s, mu, phi):
    """compute the angle between the blob (with zenith mu_s) and a photon with
    zenith and azimuth (mu, phi). The system is symmetric in azimuth for the
    electron phi_s = 0, Eq. 8 in [Finke2016]_."""
    term_1 = mu * mu_s
    term_2 = np.sqrt(1 - np.power(mu, 2)) * np.sqrt(1 - np.power(mu_s, 2))
    term_3 = np.cos(phi)
    return term_1 + term_2 * term_3


def x_re_shell(mu, R_re, r):
    """distance between the blob and a spherical reprocessing material,
    see Fig. 9 and Eq. 76 in [Finke2016]_.

    Parameters
    ----------
    mu : :class:`~numpy.ndarray`
        (array of) cosine of the zenith angle subtended by the target
    R_re : :class:`~astropy.units.Quantity`
        distance from the BH to the reprocessing material
    r : :class:`~astropy.units.Quantity`
        height of the emission region in the jet
    """
    return np.sqrt(np.power(R_re, 2) + np.power(r, 2) - 2 * r * R_re * mu)


def mu_star_shell(mu, R_re, r):
    """cosine of the angle between the blob and a sphere of reprocessing
    material, see Fig. 9 and Eq. 78 in [Finke2016]_.
    works only if the blob is at the jet axis

    Parameters
    ----------
    mu : :class:`~numpy.ndarray`
        (array of) cosine of the zenith angle subtended by the target
    R_re : :class:`~astropy.units.Quantity`
        distance (in cm) from the BH to the reprocessing material
    r : :class:`~astropy.units.Quantity`
        height (in cm) of the emission region in the jet
    """
    addend = np.power(R_re / x_re_shell(mu, R_re, r), 2) * (1 - np.power(mu, 2))
    mu_star = np.sqrt(1 - addend)

    # if r<mu*R_re you need to multiply by -1 because the blob is before the shell element
    mu_sign = ((r > mu * R_re) - 0.5) * 2
    mu_star *= mu_sign
    return mu_star


def x_re_ring(R_re, r):
    """distance between the blob and a ring of reprocessing material"""
    return np.sqrt(np.power(R_re, 2) + np.power(r, 2))


def x_re_ring_mu_s(R_re, r, phi_re, u, mu_s):
    """distance between the blob and a ring of reprocessing material
    if the photon moved u along the mu_s direction starting from r position

    Parameters
    ----------
    R_re : :class:`~astropy.units.Quantity`
        distance (in cm) from the BH to the reprocessing material
    r : :class:`~astropy.units.Quantity`
        distance (in cm) from the BH to the starting place of the photon
    phi_re : :class:`~numpy.ndarray`
        (array of) azimuth angle of the reprocessing material 
    u : :class:`~numpy.ndarray`
        (array of) distances (in cm) that the gamma ray has travelled 
    mu_s : :class:`~astropy.units.Quantity`
        direction of gamma ray  motion: cos angle to the jet axis. 
        The gamma ray is moving in the direction of phi_re=0
    """

    sin_theta_s = np.sqrt(1 - mu_s * mu_s)
    x2 = (
        u * u
        + R_re * R_re
        + r * r
        - 2 * u * R_re * sin_theta_s * np.cos(phi_re)
        + 2 * r * u * mu_s
    )
    return np.sqrt(x2)


def phi_mu_re_ring(R_re, r, phi_re, u, mu_s):
    """azimuth and  angle of the soft photon produced from  DT/BLR ring
    photon is produced at (R_re*cos(phi_re), R_re*sin(phi_re),0)
    and reached gamma ray at (u*sin(theta_s), 0, r+u*mu_s) distance between the blob and a ring of reprocessing material
    if the photon moved u along the mu_s direction starting from r position

    Parameters
    ----------
    R_re : :class:`~astropy.units.Quantity`
        distance (in cm) from the BH to the reprocessing material
    r : :class:`~astropy.units.Quantity`
        distance (in cm) from the BH to the starting place of the photon
    phi_re : :class:`~numpy.ndarray`
        (array of) azimuth angle of the reprocessing material 
    u : :class:`~numpy.ndarray`
        (array of) distances (in cm) that the gamma ray has travelled 
    mu_s : :class:`~astropy.units.Quantity`
        direction of gamma ray  motion: cos angle to the jet axis. 
        The gamma ray is moving in the direction of phi_re=0
    """
    sin_theta_s = np.sqrt(1 - mu_s * mu_s)
    dx = u * sin_theta_s - R_re * np.cos(phi_re)
    dy = 0 - R_re * np.sin(phi_re)
    dz = r + u * mu_s
    phi = np.arctan2(dy, dx)
    x = x_re_ring_mu_s(R_re, r, phi_re, u, mu_s)
    mu = dz / x
    return phi, mu
