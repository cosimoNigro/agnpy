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
    material, see Fig. 9 and Eq. 76 in [Finke2016]_.

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
    return np.sqrt(1 - addend)


def x_re_ring(R_re, r):
    """distance between the blob and a ring of reprocessing material"""
    return np.sqrt(np.power(R_re, 2) + np.power(r, 2))
