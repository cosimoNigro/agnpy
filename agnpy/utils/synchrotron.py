import numpy as np
from astropy.constants import e, c, m_e
from ..utils.conversion import B_to_cgs, mec2

e = e.gauss
c=c.cgs
mec2=mec2.cgs

def nu_synch_peak(B, gamma, mass=m_e):
    r"""Critical synchrotron frequency for a relativistic electron.

    Defined as the frequency at which a single electron of Lorentz factor
    :math:`\gamma` and mass :math:`m` in magnetic field :math:`B` emits the
    peak of its synchrotron power (Dermer & Menon 2009, Eq. 7.19):

    .. math::
        \nu_c(\gamma, x) = \frac{3\,e\,B(x)}{4\pi\,m\,c}\,\gamma^2.

    Parameters
    ----------
    B : :class:`~astropy.units.Quantity`
        Magnetic field strength. Converted to cgs Gauss internally.
        Shape must be broadcastable with ``gamma``.
    gamma : array-like or :class:`~astropy.units.Quantity`
        Electron Lorentz factors. Shape must be broadcastable with ``B``.
    mass : :class:`~astropy.constants.Constant`, optional
        Particle mass. Default is :data:`~astropy.constants.m_e` (electron).

    Returns
    -------
    :class:`~astropy.units.Quantity`
        Critical synchrotron frequency in Hz, with shape determined by
        broadcasting ``B`` and ``gamma``.
    """
    B = B_to_cgs(B)
    nu_peak = (3 * e * B / (4 * np.pi * mass * c)) * np.power(gamma, 2)
    return nu_peak.to("Hz")

def Z(eta):
    """Eq. 7.45 in [Dermer2009]_, angle-averaged integrand of the radiated power, the
    approximation of this function, given in Eq. D7 of [Aharonian2010]_, is used.
    """
    term_1_num = 1.808 * np.power(eta, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(eta, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(eta, 2 / 3) + 0.347 * np.power(eta, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(eta, 2 / 3) + 0.217 * np.power(eta, 4 / 3)
    return term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-eta)


def tau_to_attenuation(tau):
    """Convert SSA optical depth → attenuation factor.
    Eq. 7.122 in [DermerMenon2009]_."""
    u = 0.5 + np.exp(-tau) / tau - (1 - np.exp(-tau)) / np.power(tau, 2)
    return np.where(tau < 1e-3, 1, 3 * u / tau)
