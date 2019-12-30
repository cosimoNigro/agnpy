"""classes and functions describing the external Compton target photon fields"""
import numpy as np
import astropy.constants as const
import astropy.units as u


# cgs is not well-handled by astropy.units
# every variable indicated with capital letters is dimensionsless
# will be used in SED computations for speed-up
E = 4.80320425e-10  # statC (not handled by astropy units)
H = const.h.cgs.value
C = const.c.cgs.value
ME = const.m_e.cgs.value
MEC = (const.m_e * const.c).cgs.value
MEC2 = (const.m_e * const.c * const.c).cgs.value
G = const.G.cgs.value
M_SUN = const.M_sun.cgs.value
K_B = const.k_B.cgs.value
SIGMA_SB = const.sigma_sb.cgs.value

__all__ = ["SSDisk", "SphericalShellBLR"]


class SSDisk:
    """Shakura Sunyaev disk.

    Parameters
    ----------
    M_BH : `~astropy.units.Quantity`
        Black Hole mass
    L_disk = `~astropy.units.Quantity`
        luminosity of the disk in erg s-1
    eta : float
        accretion efficiency
    R_in : `~astropy.units.Quantity`
        inner disk radius
    R_out : `~astropy.units.Quantity`
        outer disk radius

    properties with trailing underscore are dimensionless, for fast computation
    """

    def __init__(self, M_BH, L_disk, eta, R_in, R_out):
        self.type = "SSDisk"
        # masses and luminosities
        self.M_BH = M_BH.cgs
        self._M_BH = self.M_BH.value
        self.M_8 = self.M_BH.value / (1e8 * M_SUN)
        self.L_Edd = 1.26 * 1e46 * self.M_8 * u.Unit("erg s-1")
        self.L_disk = L_disk
        # fraction of the Eddington luminosity at which the disk is accreting
        self.l_Edd = (self.L_disk / self.L_Edd).decompose()
        self.eta = eta
        self.m_dot = (self.L_disk / (self.eta * const.c * const.c)).cgs
        self._m_dot = self.m_dot.value
        # gravitational radius
        self._R_g = G * self.M_BH.value / np.power(C, 2)
        self.R_g = self._R_g * u.cm
        self.R_in = R_in.cgs
        self.R_out = R_out.cgs
        # for fast computation of the temperature dependency
        self._R_in = self.R_in.value
        self._R_out = self.R_out.value

    def __str__(self):
        summary = (
            f"* Shakura Sunyaev accretion disk:\n"
            + f" - M_BH (central black hole mass): {self.M_BH:.2e}\n"
            + f" - L_disk (disk luminosity): {self.L_disk:.2e}\n"
            + f" - eta (accretion efficiency): {self.eta:.2e}\n"
            + f" - dot(m) (mass accretion rate): {self.m_dot:.2e}\n"
            + f" - R_in (disk inner radius): {self.R_in:.2e}\n"
            + f" - R_out (disk inner radius): {self.R_out:.2e}"
        )
        return summary

    def _phi_disk(self, R):
        """Radial dependency of disk temperature
        Eq. 63 in [4].

        Parameters
        ----------
        R : array_like
            radial coordinate along the disk, in cm
            dimensionless to speed-up computation in the external Compton SED
        """
        return 1 - np.sqrt(self._R_in / R)

    def _phi_disk_mu(self, mu, r):
        """same as _phi_disk but computed with cosine of zenith mu and distance
        from the black hole r. Eq. 67 in [4]."""
        R = r * np.sqrt(np.power(mu, -2) - 1)
        return self._phi_disk(R)

    def _epsilon(self, R):
        """Monochromatic approximation for the mean photon energy at radius R
        of the accretion disk. Eq. 64 in [4]. R is dimensionless.
        """
        _term_1 = np.power(self.l_Edd / (self.M_8 * self.eta), 1 / 4)
        _term_2 = np.power(R / self._R_g, -3 / 4)
        _prefactor = 2.7 * 1e-4
        return _prefactor * _term_1 * _term_2

    def _epsilon_mu(self, mu, r):
        """same as _epsilon but computed with cosine of zenith mu and distance
        from the black hole r. Eq. 67 in [4]."""
        R = r * np.sqrt(np.power(mu, -2) - 1)
        return self._epsilon(R)

    def T(self, R):
        """Temperature of the disk at distance R. Eq. 64 in [4]."""
        value = const.m_e * const.c * const.c / (2.7 * const.k_B) * self.epsilon(R)
        return value.to("K")


class SphericalShellBLR:
    """Spherical Shell Broad Line Region.
    Each line is emitted from an infinitesimally thin spherical shell. 

    Parameters
    ----------
    disk : `~agnpy.targets.SSDisk`
        disk whose radiation is being reprocessed by the BLR
    csi_line : float
        fraction of the disk radiation reprocessed by the disk
    epsilon_line : float
        dimensionless energy of the emitted line
    R_line : `~astropy.units.Quantity`
        radius of the spherical shell
    """

    def __init__(self, disk, csi_line, epsilon_line, R_line):
        self.type = "SphericalShellBLR"
        self.parent_disk = disk
        self.csi_line = csi_line
        self.epsilon_line = epsilon_line
        self.R_line = R_line.cgs
        self._R_line = self.R_line.value


