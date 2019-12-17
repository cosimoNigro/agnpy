import numpy as np
import astropy.units as u
import astropy.constants as const
from .utils import _power_law, I_bb_epsilon

G = const.G.cgs.value
C = const.c.cgs.value
ME = const.m_e.cgs.value
MEC2 = (const.m_e * const.c * const.c).cgs.value
M_SUN = const.M_sun.cgs.value
H = const.h.cgs.value
SIGMA_T = const.sigma_T.cgs.value
K_B = const.k_B.cgs.value
LAMBDA_C = 2.4263e-10  # Compton Wavelength of electron
unit_sed = u.Unit("erg cm-2 s-1")

__all__ = [
    "Target",
    "Monochromatic",
    "PowerLaw",
    "Disk",
    "SphericalShellBLR",
    "RingDustTorus",
]


class Target:
    """Generic class to represent a photon target for the external Compton

    Parameters
    ----------
    target_type : string
        type of target (isotropic monochromatic, broad line region, ...)
    mu_min : float

    mu_size : int
        size of the array of the zenith dependence of the target field
    phi_size : array_like
        size of the array of the azimuth dependence of the target field
    epsilon_size : int
        size of the dimensionless photon target energy
    """

    def __init__(
        self,
        target_type,
        mu_min=-1,
        mu_max=1,
        mu_size=50,
        phi_size=50,
        epsilon_size=100,
    ):
        self.target_type = target_type
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_size = mu_size
        self.phi_size = phi_size
        self.mu = np.linspace(self.mu_min, self.mu_max, self.mu_size)
        self.phi = np.linspace(0, 2 * np.pi, self.phi_size)
        self.epsilon_size = epsilon_size

    def set_mu_size(self, mu_size):
        """set the size of the mu grid and recompute it"""
        self.mu_size = mu_size
        self.mu = np.linspace(self.mu_min, self.mu_max, self.mu_size)

    def set_phi_size(self, phi_size):
        """set the size of the mu grid and recompute it"""
        self.phi_size = phi_size
        self.phi = np.linspace(0, 2 * np.pi, phi_size)


class Monochromatic(Target):
    """monochromatic isotropic photon field

    Parameters
    ----------
    u_0 : `~astropy.units.Quantity`
        energy density of the target photon distribution in erg cm-3
    epsilon : float
        monochromatic energy of the photon field in m_e c^2 unit

    Reference
    ---------
    [1] : Dermer, Menon; High Energy Radiation From Black Holes;
    Princeton Series in Astrophysics
    """

    def __init__(self, u_0, epsilon):
        Target.__init__(self, target_type="monochromatic isotropic")
        self.u_0 = u_0.to("erg cm-3").value
        self.epsilon = epsilon


class PowerLaw(Target):
    """power-law isotropic photon field, Eq. (6.85) in [1]

    Parameters
    ----------
    u_0 : `~astropy.units.Quantity`
        energy density of the target photon distribution in erg cm-3
    alpha : float
        "energy" spectral index, the total spectral index is -1 -alpha
    epsilon_1 : float
        minimum energy of the photon field in m_e c^2 unit
    epsilon_2 : float
        maximum energy of the photon field in m_e c^2 unit

    Reference
    ---------
    [1] : Dermer, Menon; High Energy Radiation From Black Holes;
    Princeton Series in Astrophysics
    """

    def __init__(self, u_0, alpha, epsilon_1, epsilon_2):
        Target.__init__(self, target_type="power law isotropic")
        self.u_0 = u_0.to("erg cm-3").value
        self.alpha = alpha
        self.epsilon_1 = epsilon_1
        self.epsilon_2 = epsilon_2
        # array of photon field energies
        self.epsilon = np.logspace(
            np.log10(self.epsilon_1), np.log10(self.epsilon_2), self.epsilon_size
        )
        # normalization of the photons, Eq. (6.86) in [1]
        n_0_ph_num = (1 - self.alpha) * self.u_0
        n_0_ph_denom = MEC2 * (
            np.power(self.epsilon_2, 1 - self.alpha)
            - np.power(self.epsilon_1, 1 - self.alpha)
        )
        self.n_0_ph = n_0_ph_num / n_0_ph_denom
        # mind the sign in the power law is inverted by _power_law
        self.n_ph = _power_law(
            self.epsilon, self.n_0_ph, 1 + self.alpha, self.epsilon_1, self.epsilon_2
        )
        self.u = MEC2 * self.epsilon * self.n_ph


class Disk(Target):
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
    r : `~astropy.units.Quantity`
        distance of the blob from the Black Hole

    Reference
    ---------
    [3] : https://ui.adsabs.harvard.edu/#abs/2016ApJ...830...94F/abstract
    """

    def __init__(self, M_BH, L_disk, eta, R_in, R_out, r):
        Target.__init__(
            self,
            target_type="Shakura Sunyaev accretion disk",
            mu_min=1 / np.sqrt(1 + np.power(R_out.cgs.value / r.cgs.value, 2)),
            mu_max=1 / np.sqrt(1 + np.power(R_in.cgs.value / r.cgs.value, 2)),
        )
        # define some disk parameter and try to plot
        self.M_BH = M_BH.cgs.value
        self.M_8 = self.M_BH / (1e8 * M_SUN)
        self.L_Edd = 1.26 * 1e46 * self.M_8  # erg s-1
        self.L_disk = L_disk.to("erg s-1").value
        # fraction of the Eddington luminosity at which the disk is accreting
        self.l_Edd = self.L_disk / self.L_Edd
        self.eta = eta
        # gravitational radius
        self.R_g = G * self.M_BH / np.power(C, 2)
        self.R_in = R_in.cgs.value
        self.R_out = R_out.cgs.value
        # array of distances along the disk, same size of the zenith angles array
        self.R = np.logspace(np.log10(self.R_in), np.log10(self.R_out), self.mu_size)
        # same array, scaled with the gravitational radius
        self.R_tilde = self.R / self.R_g
        self.r = r.cgs.value
        self.r_tilde = self.r / self.R_g

    def phi_disk(self, R):
        return 1 - np.sqrt(self.R_in / R)

    def phi_disk_mu(self, mu):
        """Shakura Sunyaev density of material in the disk, Eq. (66) in [3]

        Parameters
        ----------
        mu : `array_like`
            array of zenith angle of the target, see Figure 1 in [3]
        """
        R = self.r * np.sqrt(np.power(mu, -2) - 1)
        return self.phi_disk(R)

    def epsilon_disk(self, R):
        _term_1 = np.power(self.l_Edd / (self.M_8 * self.eta), 1 / 4)
        _term_2 = np.power(R / self.R_g, -3 / 4)
        _prefactor = 2.7 * 1e-4
        return _prefactor * _term_1 * _term_2

    def epsilon_disk_mu(self, mu):
        """Energy of the photons as a function of the distance from the disk
        center. Eq. (64) in [3]

        Parameters
        ----------
        mu : `array_like`
            array of zenith angle of the target, see Figure 1 in [3]
        """
        R = self.r * np.sqrt(np.power(mu, -2) - 1)
        return self.epsilon_disk(R)

    def sed(self, nu, blob):
        """nuFnu SED of the accretion disk
        we assume the disk is viewed from the side
        """
        cos_i = blob.mu_s  # angle of the line of sight wrt disk axis
        nu = nu.to("Hz").value * (1 + blob.z)
        _nu = nu.reshape(nu.size, 1)
        _R = self.R.reshape(1, self.R.size)
        _KT = self.epsilon_disk(_R) * MEC2 / 2.7
        _x = H * _nu / _KT
        _integrand = _R / (np.exp(_x) - 1)
        integral = np.trapz(_integrand, self.R, axis=-1)
        prefactor = (
            4
            * np.pi
            * cos_i
            * H
            * np.power(nu, 4)
            / (np.power(C, 2) * np.power(blob.d_L, 2))
        )
        return prefactor * integral * unit_sed


class SphericalShellBLR(Target):
    """Spherical Shell Broad Line Region.
    Single emission line at single radius

    Parameters
    ----------
    csi_line : float
        fraction of the disk radiation reprocessed
    epsilon_line : float
        dimensionless energy of the monochromatic radiation field
    R_line : `~astropy.units.Quantity`
        radius where the shell is located
    L_disk = `~astropy.units.Quantity`
        luminosity of the disk in erg s-1
    r : `~astropy.units.Quantity`
        distance of the blob from the Black Hole

    Reference
    ---------
    [3] : https://ui.adsabs.harvard.edu/#abs/2016ApJ...830...94F/abstract
    """

    def __init__(self, M_BH, L_disk, csi_line, epsilon_line, R_line, r):
        Target.__init__(self, target_type="spherical shell broad line region")
        self.M_BH = M_BH.cgs.value
        self.M_8 = self.M_BH / (1e8 * M_SUN)
        self.R_g = G * self.M_BH / np.power(C, 2)
        self.L_Edd = 1.26 * 1e46 * self.M_8  # erg s-1
        self.L_disk = L_disk.to("erg s-1").value
        self.l_Edd = self.L_disk / self.L_Edd
        self.csi_line = csi_line
        self.epsilon_line = epsilon_line
        self.R_line = R_line.cgs.value
        self.r = r.cgs.value
        self.r_tilde = self.r / self.R_g

    def x(self, mu):
        """distance between the blob and the reprocessing material

        Parameters
        ----------
        mu :  array_like
            cosine of the zenith from the central BH to the reprocessing material
            see Figure 9 in [3]
        """
        value = np.sqrt(
            np.power(self.R_line, 2)
            + np.power(self.r, 2)
            - 2 * self.r * self.R_line * mu
        )
        return value

    def mu_star(self, mu):
        """cosine of the angle between the blob and the reprocessing material
        Eq. (76) in [3]

        Parameters
        ----------
        mu :  array_like
            cosine of the zenith from the central BH to the reprocessing material
            see Figure 9 in [3]
        """
        addend = np.power(self.R_line / self.x(mu), 2) * (1 - np.power(mu, 2))
        mu_star = np.sqrt(1 - addend)
        return mu_star


class RingDustTorus(Target):
    """Dust Torus as infinitesimally thin annulus.
    Monochromatic emission at the peak of the Black Body spectrum.

    Parameters
    ----------
    csi_dt : float
        fraction of the disk radiation reprocessed
    epsilon_dt : float
        dimensionless energy peak of the black body distribution
    L_disk = `~astropy.units.Quantity`
        luminosity of the disk in erg s-1
    r : `~astropy.units.Quantity`
        distance of the blob from the Black Hole

    Reference
    ---------
    [3] : https://ui.adsabs.harvard.edu/#abs/2016ApJ...830...94F/abstract
    """

    def __init__(self, M_BH, L_disk, csi_dt, epsilon_dt, r, R_dt=None):
        Target.__init__(self, target_type="ring dust torus")
        self.M_BH = M_BH.cgs.value
        self.M_8 = self.M_BH / (1e8 * M_SUN)
        self.R_g = G * self.M_BH / np.power(C, 2)
        self.L_Edd = 1.26 * 1e46 * self.M_8  # erg s-1
        self.L_disk = L_disk.to("erg s-1").value
        self.l_Edd = self.L_disk / self.L_Edd
        self.csi_dt = csi_dt
        self.epsilon_dt = epsilon_dt
        # dimensionless temperature of the Dust Torus
        self.Theta = self.epsilon_dt / 2.7
        # temperatue in K
        self.T_dt = self.Theta * MEC2 / K_B
        self.L_disk = L_disk.to("erg s-1").value
        self.r = r.cgs.value
        self.r_tilde = self.r / self.R_g
        # if the radius is not specified use saturation radius Eq. (96) of [3]
        if R_dt is None:
            self.R_dt = (
                3.5
                * 1e18
                * np.sqrt(self.L_disk / 1e45)
                * np.power(self.T_dt / 1e3, -2.6)
            )
        else:
            self.R_dt = R_dt.cgs.value
        self.x = np.sqrt(np.power(self.R_dt, 2) + np.power(self.r, 2))

    def sed(self, nu, blob):
        epsilon = H * nu.to("Hz").value * (1 + blob.z) / MEC2
        prefactor = epsilon * np.pi * np.power(self.R_dt / blob.d_L, 2)
        return prefactor * I_bb_epsilon(epsilon, self.Theta) * unit_sed
