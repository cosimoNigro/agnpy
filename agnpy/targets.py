import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import Distance


# every variable indicated with capital letters is unitless
# will be used in SED computations for speed-up
E = const.e.gauss.value
H = const.h.cgs.value
C = const.c.cgs.value
ME = const.m_e.cgs.value
MEC = (const.m_e * const.c).cgs.value
MEC2 = (const.m_e * const.c * const.c).cgs.value
G = const.G.cgs.value
M_SUN = const.M_sun.cgs.value
K_B = const.k_B.cgs.value
SIGMA_SB = const.sigma_sb.cgs.value
EMISSIVITY_UNIT = "erg s-1"
SED_UNIT = "erg cm-2 s-1"


lines_dictionary = {
    "Lyepsilon": {"lambda": 937.80 * u.Angstrom, "R_Hbeta_ratio": 2.7},
    "Lydelta": {"lambda": 949.74 * u.Angstrom, "R_Hbeta_ratio": 2.8},
    "CIII": {"lambda": 977.02 * u.Angstrom, "R_Hbeta_ratio": 0.83},
    "NIII": {"lambda": 990.69 * u.Angstrom, "R_Hbeta_ratio": 0.85},
    "Lybeta": {"lambda": 1025.72 * u.Angstrom, "R_Hbeta_ratio": 1.2},
    "OVI": {"lambda": 1033.83 * u.Angstrom, "R_Hbeta_ratio": 1.2},
    "ArI": {"lambda": 1066.66 * u.Angstrom, "R_Hbeta_ratio": 4.5},
    "Lyalpha": {"lambda": 1215.67 * u.Angstrom, "R_Hbeta_ratio": 0.27},
    "OI": {"lambda": 1304.35 * u.Angstrom, "R_Hbeta_ratio": 4.0},
    "SiII": {"lambda": 1306.82 * u.Angstrom, "R_Hbeta_ratio": 4.0},
    "SiIV": {"lambda": 1396.76 * u.Angstrom, "R_Hbeta_ratio": 0.83},
    "OIV]": {"lambda": 1402.06 * u.Angstrom, "R_Hbeta_ratio": 0.83},
    "CIV": {"lambda": 1549.06 * u.Angstrom, "R_Hbeta_ratio": 0.83},
    "NIV": {"lambda": 1718.55 * u.Angstrom, "R_Hbeta_ratio": 3.8},
    "AlII": {"lambda": 1721.89 * u.Angstrom, "R_Hbeta_ratio": 3.8},
    "CIII]": {"lambda": 1908.73 * u.Angstrom, "R_Hbeta_ratio": 0.46},
    "[NeIV]": {"lambda": 2423.83 * u.Angstrom, "R_Hbeta_ratio": 5.8},
    "MgII": {"lambda": 2798.75 * u.Angstrom, "R_Hbeta_ratio": 0.45},
    "HeI": {"lambda": 3188.67 * u.Angstrom, "R_Hbeta_ratio": 4.3},
    "Hdelta": {"lambda": 4102.89 * u.Angstrom, "R_Hbeta_ratio": 3.4},
    "Hgamma": {"lambda": 4341.68 * u.Angstrom, "R_Hbeta_ratio": 3.2},
    "HeII": {"lambda": 4687.02 * u.Angstrom, "R_Hbeta_ratio": 0.63},
    "Hbeta": {"lambda": 4862.68 * u.Angstrom, "R_Hbeta_ratio": 1.0},
    "[ClIII]": {"lambda": 5539.43 * u.Angstrom, "R_Hbeta_ratio": 4.8},
    "HeI": {"lambda": 5877.29 * u.Angstrom, "R_Hbeta_ratio": 0.39},
    "Halpha": {"lambda": 6564.61 * u.Angstrom, "R_Hbeta_ratio": 1.3},
}


__all__ = ["SSDisk", "SphericalShellBLR", "RingDustTorus", "print_lines_list"]


def print_lines_list():
    """Print the list of the available spectral lines.
    The dictionary with the possible emission lines is taken from Table 5 in 
    [Finke2016]_ and contains the value of the line wavelength and the ratio of 
    its radius to the radius of the :math:`H_{\\beta}` shell, not used at the moment.
    """
    for line in lines_dictionary.keys():
        print(f"{line}: {lines_dictionary[line]}")


def I_nu_bb(nu, T):
    """Black-Body intensity :math:`I_{\\nu}^{bb}`, Eq. 5.14 of [DermerMenon2009]_.
    Unitless parameters to speed-up calculations.

    Parameters
    ----------
    nu : :class:`~nump.ndarray`
        array of the frequencies (in Hz) to evaluate the Black Body intensity
    T : float or :class:`~nump.ndarray`
        temperature of the Black Body in K (might be function of the radius)
    """
    num = 2 * H * np.power(nu, 3)
    denum = np.power(C, 2) * (np.exp(H * nu / (K_B * T)) - 1)
    return num / denum


class SSDisk:
    """[Shakura1973]_ accretion disk.

    Parameters
    ----------
    M_BH : :class:`~astropy.units.Quantity`
        Black Hole mass    
    L_disk : :class:`~astropy.units.Quantity`
        luminosity of the disk 
    eta : float
        accretion efficiency
    R_in : :class:`~astropy.units.Quantity`
        inner disk radius
    R_out : :class:`~astropy.units.Quantity`
        outer disk radius
    """

    # properties with underscore are unitless, for fast computation

    def __init__(self, M_BH, L_disk, eta, R_in, R_out):
        self.type = "SSDisk"
        # masses and luminosities
        self.M_BH = M_BH.cgs
        self._M_BH = self.M_BH.value
        self.M_8 = self._M_BH / (1e8 * M_SUN)
        self.L_Edd = 1.26 * 1e46 * self.M_8 * u.Unit("erg s-1")
        self.L_disk = L_disk.cgs
        self._L_disk = self.L_disk.value
        # fraction of the Eddington luminosity at which the disk is accreting
        self.l_Edd = (self.L_disk / self.L_Edd).decompose().value
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
        Eq. 63 in [Dermer2009]_.

        Parameters
        ----------
        R : :class:`~numpy.ndarray`
            radial coordinate along the disk, in cm
            dimensionless to speed-up computation in the external Compton SED
        """
        return 1 - np.sqrt(self._R_in / R)

    def _phi_disk_mu(self, mu, r):
        """same as _phi_disk but computed with cosine of zenith mu and distance
        from the black hole r. Eq. 67 in [Dermer2009]_."""
        R = r * np.sqrt(np.power(mu, -2) - 1)
        return self._phi_disk(R)

    def _epsilon(self, R):
        """Monochromatic approximation for the mean photon energy at radius R
        of the accretion disk. Eq. 64 in [Dermer2009]_. R is unitless.
        """
        _term_1 = np.power(self.l_Edd / (self.M_8 * self.eta), 1 / 4)
        _term_2 = np.power(R / self._R_g, -3 / 4)
        _prefactor = 2.7 * 1e-4
        return _prefactor * _term_1 * _term_2

    def _epsilon_mu(self, mu, r):
        """same as _epsilon but computed with cosine of zenith mu and distance
        from the black hole r. Eq. 67 in [Dermer2009]_."""
        R = r * np.sqrt(np.power(mu, -2) - 1)
        return self._epsilon(R)

    def T(self, R):
        """Temperature of the disk at distance R. Eq. 64 in [Dermer2009]_."""
        value = MEC2 / (2.7 * K_B) * self._epsilon(R.value)
        return value * u.K

    def sed_flux(self, nu, z, mu_s):
        """Black Body SED generated by the SS Disk:

        .. math::
            \\nu F_{\\nu} \, [\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]
        
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed, **note** these are 
            observed frequencies (observer frame).
        z : float
            redshift of the galaxy, to correct the observed frequencies and to 
            compute the flux once the distance is obtained
        mu_s : float
            cosine of the angle between the disk axis and the line of sight,
            same as the jet viewing angle
        """
        nu = nu.to("Hz").value * (1 + z)
        d_L = Distance(z=z).to("cm").value
        prefactor = 2 * np.pi * mu_s / np.power(d_L, 2)
        R = np.logspace(np.log10(self._R_in), np.log10(self._R_out)) * u.cm
        T = self.T(R)
        _R = R.value.reshape(R.size, 1)
        _T = T.value.reshape(T.size, 1)
        _nu = nu.reshape(1, nu.size)
        _integrand = _R * I_nu_bb(_nu, _T)
        return prefactor * nu * np.trapz(_integrand, R, axis=0) * u.Unit(SED_UNIT)


class SphericalShellBLR:
    """Spherical Shell Broad Line Region, from [Finke2016]_.
    Each line is emitted from an infinitesimally thin spherical shell. 

    Parameters
    ----------
    L_disk : :class:`~astropy.units.Quantity`
        Luminosity of the disk whose radiation is being reprocessed by the BLR
    xi_line : float
        fraction of the disk radiation reprocessed by the BLR
    line : string
        type of line emitted
    R_line : :class:`~astropy.units.Quantity`
        radius of the BLR spherical shell
    """

    def __init__(self, L_disk, xi_line, line, R_line):
        self.type = "SphericalShellBLR"
        self.L_disk = L_disk
        self.xi_line = xi_line
        if line in lines_dictionary:
            self.line = line
            self.lambda_line = lines_dictionary[line]["lambda"]
        else:
            raise NameError(f"{line} not available in the line dictionary")
        self.epsilon_line = (
            self.lambda_line.to("erg", equivalencies=u.spectral()).value / MEC2
        )
        self.R_line = R_line

    def __str__(self):
        summary = (
            f"* Spherical Shell Broad Line Region:\n"
            + f" - L_disk (accretion disk luminosity): {self.L_disk:.2e}\n"
            + f" - xi_line (fraction of the disk radiation reprocessed by the BLR): {self.xi_line:.2e}\n"
            + f" - line (type of emitted line): {self.line}, lambda = {self.lambda_line:.2f}\n"
            + f" - R_line (radius of the BLR shell): {self.R_line:.2e}\n"
        )
        return summary

    def u(self, r):
        """Density of radiation produced by the BLR at the distance r along the 
        jet axis.
        Eq. 80 in [Finke2016]_

        Parameters
        ----------
        r : :class:`~astropy.units.Quantity`
            array of distances along the jet axis
        """
        mu = np.linspace(-1, 1)
        _mu = mu.reshape(mu.size, 1)
        _r = r.reshape(1, r.size)
        x2 = np.power(_r, 2) + np.power(self.R_line, 2) - 2 * _r * self.R_line * _mu
        integral = np.trapz(1 / x2, mu, axis=0)
        prefactor = self.xi_line * self.L_disk / (np.power(4 * np.pi, 2) * const.c)
        return (prefactor * integral).to("erg cm-3")


class RingDustTorus:
    """Dust Torus as infinitesimally thin annulus, from [Finke2016]_.
    For the Compton scattering monochromatic emission at the peak energy of the 
    Black Body spectrum is considered.

    Parameters
    ----------
    L_disk : :class:`~astropy.units.Quantity`
       Luminosity of the disk whose radiation is being reprocessed by the Torus
    xi_dt : float
        fraction of the disk radiation reprocessed
    T_dt : :class:`~astropy.units.Quantity`
        peak temperature of the black body emission of the Torus
    R_dt : :class:`~astropy.units.Quantity`
        radius of the Torus, if not specified the saturation radius of Eq. 96 in
        [Finke2016]_ will be used
    """

    def __init__(self, L_disk, xi_dt, T_dt, R_dt=None):
        self.type = "RingDustTorus"
        self.L_disk = L_disk
        self.xi_dt = xi_dt
        self.T_dt = T_dt
        # dimensionless temperature of the torus
        self.Theta = (const.k_B * self.T_dt).to("erg").value / MEC2
        self.epsilon_dt = 2.7 * self.Theta

        # if the radius is not specified use saturation radius Eq. 96 of [Finke2016]_
        if R_dt is None:
            self.R_dt = (
                3.5
                * 1e18
                * np.sqrt(self.L_disk.cgs.value / 1e45)
                * np.power(self.T_dt.to("K").value / 1e3, -2.6)
            ) * u.cm
        else:
            self.R_dt = R_dt.cgs

    def __str__(self):
        summary = (
            f"* Ring Dust Torus:\n"
            + f" - L_disk (accretion disk luminosity): {self.L_disk:.2e}\n"
            + f" - xi_dt (fraction of the disk radiation reprocessed by the torus): {self.xi_dt:.2e}\n"
            + f" - T_dt (temperature of the dust torus): {self.T_dt:.2e}\n"
            + f" - R_dt (radius of the torus): {self.R_dt:.2e}\n"
        )
        return summary

    def u(self, r):
        """Density of radiation produced by the Torus at the distance r along the 
        jet axis.
        Eq. 85 in [Finke2016]_

        Parameters
        ----------
        r : :class:`~astropy.units.Quantity`
            array of distances along the jet axis
        """
        x2 = np.power(self.R_dt, 2) + np.power(r, 2)
        prefactor = self.xi_dt * self.L_disk / (np.power(4 * np.pi, 2) * const.c)
        return (prefactor * 1 / x2).to("erg cm-3")

    def sed_flux(self, nu, z):
        """Black Body SED generated by the Dust Torus:

        .. math::
            \\nu F_{\\nu} \, [\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]
        
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed, **note** these are 
            observed frequencies (observer frame).
        z : float
            redshift of the galaxy, to correct the observed frequencies and to 
            compute the flux once the distance is obtained
        """
        nu = nu.to("Hz").value * (1 + z)
        d_L = Distance(z=z).to("cm").value
        prefactor = np.pi * np.power(self._R_dt / d_L, 2)
        sed = prefactor * nu * I_nu_bb(nu, self._T_dt)
        return sed * u.Unit("erg cm-2 s-1")
