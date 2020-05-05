import numpy as np
from astropy.constants import m_e, h, c, k_B, M_sun, G
import astropy.units as u
from astropy.coordinates import Distance


mec2 = m_e.to("erg", equivalencies=u.mass_energy())
lambda_c = (h / (m_e * c)).to("cm")  # Compton wavelength
# equivalency to transform frequencies to energies in electron rest mass units
epsilon_equivalency = [
    (u.Hz, u.Unit(""), lambda x: h.cgs * x / mec2, lambda x: x * mec2 / h.cgs)
]


# dictionary with all the available spectral lines
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


def I_epsilon_bb(epsilon, Theta):
    """Black-Body intensity :math:`I_{\\nu}^{bb}`, Eq. 5.15 of [DermerMenon2009]_.

    Parameters
    ----------
    epsilon : :class:`~numpy.ndarray`
        array of dimensionless energies (in electron rest mass units) 
    Theta : float 
        dimensionless temperature of the Black Body 
    """
    num = 2 * m_e * np.power(c, 3) * np.power(epsilon, 3)
    denum = np.power(lambda_c, 3) * (np.exp(epsilon / Theta) - 1)
    I = num / denum
    return I.to("erg cm-2 s-1")


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
    R_in : :class:`~astropy.units.Quantity` / float
        inner disk radius
    R_out : :class:`~astropy.units.Quantity` / float
        outer disk radius
    R_g_units : bool
        whether or not input radiuses are specified in units of the gravitational radius
    """

    def __init__(self, M_BH, L_disk, eta, R_in, R_out, R_g_units=False):
        self.type = "SSDisk"
        # masses and luminosities
        self.M_BH = M_BH
        self.M_8 = (M_BH / (1e8 * M_sun)).to_value("")
        self.L_Edd = 1.26 * 1e46 * self.M_8 * u.Unit("erg s-1")
        self.L_disk = L_disk
        # fraction of the Eddington luminosity at which the disk is accreting
        self.l_Edd = (self.L_disk / self.L_Edd).to_value("")
        self.eta = eta
        self.m_dot = self.L_disk / (self.eta * np.power(c, 2))
        # gravitational radius
        self.R_g = (G * self.M_BH / np.power(c, 2)).to("cm")
        if R_g_units:
            # check that numbers have been passed
            R_in_unit_check = isinstance(R_in, int) or isinstance(R_in, float)
            R_out_unit_check = isinstance(R_out, int) or isinstance(R_out, float)
            if R_in_unit_check and R_out_unit_check:
                self.R_in = R_in * self.R_g
                self.R_out = R_out * self.R_g
                self.R_in_tilde = R_in
                self.R_out_tilde = R_out
            else:
                raise TypeError("R_in / R_out passed with units, int / float expected")
        else:
            # check that quantities have been passed
            R_in_unit_check = isinstance(R_in, u.Quantity)
            R_out_unit_check = isinstance(R_out, u.Quantity)
            if R_in_unit_check and R_out_unit_check:
                self.R_in = R_in
                self.R_out = R_out
                self.R_in_tilde = (self.R_in / self.R_g).to_value("")
                self.R_out_tilde = (self.R_out / self.R_g).to_value("")
            else:
                raise TypeError("R_in / R_out passed without units")
        # array of R_tile values
        self.R_tilde = np.linspace(self.R_in_tilde, self.R_out_tilde)

    def __str__(self):
        summary = (
            f"* Shakura Sunyaev accretion disk:\n"
            + f" - M_BH (central black hole mass): {self.M_BH.cgs:.2e}\n"
            + f" - L_disk (disk luminosity): {self.L_disk.cgs:.2e}\n"
            + f" - eta (accretion efficiency): {self.eta:.2e}\n"
            + f" - dot(m) (mass accretion rate): {self.m_dot.cgs:.2e}\n"
            + f" - R_in (disk inner radius): {self.R_in.cgs:.2e}\n"
            + f" - R_out (disk inner radius): {self.R_out.cgs:.2e}"
        )
        return summary

    def mu_from_r_tilde(self, r_tilde, size=100):
        """array of cosine angles, spanning from :math:`R_{\mathrm{in}}` to 
        :math:`R_{\mathrm{out}}`, viewed from a given distance :math:`\\tilde{r}` 
        along the jet axis, Eq. 72 and 73 in [Finke2016]_."""
        mu_min = 1 / np.sqrt(1 + np.power((self.R_out_tilde / r_tilde), 2))
        mu_max = 1 / np.sqrt(1 + np.power((self.R_in_tilde / r_tilde), 2))
        return np.linspace(mu_min, mu_max, size)

    def phi_disk(self, R_tilde):
        """Radial dependency of disk temperature
        Eq. 63 in [Dermer2009]_.

        Parameters
        ----------
        R_tilde : :class:`~nump.ndarray`
            radial coordinate along the disk normalised to R_g
        """
        return 1 - np.sqrt(self.R_in_tilde / R_tilde)

    def phi_disk_mu(self, mu, r_tilde):
        """same as phi_disk but computed with cosine of zenith mu and normalised 
        distance from the black hole :math:`\\tilde{r}` Eq. 67 in [Dermer2009]_."""
        R_tilde = r_tilde * np.sqrt(np.power(mu, -2) - 1)
        return self.phi_disk(R_tilde)

    def epsilon(self, R_tilde):
        """monochromatic approximation for the mean photon energy at radius 
        :math:`\\tilde{R}` of the accretion disk. Eq. 65 in [Dermer2009]_."""
        xi = np.power(self.l_Edd / (self.M_8 * self.eta), 1 / 4)
        return 2.7 * 1e-4 * xi * np.power(R_tilde, -3 / 4)

    def epsilon_mu(self, mu, r_tilde):
        """same as epsilon but computed with cosine of zenith mu and distance
        from the black hole :math:`\\tilde{r}`. Eq. 67 in [Dermer2009]_."""
        R_tilde = r_tilde * np.sqrt(np.power(mu, -2) - 1)
        return self.epsilon(R_tilde)

    def T(self, R_tilde):
        """Temperature of the disk at distance :math:`\\tilde{R}`. 
        Eq. 64 in [Dermer2009]_."""
        value = mec2 / (2.7 * k_B) * self.epsilon(R_tilde)
        return value.to("K")

    def Theta(self, R_tilde):
        """Dimensionless temperature of the black body at distance :math:`\\tilde{R}`"""
        theta = k_B * self.T(R_tilde) / mec2
        return theta.to_value("")

    def u_ph(self, r):
        """Density of radiation produced by the Disk at the distance r along the 
        jet axis. Integral over the solid angle of Eq. 69 in [Dermer2009]_.
        
        Parameters
        ----------
        r : :class:`~astropy.units.Quantity`
            distance along the jet axis
        """
        r_tilde = (r / self.R_g).to_value("")
        mu = self.mu_from_r_tilde(r_tilde)
        integrand = np.power(np.power(mu, -2) - 1, -3 / 2) * self.phi_disk_mu(
            mu, r_tilde
        )
        prefactor_denum = (
            16
            * np.power(np.pi, 2)
            * c
            * self.eta
            * np.power(self.R_g, 2)
            * np.power(r_tilde, 3)
        )
        prefactor = 3 * self.L_disk / prefactor_denum
        density = prefactor * np.trapz(integrand, mu, axis=0)
        return density.to("erg cm-3")

    def sed_flux(self, nu, z):
        """Black Body SED generated by the SS Disk, considered as a 
        multi-dimensional black body. I obtain the formula following 
        Chapter 5 of [DermerMenon2009]_

        .. math::
            f_{\epsilon} (= \\nu F_{\\nu}) &= 
            \epsilon \, \int_{\Omega_s} \mu I_{\epsilon} \mathrm{d}\Omega \\\\
            &= \epsilon \, 2 \pi \int_{\mu_{\mathrm{min}}}^{\mu_{\mathrm{max}}}
            \mu I_{\epsilon} \mathrm{d}\mu

        where the cosine of the angle under which an observer at :math:`d_L` 
        sees the disk is :math:`\mu = 1 / \sqrt{1 + (R / d_L)^2}`, integrating
        over :math:`R` rather than :math:`\mu`

        .. math::
            f_{\epsilon} &= \epsilon \, 2 \pi \int_{R_{\mathrm{in}}}^{R_{\mathrm{out}}}
            (1 + R^2 / d_L^2)^{-3/2} \\frac{R}{d_L^2} \, I_{\epsilon}(R) \, \mathrm{d}R \\\\
            &= \epsilon \, 2 \pi \\frac{R_g^2}{d_L^2} 
            \int_{\\tilde{R}_{\mathrm{in}}}^{\\tilde{R}_{\mathrm{out}}}
            \\left(1 + \\tilde{R}^2 / \\tilde{d_L}^2 \\right)^{-3/2} \, 
            \\tilde{R} \, I_{\epsilon}(\\tilde{R}) \, \mathrm{d}\\tilde{R}
      
        where in the last integral distances with :math:`\\tilde{}` have been 
        scaled to the gravitational radius :math:`R_g`.

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed, **note** these are 
            observed frequencies (observer frame).
        z : float
            redshift of the galaxy, to correct the observed frequencies and to 
            compute the flux once the distance is obtained
        """
        nu *= 1 + z
        epsilon = nu.to("", equivalencies=epsilon_equivalency)
        d_L = Distance(z=z).to("cm")
        d_L_tilde = (d_L / self.R_g).to_value("")
        Theta = self.Theta(self.R_tilde)
        # for multidimensional integration
        # axis 0: radiuses (and temperatures)
        # axis 1: photons epsilon
        _R_tilde = np.reshape(self.R_tilde, (self.R_tilde.size, 1))
        _Theta = np.reshape(Theta, (Theta.size, 1))
        _epsilon = np.reshape(epsilon, (1, epsilon.size))
        _integrand = (
            np.power(1 + np.power(_R_tilde / d_L_tilde, 2), -3 / 2)
            * _R_tilde
            * I_epsilon_bb(_epsilon, _Theta)
        )
        prefactor = 2 * np.pi * np.power(self.R_g, 2) / np.power(d_L, 2)
        sed = epsilon * prefactor * np.trapz(_integrand, self.R_tilde, axis=0)
        return sed.to("erg cm-2 s-1")


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
            self.lambda_line.to("erg", equivalencies=u.spectral()) / mec2
        ).to_value("")
        self.R_line = R_line

    def __str__(self):
        summary = (
            f"* Spherical Shell Broad Line Region:\n"
            + f" - L_disk (accretion disk luminosity): {self.L_disk.cgs:.2e}\n"
            + f" - xi_line (fraction of the disk radiation reprocessed by the BLR): {self.xi_line:.2e}\n"
            + f" - line (type of emitted line): {self.line}, lambda = {self.lambda_line.cgs:.2f}\n"
            + f" - R_line (radius of the BLR shell): {self.R_line.cgs:.2e}\n"
        )
        return summary

    def u_ph(self, r):
        """Density of radiation produced by the BLR at the distance r along the 
        jet axis. Integral over the solid angle of Eq. 80 in [Finke2016]_.

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
        prefactor = self.xi_line * self.L_disk / (np.power(4 * np.pi, 2) * c)
        return (2 * np.pi * prefactor * integral).to("erg cm-3")


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
        self.Theta = (k_B * self.T_dt / mec2).to_value("")
        self.epsilon_dt = 2.7 * self.Theta

        # if the radius is not specified use saturation radius Eq. 96 of [Finke2016]_
        if R_dt is None:
            self.R_dt = (
                3.5
                * 1e18
                * np.sqrt((self.L_disk / (1e45 * u.Unit("erg s-1"))).to_value(""))
                * np.power((self.T_dt / (1e3 * u.K)).to_value(""), -2.6)
            ) * u.cm
        else:
            self.R_dt = R_dt

    def __str__(self):
        summary = (
            f"* Ring Dust Torus:\n"
            + f" - L_disk (accretion disk luminosity): {self.L_disk.cgs:.2e}\n"
            + f" - xi_dt (fraction of the disk radiation reprocessed by the torus): {self.xi_dt:.2e}\n"
            + f" - T_dt (temperature of the dust torus): {self.T_dt:.2e}\n"
            + f" - R_dt (radius of the torus): {self.R_dt.cgs:.2e}\n"
        )
        return summary

    def u_ph(self, r):
        """Density of radiation produced by the Torus at the distance r along the 
        jet axis. Integral over the solid angle of Eq. 85 in [Finke2016]_

        Parameters
        ----------
        r : :class:`~astropy.units.Quantity`
            array of distances along the jet axis
        """
        x2 = np.power(self.R_dt, 2) + np.power(r, 2)
        prefactor = self.xi_dt * self.L_disk / (np.power(4 * np.pi, 2) * c)
        return (2 * np.pi * prefactor * 1 / x2).to("erg cm-3")

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
        nu *= 1 + z
        epsilon = nu.to("", equivalencies=epsilon_equivalency)
        d_L = Distance(z=z).to("cm")
        prefactor = np.pi * np.power((self.R_dt / d_L).to_value(""), 2)
        sed = prefactor * epsilon * I_epsilon_bb(epsilon, self.Theta)
        return sed * u.Unit("erg cm-2 s-1")
