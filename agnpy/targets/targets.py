# module containing the targets for the external compton radiation
import numpy as np
from astropy.constants import c, G, M_sun, k_B, sigma_sb
import astropy.units as u
from astropy.coordinates import Distance
from astropy.modeling.models import BlackBody
from ..utils.conversion import mec2
from ..utils.math import axes_reshaper


__all__ = [
    "lines_dictionary",
    "CMB",
    "SSDisk",
    "PointSourceBehindJet",
    "SphericalShellBLR",
    "RingDustTorus",
]


# dictionary with all the available spectral lines
lines_dictionary = {
    "Lyepsilon": {
        "lambda": 937.80 * u.Angstrom,
        "R_Hbeta_ratio": 2.7,
        "L_Hbeta_ratio": 0.24,
    },
    "Lydelta": {
        "lambda": 949.74 * u.Angstrom,
        "R_Hbeta_ratio": 2.8,
        "L_Hbeta_ratio": 0.24,
    },
    "CIII": {
        "lambda": 977.02 * u.Angstrom,
        "R_Hbeta_ratio": 0.83,
        "L_Hbeta_ratio": 0.60,
    },
    "NIII": {
        "lambda": 990.69 * u.Angstrom,
        "R_Hbeta_ratio": 0.85,
        "L_Hbeta_ratio": 0.60,
    },
    "Lybeta": {
        "lambda": 1025.72 * u.Angstrom,
        "R_Hbeta_ratio": 1.2,
        "L_Hbeta_ratio": 1.1,
    },
    "OVI": {
        "lambda": 1033.83 * u.Angstrom,
        "R_Hbeta_ratio": 1.2,
        "L_Hbeta_ratio": 1.1,
    },
    "ArI": {
        "lambda": 1066.66 * u.Angstrom,
        "R_Hbeta_ratio": 4.5,
        "L_Hbeta_ratio": 0.094,
    },
    "Lyalpha": {
        "lambda": 1215.67 * u.Angstrom,
        "R_Hbeta_ratio": 0.27,
        "L_Hbeta_ratio": 12,
    },
    "OI": {
        "lambda": 1304.35 * u.Angstrom,
        "R_Hbeta_ratio": 4.0,
        "L_Hbeta_ratio": 0.23,
    },
    "SiII": {
        "lambda": 1306.82 * u.Angstrom,
        "R_Hbeta_ratio": 4.0,
        "L_Hbeta_ratio": 0.23,
    },
    "SiIV": {
        "lambda": 1396.76 * u.Angstrom,
        "R_Hbeta_ratio": 0.83,
        "L_Hbeta_ratio": 1.0,
    },
    "OIV]": {
        "lambda": 1402.06 * u.Angstrom,
        "R_Hbeta_ratio": 0.83,
        "L_Hbeta_ratio": 1.0,
    },
    "CIV": {
        "lambda": 1549.06 * u.Angstrom,
        "R_Hbeta_ratio": 0.83,
        "L_Hbeta_ratio": 2.9,
    },
    "NIV": {
        "lambda": 1718.55 * u.Angstrom,
        "R_Hbeta_ratio": 3.8,
        "L_Hbeta_ratio": 0.30,
    },
    "AlII": {
        "lambda": 1721.89 * u.Angstrom,
        "R_Hbeta_ratio": 3.8,
        "L_Hbeta_ratio": 0.30,
    },
    "CIII]": {
        "lambda": 1908.73 * u.Angstrom,
        "R_Hbeta_ratio": 0.46,
        "L_Hbeta_ratio": 1.8,
    },
    "[NeIV]": {
        "lambda": 2423.83 * u.Angstrom,
        "R_Hbeta_ratio": 5.8,
        "L_Hbeta_ratio": 0.051,
    },
    "MgII": {
        "lambda": 2798.75 * u.Angstrom,
        "R_Hbeta_ratio": 0.45,
        "L_Hbeta_ratio": 1.7,
    },
    "Hdelta": {
        "lambda": 4102.89 * u.Angstrom,
        "R_Hbeta_ratio": 3.4,
        "L_Hbeta_ratio": 0.12,
    },
    "Hgamma": {
        "lambda": 4341.68 * u.Angstrom,
        "R_Hbeta_ratio": 3.2,
        "L_Hbeta_ratio": 0.30,
    },
    "HeII": {
        "lambda": 4687.02 * u.Angstrom,
        "R_Hbeta_ratio": 0.63,
        "L_Hbeta_ratio": 0.016,
    },
    "Hbeta": {
        "lambda": 4862.68 * u.Angstrom,
        "R_Hbeta_ratio": 1.0,
        "L_Hbeta_ratio": 1.0,
    },
    "[ClIII]": {
        "lambda": 5539.43 * u.Angstrom,
        "R_Hbeta_ratio": 4.8,
        "L_Hbeta_ratio": 0.039,
    },
    "HeI": {
        "lambda": 5877.29 * u.Angstrom,
        "R_Hbeta_ratio": 0.39,
        "L_Hbeta_ratio": 0.092,
    },
    "Halpha": {
        "lambda": 6564.61 * u.Angstrom,
        "R_Hbeta_ratio": 1.3,
        "L_Hbeta_ratio": 3.6,
    },
}


class CMB:
    """Cosmic Microwave Background radiation, approximated as an isotropic
    monochromatic target.

    Parameters
    ----------
    z : float
        redshift at which the CMB is considered
    """

    def __init__(self, z):
        self.name = "Cosmic Microwave Background Radiation"
        a = 7.5657 * 1e-15 * u.Unit("erg cm-3 K-4")  # radiation constant
        T = 2.72548 * u.K
        self.u_0 = (a * np.power(T, 4)).to("erg cm-3") * np.power(1 + z, 4)
        self.epsilon_0 = (2.7 * k_B * T / mec2).to_value("") * (1 + z)

    def u(self, blob=None):
        """integral energy density of the CMB

        Parameters
        ----------
        blob : :class:`~agnpy.emission_regions.Blob`
            if provided, the energy density is computed in a reference frame
            comvoing with the blob
        """
        if blob:
            return self.u_0 * np.power(blob.Gamma, 2) * (1 + np.power(blob.Beta, 2) / 3)
        else:
            return self.u_0


class PointSourceBehindJet:
    """Monochromatic point source behind the jet.

    Parameters
    ----------
    L_0 : :class:`~astropy.units.Quantity`
        luminosity of the source
    epsilon_0 : float
        dimensionless monochromatic energy of the source
    """

    def __init__(self, L_0, epsilon_0):
        self.name = "Monochromatic Point Source Behind the Jet"
        self.L_0 = L_0
        self.epsilon_0 = epsilon_0

    def u(self, r, blob=None):
        """integral energy density of the point source at distance r along the
        jet axis

        Parameters
        ----------
        r : :class:`~astropy.units.Quantity`
            array of distances along the jet axis
        blob : :class:`~agnpy.emission_regions.Blob`
            if provided, the energy density is computed in a reference frame
            comvoing with the blob
        """
        u_0 = (self.L_0 / (4 * np.pi * c * np.power(r, 2))).to("erg cm-3")
        if blob:
            return u_0 / (np.power(blob.Gamma, 2) * np.power(1 + blob.Beta, 2))
        else:
            return u_0


class SSDisk:
    """[Shakura1973]_ accretion disk as modeled by [Dermer2002]_

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
        self.name = "Shakura Sunyaev Accretion Disk"
        # masses and luminosities
        self.M_BH = M_BH
        self.M_8 = (M_BH / (1e8 * M_sun)).to_value("")
        self.L_Edd = 1.26 * 1e46 * self.M_8 * u.Unit("erg s-1")
        self.L_disk = L_disk
        # fraction of the Eddington luminosity at which the disk is accreting
        self.l_Edd = (self.L_disk / self.L_Edd).to_value("")
        self.eta = eta
        self.m_dot = (self.L_disk / (self.eta * np.power(c, 2))).to("g s-1")
        # gravitational radius
        self.R_g = (G * self.M_BH / np.power(c, 2)).to("cm")
        if R_g_units:
            # check that numbers have been passed
            R_in_unit_check = isinstance(R_in, (int, float))
            R_out_unit_check = isinstance(R_out, (int, float))
            if not R_in_unit_check or not R_out_unit_check:
                raise TypeError("R_in / R_out passed with units, int / float expected")
            self.R_in = R_in * self.R_g
            self.R_out = R_out * self.R_g
            self.R_in_tilde = R_in
            self.R_out_tilde = R_out
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
        return (
            f"* Shakura Sunyaev accretion disk:\n"
            + f" - M_BH (central black hole mass): {self.M_BH.cgs:.2e}\n"
            + f" - L_disk (disk luminosity): {self.L_disk.cgs:.2e}\n"
            + f" - eta (accretion efficiency): {self.eta:.2e}\n"
            + f" - dot(m) (mass accretion rate): {self.m_dot.cgs:.2e}\n"
            + f" - R_in (disk inner radius): {self.R_in.cgs:.2e}\n"
            + f" - R_out (disk inner radius): {self.R_out.cgs:.2e}"
        )

    # staticmethods to be used in SED calculations without using a class instance
    @staticmethod
    def evaluate_mu_from_r_tilde(R_in_tilde, R_out_tilde, r_tilde, size=100):
        r"""array of cosine angles, spanning from :math:`R_{\mathrm{in}}` to
        :math:`R_{\mathrm{out}}`, viewed from a given height :math:`\tilde{r}`
        above the disk, Eq. 72 and 73 in [Finke2016]_."""
        mu_min = 1 / np.sqrt(1 + np.power((R_out_tilde / r_tilde), 2))
        mu_max = 1 / np.sqrt(1 + np.power((R_in_tilde / r_tilde), 2))
        return np.linspace(mu_min, mu_max, size)

    @staticmethod
    def evaluate_phi_disk_mu(mu, R_in_tilde, r_tilde):
        """dependency of the radiant surface-energy flux from the disk radius,
        here obtained from the cosine of the zenith `mu` and the height above
        the disk `r_tilde` (in graviational radius units),
        Eq. 63 [Dermer2009]_"""
        R_tilde = r_tilde * np.sqrt(np.power(mu, -2) - 1)
        return 1 - np.sqrt(R_in_tilde / R_tilde)

    @staticmethod
    def evaluate_epsilon(L_disk, M_BH, eta, R_tilde):
        """evaluate the dimensionless energy emitted at the radius `R_tilde`
        Eq. 65 [Dermer2009]_"""
        M_8 = (M_BH / (1e8 * M_sun)).to_value("")
        L_Edd = 1.26 * 1e46 * M_8 << u.Unit("erg s-1")
        l_Edd = (L_disk / L_Edd).to_value("")
        xi = np.power(l_Edd / (M_8 * eta), 1 / 4)
        return 2.7 * 1e-4 * xi * np.power(R_tilde, -3 / 4)

    @staticmethod
    def evaluate_epsilon_mu(L_disk, M_BH, eta, mu, r_tilde):
        """same as :func:`~agnpy.targets.SSDisk.evaluate_epsilon` but
        considering the cosine of the subtended zenith `mu` and the height
        above the disk `r` instead of the radius `R_tilde`"""
        R_tilde = r_tilde * np.sqrt(np.power(mu, -2) - 1)
        return SSDisk.evaluate_epsilon(L_disk, M_BH, eta, R_tilde)

    @staticmethod
    def evaluate_T(R, M_BH, m_dot, R_in):
        """black body temperature (K) at the radial coordinate R"""
        phi = 1 - np.sqrt((R_in / R).to_value(""))
        val = (3 * G * M_BH * m_dot * phi) / (8 * np.pi * np.power(R, 3) * sigma_sb)
        return np.power(val, 1 / 4).to("K")

    def epsilon_mu(self, mu, r_tilde):
        return self.evaluate_epsilon_mu(self.L_disk, self.M_BH, self.eta, mu, r_tilde)

    def epsilon(self, R_tilde):
        r"""dimensionless energy emitted at the disk radius :math:`\tilde{R}`"""
        return self.evaluate_epsilon(self.L_disk, self.M_BH, self.eta, R_tilde)

    def phi_disk_mu(self, mu, r_tilde):
        return self.evaluate_phi_disk_mu(mu, self.R_in_tilde, r_tilde)

    def phi_disk(self, R_tilde):
        return 1 - np.sqrt(self.R_in_tilde / R_tilde)

    def T(self, R):
        r"""temperature of the disk at radius :math:`R`.
        Eq. 64 in [Dermer2009]_."""
        return self.evaluate_T(R, self.M_BH, self.m_dot, self.R_in)

    @staticmethod
    def evaluate_multi_T_bb_sed(nu, z, M_BH, m_dot, R_in, R_out, d_L, mu_s=1):
        r"""Evaluate a multi-temperature black body SED in the case of the SS Disk.
        The SED is calculated for an observer far away from the disk (:math:`d_L \gg R`)
        with the following:

        .. math::
            \nu F_{\nu} \approx \mu_s \, \nu \frac{2 \pi}{d_L^2}
            \int_{R_{\rm in}}^{R_{\rm out}}{\rm d}R \, R \, I_{\nu}(T(R)),

        where :math:`I_{\nu}` is Planck's law, :math:`R` the radial coordinate
        along the disk, and :math:`d_L` the luminosity distance. :math:`\mu_s`
        is the cosine of the angle between the disk axis and the observer's line of sight.

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        M_BH : :class:`~astropy.units.Quantity`
            Black Hole mass
        m_dot : float
            mass accretion rate
        R_in : :class:`~astropy.units.Quantity`
            inner disk radius
        R_out : :class:`~astropy.units.Quantity`
            outer disk radius
        d_L : :class:`~astropy.units.Quantity`
            luminosity of the source
        mu_s : float
            cosine of the angle between the observer line of sight and the disk axis
        """
        # correct for redshift
        nu *= 1 + z
        # array to integrate R
        R = np.linspace(R_in, R_out, 100)
        _R, _nu = axes_reshaper(R, nu)
        _T = SSDisk.evaluate_T(_R, M_BH, m_dot, R_in)
        _I_nu = BlackBody().evaluate(_nu, _T, scale=1)
        integrand = _R / np.power(d_L, 2) * _I_nu * u.sr
        F_nu = 2 * np.pi * np.trapz(integrand, R, axis=0)
        return mu_s * (nu * F_nu).to("erg cm-2 s-1")

    @staticmethod
    def evaluate_multi_T_bb_norm_sed(
        nu, z, L_disk, M_BH, m_dot, R_in, R_out, d_L, mu_s=1
    ):
        """Evaluate a normalised, multi-temperature black body SED as in
        :func:`~targets.SSDisk.evaluate_multi_T_bb_sed`, but the integral luminosity
        is set to be equal to `L_disk`."""
        sed_disk = SSDisk.evaluate_multi_T_bb_sed(
            nu, z, M_BH, m_dot, R_in, R_out, d_L, mu_s
        )
        # renormalise, the factor 2 includes the two sides of the Disk
        L = 2 * (np.trapz(sed_disk / nu, nu) * 4 * np.pi * d_L ** 2).to("erg s-1")
        norm = (L_disk / L).to_value("")
        return norm * sed_disk

    def sed_flux(self, nu, z, mu_s=1):
        r"""evaluate the multi-temperature black body SED for this SS Disk, refer
        to :func:`~targets.SSDisk.evaluate_multi_T_bb_sed` and to
        :func:`~targets.SSDisk.evaluate_multi_T_bb_norm_sed`
        """
        d_L = Distance(z=z).to("cm")
        return SSDisk.evaluate_multi_T_bb_norm_sed(
            nu, z, self.L_disk, self.M_BH, self.m_dot, self.R_in, self.R_out, d_L, mu_s
        )

    def u(self, r, blob=None):
        """integral energy density of radiation produced by the Disk at the distance
        r along the jet axis. Integral over the solid angle of Eq. 69 in [Dermer2009]_.

        Parameters
        ----------
        r : :class:`~astropy.units.Quantity`
            array of distances along the jet axis
        blob : :class:`~agnpy.emission_regions.Blob`
            if provided, the energy density is computed in a reference frame
            comvoing with the blob
        """
        r_tilde = (r / self.R_g).to_value("")
        mu = self.evaluate_mu_from_r_tilde(self.R_in_tilde, self.R_out_tilde, r_tilde)
        prefactor = (
            3
            * self.l_Edd
            * self.L_Edd
            * self.R_g
            / (8 * np.pi * c * self.eta * np.power(r, 3))
        )
        integrand = (
            1
            / mu
            * np.power(np.power(mu, -2) - 1, -3 / 2)
            * self.evaluate_phi_disk_mu(mu, self.R_in_tilde, r_tilde)
        )
        if blob:
            mu_prime = (mu - blob.Beta) / (1 - blob.Beta * mu)
            integrand_prefactor = 1 / (
                np.power(blob.Gamma, 6)
                * np.power(1 - blob.Beta * mu, 2)
                * np.power(1 + blob.Beta * mu_prime, 4)
            )
            integrand *= integrand_prefactor

        integral = np.trapz(integrand, mu, axis=0)
        return (prefactor * integral).to("erg cm-3")


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
        self.name = "SphericalShellBLR"
        self.L_disk = L_disk
        self.xi_line = xi_line
        if line not in lines_dictionary:
            raise NameError(f"{line} not available in the line dictionary")
        self.line = line
        self.lambda_line = lines_dictionary[line]["lambda"]
        self.epsilon_line = (
            self.lambda_line.to("erg", equivalencies=u.spectral()) / mec2
        ).to_value("")
        self.R_line = R_line

    def __str__(self):
        return (
            f"* Spherical Shell Broad Line Region:\n"
            + f" - L_disk (accretion disk luminosity): {self.L_disk.cgs:.2e}\n"
            + f" - xi_line (fraction of the disk radiation reprocessed by the BLR): {self.xi_line:.2e}\n"
            + f" - line (type of emitted line): {self.line}, lambda = {self.lambda_line.cgs:.2e}\n"
            + f" - R_line (radius of the BLR shell): {self.R_line.cgs:.2e}\n"
        )

    def print_lines_list():
        r"""Print the list of the available spectral lines.
        The dictionary with the possible emission lines is taken from Table 5 in
        [Finke2016]_ and contains the value of the line wavelength and the ratio of
        its radius to the radius of the :math:`H_{\beta}` shell, not used at the moment.
        """
        for line in lines_dictionary.keys():
            print(f"{line}: {lines_dictionary[line]}")

    def u(self, r, blob=None):
        """Density of radiation produced by the BLR at the distance r along the
        jet axis. Integral over the solid angle of Eq. 80 in [Finke2016]_.

        Parameters
        ----------
        r : :class:`~astropy.units.Quantity`
            array of distances along the jet axis
        blob : :class:`~agnpy.emission_regions.Blob`
            if provided, the energy density is computed in a reference frame
            comvoing with the blob
        """
        mu = np.linspace(-1, 1)
        _mu = mu.reshape(mu.size, 1)
        _r = r.reshape(1, r.size)
        _x2 = np.power(_r, 2) + np.power(self.R_line, 2) - 2 * _r * self.R_line * _mu
        prefactor = self.xi_line * self.L_disk / (8 * np.pi * c)
        integrand = 1 / _x2
        if blob:
            _mu_star = np.sqrt(
                1 - np.power(self.R_line, 2) / _x2 * (1 - np.power(_mu, 2))
            )
            integrand_prefactor = np.power(blob.Gamma, 2) * np.power(
                1 - blob.Beta * _mu_star, 2
            )
            integrand *= integrand_prefactor
        integral = np.trapz(integrand, mu, axis=0)
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
        self.name = "RingDustTorus"
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
        return (
            f"* Ring Dust Torus:\n"
            + f" - L_disk (accretion disk luminosity): {self.L_disk.cgs:.2e}\n"
            + f" - xi_dt (fraction of the disk radiation reprocessed by the torus): {self.xi_dt:.2e}\n"
            + f" - T_dt (temperature of the dust torus): {self.T_dt:.2e}\n"
            + f" - R_dt (radius of the torus): {self.R_dt.cgs:.2e}\n"
        )

    @staticmethod
    def evaluate_bb_sed(nu, z, T_dt, R_dt, d_L):
        """evaluate the black body SED corresponding to the torus temperature"""
        nu *= 1 + z
        # geometrical factor for a source of size R_dt at distance d_L
        prefactor = np.pi * np.power((R_dt / d_L).to_value(""), 2) * u.sr
        I_nu = BlackBody().evaluate(nu, T_dt, scale=1)
        return (prefactor * nu * I_nu).to("erg cm-2 s-1")

    @staticmethod
    def evaluate_bb_norm_sed(nu, z, L_dt, T_dt, R_dt, d_L):
        """evaluate the torus black-body SED such that its integral luminosity
        is equal to the torus luminosity (`xi_dt * L_disk`)"""
        sed_dt = RingDustTorus.evaluate_bb_sed(nu, z, T_dt, R_dt, d_L)
        # renormalise
        L = (np.trapz(sed_dt / nu, nu) * 4 * np.pi * d_L ** 2).to("erg s-1")
        norm = (L_dt / L).to_value("")
        return norm * sed_dt

    def sed_flux(self, nu, z):
        """evaluate the black-body SED for the Dust Torus"""
        d_L = Distance(z=z).to("cm")
        L_dt = self.xi_dt * self.L_disk
        return RingDustTorus.evaluate_bb_norm_sed(
            nu, z, L_dt, self.T_dt, self.R_dt, d_L
        )

    def u(self, r, blob=None):
        r"""Density of radiation produced by the Torus at the distance r along the
        jet axis. Integral over the solid angle of Eq. 85 in [Finke2016]_

        Parameters
        ----------
        r : :class:`~astropy.units.Quantity`
            array of distances along the jet axis
        blob : :class:`~agnpy.emission_regions.Blob`
            if provided, the energy density is computed in a reference frame
            comvoing with the blob
        """
        x2 = np.power(self.R_dt, 2) + np.power(r, 2)
        x = np.sqrt(x2)
        integral = self.xi_dt * self.L_disk / (4 * np.pi * c * x2)
        if blob:
            mu = (r / x).to_value("")
            integral *= np.power(blob.Gamma * (1 - blob.Beta * mu), 2)
        return integral.to("erg cm-3")
