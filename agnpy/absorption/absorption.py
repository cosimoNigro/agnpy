# module containing the gamma-gamma absorption
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.constants import c, G, m_e, sigma_T
from scipy.interpolate import interp2d
from ..utils.math import (
    axes_reshaper,
    log,
    mu_to_integrate,
    phi_to_integrate,
    min_rel_distance,
)
from ..utils.geometry import (
    cos_psi,
    x_re_shell,
    mu_star_shell,
    x_re_ring,
    x_re_ring_mu_s,
    phi_mu_re_shell,
    phi_mu_re_ring,
    x_re_shell_mu_s,
)
from ..utils.conversion import nu_to_epsilon_prime, to_R_g_units
from ..targets import PointSourceBehindJet, SSDisk, SphericalShellBLR, RingDustTorus
from ..emission_regions import Blob
from ..synchrotron import nu_synch_peak, Synchrotron


__all__ = ["sigma", "Absorption", "ebl_files_dict", "EBL"]

agnpy_dir = Path(__file__).parent.parent
ebl_files_dict = {
    "franceschini": f"{agnpy_dir}/data/ebl_models/ebl_franceschini08.fits.gz",
    "dominguez": f"{agnpy_dir}/data/ebl_models/ebl_dominguez11.fits.gz",
    "finke": f"{agnpy_dir}/data/ebl_models/ebl_finke10.fits.gz",
    "saldana-lopez": f"{agnpy_dir}/data/ebl_models/ebl_saldana-lopez21.fits.gz",
}


def sigma(s):
    """photon-photon pair production cross section, Eq. 17 of [Dermer2009]"""
    beta_cm = np.sqrt(1 - 1 / s)
    prefactor = 3 / 16 * sigma_T * (1 - np.power(beta_cm, 2))
    term1 = (3 - np.power(beta_cm, 4)) * log((1 + beta_cm) / (1 - beta_cm))
    term2 = -2 * beta_cm * (2 - np.power(beta_cm, 2))
    values = prefactor * (term1 + term2)
    values[s < 1] = 0
    return values


class Absorption:
    """class to compute the absorption due to gamma-gamma pair production

    Parameters
    ----------
    blob : :class:`~agnpy.emission_regions.Blob`
        emission region and electron distribution hitting the photon target
    target : :class:`~agnpy.targets` or class:`~agnpy.emission_regions.Blob`
        class describing the target photon field
    r : :class:`~astropy.units.Quantity`
        distance of the blob from the Black Hole (i.e. from the target photons)
        the distance is irrelevant in the case of absorption
    """

    def __init__(self, target, r=None, z=0, mu_s=1):
        self.target = target
        self.r = r
        self.z = z
        self.mu_s = mu_s
        self.set_mu()
        self.set_phi()
        self.set_l()
        # r can be only ignored for absorption on synchrotron radiation
        if r is None and not isinstance(self.target, Blob):
            raise ValueError(
                "No distance provided for absorption on "
                + str(target.__class__)
                + ", this can be only done for Blob class"
            )

    def set_mu(self, mu_size=100):
        self.mu_size = mu_size
        self.mu = np.linspace(-1, 1, self.mu_size)

    def set_phi(self, phi_size=50):
        "Set array of azimuth angles to integrate over"
        self.phi_size = phi_size
        self.phi = np.linspace(0, 2 * np.pi, self.phi_size)

    def set_l(self, l_size=50):
        self.l_size = l_size

    @staticmethod
    def evaluate_tau_ps_behind_blob(nu, z, mu_s, epsilon_0, L_0, r):
        r"""Evaluates the absorption produced by the photon field of a point
        source of photons behind the blob, for a general set of model parameters

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the opacity
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        mu_s : float
            cosine of the angle between the blob motion and the jet axis
        epsilon_0 : float
            dimensionless energy (in electron rest mass energy units) of the
            target photon field
        L_0 : :class:`~astropy.units.Quantity`
            luminosity [erg cm-3] of the point source behind the jet
        r : :class:`~astropy.units.Quantity`
            distance between the point source and the blob

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the tau values corresponding to each frequency
        """
        epsilon_1 = nu_to_epsilon_prime(nu, z)
        s = epsilon_0 * epsilon_1 * (1 - mu_s) / 2
        integral = (1 - mu_s) * sigma(s) / r
        prefactor = L_0 / (4 * np.pi * epsilon_0 * m_e * c ** 3)
        return (prefactor * integral).to_value("")

    def tau_ps_behind_blob(self, nu):
        """Evaluates the absorption produced by the photon field of a point
        source of photons behind the blob"""
        return self.evaluate_tau_ps_behind_blob(
            nu, self.z, self.mu_s, self.target.epsilon_0, self.target.L_0, self.r
        )

    @staticmethod
    def evaluate_tau_ps_behind_blob_mu_s(nu, z, mu_s, epsilon_0, L_0, r, u_size=100):
        r"""Evaluates the absorption produced by the photon field of a point
        source of photons behind the blob, for a general set of model parameters

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the opacity
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        mu_s : float
            cosine of the angle between the blob motion and the jet axis
        epsilon_0 : float
            dimensionless energy (in electron rest mass energy units) of the
            target photon field
        L_0 : :class:`~astropy.units.Quantity`
            luminosity [erg cm-3] of the point source behind the jet
        r : :class:`~astropy.units.Quantity`
            distance between the point source and the blob
        u_size : int
            size of the array of distances from the photon origin to integrate over

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the tau values corresponding to each frequency
        """
        epsilon_1 = nu_to_epsilon_prime(nu, z)

        uu = np.logspace(-5, 5, u_size) * r
        _u, _epsilon_1 = axes_reshaper(uu, epsilon_1)
        # distance between soft photon and gamma ray
        x = np.sqrt(r * r + _u * _u + 2 * _u * r * mu_s)

        # cos angle of the soft photon to the z axis
        _mu = (r + _u * mu_s) / x
        phi = 0  # both gamma ray and soft photon move in XZ plane
        _cos_psi = cos_psi(mu_s, _mu, phi)
        s = _epsilon_1 * epsilon_0 * (1 - _cos_psi) / 2

        integrand = (1 - _cos_psi) / x ** 2 * sigma(s)
        # integrate
        integral = np.trapz(integrand, uu, axis=0)
        prefactor = L_0 / (4 * np.pi * epsilon_0 * m_e * c ** 3)
        return (prefactor * integral).to_value("")

    def tau_ps_behind_blob_mu_s(self, nu):
        """Evaluates the absorption produced by the photon field of a point
        source of photons behind the blob"""
        return self.evaluate_tau_ps_behind_blob_mu_s(
            nu, self.z, self.mu_s, self.target.epsilon_0, self.target.L_0, self.r
        )

    @staticmethod
    def evaluate_tau_ss_disk(
        nu,
        z,
        mu_s,
        M_BH,
        L_disk,
        eta,
        R_in,
        R_out,
        r,
        R_tilde_size=100,
        l_tilde_size=50,
        phi=phi_to_integrate,
    ):
        """Evaluates the gamma-gamma absorption produced by the photon field of
        a Shakura-Sunyaev accretion disk

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the opacity
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        mu_s : float
            cosine of the angle between the blob motion and the jet axis
        M_BH : :class:`~astropy.units.Quantity`
            Black Hole mass
        L_disk : :class:`~astropy.units.Quantity`
            luminosity of the disk
        eta : float
            accretion efficiency
        R_in : :class:`~astropy.units.Quantity`
            inner disk radius
        R_out : :class:`~astropy.units.Quantity`
            inner disk radius
        R_tilde_size : int
            size of the array of disk coordinates to integrate over
        r : :class:`~astropy.units.Quantity`
            distance between the point source and the blob
        l_tilde_size : int
            size of the array of distances from the BH to integrate over
        phi : :class:`~numpy.ndarray`
            array of azimuth angles to integrate over

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the tau values corresponding to each frequency
        """
        # conversions
        R_g = (G * M_BH / c ** 2).to("cm")
        r_tilde = to_R_g_units(r, M_BH)
        R_in_tilde = to_R_g_units(R_in, M_BH)
        R_out_tilde = to_R_g_units(R_out, M_BH)
        # multidimensional integration
        R_tilde = np.linspace(R_in_tilde, R_out_tilde, R_tilde_size)
        l_tilde = np.logspace(0, 5, l_tilde_size) * r_tilde
        epsilon_1 = nu_to_epsilon_prime(nu, z)
        _R_tilde, _phi, _l_tilde, _epsilon_1 = axes_reshaper(
            R_tilde, phi, l_tilde, epsilon_1
        )
        _epsilon = SSDisk.evaluate_epsilon(L_disk, M_BH, eta, _R_tilde)
        _phi_disk = 1 - (R_in_tilde / _R_tilde) ** (1 / 2)
        _mu = (1 + (_R_tilde ** 2 / _l_tilde ** 2)) ** (-1 / 2)
        _cos_psi = cos_psi(mu_s, _mu, _phi)
        s = _epsilon * _epsilon_1 * (1 - _cos_psi) / 2
        integrand = (
            1
            / _l_tilde ** 2
            / _R_tilde ** 2
            / (1 + (_R_tilde ** 2 / _l_tilde ** 2)) ** (3 / 2)
            * _phi_disk
            / _epsilon
            * sigma(s)
            * (1 - _cos_psi)
        )
        integral_R_tilde = np.trapz(integrand, R_tilde, axis=0)
        integral_phi = np.trapz(integral_R_tilde, phi, axis=0)
        integral = np.trapz(integral_phi, l_tilde, axis=0)
        prefactor = 3 * L_disk / ((4 * np.pi) ** 2 * eta * m_e * c ** 3 * R_g)
        return (prefactor * integral).to_value("")

    def tau_ss_disk(self, nu):
        """Evaluates the gamma-gamma absorption produced by the photon field of
        a Shakura-Sunyaev accretion disk"""
        return self.evaluate_tau_ss_disk(
            nu,
            self.z,
            self.mu_s,
            self.target.M_BH,
            self.target.L_disk,
            self.target.eta,
            self.target.R_in,
            self.target.R_out,
            self.r,
            R_tilde_size=100,
            l_tilde_size=self.l_size,
            phi=self.phi,
        )

    @staticmethod
    def evaluate_tau_blr(
        nu,
        z,
        mu_s,
        L_disk,
        xi_line,
        epsilon_line,
        R_line,
        r,
        l_size=50,
        mu=mu_to_integrate,
        phi=phi_to_integrate,
    ):
        """Evaluates the gamma-gamma absorption produced by a spherical shell
        BLR for a general set of model parameters

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the tau
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        mu_s : float
            cosine of the angle between the blob motion and the jet axis
        L_disk : :class:`~astropy.units.Quantity`
            Luminosity of the disk whose radiation is being reprocessed by the BLR
        xi_line : float
            fraction of the disk radiation reprocessed by the BLR
        epsilon_line : string
            dimensionless energy of the emitted line
        R_line : :class:`~astropy.units.Quantity`
            radius of the BLR spherical shell
        r : :class:`~astropy.units.Quantity`
            distance between the Broad Line Region and the blob
        l_size : int
            size of the array of distances from the BH to integrate over
        mu, phi : :class:`~numpy.ndarray`
            arrays of cosine of zenith and azimuth angles to integrate over

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the tau values corresponding to each frequency
        """
        # conversions
        epsilon_1 = nu_to_epsilon_prime(nu, z)
        # multidimensional integration
        l = np.logspace(0, 5, l_size) * r

        # check if any point is too close to R_line, the function works only for mu=1, so
        # we can check directly if R_line is within 'l' array
        idx = np.isclose(l, R_line, rtol=min_rel_distance)
        l[idx] += min_rel_distance * R_line

        _mu, _phi, _l, _epsilon_1 = axes_reshaper(mu, phi, l, epsilon_1)
        x = x_re_shell(_mu, R_line, _l)
        _mu_star = mu_star_shell(_mu, R_line, _l)
        _cos_psi = cos_psi(mu_s, _mu_star, _phi)
        s = _epsilon_1 * epsilon_line * (1 - _cos_psi) / 2
        integrand = (1 - _cos_psi) / x ** 2 * sigma(s)
        # integrate
        integral_mu = np.trapz(integrand, mu, axis=0)
        integral_phi = np.trapz(integral_mu, phi, axis=0)
        integral = np.trapz(integral_phi, l, axis=0)
        prefactor = (L_disk * xi_line) / (
            (4 * np.pi) ** 2 * epsilon_line * m_e * c ** 3
        )
        return (prefactor * integral).to_value("")

    @staticmethod
    def evaluate_tau_blr_mu_s(
        nu,
        z,
        mu_s,
        L_disk,
        xi_line,
        epsilon_line,
        R_line,
        r,
        u_size=100,
        mu=mu_to_integrate,
        phi=phi_to_integrate,
    ):
        """Evaluates the gamma-gamma absorption produced by a spherical shell
        BLR for a general set of model parameters and arbitrary mu_s

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the tau
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        mu_s : float
            cosine of the angle between the blob motion and the jet axis
        L_disk : :class:`~astropy.units.Quantity`
            Luminosity of the disk whose radiation is being reprocessed by the BLR
        xi_line : float
            fraction of the disk radiation reprocessed by the BLR
        epsilon_line : string
            dimensionless energy of the emitted line
        R_line : :class:`~astropy.units.Quantity`
            radius of the BLR spherical shell
        r : :class:`~astropy.units.Quantity`
            distance between the Broad Line Region and the blob
        l_size : int
            size of the array of distances from the BH to integrate over
        mu, phi : :class:`~numpy.ndarray`
            arrays of cosine of zenith and azimuth angles to integrate over

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the tau values corresponding to each frequency
        """
        # conversions
        epsilon_1 = nu_to_epsilon_prime(nu, z)
        # multidimensional integration
        # here uu is the distance that the photon traversed
        uu = np.logspace(-5, 5, u_size) * r

        # check if for any uu value the position of the photon is too close to the BLR
        x_cross = np.sqrt(r ** 2 + uu ** 2 + 2 * uu * r * mu_s)
        idx = np.isclose(x_cross, R_line, rtol=min_rel_distance)
        if idx.any():
            uu[idx] += min_rel_distance * R_line
            # it might happen that some of the points get more shifted then the next one,
            # possibly making integration messy, so we sort the points
            uu = np.sort(uu)

        _mu_re, _phi_re, _u, _epsilon_1 = axes_reshaper(mu, phi, uu, epsilon_1)

        # distance between soft photon and gamma ray
        x = x_re_shell_mu_s(R_line, r, _phi_re, _mu_re, _u, mu_s)

        # convert the phi and mu angles of the position in the sphere into the actual phi and mu angles
        # the actual phi and mu angles of the soft photon catching up with the gamma ray
        _phi, _mu_star = phi_mu_re_shell(R_line, r, _phi_re, _mu_re, _u, mu_s)

        # angle between the soft photon and gamma ray
        _cos_psi = cos_psi(mu_s, _mu_star, _phi)
        s = _epsilon_1 * epsilon_line * (1 - _cos_psi) / 2
        integrand = (1 - _cos_psi) / x ** 2 * sigma(s)
        # integrate
        integral_mu = np.trapz(integrand, mu, axis=0)
        integral_phi = np.trapz(integral_mu, phi, axis=0)
        integral = np.trapz(integral_phi, uu, axis=0)
        prefactor = (L_disk * xi_line) / (
            (4 * np.pi) ** 2 * epsilon_line * m_e * c ** 3
        )
        return (prefactor * integral).to_value("")

    def tau_blr(self, nu):
        """Evaluates the gamma-gamma absorption produced by a spherical shell
        BLR for a general set of model parameters
        """
        return self.evaluate_tau_blr(
            nu,
            self.z,
            self.mu_s,
            self.target.L_disk,
            self.target.xi_line,
            self.target.epsilon_line,
            self.target.R_line,
            self.r,
            l_size=self.l_size,
            mu=self.mu,
            phi=self.phi,
        )

    def tau_blr_mu_s(self, nu):
        """Evaluates the gamma-gamma absorption produced by a spherical shell
        BLR for a general set of model parameters and arbitrary mu_s
        """
        return self.evaluate_tau_blr_mu_s(
            nu,
            self.z,
            self.mu_s,
            self.target.L_disk,
            self.target.xi_line,
            self.target.epsilon_line,
            self.target.R_line,
            self.r,
            u_size=2 * self.l_size,
            mu=self.mu,
            phi=self.phi,
        )

    @staticmethod
    def evaluate_tau_dt(
        nu,
        z,
        mu_s,
        L_disk,
        xi_dt,
        epsilon_dt,
        R_dt,
        r,
        l_size=50,
        phi=phi_to_integrate,
    ):
        r"""Evaluates the gamma-gamma absorption produced by a ring dust torus

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        mu_s : float
            cosine of the angle between the blob motion and the jet axis
        L_disk : :class:`~astropy.units.Quantity`
            Luminosity of the disk whose radiation is being reprocessed by the BLR
        xi_dt : float
            fraction of the disk radiation reprocessed by the disk
        epsilon_dt : string
            peak (dimensionless) energy of the black body radiated by the torus
        R_dt : :class:`~astropy.units.Quantity`
            radius of the ting-like torus
        r : :class:`~astropy.units.Quantity`
            distance between the dust torus and the blob
        l_size : int
            size of the array of distances from the BH to integrate over
        phi : :class:`~numpy.ndarray`
            arrays of azimuth angles to integrate over

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        # conversions
        epsilon_1 = nu_to_epsilon_prime(nu, z)
        # multidimensional integration
        l = np.logspace(0, 5, l_size) * r
        _phi, _l, _epsilon_1 = axes_reshaper(phi, l, epsilon_1)
        x = x_re_ring(R_dt, _l)
        _mu = _l / x
        _cos_psi = cos_psi(mu_s, _mu, _phi)
        s = _epsilon_1 * epsilon_dt * (1 - _cos_psi) / 2
        integrand = (1 - _cos_psi) / x ** 2 * sigma(s)
        # integrate
        integral_phi = np.trapz(integrand, phi, axis=0)
        integral = np.trapz(integral_phi, l, axis=0)
        prefactor = (L_disk * xi_dt) / (8 * np.pi ** 2 * epsilon_dt * m_e * c ** 3)
        return (prefactor * integral).to_value("")

    @staticmethod
    def evaluate_tau_dt_mu_s(
        nu,
        z,
        mu_s,
        L_disk,
        xi_dt,
        epsilon_dt,
        R_dt,
        r,
        u_size=100,
        phi_re=phi_to_integrate,
    ):
        r"""Evaluates the gamma-gamma absorption produced by a ring dust torus
        for the case of photon moving at an angle to the jet

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        mu_s : float
            cosine of the angle between the blob motion and the jet axis
        L_disk : :class:`~astropy.units.Quantity`
            Luminosity of the disk whose radiation is being reprocessed by the BLR
        xi_dt : float
            fraction of the disk radiation reprocessed by the disk
        epsilon_dt : string
            peak (dimensionless) energy of the black body radiated by the torus
        R_dt : :class:`~astropy.units.Quantity`
            radius of the ting-like torus
        r : :class:`~astropy.units.Quantity`
            distance between the dust torus and the blob
        u_size : int
            size of the array of distances from the photon origin to integrate over
        phi_re : :class:`~numpy.ndarray`
            arrays of azimuth angles of the dust torus to integrate over

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        # conversions
        epsilon_1 = nu_to_epsilon_prime(nu, z)
        # multidimensional integration
        # here uu is the distance that the photon traversed
        uu = np.logspace(-5, 5, u_size) * r
        _phi_re, _u, _epsilon_1 = axes_reshaper(phi_re, uu, epsilon_1)
        # distance between soft photon and gamma ray
        x = x_re_ring_mu_s(R_dt, r, _phi_re, _u, mu_s)
        # convert the phi angles of the ring into the actual phi angles
        # of the soft photon catching up with the gamma ray
        _phi, _mu = phi_mu_re_ring(R_dt, r, _phi_re, _u, mu_s)
        _cos_psi = cos_psi(mu_s, _mu, _phi)
        s = _epsilon_1 * epsilon_dt * (1 - _cos_psi) / 2
        integrand = (1 - _cos_psi) / x ** 2 * sigma(s)
        # integrate
        integral_phi = np.trapz(integrand, phi_re, axis=0)
        integral = np.trapz(integral_phi, uu, axis=0)
        prefactor = (L_disk * xi_dt) / (8 * np.pi ** 2 * epsilon_dt * m_e * c ** 3)
        return (prefactor * integral).to_value("")

    def tau_dt(self, nu):
        """evaluates the gamma-gamma absorption produced by a ring dust torus"""
        return self.evaluate_tau_dt(
            nu,
            self.z,
            self.mu_s,
            self.target.L_disk,
            self.target.xi_dt,
            self.target.epsilon_dt,
            self.target.R_dt,
            self.r,
            l_size=self.l_size,
            phi=self.phi,
        )

    def tau_dt_mu_s(self, nu):
        """evaluates the gamma-gamma absorption produced by a ring dust torus"""
        return self.evaluate_tau_dt_mu_s(
            nu,
            self.z,
            self.mu_s,
            self.target.L_disk,
            self.target.xi_dt,
            self.target.epsilon_dt,
            self.target.R_dt,
            self.r,
            u_size=2 * self.l_size,
            phi_re=self.phi,
        )

    def tau_on_synchrotron(self, blob, nu, nu_s_size=200, delta_margin_low=1.0e-2):
        r"""Optical depth for absorption of gamma rays in synchrotron radiation of the blob.
        It assumes the same radiation field as the SSC class.

        Parameters
        ----------
        blob : :class:`~agnpy.emission_regions.Blob`
            emission region and electron distribution hitting the photon target
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the opacity
            **note** these are observed frequencies (observer frame)
        nu_s_size : int
            size of the array over the synchrotron frequencies
        delta_margin_low : float
            extension of the integration range of the synchrotron radiation beyond
            the delta approximation, default = 0.01, but lower value might be needed
            if the calculations are performed up to very high energies
        """
        # energy of the gamma rays in blob frame
        epsilon1 = nu_to_epsilon_prime(nu, blob.z, blob.delta_D)

        # first derive the ranges of the synchrotron spectrum using delta approximation
        # add margin on both sides to allow for the energy distribution
        nu_s_min = nu_synch_peak(blob.B, blob.gamma_min) * delta_margin_low
        nu_s_max = nu_synch_peak(blob.B, blob.gamma_max) * 1.0e2

        # frequencies in the blob frame
        nu_s = (
            np.logspace(
                np.log10(nu_s_min.to_value("Hz")),
                np.log10(nu_s_max.to_value("Hz")),
                nu_s_size,
            )
            * u.Hz
        )

        # and in observers frame
        nu_s_obs = nu_s * blob.delta_D / (1 + blob.z)
        # energy of the synchrotron photons in blob frame
        epsilon = nu_to_epsilon_prime(nu_s_obs, blob.z, blob.delta_D)

        synch = Synchrotron(blob, ssa=True)
        sed_synch = synch.sed_flux(nu_s_obs)

        # Eq. 8 [Finke2008]_ divided by extra epsilon mc^2
        n_synch = (
            (3 * np.power(blob.d_L, 2) * sed_synch)
            / (
                c
                * np.power(blob.R_b, 2)
                * np.power(blob.delta_D, 4)
                * epsilon ** 2
                * m_e
                * c ** 2
            )
        ).to("cm-3")

        # factor 3 / 4 accounts for averaging in a sphere
        # not included in Dermer and Finke's papers
        n_synch *= 3 / 4

        _epsilon, _epsilon1 = axes_reshaper(epsilon, epsilon1)
        _s = _epsilon * _epsilon1 / 2
        _n_synch = n_synch[..., np.newaxis]

        return (2 * blob.R_b * np.trapz(_n_synch * sigma(_s), epsilon, axis=0)).to("")

    def tau(self, nu):
        """optical depth

        .. math::
            \\tau_{\\gamma \\gamma}(\\nu)

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the opacity, **note** these are
            observed frequencies (observer frame).
        """

        if isinstance(self.target, Blob):
            return self.tau_on_synchrotron(self.target, nu)

        if self.mu_s == 1:  # default value
            if isinstance(self.target, PointSourceBehindJet):
                return self.tau_ps_behind_blob(nu)  # this is always 0 for mu_s=1!
            if isinstance(self.target, SSDisk):
                return self.tau_ss_disk(nu)
            if isinstance(self.target, SphericalShellBLR):
                return self.tau_blr(nu)
            if isinstance(self.target, RingDustTorus):
                return self.tau_dt(nu)
        else:
            if isinstance(self.target, PointSourceBehindJet):
                return self.tau_ps_behind_blob_mu_s(nu)
            # those are yet to be implemented
            # if isinstance(self.target, SSDisk):
            #    return self.tau_ss_disk(nu)
            if isinstance(self.target, SphericalShellBLR):
                return self.tau_blr_mu_s(nu)
            if isinstance(self.target, RingDustTorus):
                return self.tau_dt_mu_s(nu)

    def absorption(self, nu):
        """This function returns the attenuation of the emission assuming that
        the optical depth tau is computed from the production place to the observer.
        """
        return np.exp(-self.tau(nu))

    def absorption_homogeneous(self, nu):
        """This function returns the attenuation of the emission assuming that
        the emission is produced homogenously inside absorbing material.
        The calculations is only accurate for a slab of absorbing material with the
        total optical depth tau, but the same formula is often used also e.g.
        in the context of absorption of gamma-ray emission by synchrotron radiation in blobs
        See e.g. section 2.5.1. of Finke et al. 2008.
        """
        t = self.tau(nu)
        return (1 - np.exp(-t)) / t


class EBL:
    """Class representing for the Extragalactic Background Light absorption.
    Tabulated values of absorption as a function of redshift and energy according
    to the models of [Franceschini2008]_, [Finke2010]_, [Dominguez2011]_, [Saldana-Lopez2021]_ are available
    in `data/ebl_models`.
    They are interpolated by `agnpy` and can be later evaluated for a given redshift
    and range of frequencies.

    Parameters
    ----------
    model : ["franceschini", "dominguez", "finke", "saldana-lopez"]
        choose the reference for the EBL model
    """

    def __init__(self, model="franceschini"):
        if model not in ["franceschini", "dominguez", "finke", "saldana-lopez"]:
            raise ValueError("No EBL model for the reference you specified")
        self.model_file = ebl_files_dict[model]
        # load the absorption table
        self.load_absorption_table()
        self.interpolate_absorption_table()

    def load_absorption_table(self):
        """load the reference values from the table file to be interpolated later"""
        f = fits.open(self.model_file)
        self.energy_ref = (
            np.sqrt(f["ENERGIES"].data["ENERG_LO"] * f["ENERGIES"].data["ENERG_HI"])
            * u.eV
        )
        # Franceschini file has two columns repeated, eliminate them
        self.z_ref = np.unique(f["SPECTRA"].data["PARAMVAL"])
        self.values_ref = np.unique(f["SPECTRA"].data["INTPSPEC"], axis=0)

    def interpolate_absorption_table(self, kind="linear"):
        """interpolate the reference values, choose the kind of interpolation"""
        log10_energy_ref = np.log10(self.energy_ref.to_value("eV"))
        self.interpolated_model = interp2d(
            log10_energy_ref, self.z_ref, self.values_ref, kind=kind
        )

    def absorption(self, z, nu):
        "This function returns the attenuation of the emission by EBL"
        energy = nu.to_value("eV", equivalencies=u.spectral())
        log10_energy = np.log10(energy)
        return self.interpolated_model(log10_energy, z)
