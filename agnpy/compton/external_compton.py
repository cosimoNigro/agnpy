# module containing the External Compton radiative process
import numpy as np
from astropy.constants import c, sigma_T, G
from ..utils.math import (
    axes_reshaper,
    gamma_to_integrate,
    mu_to_integrate,
    phi_to_integrate,
)
from ..utils.conversion import nu_to_epsilon_prime, to_R_g_units
from ..utils.geometry import x_re_shell, mu_star_shell, x_re_ring
from ..targets import (
    CMB,
    PointSourceBehindJet,
    SSDisk,
    SphericalShellBLR,
    RingDustTorus,
)
from .kernels import compton_kernel

__all__ = ["ExternalCompton"]


class ExternalCompton:
    """class for External Compton radiation computation

    Parameters
    ----------
    blob : :class:`~agnpy.emission_regions.Blob`
        emission region and electron distribution hitting the photon target
    target : :class:`~agnpy.targets`
        class describing the target photon field    
    r : :class:`~astropy.units.Quantity`
        distance of the blob from the Black Hole (i.e. from the target photons)
    integrator : func
        function to be used for integration (default = `np.trapz`)
    """

    def __init__(self, blob, target, r=None, integrator=np.trapz):
        self.blob = blob
        # we integrate on a larger grid to account for the transformation
        # of the electron density in the reference frame of the BH
        self.gamma = self.blob.gamma_to_integrate
        self.target = target
        self.r = r
        self.integrator = integrator
        self.set_mu()
        self.set_phi()

    def set_mu(self, mu_size=100):
        self.mu_size = mu_size
        if isinstance(self.target, SSDisk):
            # in case of hte disk the mu interval does not go from -1 to 1
            r_tilde = (self.r / self.target.R_g).to_value("")
            self.mu = self.target.evaluate_mu_from_r_tilde(
                self.target.R_in_tilde, self.target.R_out_tilde, r_tilde
            )
        else:
            self.mu = np.linspace(-1, 1, self.mu_size)

    def set_phi(self, phi_size=50):
        self.phi_size = phi_size
        self.phi = np.linspace(0, 2 * np.pi, self.phi_size)

    @staticmethod
    def evaluate_sed_flux_iso_mono(
        nu,
        z,
        d_L,
        delta_D,
        mu_s,
        R_b,
        epsilon_0,
        u_0,
        n_e,
        *args,
        integrator=np.trapz,
        gamma=gamma_to_integrate,
        mu=mu_to_integrate,
        phi=phi_to_integrate
    ):
        r"""Evaluates the flux SED (:math:`\nu F_{\nu}`) for external Compton 
        on a monochromatic isotropic target photon field, for a general set of model parameters

        **Note** parameters after \*args need to be passed with a keyword

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed 
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        d_L : :class:`~astropy.units.Quantity` 
            luminosity distance of the source
        delta_D : float
            Doppler factor of the relativistic outflow
        mu_s : float
            cosine of the angle between the blob motion and the jet axis
        R_b : :class:`~astropy.units.Quantity`
            size of the emitting region (spherical blob assumed)
        epsilon_0 : float
            dimensionless energy (in electron rest mass energy units) of the
            target photon field
        u_0 : :class:`~astropy.units.Quantity`
            energy density [erg cm-3] of the target photon field
        n_e : :class:`~agnpy.spectra.ElectronDistribution`
            electron energy distribution
        \*args
            parameters of the electron energy distribution (k_e, p, ...)
        integrator : func
            which function to use for integration, default `numpy.trapz`
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to integrate the electron 
            distribution
        mu, phi : :class:`~numpy.ndarray`
            arrays of cosine of zenith and azimuth angles to integrate over

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        # conversion
        epsilon_s = nu_to_epsilon_prime(nu, z)
        # multi-dimensional integration
        _gamma, _mu, _phi, _epsilon_s = axes_reshaper(gamma, mu, phi, epsilon_s)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_e.evaluate(_gamma / delta_D, *args)
        kernel = compton_kernel(_gamma, _epsilon_s, epsilon_0, mu_s, _mu, _phi)
        integrand = N_e / np.power(_gamma, 2) * kernel
        integral_gamma = integrator(integrand, gamma, axis=0)
        integral_mu = np.trapz(integral_gamma, mu, axis=0)
        integral_phi = np.trapz(integral_mu, phi, axis=0)
        prefactor_num = (
            3 * c * sigma_T * u_0 * np.power(epsilon_s, 2) * np.power(delta_D, 3)
        )
        prefactor_denom = (
            np.power(2, 7)
            * np.power(np.pi, 2)
            * np.power(d_L, 2)
            * np.power(epsilon_0, 2)
        )
        return (prefactor_num / prefactor_denom * integral_phi).to("erg cm-2 s-1")

    def sed_flux_cmb(self, nu):
        """evaluates the flux SED for External Compton on the CMB"""
        return self.evaluate_sed_flux_iso_mono(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.mu_s,
            self.blob.R_b,
            self.target.epsilon_0,
            self.target.u_0,
            self.blob.n_e,
            *self.blob.n_e.parameters,
            integrator=self.integrator,
            gamma=self.gamma,
            mu=self.mu,
            phi=self.phi
        )

    @staticmethod
    def evaluate_sed_flux_ps_behind_jet(
        nu,
        z,
        d_L,
        delta_D,
        mu_s,
        R_b,
        epsilon_0,
        L_0,
        r,
        n_e,
        *args,
        integrator=np.trapz,
        gamma=gamma_to_integrate
    ):
        r"""Evaluates the flux SED (:math:`\nu F_{\nu}`) for external Compton 
        on a point source of photons behind the jet, for a general set of model parameters

        **Note** parameters after \*args need to be passed with a keyword

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed 
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        d_L : :class:`~astropy.units.Quantity` 
            luminosity distance of the source
        delta_D : float
            Doppler factor of the relativistic outflow
        mu_s : float
            cosine of the angle between the blob motion and the jet axis
        R_b : :class:`~astropy.units.Quantity`
            size of the emitting region (spherical blob assumed)
        epsilon_0 : float
            dimensionless energy (in electron rest mass energy units) of the
            target photon field
        L_0 : :class:`~astropy.units.Quantity`
            luminosity [erg cm-3] of the point source behind the jet
        r : :class:`~astropy.units.Quantity`
            distance between the point source and the blob
        n_e : :class:`~agnpy.spectra.ElectronDistribution`
            electron energy distribution
        \*args
            parameters of the electron energy distribution (k_e, p, ...)
        integrator : func
            which function to use for integration, default `numpy.trapz`
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to integrate the electron 
            distribution

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        # conversion
        epsilon_s = nu_to_epsilon_prime(nu, z)
        # multi-dimensional integration
        _gamma, _epsilon_s = axes_reshaper(gamma, epsilon_s)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_e.evaluate(_gamma / delta_D, *args)
        kernel = compton_kernel(_gamma, _epsilon_s, epsilon_0, mu_s, 1, 0)
        integrand = N_e / np.power(_gamma, 2) * kernel
        integral = integrator(integrand, gamma, axis=0)
        prefactor_num = (
            3 * sigma_T * L_0 * np.power(epsilon_s, 2) * np.power(delta_D, 3)
        )
        prefactor_denom = (
            np.power(2, 7)
            * np.power(np.pi, 2)
            * np.power(d_L, 2)
            * np.power(r, 2)
            * np.power(epsilon_0, 2)
        )
        return (prefactor_num / prefactor_denom * integral).to("erg cm-2 s-1")

    def sed_flux_ps_behind_jet(self, nu):
        """evaluates the flux SED for External Compton on a point source behind 
        the jet"""
        return self.evaluate_sed_flux_ps_behind_jet(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.mu_s,
            self.blob.R_b,
            self.target.epsilon_0,
            self.target.L_0,
            self.r,
            self.blob.n_e,
            *self.blob.n_e.parameters,
            integrator=self.integrator,
            gamma=self.gamma
        )

    @staticmethod
    def evaluate_sed_flux_ss_disk(
        nu,
        z,
        d_L,
        delta_D,
        mu_s,
        R_b,
        M_BH,
        L_disk,
        eta,
        R_in,
        R_out,
        r,
        n_e,
        *args,
        integrator=np.trapz,
        gamma=gamma_to_integrate,
        mu_size=100,
        phi=phi_to_integrate
    ):
        r"""Evaluates the flux SED (:math:`\nu F_{\nu}`) for external Compton 
        on the photon field of a Shakura Sunyaev disk, for a general set of model parameters

        **Note** parameters after \*args need to be passed with a keyword

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed 
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        d_L : :class:`~astropy.units.Quantity` 
            luminosity distance of the source
        delta_D: float
            Doppler factor of the relativistic outflow
        mu_s : float
            cosine of the angle between the blob motion and the jet axis
        R_b : :class:`~astropy.units.Quantity`
            size of the emitting region (spherical blob assumed)
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
        r : :class:`~astropy.units.Quantity`
            distance between the disk and the blob
        n_e : :class:`~agnpy.spectra.ElectronDistribution`
            electron energy distribution
        \*args
            parameters of the electron energy distribution (k_e, p, ...)
        integrator : func
            which function to use for integration, default `numpy.trapz`
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to integrate the electron 
            distribution
        mu_size : int
            size of the array of zenith angles to integrate over 
        phi : :class:`~numpy.ndarray`
            arrays of azimuth angles to integrate over

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        # conversions
        epsilon_s = nu_to_epsilon_prime(nu, z)
        r_tilde = to_R_g_units(r, M_BH)
        R_in_tilde = to_R_g_units(R_in, M_BH)
        R_out_tilde = to_R_g_units(R_out, M_BH)
        m_dot = (L_disk / (eta * np.power(c, 2))).to("g / s")
        # multidimensional integration
        # for the disk we do not integrate mu from -1 to 1 but choose the range
        # of zenith angles subtended from a given distance
        mu = SSDisk.evaluate_mu_from_r_tilde(R_in_tilde, R_out_tilde, r_tilde, mu_size)
        _gamma, _mu, _phi, _epsilon_s = axes_reshaper(gamma, mu, phi, epsilon_s)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_e.evaluate(_gamma / delta_D, *args)
        epsilon = SSDisk.evaluate_epsilon_mu(L_disk, M_BH, eta, _mu, r_tilde)
        phi_disk = SSDisk.evaluate_phi_disk_mu(_mu, R_in_tilde, r_tilde)
        kernel = compton_kernel(_gamma, _epsilon_s, epsilon, mu_s, _mu, _phi)
        integrand = (
            phi_disk
            / np.power(epsilon, 2)
            / _mu
            / np.power(np.power(_mu, -2) - 1, 3 / 2)
            * N_e
            / np.power(_gamma, 2)
            * kernel
        )
        integral_gamma = integrator(integrand, gamma, axis=0)
        integral_mu = np.trapz(integral_gamma, mu, axis=0)
        integral_phi = np.trapz(integral_mu, phi, axis=0)
        prefactor_num = (
            9
            * sigma_T
            * G
            * M_BH
            * m_dot
            * np.power(epsilon_s, 2)
            * np.power(delta_D, 3)
        )
        prefactor_denom = (
            np.power(2, 9) * np.power(np.pi, 3) * np.power(d_L, 2) * np.power(r, 3)
        )
        return (prefactor_num / prefactor_denom * integral_phi).to("erg cm-2 s-1")

    def sed_flux_ss_disk(self, nu):
        """evaluates the flux SED for External Compton on a [Shakura1973]_ disk"""
        return self.evaluate_sed_flux_ss_disk(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.mu_s,
            self.blob.R_b,
            self.target.M_BH,
            self.target.L_disk,
            self.target.eta,
            self.target.R_in,
            self.target.R_out,
            self.r,
            self.blob.n_e,
            *self.blob.n_e.parameters,
            integrator=self.integrator,
            gamma=self.gamma,
            mu_size=self.mu_size,
            phi=self.phi
        )

    @staticmethod
    def evaluate_sed_flux_blr(
        nu,
        z,
        d_L,
        delta_D,
        mu_s,
        R_b,
        L_disk,
        xi_line,
        epsilon_line,
        R_line,
        r,
        n_e,
        *args,
        integrator=np.trapz,
        gamma=gamma_to_integrate,
        mu=mu_to_integrate,
        phi=phi_to_integrate
    ):
        r"""Evaluates the flux SED (:math:`\nu F_{\nu}`) for External Compton on 
        the photon field of a spherical shell BLR, for a general set of model parameters

        **Note** parameters after \*args need to be passed with a keyword

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed 
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        d_L : :class:`~astropy.units.Quantity` 
            luminosity distance of the source
        delta_D: float
            Doppler factor of the relativistic outflow
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
        n_e : :class:`~agnpy.spectra.ElectronDistribution`
            electron energy distribution
        \*args
            parameters of the electron energy distribution (k_e, p, ...)
        integrator : func
            which function to use for integration, default `numpy.trapz`
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to integrate the electron 
            distribution
        mu, phi : :class:`~numpy.ndarray`
            arrays of cosine of zenith and azimuth angles to integrate over

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        # conversions
        epsilon_s = nu_to_epsilon_prime(nu, z)
        # multidimensional integration
        _gamma, _mu, _phi, _epsilon_s = axes_reshaper(gamma, mu, phi, epsilon_s)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_e.evaluate(_gamma / delta_D, *args)
        x = x_re_shell(_mu, R_line, r)
        mu_star = mu_star_shell(_mu, R_line, r)
        kernel = compton_kernel(_gamma, _epsilon_s, epsilon_line, mu_s, mu_star, _phi)
        integrand = 1 / np.power(x, 2) * N_e / np.power(_gamma, 2) * kernel
        integral_gamma = integrator(integrand, gamma, axis=0)
        integral_mu = np.trapz(integral_gamma, mu, axis=0)
        integral_phi = np.trapz(integral_mu, phi, axis=0)
        prefactor_num = (
            3
            * sigma_T
            * xi_line
            * L_disk
            * np.power(epsilon_s, 2)
            * np.power(delta_D, 3)
        )
        prefactor_denom = (
            np.power(2, 9)
            * np.power(np.pi, 3)
            * np.power(d_L, 2)
            * np.power(epsilon_line, 2)
        )
        return (prefactor_num / prefactor_denom * integral_phi).to("erg cm-2 s-1")

    def sed_flux_blr(self, nu):
        """evaluates the flux SED for External Compton on a spherical BLR"""
        return self.evaluate_sed_flux_blr(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.mu_s,
            self.blob.R_b,
            self.target.L_disk,
            self.target.xi_line,
            self.target.epsilon_line,
            self.target.R_line,
            self.r,
            self.blob.n_e,
            *self.blob.n_e.parameters,
            integrator=self.integrator,
            gamma=self.gamma,
            mu=self.mu,
            phi=self.phi
        )

    @staticmethod
    def evaluate_sed_flux_dt(
        nu,
        z,
        d_L,
        delta_D,
        mu_s,
        R_b,
        L_disk,
        xi_dt,
        epsilon_dt,
        R_dt,
        r,
        n_e,
        *args,
        integrator=np.trapz,
        gamma=gamma_to_integrate,
        phi=phi_to_integrate
    ):
        r"""Evaluates the flux SED (:math:`\nu F_{\nu}`) for External Compton on 
        the photon field of a ring dust torus, for a general set of model parameters

        **Note** parameters after \*args need to be passed with a keyword

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed 
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        d_L : :class:`~astropy.units.Quantity` 
            luminosity distance of the source
        delta_D: float
            Doppler factor of the relativistic outflow
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
            distance between the Broad Line Region and the blob
        n_e : :class:`~agnpy.spectra.ElectronDistribution`
            electron energy distribution
        \*args
            parameters of the electron energy distribution (k_e, p, ...)
        integrator : func
            which function to use for integration, default `numpy.trapz`
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to integrate the electron 
            distribution
        mu, phi : :class:`~numpy.ndarray`
            arrays of cosine of zenith and azimuth angles to integrate over

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        # conversions
        epsilon_s = nu_to_epsilon_prime(nu, z)
        # multidimensional integration
        _gamma, _phi, _epsilon_s = axes_reshaper(gamma, phi, epsilon_s)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_e.evaluate(_gamma / delta_D, *args)
        x_re = x_re_ring(R_dt, r)
        mu = (r / x_re).to_value("")
        kernel = compton_kernel(_gamma, _epsilon_s, epsilon_dt, mu_s, mu, _phi)
        integrand = N_e / np.power(_gamma, 2) * kernel
        integral_gamma = integrator(integrand, gamma, axis=0)
        integral_phi = np.trapz(integral_gamma, phi, axis=0)
        prefactor_num = (
            3 * sigma_T * xi_dt * L_disk * np.power(epsilon_s, 2) * np.power(delta_D, 3)
        )
        prefactor_denom = (
            np.power(2, 8)
            * np.power(np.pi, 3)
            * np.power(d_L, 2)
            * np.power(x_re, 2)
            * np.power(epsilon_dt, 2)
        )
        return (prefactor_num / prefactor_denom * integral_phi).to("erg cm-2 s-1")

    def sed_flux_dt(self, nu):
        """evaluates the flux SED for External Compton on a ring dust torus"""
        return self.evaluate_sed_flux_dt(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.mu_s,
            self.blob.R_b,
            self.target.L_disk,
            self.target.xi_dt,
            self.target.epsilon_dt,
            self.target.R_dt,
            self.r,
            self.blob.n_e,
            *self.blob.n_e.parameters,
            integrator=self.integrator,
            gamma=self.gamma,
            phi=self.phi
        )

    def sed_flux(self, nu):
        """SEDs for external Compton"""
        if isinstance(self.target, CMB):
            return self.sed_flux_cmb(nu)
        if isinstance(self.target, PointSourceBehindJet):
            return self.sed_flux_ps_behind_jet(nu)
        if isinstance(self.target, SSDisk):
            return self.sed_flux_ss_disk(nu)
        if isinstance(self.target, SphericalShellBLR):
            return self.sed_flux_blr(nu)
        if isinstance(self.target, RingDustTorus):
            return self.sed_flux_dt(nu)

    def sed_luminosity(self, nu):
        r"""Evaluates the external Compton luminosity SED
        :math:`\nu L_{\nu} \, [\mathrm{erg}\,\mathrm{s}^{-1}]`"""
        sphere = 4 * np.pi * np.power(self.blob.d_L, 2)
        return (sphere * self.sed_flux(nu)).to("erg s-1")
