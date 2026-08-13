# class for photomeson production
import astropy.units as u
import numpy as np
from astropy.constants import h

from ..utils.conversion import mec2, mpc2
from ..utils.math import axes_reshaper, log10
from .kernels import PhiKernel, eta_0, gKernel, secondaries


class PhotoMesonProduction:
    """Class for computation of the energetic spectra of secondaries of photomeson interactions.

    Parameters
    ----------
    blob : :class:`~agnpy.emission_region.Blob`
        emitting region with a proton distribution
    target : ...TBD...
        ...TBD...
    integrator : func
        function to be used for integration (default = `np.trapz`)
    """

    def __init__(self, blob, target, integrator=np.trapz):
        self.blob = blob
        # check that this blob has a proton distribution
        if self.blob._n_p is None:
            raise AttributeError(
                "There is no proton distribution in this emission region"
            )
        self.target = target
        self.integrator = integrator
        return

    def H(self, eta, E, phi_kernel, integrator=np.trapz):
        r"""Compute the H function in Eq. (70) [KelnerAharonian2008]_.

        Parameters
        ----------
        E : :class:`~astropy.units.Quantity`
            energy of the secondary particles
        eta : float
            kinematic variable (:math:`eta = 4 \epsilon \gamma_{\rm p}`)
        phi_kernel : `~agnpy.photo_meson.PhiKernel`
            kernel to be used for the integration (depends on the particle)
        integrator : func
            function to be used for integration (default = `np.trapz`)
        """
        # Integral on E_p to be made from E to infinity
        _eta, _E = axes_reshaper(eta, E)  # shape (len(eta), 1), (1, len(E))

        Emin = self.blob.n_p.gamma_min * mpc2
        Emax = self.blob.n_p.gamma_max * mpc2

        _E_min = np.full_like(_E, Emin)
        _E_max = np.full_like(_E, Emax)

        _E_p = np.logspace(
            log10(_E_min.to_value("eV")), log10(_E_max.to_value("eV")), 200
        ) * u.Unit(
            "eV"
        )  # shape (200, 1, len(E))

        _gamma_p = _E_p / mpc2
        _epsilon = _eta * mpc2**2 / (4 * _E_p)
        _nu = _epsilon / h
        _x = _E / _E_p
        _H_integrand = (
            mpc2**2
            / 4  # erg2
            * ((mpc2**-1) * self.blob.n_p(_gamma_p) / _E_p**2)  # cm-3 erg-3
            * self.target(_nu)  # cm-3 erg-1
            * phi_kernel(_eta, _x)  # cm3 s-1
        ).to("erg-2 cm-3 s-1")

        _H = integrator(
            _H_integrand,
            _E_p,
            axis=0,
        ).to("erg-1 cm-3 s-1")
        return _H

    def evaluate_spectrum(self, E, particle, eta_log_range=5, integrator=np.trapz):
        """Evaluate the spectrum of secondaries in the emission region reference frame
        as in Eq. (69) [KelnerAharonian2008]_.

        Parameters
        ----------
        E : float
            energy of the secondary particles
        particle: str
            name of the secondary particle, to be chosen among
            "gamma", "electron", "positron", "electron_neutrino",
            "electron_antineutrino", "muon_neutrino", "muon_antineutrino"
        integrator : func
            function to be used for integration (default = `np.trapz`)
        """
        if particle not in secondaries:
            raise AttributeError(
                f"There is no secondary particle from photomeson interactions named {particle}."
            )

        phi_kernel = PhiKernel(particle)
        # Integral on eta to be done from eta_0 to infinity
        eta = np.logspace(
            log10(eta_0),
            log10(eta_0) + eta_log_range,
            100,
        )
        _H = self.H(
            eta,
            E,
            phi_kernel,
            integrator=integrator,
        )
        dN_dEdVdt = integrator(_H, eta, axis=0).to("erg-1 cm-3 s-1")
        return dN_dEdVdt


class PhotoMesonProductionAngular:
    r"""
    Parameters
    ----------
    theta_s : class:`~astropy.units.Quantity`
              observer viewing angle

    r       : class:`~astropy.units.Quantity`
              distance along the jet axis

    """

    def __init__(self, blob, target, integrator=np.trapz):
        self.blob = blob
        # check that this blob has a proton distribution
        if self.blob._n_p is None:
            raise AttributeError(
                "There is no proton distribution in this emission region"
            )
        self.delta_D = blob.delta_D
        self.theta_s = blob.theta_s
        self.target = target
        self.integrator = integrator

        return

    def H(self, phi, E, r, g_kernel, integrator=np.trapz):

        # Integral on E_p to be made from E to infinity
        _phi, _E = axes_reshaper(phi, E)  # shape (len(phi), 1), (1, len(E))

        Emin = self.blob.n_p.gamma_min * mpc2
        Emax = self.blob.n_p.gamma_max * mpc2

        _E_min = np.full_like(_E, Emin)
        _E_max = np.full_like(_E, Emax)

        _E_p = np.logspace(
            log10(_E_min.to_value("eV")), log10(_E_max.to_value("eV")), 200
        ) * u.Unit(
            "eV"
        )  # shape (200, 1, len(E))

        _gamma_p_prim = _E_p / (mpc2 * self.delta_D)
        _x = _E / _E_p
        _eta = 4.0 * _E_p * self.target.epsilon_dt * mec2 / mpc2**2
        _eta = _eta.to("")

        if type(self.target).__name__ == "RingDustTorus":
            n_ph = self.target.u(r) / (2.0 * np.pi * mec2 * self.target.epsilon_dt)

            prefactor = self.blob.V_b / mpc2

            N_p_prim = prefactor * self.blob.n_p(_gamma_p_prim)

            rR = r / self.target.R_dt
            rR.to("")
            sqr = np.sqrt(1 + rR**2)
            theta_pgam = (
                np.acos(
                    rR * np.cos(self.theta_s) / sqr
                    + np.sin(self.theta_s) * np.cos(np.pi / 2 + _phi) / sqr
                )
                * 180.0
                / np.pi
            )  # in deg

            _H_integrand = (
                1
                / _E_p
                * n_ph
                * self.delta_D**3
                * N_p_prim
                / (4.0 * np.pi)
                * g_kernel(_eta * eta_0, theta_pgam, _x)  # gKernel angle in degrees!!!
            ).to("erg-2 cm-3")
        else:
            raise AttributeError("Target not included in calculations")

        _H = integrator(
            _H_integrand,
            _E_p,
            axis=0,
        ).to("erg-1 cm-3")
        return _H

    def evaluate_spectrum(self, E, r, particle, integrator=np.trapz):

        if particle not in secondaries:
            raise AttributeError(
                f"There is no secondary particle from photomeson interactions named {particle}."
            )

        g_kernel = gKernel(particle)

        # Integral on phi angle to be done from 0 to 2 pi
        phi = np.linspace(
            0.0,
            2 * np.pi,
            100,
        )
        _H = self.H(
            phi,
            E,
            r,
            g_kernel,
            integrator=integrator,
        )
        dN_dEdtdOmegap = integrator(_H, phi, axis=0).to("erg-1 cm-3")
        return dN_dEdtdOmegap
