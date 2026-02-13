# class for photomeson production
import numpy as np
import astropy.units as u
from astropy.constants import c, h, m_e
from .kernels import PhiKernel
from ..utils.math import axes_reshaper, log10
from ..utils.conversion import mpc2

secondaries = [
    "gamma",
    "electron",
    "positron",
    "electron_neutrino",
    "electron_antineutrino",
    "muon_neutrino",
    "muon_antineutrino",
]

eta_0 = 0.313

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

    def __init__(
        self,
        blob,
        target,
        integrator = np.trapz
    ):
        self.blob = blob
        # check that this blob has a proton distribution
        if self.blob._n_p is None:
            raise AttributeError(
                "There is no proton distribution in this emission region"
            )
        self.target = target
        self.integrator = integrator
        return


    def H(
        self,
        eta,
        E,
        phi_kernel,
        integrator = np.trapz
    ):
        r""" Compute the H function in Eq. (70) [KelnerAharonian2008]_.

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
        _eta, _E = axes_reshaper(eta, E)   # shape (len(eta), 1), (1, len(E))
        _E_p = np.logspace(
            log10(_E.to_value("eV")),
            log10(_E.to_value("eV")) + 8,
            200
        ) * u.Unit("eV")                   # shape (200, 1, len(E))
        _gamma_p = _E_p / mpc2
        _epsilon = _eta * mpc2**2 / (4*_E_p)
        _nu = _epsilon / h
        _x = _E / _E_p
        _H_integrand = (
            mpc2**2 / 4                           # erg2
            * ((mpc2**-1)*self.blob.n_p(_gamma_p) / _E_p**2) # cm-3 erg-3
            * self.target(_nu)                    # cm-3 erg-1
            * phi_kernel(_eta, _x)                # cm3 s-1
        ).to("erg-2 cm-3 s-1")

        _H = integrator(
            _H_integrand,
            _E_p,
            axis = 0,
        ).to("erg-1 cm-3 s-1")
        return _H


    def evaluate_spectrum (
        self,
        E,
        particle,
        integrator = np.trapz
    ):
        """ Evaluate the spectrum of secondaries in the emission region reference frame
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
            log10(eta_0) + 5,
            100,
        )
        _H = self.H(
            eta,
            E,
            phi_kernel,
            integrator = integrator,
        )
        dN_dEdVdt = integrator(
            _H,
            eta,
            axis = 0
        ).to("erg-1 cm-3 s-1")
        return dN_dEdVdt
