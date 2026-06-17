#Module describing the conical emission region

import numbers
from astropy.constants import c, m_e
from agnpy.utils.conversion import B_to_cgs, mec2
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from astropy.constants import c, m_e

from .. import ParticleDistribution
from ..spectra import PowerLaw
from ..utils.conversion import mec2, B_to_cgs
from ..utils.numerical_methods import *

class Cone:
    r"""Uniform Conical Jet Model.

    **Note:** We will assume that the jet has an opening angle θ_{opening},
    initial radius R_0 and length L, all defined in the centre of momentum
    frame of the fluid.

    Parameters
    ----------
    R_0 : :class:`~astropy.units.Quantity`
        Initial radius at the base of the cone
    L : :class:`~astropy.units.Quantity`
        Length of the Jet
    theta : class:`~astropy.units.Quantity`
        Opening angle of the cone
    z : float
        redshift of the source
    delta_D : float
        Doppler factor of the relativistic outflow
    Gamma : float
        Lorentz factor of the relativistic outflow
    B_0 : :class:`~astropy.units.Quantity`
        magnetic field at the base of the cone
    n_e : :class:`~agnpy.spectra.ParticleDistribution`
        electron distribution initialized at the base of the cone
    A_equi : float
        Equipartition parameter at the base of the jet, 
        defined as the ratio of electron energy density to magnetic
    electron_escape : bool
        If True, includes particle escape losses in the electron transport equation. 
        Escaping particles are removed from the electron population as they propagate along the jet.
    escape_coefficient : float, optional
        Dimensionless coefficient controlling the strength of particle escape. 
    """

    def __init__(
        self,
        L=1e18 * u.cm,
        R_0=1e14 * u.cm,
        B_0=1 * u.G,
        theta=5 * u.deg,
        z=0.033,
        delta_D=10,
        Gamma=10,
        n_e: ParticleDistribution = PowerLaw(mass=m_e),
        xi=1.0,
        gamma_e_size=200,
        x_size=100,
        A_equi=1,
        cosmology=None,
        electron_escape=False,
        escape_coefficient=None,
    ):
        if not isinstance(delta_D, numbers.Number) or delta_D <= 0:
            raise ValueError("delta_D must be a positive number")

        self.L = L.to("cm")
        self.R_0 = R_0
        self.B_0 = B_to_cgs(B_0)
        self.theta = theta.to("rad")
        self.z = z
        self.d_L = Distance(z=self.z, cosmology=cosmology).cgs
        self.delta_D = delta_D
        self.Gamma = Gamma
        self.A_equi = A_equi
        self._n_e: ParticleDistribution = n_e
        self.xi = xi
        self.gamma_e_size = gamma_e_size
        self.x_size = x_size
        self.electron_escape = electron_escape
        if self.electron_escape:
            if escape_coefficient is None:
                raise ValueError("escape_coefficient must be provided")
            if escape_coefficient == 0:
                raise ValueError(
                    "escape_coefficient cannot be 0 when electron_escape=True (use electron_escape=False instead)"
                )
            self.escape_coefficient = escape_coefficient
        else:
            self.escape_coefficient = None

    @classmethod
    def from_jet_power(cls, W_j, **kwargs):
        B = kwargs.get("B_0")
        gamma = kwargs.get("Gamma")
        u_B = ((B_to_cgs(B) ** 2) / (8 * np.pi)).to("erg cm-3")
        R_0_squared = (W_j) / (2 * u_B * np.pi * (gamma**2) * c.cgs)
        R_0 = np.sqrt(R_0_squared).to("cm")
        return cls(R_0=R_0, **kwargs)

    @property
    def V_c(self):
        """Volume of the truncated cone."""
        R_L = self.R_0 + self.L * np.tan(self.theta)
        return 1 / 3 * (np.pi * self.L) * (self.R_0**2 + self.R_0 * R_L + R_L**2)

    @property
    def R_x(self):
        """Radius of the cone as a function of distance from the base"""
        return self.R_0 + self.x * np.tan(self.theta)

    @property
    def B_x(self):
        """Magnetic field of the cone as a function of distance from the base"""
        return self.B_0 * (self.R_0 / self.R_x)

    @property
    def x(self, x_min=0.000000001 * u.pc):
        """Spatial grid"""
        x = (
            np.logspace(
                np.log10(x_min.to_value("pc")),
                np.log10(self.L.to_value("pc")),
                self.x_size)* u.pc)
        return x.to("cm")


    @property
    def gamma_e_cc(self):
        """Array of electrons Lorentz factors, to be used for integration in the
        reference frame comoving with the emission region."""
        return np.logspace(
            np.log10(self._n_e.gamma_min),
            np.log10(self._n_e.gamma_max),
            self.gamma_e_size,
        )

    @property
    def gamma_e(self):
        """Array of electrons Lorentz factors modified for Chang and Cooper numerical method,
        to be used for integration in the reference frame comoving with the emission region.
        """
        return self.gamma_e_cc[1:-1:2]

    @property
    def gamma_e_external_frame(self):
        """Array of electrons Lorentz factors, to be used for integration in the
        reference frame external to the emission region."""
        return np.logspace(1, 9, self.gamma_e_size)

    @property
    def norm_equi(self):
        """Modified normalization to convolve with initial electron distributionto impose equipartition based on A_equi. 
        Obtained by equating magnetic energy density at the base to
        electron energy density.
        units: cm-3"""
        e_energy_initial = mec2.to("eV") * np.trapz(
            self.gamma_e_cc * self._n_e(self.gamma_e_cc), self.gamma_e_cc
        )
        e_energy_initial *= u.Unit("cm-3")
        K_dash = (self.A_equi * (self.B_0**2)) / (8 * np.pi * e_energy_initial)
        return K_dash.cgs
    
    @property
    def n_e_base(self):
        """Electron distribution as obtained by agnpy.spectra, with modified normalization
        imposed by equipartition parameter
        units: cm-3
        """
        return (
            (self._n_e(self.gamma_e_cc)[1:-1:2])
            * self.norm_equi
            * u.cm**-3
            * np.pi
            * self.R_0**2
        )

    @property
    def N_e_xg(self):
        """Solution to the Electron Evolution Equation using Chang and Cooper scheme.
        Returns the number of electrons per cm"""
        cc_solver = ChangCooperSolver(
            gamma_e=self.gamma_e_cc,
            x=self.x,
            R_0=self.R_0,
            B_0=self.B_0,
            theta_open=self.theta,
            n_e=self.n_e_base,
            electron_escape=self.electron_escape,
            escape_coefficient=self.escape_coefficient,
        )
        N_e_xg = cc_solver.run()
        return N_e_xg

    @property
    def N_e_gamma(self):
        """Number of electrons as a function of gamma across the Jet"""
        N_e_gamma = np.trapz(self.N_e_xg, self.x, axis=0)
        return N_e_gamma

    @property
    def N_e_x(self):
        """Number of electrons per cm at position x along the jet axis"""
        N_e_x = np.trapz(self.N_e_xg, self.gamma_e, axis=1)
        return N_e_x

    @property
    def U_e_x(self):
        r"""Energy of electrons in a slice of width 1 cm"""
        U_e_x = mec2.cgs * np.trapz(
            self.gamma_e[None, :] * self.N_e_xg, self.gamma_e, axis=1
        )
        return U_e_x

    @property
    def U_B_x(self):
        r"""Energy of magnetic field in a slice of width 1 cm as a function of distance
        from the base of the Jet
        """
        return np.pi * self.R_x**2 * np.power(self.B_x, 2) / (8 * np.pi)

    @property
    def W_e(self):
        r"""Total energy of electrons"""
        W_e = mec2.cgs * np.trapz(self.gamma_e * self.N_e_gamma, self.gamma_e)
        return W_e

    @property
    def N_e(self):
        r"""Total Number of electrons in the Jet,
        :math:`N_{\rm p}(\gamma') = V_{\rm b}\,n_{\rm p}(\gamma')`.

        Parameters
        ----------
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to evaluate the number of electrons
        """
        N_e = np.trapz(self.N_e_x, self.x)
        return N_e

    @property
    def Beta(self):
        """Bulk Lorentz factor of the Cone."""
        return np.sqrt(1 - 1 / np.power(self.Gamma, 2))

    @property
    def mu_s(self):
        """Cosine of the viewing angle from the jet axis to the observer."""
        return (1 - 1 / (self.Gamma * self.delta_D)) / self.Beta

    @property
    def theta_s(self):
        """Viewing angle from the jet axis to the observer."""
        return (np.arccos(self.mu_s) * u.rad).to("deg")

    def set_delta_D(self, Gamma, theta_s):
        """Set the Doppler factor by specifying the bulk Lorentz factor of the
        outflow and the viewing angle.

        Parameters
        ----------
        Gamma : float
            Lorentz factor of the relativistic outflow
        theta_s : :class:`~astropy.units.Quantity`
            viewing angle of the jet
        """
        self.Gamma = Gamma
        mu_s = np.cos(theta_s.to("rad").value)
        self.delta_D = 1 / (self.Gamma * (1 - self.Beta * mu_s))

    def __str__(self):
        """Printable summary of the Conical Jet."""
        resume = (
            "* Conical emission region\n"
            + f" - R_0 (Radius at the base of the cone): {self.R_0.cgs:.2e}\n"
            + f" - L (Length of the Jet): {self.L.cgs:.2e}\n"
            + f" - θ (Opening Angle of the Conical Jet): {self.theta:.2e}\n"
            + f" - V_c (Volume of the Cone): {self.V_c.cgs:.2e}\n"
            + f" - z (source redshift): {self.z:.2f}\n"
            + f" - d_L (source luminosity distance):{self.d_L.cgs:.2e}\n"
            + f" - delta_D (Cone Doppler factor): {self.delta_D:.2e}\n"
            + f" - Gamma (Bulk Lorentz factor): {self.Gamma:.2e}\n"
            + f" - Beta (Bulk relativistic velocity): {self.Beta:.2e}\n"
            + f" - theta_s (jet viewing angle): {self.theta_s:.2e}\n"
            + f" - B (Magnetic field at the base the jet): {self.B_0:.2e}\n"
            + f" - xi (coefficient for 1st order Fermi acceleration) : {self.xi:.2e}\n"
        )
