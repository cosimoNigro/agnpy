#Module describing the conical emission region

import numbers
from astropy.constants import c, m_e
from agnpy.utils.conversion import B_to_cgs, mec2
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from astropy.constants import c, m_e
from ..utils.numerical_methods import cooling_time

from .. import ParticleDistribution
from ..spectra import PowerLaw
from ..utils.conversion import mec2, B_to_cgs
from ..utils.numerical_methods import *

class Cone:
    r"""Uniform Conical Jet Model.

    **Note:** We will assume that the jet has an opening angle theta_open,
    initial radius R_0 and length L, all defined in the centre of momentum
    frame of the fluid.

    Parameters
    ----------
    R_0 : :class:`~astropy.units.Quantity`
        Initial radius at the base of the cone
    L : :class:`~astropy.units.Quantity`
        Length of the Jet
    theta_open : class:`~astropy.units.Quantity`
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
        theta_open=5 * u.deg,
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
        self.theta_open = theta_open.to("rad")
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
        """ Initialize a conical jet from its total jet power.

        Computes the radius at the base of the jet by assuming that the total
        jet power is carried by the magnetic field and the electron population,
        with their energy densities related through the equipartition parameter
        ``A_equi = U_e / U_B``.
        """
        B = kwargs.get("B_0")
        A = kwargs.setdefault("A_equi", 1)
        gamma = kwargs.get("Gamma")
        u_B = ((B_to_cgs(B) ** 2) / (8 * np.pi)).to("erg cm-3")
        R_0_squared = (W_j) / ((A+1) * u_B * np.pi * (gamma**2) * c.cgs)
        R_0 = np.sqrt(R_0_squared).to("cm")
        return cls(R_0=R_0, **kwargs)
    
    @classmethod
    def from_volume_emission_region(cls, V, **kwargs):
        """Initialize a conical jet from its total emission-region volume.

        Computes the base radius of a truncated conical jet from its total
        volume, length, and opening angle by solving the truncated cone
        volume equation"""

        L = kwargs.get("L")
        theta_open = kwargs.get("theta")
        A = 1
        B = L * np.tan(theta_open)
        C = (L**2 * np.tan(theta_open)**2) / 3 - V / (np.pi * L)
        R0 = (-B + np.sqrt(B**2 - 4*A*C)) / (2*A)
        return cls(R_0=R0,**kwargs)

    @property
    def V_c(self):
        r"""Volume of the truncated conical jet.

        The volume is
        :math:`V_{\rm c} = \frac{\pi L}{3}\left(R_0^2 + R_0 R_L + R_L^2\right)`,
        where
        :math:`R_L = R_0 + L \tan\theta_{\rm open}`.
        
        Returns
        -------
        astropy.units.Quantity
            Total volume of the conical emission region."""
        
        R_L = self.R_0 + self.L * np.tan(self.theta_open)
        return 1 / 3 * (np.pi * self.L) * (self.R_0**2 + self.R_0 * R_L + R_L**2)

    @property
    def R_x(self):
        r"""Jet radius as a function of distance along the jet axis.

        The radius expands linearly:
        :math:`R(x) = R_0 + x \tan\theta_{\rm open}`.
        
        Returns
        -------
        astropy.units.Quantity
            Jet radius at position :math:`x`."""
        
        return self.R_0 + self.x * np.tan(self.theta_open)

    @property
    def B_x(self):
        r"""Magnetic field profile along the jet.
        
        Assuming magnetic flux conservation in a conical expansion:
        :math:`B(x) = B_0 \left(\frac{R_0}{R(x)}\right)`.
       
        Returns
        -------
        astropy.units.Quantity
            Magnetic field at position :math:`x`."""
        
        return self.B_0 * (self.R_0 / self.R_x)

    @property
    def x(self, x_min=1e-9 * u.pc):
        r"""Spatial grid along the jet axis. 

        The grid is logarithmically spaced: 
            :math:`x \in [x_{\min}, L]`, with :math:`x = 10^{\log_{10}(x_{\min}) \to \log_{10}(L)}`. 

        Returns 
        ------- 
        astropy.units.Quantity 
            1D array of positions along the jet in cm. """
        
        x = (
            np.logspace(
                np.log10(x_min.to_value("pc")),
                np.log10(self.L.to_value("pc")),
                self.x_size)* u.pc)
        return x.to("cm")

    @property
    def gamma_e_cc(self):
        r"""Electron Lorentz factor grid used for Chang and Cooper solver. 
        
        The grid is logarithmically spaced:  
            :math:`\gamma_i \in [\gamma_{\min}, \gamma_{\max}]`. 
        
        Returns 
        ------- 
        numpy.ndarray 
            Lorentz factor grid for numerical solution. """

        return np.logspace(
            np.log10(self._n_e.gamma_min),
            np.log10(self._n_e.gamma_max),
            self.gamma_e_size,
        )

    @property
    def gamma_e(self):
        r"""Reduced Lorentz factor grid obtained from Chang and Cooper solver,
        the electron distribution and subsequent quantities are calculated on this grid 
        
        Defined as every second interior point to represent the bin-edges: 
            :math:`\gamma = \gamma_{\rm cc}[1:-1:2]`. 
        
        Returns 
        ------- 
        numpy.ndarray 
            Reduced electron Lorentz factor grid. """
        return self.gamma_e_cc[1:-1:2]

    @property
    def norm_equi(self):
        r"""Normalization factor enforcing equipartition.
        Defined by equating electron and magnetic energy densities:
            :math:`K = \frac{A_{\rm equi} U_B}{U_e}`,
        where 
        :math:`U_B = \frac{B_0^2}{8\pi}`,
        and
        :math:`U_e = m_e c^2 \int \gamma n_e(\gamma)\,d\gamma`.

        Returns
        -------
        float
            Dimensionless normalization constant."""
        
        e_energy_initial = mec2.to("eV") * np.trapz(
            self.gamma_e_cc * self._n_e(self.gamma_e_cc), self.gamma_e_cc
        )

        e_energy_initial *= u.Unit("cm-3")
        K_dash = (self.A_equi * (self.B_0**2)) / (8 * np.pi * e_energy_initial)
        return K_dash.cgs
    
    @property
    def n_e_base(self):
        r"""Initial electron distribution at the jet base.
        Given by:
            :math:`N_e(\gamma) = K \, n_e(\gamma)\, \pi R_0^2`,
        where :math:`K` enforces equipartition.

        Returns
        -------
        astropy.units.Quantity
            Number of electrons in a slice of width 1 cm at the jet base."""
        
        return (
            (self._n_e(self.gamma_e_cc)[1:-1:2])
            * self.norm_equi
            * u.cm**-3
            * np.pi
            * self.R_0**2
        )

    @property
    def N_e_xg(self):
        r"""Electron distribution along the jet. Solution of the 
        transport equation using the Chang and Cooper numerical scheme:
            :math:`\frac{\partial N_e}{\partial x}
            = \mathcal{L}[N_e(\gamma, x)] - \text{escape terms}`.

        Returns
        -------
        astropy.units.Quantity
            Number of electrons in a slice of width 1 cm. 
            :math:`N_e(x,\gamma)`."""
        
        cc_solver = ChangCooperSolver(
            gamma_e=self.gamma_e_cc,
            x=self.x,
            R_0=self.R_0,
            B_0=self.B_0,
            theta_open=self.theta_open,
            n_e=self.n_e_base,
            electron_escape=self.electron_escape,
            escape_coefficient=self.escape_coefficient,
        )
        N_e_xg = cc_solver.run()
        return N_e_xg

    @property
    def synch_cooling_time(self):
        t_cool = cooling_time(self.gamma_e[None,:], self.B_x[:,None])
        return t_cool
    
    @property
    def N_e_gamma(self):
        r"""Electron spectrum integrated over jet length.
            :math:`N_e(\gamma) = \int N_e(x,\gamma)\,dx`.

        Returns
        -------
        astropy.units.Quantity
            Electron distribution as a function of Lorentz factor."""
        
        N_e_gamma = np.trapz(self.N_e_xg, self.x, axis=0)
        return N_e_gamma

    @property
    def N_e_x(self):
        r"""Electron in a slice of width 1 cm along the jet axis.
            :math:`N_e(x) = \int N_e(x,\gamma)\,d\gamma`.

        Returns
        -------
        astropy.units.Quantity
            Electron number per unit length."""

        N_e_x = np.trapz(self.N_e_xg, self.gamma_e, axis=1)
        return N_e_x

    @property
    def U_e_x(self):
        r"""Electron energy density along the jet.
        Computed as:
            :math:`U_e(x) = m_e c^2 \int \gamma N_e(x,\gamma)\,d\gamma`.

        Returns
        -------
        astropy.units.Quantity
            Electron energy per unit jet slice.
        """
        U_e_x = mec2.cgs * np.trapz(
            self.gamma_e[None, :] * self.N_e_xg, self.gamma_e, axis=1
        )
        return U_e_x

    @property
    def U_B_x(self):
        r"""Magnetic energy in a jet slice.
        Given by:
            :math:`U_B(x) = \frac{B(x)^2}{8\pi} \, \pi R(x)^2`.

        Returns
        -------
        astropy.units.Quantity
            Magnetic energy per unit length."""
        
        return np.pi * self.R_x**2 * np.power(self.B_x, 2) / (8 * np.pi)

    @property
    def W_e(self):
        r"""Total electron energy in the jet.
            :math:`W_e = m_e c^2 \int \gamma N_e(\gamma)\,d\gamma`.

        Returns
        -------
        astropy.units.Quantity
            Total electron energy."""

        W_e = mec2.cgs * np.trapz(self.gamma_e * self.N_e_gamma, self.gamma_e)
        return W_e

    @property
    def N_e(self):
        r"""Total number of electrons in the jet.
        Obtained by integrating over position and energy:
            :math:`N_e = \int \int N_e(x,\gamma)\, d\gamma \, dx`.

        Returns
        -------
        float
            Total electron number in the emission region. """
       
        N_e = np.trapz(self.N_e_x, self.x)
        return N_e

    @property
    def Beta(self):
        r"""Bulk velocity of the flow in units of c.
            :math:`\beta = \sqrt{1 - \Gamma^{-2}}`.

        Returns
        -------
        float
            Dimensionless bulk velocity."""
        
        return np.sqrt(1 - 1 / np.power(self.Gamma, 2))

    @property
    def mu_s(self):
        r"""Cosine of the viewing angle.
        Defined from relativistic beaming relation:
            :math:`\mu_s = \frac{1 - 1/(\Gamma \delta_D)}{\beta}`.

        Returns
        -------
        float
            Cosine of the viewing angle."""
        
        return (1 - 1 / (self.Gamma * self.delta_D)) / self.Beta

    @property
    def theta_s(self):
        r"""Viewing angle of the jet.
        Computed as:
            :math:`\theta_s = \arccos(\mu_s)`.

        Returns
        -------
        astropy.units.Quantity
            Jet viewing angle in degrees."""
        
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
            + f" - theta_open (Opening Angle of the Conical Jet): {self.theta_open:.2e}\n"
            + f" - V_c (Volume of the Cone): {self.V_c.cgs:.2e}\n"
            + f" - z (source redshift): {self.z:.2f}\n"
            + f" - d_L (source luminosity distance):{self.d_L.cgs:.2e}\n"
            + f" - delta_D (Cone Doppler factor): {self.delta_D:.2e}\n"
            + f" - Gamma (Bulk Lorentz factor): {self.Gamma:.2e}\n"
            + f" - Beta (Bulk relativistic velocity): {self.Beta:.2e}\n"
            + f" - theta_s (jet viewing angle): {self.theta_s:.2e}\n"
            + f" - B (Magnetic field at the base the jet): {self.B_0:.2e}\n"
        )
