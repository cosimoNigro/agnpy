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
    r"""Uniform conical jet emission region.

    Models the jet as a truncated cone of constant opening angle, with a
    magnetic field that declines with radius to conserve magnetic flux, and
    an electron population evolved along the jet axis using the Chang &
    Cooper numerical scheme.

    All geometric quantities (:attr:`R_0`, :attr:`L`, :attr:`theta_open`)
    are defined in the centre-of-momentum (fluid) frame of the plasma.

    Parameters
    ----------
    L : :class:`~astropy.units.Quantity`
        Length of the jet along its axis in the fluid frame.
        Default ``1e18 cm``.
    R_0 : :class:`~astropy.units.Quantity`
        Radius of the jet at its base (:math:`x = 0`).
        Default ``1e14 cm``.
    B_0 : :class:`~astropy.units.Quantity`
        Magnetic field strength at the jet base.
        Default ``1 G``.
    theta_open : :class:`~astropy.units.Quantity`
        Half-opening angle of the cone.
        Default ``5 deg``.
    z : float
        Redshift of the source. Default ``0.033``.
    delta_D : float
        Doppler factor of the relativistic outflow. Must be positive.
        Default ``10``.
    Gamma : float
        Bulk Lorentz factor of the outflow. Default ``10``.
    n_e : :class:`~agnpy.spectra.ParticleDistribution`
        Electron energy distribution injected at the jet base.
        Default is a :class:`~agnpy.spectra.PowerLaw` with electron mass.
    gamma_e_size : int
        Number of points in the Lorentz factor grid used by the
        Chang & Cooper solver. Default ``200``.
    x_size : int
        Number of points in the spatial grid along the jet axis.
        Default ``100``.
    A_equi : float
        Equipartition parameter at the jet base, defined as
        :math:`A_{\rm equi} = U_e / U_B`. Default ``1``.
    cosmology : :class:`~astropy.cosmology.Cosmology`, optional
        Cosmology used to convert ``z`` to a luminosity distance.
        Defaults to the astropy default cosmology.
    electron_escape : bool
        If ``True``, particle escape losses are included in the electron
        transport equation; escaping electrons are removed as they
        propagate along the jet. Default ``False``.
    escape_coefficient : float, optional
        Dimensionless coefficient controlling the strength of particle
        escape. Required (and must be non-zero) when
        ``electron_escape=True``.
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
        r"""Initialise a conical jet from its total jet power.

        Derives the base radius :math:`R_0` by requiring that the jet
        power is carried by the magnetic field and electron population
        in the ratio set by ``A_equi``:

        .. math::
            R_0 = \sqrt{\frac{W_j}{(A_{\rm equi}+1)\,U_B\,\pi\,\Gamma^2\,c}},

        where :math:`U_B = B_0^2 / (8\pi)`.

        Parameters
        ----------
        W_j : :class:`~astropy.units.Quantity`
            Total jet power in the lab frame (e.g. ``erg s-1``).
        **kwargs
            All parameters accepted by :class:`Cone`, except ``R_0``
            which is computed here. ``B_0`` and ``Gamma`` are required.

        Returns
        -------
        :class:`Cone`
            Cone instance with ``R_0`` set from the jet power.
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
        r"""Initialise a conical jet from its total emission-region volume.

        Derives the base radius :math:`R_0` by solving the truncated-cone
        volume equation:

        .. math::
            V = \pi L \left(R_0^2 + R_0 R_0 \tan\theta + \frac{L^2 \tan^2\theta}{3}\right),

        which is a quadratic in :math:`R_0` with solution:

        .. math::
            R_0 = \frac{-B + \sqrt{B^2 - 4C}}{2}, \quad
            B = L\tan\theta, \quad
            C = \frac{L^2\tan^2\theta}{3} - \frac{V}{\pi L}.

        Parameters
        ----------
        V : :class:`~astropy.units.Quantity`
            Total volume of the emission region (e.g. ``cm3``).
        **kwargs
            All parameters accepted by :class:`Cone`, except ``R_0``
            which is computed here. ``L`` and ``theta`` are required.

        Returns
        -------
        :class:`Cone`
            Cone instance with ``R_0`` set from the volume.
        """

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

        Computed as:

        .. math::
            V_c = \frac{\pi L}{3}\left(R_0^2 + R_0 R_L + R_L^2\right),

        where :math:`R_L = R_0 + L\tan\theta_{\rm open}` is the radius
        at the far end of the jet.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Total volume of the conical emission region in cm³.
        """
        
        R_L = self.R_0 + self.L * np.tan(self.theta_open)
        return 1 / 3 * (np.pi * self.L) * (self.R_0**2 + self.R_0 * R_L + R_L**2)

    @property
    def R_x(self):
        r"""Jet radius as a function of position along the axis.

        Follows the linear conical expansion:

        .. math::
            R(x) = R_0 + x\,\tan\theta_{\rm open}.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Jet radius at each grid position :attr:`x`, shape ``(N_x,)``.
        """
        
        return self.R_0 + self.x * np.tan(self.theta_open)

    @property
    def B_x(self):
        r"""Magnetic field profile along the jet axis.

        Assumes magnetic flux conservation in the conical expansion,
        giving a :math:`1/R` decline:

        .. math::
            B(x) = B_0\,\frac{R_0}{R(x)}.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Magnetic field strength at each grid position :attr:`x`,
            shape ``(N_x,)``, in Gauss (cgs).
        """
        
        return self.B_0 * (self.R_0 / self.R_x)

    @property
    def x(self, x_min=1e-9 * u.pc):
        r"""Logarithmically spaced spatial grid along the jet axis.

        The grid spans from a an inner boundary to the jet length:

        .. math::
            x \in [x_{\min},\, L], \quad
            x_i = 10^{\log_{10}(x_{\min}) + i\,\Delta},

        with :attr:`x_size` points.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            1D array of positions along the jet axis in cm,
            shape ``(N_x,)`` where ``N_x = x_size``.
        """
        
        x = (
            np.logspace(
                np.log10(x_min.to_value("pc")),
                np.log10(self.L.to_value("pc")),
                self.x_size)* u.pc)
        return x.to("cm")

    @property
    def gamma_e_cc(self):
        r"""Electron Lorentz factor grid for the Chang & Cooper solver.

        Logarithmically spaced between :attr:`~agnpy.spectra.ParticleDistribution.gamma_min`
        and :attr:`~agnpy.spectra.ParticleDistribution.gamma_max` of the
        injected distribution, with :attr:`gamma_e_size` points.

        Returns
        -------
        :class:`~numpy.ndarray`
            1D Lorentz factor array used as the full solver grid,
            shape ``(gamma_e_size,)``.
        """

        return np.logspace(
            np.log10(self._n_e.gamma_min),
            np.log10(self._n_e.gamma_max),
            self.gamma_e_size,
        )

    @property
    def gamma_e(self):
        r"""Reduced electron Lorentz factor grid for post-solver quantities.

        Taken as every second interior point of :attr:`gamma_e_cc` to
        represent bin edges output by the Chang & Cooper scheme:

        .. math::
            \gamma = \gamma_{\rm cc}[1:-1:2].

        All electron distributions and derived spectra (e.g.
        :attr:`N_e_xg`) are defined on this grid.

        Returns
        -------
        :class:`~numpy.ndarray`
            1D reduced Lorentz factor array, shape
            ``((gamma_e_size - 2) // 2,)``.
        """
        return self.gamma_e_cc[1:-1:2]

    @property
    def norm_equi(self):
        r"""Equipartition normalisation factor for the electron distribution.

        Scales the injected distribution so that the electron energy density
        equals :math:`A_{\rm equi}` times the magnetic energy density at
        the jet base:

        .. math::
            K = \frac{A_{\rm equi}\,U_B}{\int \gamma\,n_e(\gamma)\,d\gamma},
            \quad U_B = \frac{B_0^2}{8\pi}.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Dimensionless normalisation constant (in cgs units with
            implicit cm³ denominator absorbed into :attr:`n_e_base`).
        """
        
        e_energy_initial = mec2.to("eV") * np.trapz(
            self.gamma_e_cc * self._n_e(self.gamma_e_cc), self.gamma_e_cc
        )

        e_energy_initial *= u.Unit("cm-3")
        K_dash = (self.A_equi * (self.B_0**2)) / (8 * np.pi * e_energy_initial)
        return K_dash.cgs
    
    @property
    def n_e_base(self):
        r"""Initial electron line density at the jet base.

        Number of electrons per cm of jet length in a slice of unit width
        at :math:`x = 0`, given by:

        .. math::
            N_e(\gamma) = K\,n_e(\gamma)\,\pi R_0^2,

        where :math:`K` is the equipartition normalisation :attr:`norm_equi`.
        Evaluated on the reduced grid :attr:`gamma_e`.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Electron line density at the base, shape ``(N_gamma,)``,
            units ``cm⁻¹``.
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
        r"""Electron line density along the jet from the transport solver.

        Solves the electron continuity equation along the jet using the
        Chang & Cooper finite-difference scheme, taking into account
        synchrotron and (optionally) inverse-Compton energy losses and
        particle escape:

        .. math::
            \frac{\partial N_e}{\partial x}
            = \mathcal{L}[N_e(\gamma, x)] - \text{escape terms}.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            2D array of electron line densities (electrons per cm of jet
            length) at each position and Lorentz factor,
            shape ``(N_x, N_gamma)``, units ``cm⁻¹``.
        """
        
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
        r"""Synchrotron cooling time at each :math:`(x, \gamma)` grid point.

        Computed as:

        .. math::
            t_{\rm cool}(\gamma, x) = \frac{E}{\dot{E}_{\rm synch}}
            = \frac{3\,m_e c}{4\,\sigma_T\,U_B(x)\,\gamma}.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            2D array of cooling times, shape ``(N_x, N_gamma)``, in seconds.
        """
        t_cool = cooling_time(self.gamma_e[None,:], self.B_x[:,None])
        return t_cool
    
    @property
    def N_e_gamma(self):
        r"""Electron distribution integrated over the jet length.

        .. math::
            N_e(\gamma) = \int_0^L N_e(x, \gamma)\,dx.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Total number of electrons per unit Lorentz factor (integrated
            over all positions), shape ``(N_gamma,)``.
        """
        N_e_gamma = np.trapz(self.N_e_xg, self.x, axis=0)
        return N_e_gamma

    @property
    def N_e_x(self):
        r"""Number of electrons per unit jet length at each position.

        .. math::
            N_e(x) = \int N_e(x, \gamma)\,d\gamma.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Electron line density integrated over :math:`\gamma`,
            shape ``(N_x,)``, units ``cm⁻¹``.
        """
        N_e_x = np.trapz(self.N_e_xg, self.gamma_e, axis=1)
        return N_e_x

    @property
    def U_e_x(self):
        r"""Electron energy per unit jet length at each position.

        .. math::
            U_e(x) = m_e c^2 \int \gamma\,N_e(x, \gamma)\,d\gamma.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Electron energy per unit length along the jet,
            shape ``(N_x,)``, units ``erg cm⁻¹``.
        """
        U_e_x = mec2.cgs * np.trapz(
            self.gamma_e[None, :] * self.N_e_xg, self.gamma_e, axis=1
        )
        return U_e_x

    @property
    def U_B_x(self):
        r"""Magnetic energy per unit jet length at each position.

        .. math::
            U_B(x) = \frac{B(x)^2}{8\pi}\,\pi R(x)^2.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Magnetic energy per unit length along the jet,
            shape ``(N_x,)``, units ``erg cm⁻¹``.
        """
        return np.pi * self.R_x**2 * np.power(self.B_x, 2) / (8 * np.pi)

    @property
    def W_e(self):
        r"""Total electron energy in the jet.

        .. math::
            W_e = m_e c^2 \int \gamma\,N_e(\gamma)\,d\gamma.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Total electron energy integrated over position and Lorentz
            factor, in erg.
        """
        W_e = mec2.cgs * np.trapz(self.gamma_e * self.N_e_gamma, self.gamma_e)
        return W_e

    @property
    def N_e(self):
        r"""Total number of electrons in the jet.

        .. math::
            N_e = \int\!\int N_e(x, \gamma)\,d\gamma\,dx.

        Returns
        -------
        float
            Total (dimensionless) electron count integrated over all
            positions and Lorentz factors.
        """
        N_e = np.trapz(self.N_e_x, self.x)
        return N_e

    @property
    def Beta(self):
        r"""Bulk velocity of the outflow in units of :math:`c`.

        .. math::
            \beta = \sqrt{1 - \Gamma^{-2}}.

        Returns
        -------
        float
            Dimensionless bulk velocity :math:`\beta \in (0, 1)`.
        """
        return np.sqrt(1 - 1 / np.power(self.Gamma, 2))

    @property
    def mu_s(self):
        r"""Cosine of the jet viewing angle.

        Derived from the Doppler beaming relation:

        .. math::
            \mu_s = \frac{1 - 1/(\Gamma\,\delta_D)}{\beta}.

        Returns
        -------
        float
            :math:`\cos\theta_s`, where :math:`\theta_s` is the angle
            between the jet axis and the line of sight.
        """
        return (1 - 1 / (self.Gamma * self.delta_D)) / self.Beta

    @property
    def theta_s(self):
        r"""Viewing angle of the jet.

        .. math::
            \theta_s = \arccos(\mu_s).

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Jet viewing angle in degrees.
        """
        return (np.arccos(self.mu_s) * u.rad).to("deg")

    def set_delta_D(self, Gamma, theta_s):
        r"""Set the Doppler factor from the bulk Lorentz factor and viewing angle.

        Computes:

        .. math::
            \delta_D = \frac{1}{\Gamma\,(1 - \beta\,\cos\theta_s)}.

        Also updates :attr:`Gamma`.

        Parameters
        ----------
        Gamma : float
            Bulk Lorentz factor of the outflow.
        theta_s : :class:`~astropy.units.Quantity`
            Viewing angle of the jet (angle between jet axis and line of
            sight).
        """
        self.Gamma = Gamma
        mu_s = np.cos(theta_s.to("rad").value)
        self.delta_D = 1 / (self.Gamma * (1 - self.Beta * mu_s))

    def __str__(self):
        """Printable summary of the conical jet parameters."""
        resume = (
            "* Conical emission region\n"
            + f" - R_0 (Radius at the base of the cone): {self.R_0.cgs:.2e}\n"
            + f" - L (Length of the Jet): {self.L.cgs:.2e}\n"
            + f" - theta_open (Opening Angle of the Conical Jet): {self.theta_open:.2e}\n"
            + f" - V_c (Volume of the Cone): {self.V_c.cgs:.2e}\n"
            + f" - z (source redshift): {self.z:.2f}\n"
            + f" - d_L (source luminosity distance): {self.d_L.cgs:.2e}\n"
            + f" - delta_D (Cone Doppler factor): {self.delta_D:.2e}\n"
            + f" - Gamma (Bulk Lorentz factor): {self.Gamma:.2e}\n"
            + f" - Beta (Bulk relativistic velocity): {self.Beta:.2e}\n"
            + f" - theta_s (jet viewing angle): {self.theta_s:.2e}\n"
            + f" - B_0 (Magnetic field at the base of the jet): {self.B_0:.2e}\n"
        )
        return resume
