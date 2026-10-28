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
    r"""Uniform conical jet emission region.

    Models the jet as a truncated cone of constant opening angle, with a
    magnetic field that declines with radius to conserve magnetic flux and
    an electron population evolved along the jet axis using the Chang &
    Cooper numerical scheme.

    All geometric quantities (:attr:`R_0`, :attr:`L`, :attr:`theta_open`)
    are defined in the host-galaxy frame of reference.

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
        cosmology=None,
        electron_escape=False,
        escape_coefficient=np.inf,
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
            self.escape_coefficient = np.inf
        self.x, self.R_x, self.B_x = self.calculate_spatial_grid_quantities()
        self.gamma_e_cc, self.gamma_e = self.define_gamma_grid()
        self.N_e_xg = self.solve_electron_evolution()
        print(self.__str__())
        

    @classmethod
    def from_jet_power(cls, W_j, A_equi, **kwargs):
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
        _n_e = kwargs.get("n_e")
        gamma = kwargs.get("Gamma")
        gamma_size = kwargs.get("gamma_e_size")
        gamma_min = _n_e.gamma_min
        gamma_max = _n_e.gamma_max
        _gamma = np.logspace(np.log10(gamma_min),np.log10(gamma_max),gamma_size)
        u_B = ((B_to_cgs(B) ** 2) / (8 * np.pi)).to("erg cm-3")
        R_0_squared = (W_j) / ((1+A_equi) * u_B * np.pi * (gamma**2) * c.cgs)
        R_0 = np.sqrt(R_0_squared).to("cm")
        u_e = (mec2.to('eV')*np.trapezoid(_gamma * _n_e(_gamma),_gamma)).to("erg cm-3")
        k_norm = A_equi*(u_B/u_e)
        _n_e.k *= k_norm
        return cls(R_0=R_0, **kwargs)
    
    @classmethod
    def from_equipartition_parameter(cls, A_equi, **kwargs):
        r"""Initialise a conical jet with a specified equipartition ratio.

        Rescales the normalization ``k`` of the injected electron
        distribution so that the ratio of electron to magnetic energy
        density at the jet base equals ``A_equi``:

        .. math::
        A_{\rm equi} = \frac{U_e}{U_B},

        where the magnetic energy density is

        .. math::
            U_B = \frac{B_0^2}{8\pi},

        and the electron energy density is calculated from the injected
        distribution as

        .. math::
            U_e = m_e c^2 \int_{\gamma_{\min}}^{\gamma_{\max}}
            \gamma\,n_e(\gamma)\,d\gamma.

        The method multiplies the electron normalization by
        :math:`A_{\rm equi} U_B/U_e` so that the resulting distribution
        has the requested electron energy to magnetic energy density ratio. 
        The remaining jet parameters are passed to :class:`Cone` through ``kwargs``.

        Parameters
        ----------
        A_equi : float
            Desired ratio of electron to magnetic energy density at the
            jet base, :math:`U_e/U_B`.
        **kwargs
            Keyword arguments forwarded to :class:`Cone`. Must include
            ``B_0`` and ``n_e``. ``gamma_e_size`` specifies the number of
            points in the Lorentz-factor integration grid.

        Returns
        -------
        :class:`Cone`
            Cone instance whose injected electron distribution is
            normalized to the requested equipartition ratio.
        """
        B = kwargs.get("B_0")
        _n_e = kwargs.get("n_e")
        gamma_size = kwargs.get("gamma_e_size")
        gamma_min = _n_e.gamma_min
        gamma_max = _n_e.gamma_max
        gamma = np.logspace(np.log10(gamma_min),np.log10(gamma_max),gamma_size)
        u_B = ((B_to_cgs(B) ** 2) / (8 * np.pi)).to("erg cm-3")
        u_e = (mec2.to('eV')*np.trapezoid(gamma * _n_e(gamma),gamma)).to("erg cm-3")
        k_norm = A_equi*(u_B/u_e)
        _n_e.k *= k_norm
        return cls( **kwargs)
    
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
        theta_open = kwargs.get("theta_open")
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
    def A_equi(self):
        u_B = ((B_to_cgs(self.B_0) ** 2) / (8 * np.pi)).to("erg cm-3")
        u_e = (mec2.to('eV')*np.trapezoid(self.gamma_e_cc*self._n_e(self.gamma_e_cc),self.gamma_e_cc)).to('erg cm-3')
        A = u_e/u_B
        return A
    
    def calculate_spatial_grid_quantities(self):
        r"""Define the spatial grid and geometric quantities along the jet axis.

            The spatial grid is logarithmically spaced between the inner boundary
            :math:`x_{\min}` and the outer boundary :math:`L`:

            .. math::
                x \in [x_{\min}, L],

            where :math:`x_{\min} = 10^{12}\,\mathrm{cm}` and ``x_size`` specifies
            the number of spatial grid points.

            The jet radius increases linearly with distance from the inner boundary,
            such that the radius at the first grid point is :math:`R_0`:

            .. math::
                R(x) = R_0 + (x - x_{\min})\tan\theta_{\rm open}.

            The magnetic field profile assumes that the field strength decreases
            inversely with the jet radius during conical expansion:

            .. math::
                B(x) = B_0\frac{R_0}{R(x)}.

            Parameters
            ----------
            None

            Returns
            -------
            x : :class:`~astropy.units.Quantity`
                One-dimensional array of spatial coordinates along the jet axis,
                in cm, with shape ``(x_size,)``.

            R : :class:`~astropy.units.Quantity`
                Jet radius evaluated at each spatial grid point, in cm, with shape
                ``(x_size,)``.

            B : :class:`~astropy.units.Quantity`
                Magnetic field strength evaluated at each spatial grid point,
                in gauss, with shape ``(x_size,)``.
            """
        x_min = 1e12 * u.cm #x_min is set at ~10R_{g} for M_BH estimates of FR0 RGs.
        x = (np.logspace((np.log10(x_min.to_value("cm"))),np.log10((self.L.to_value("cm"))),self.x_size)* u.cm)
        R = self.R_0 + (x-x[0]) * np.tan(self.theta_open)
        B = self.B_0 * (self.R_0 / R)

        return x, R, B


    def define_gamma_grid(self):
        r"""Defines the logarithmically spaced Lorentz factor grid for the Chang & Cooper 
        solver and the reduced electron Lorentz factor grid for post-solver quantities.

        The grid used as an inpit for the Chang and Cooper solver
        spans from :attr:`~agnpy.spectra.ParticleDistribution.gamma_min`
        to :attr:`~agnpy.spectra.ParticleDistribution.gamma_max` of the
        injected distribution, with :attr:`gamma_e_size` points.

        The post-solver quantities are defined at every second interior point of :attr:`gamma_e_cc` 
        to represent bin edges output by the Chang & Cooper scheme:

        .. math::
            \gamma = \gamma_{\rm cc}[1:-1:2].

        All electron distributions and derived spectra (e.g.
        :attr:`N_e_xg`) are defined on this grid.

        Returns
        -------
        :class:`~numpy.ndarray`
            1D Lorentz factor array used as the full solver grid,
            shape ``(gamma_e_size,)``.
        :class:`~numpy.ndarray`
            1D reduced Lorentz factor array, shape
            ``((gamma_e_size - 2) // 2,)``.
        """
        gamma_e_cc = np.logspace(
            np.log10(self._n_e.gamma_min),
            np.log10(self._n_e.gamma_max),
            self.gamma_e_size,
        )
        gamma_e = np.logspace(
            np.log10(5),
            np.log10(self._n_e.gamma_max),
            self.gamma_e_size,
        )[1:-1:2]
        return gamma_e_cc, gamma_e 

    def solve_electron_evolution(self):
        r"""Electron line density along the jet from the transport solver.

        Solves the electron continuity equation along the jet using the
        Chang & Cooper finite-difference scheme, taking into account
        radiative energy losses, adiabatic losses and particle escape:

        .. math::
        \frac{\partial N_e(\gamma, x)}{\partial x}
        = \frac{1}{c}\frac{\partial}{\partial \gamma}
        \left[
            \left(
                \dot{\gamma}_{\rm rad}
                + \dot{\gamma}_{\rm ad}
            \right) N_e(\gamma, x)
        \right]
        - \frac{N_e(\gamma, x)}{c\,t_{\rm esc}(x)},

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Two-dimensional array of electron line densities as a function
            of distance and Lorentz factor, with shape ``(N_x, N_gamma)``
            and units of cm\ :sup:`-1`.
        """
        self.cc_solver = ChangCooperSolver(self)
        return self.cc_solver.run()

    def check_timescale_consistency(self):
        """Warn once if any timescale is shorter than the spatial advection time."""

        warned_radiative = False
        warned_adiabatic = False
        warned_escape = False

        for i in range(len(self.x) - 1):

            # Spatial advection time for the interval x[i] -> x[i+1]
            dt = (self.x[i + 1] - self.x[i]) / c.cgs

            # Radiative cooling: check every gamma at each spatial position
            if not warned_radiative:
                if np.any(self.radiative_cooling_time[i, :] < dt):
                    print(
                        "WARNING: Radiative cooling time is shorter than the "
                        "spatial advection time at one or more grid points."
                    )
                    warned_radiative = True

            # Adiabatic cooling time
            if not warned_adiabatic:
                if self.adiabatic_loss_time[i] < dt:
                    print(
                        "WARNING: Adiabatic cooling time is shorter than the "
                        "spatial advection time at one or more grid points."
                    )
                    warned_adiabatic = True

            # Electron escape time
            if self.electron_escape and not warned_escape:
                t_escape = self.escape_coefficient * self.R_x[i] / c.cgs

                if t_escape < dt:
                    print(
                        "WARNING: Escape time is shorter than the spatial "
                        "advection time at one or more grid points."
                    )
                    warned_escape = True

            # Stop once every relevant warning has been issued
            if (
                warned_radiative
                and warned_adiabatic
                and (warned_escape or not self.electron_escape)
            ):
                break

    @property
    def radiative_cooling_time(self):
        r"""Radiative cooling time at each :math:`(x, \gamma)` grid point.

        Computed as:

        .. math::
            t_{\rm cool}(\gamma, x) = \frac{E}{\dot{E}_{\rm synch}}
            = \frac{3\,m_e c}{4\,\sigma_T\,U_B(x)\,\gamma}.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            2D array of cooling times, shape ``(N_x, N_gamma)``, in seconds.
        """
        t_cool = self.cc_solver.cooling_time()
        return t_cool
    
    @property
    def ssc_cooling_time(self):
        r"""SSC cooling time at each :math:`(x, \gamma)` grid point.

        Computed as:

        .. math::
            t_{\rm cool}(\gamma, x) = \frac{E}{\dot{E}_{\rm synch}}
            = \frac{3\,m_e c}{4\,\sigma_T\,U_B(x)\,\gamma}.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            2D array of cooling times, shape ``(N_x, N_gamma)``, in seconds.
        """
        t_cool_ssc = self.cc_solver.ssc_cooling_time()
        return t_cool_ssc
    
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
        t_cool_synch = self.cc_solver.synch_cooling_time()
        return t_cool_synch
        
    @property
    def adiabatic_loss_time(self):
        r"""
        The adiabatic cooling time is defined as the electron Lorentz factor
        divided by the magnitude of its adiabatic loss rate:
        
        .. math::
                t_{\rm ad}(x) = \frac{\gamma}{|\dot{\gamma}_{\rm ad}|}
                = \frac{3R(x)}{2c\tan\theta_{\rm open}},
        
        for a conical jet expanding with half-opening angle
        :math:`\theta_{\rm open}`, where :math:`R(x)` is the jet radius
        and :math:`c` is the speed of light. This timescale is independent
        of electron Lorentz factor for the assumed adiabatic loss rate.
        Returns
        -------
        :class:`~astropy.units.Quantity`
            One-dimensional array of adiabatic cooling times evaluated at
            the spatial grid points, with shape ``(N_x,)`` and units of
            seconds.    
        """
        t_adiabatic_loss = self.cc_solver.adiabatic_loss_time()
        return t_adiabatic_loss

    @property
    def escape_time(self):
        r"""Electron escape timescale along the jet.

        The escape timescale is defined as the escape coefficient
        multiplied by the light crossing time. 

        .. math::
            t_{\rm esc}(x) = \eta_{\rm esc}\frac{R(x)}{c},
    
        Returns
        -------
        :class:`~astropy.units.Quantity`
            One-dimensional array of escape times evaluated at the spatial
            grid points, with shape ``(N_x,)`` and units of seconds.
        """
        return self.escape_coefficient * self.R_x / c.cgs
    
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
        "Conical emission region initialized\n"
        "------------------------\n"
        f"R_0              : {self.R_0.cgs:.2e}\n"
        f"B_0              : {self.B_0:.2e}\n"
        f"L                : {self.L.cgs:.2e}\n"
        f"theta_open       : {self.theta_open.to('deg'):.2f}\n"
        f"V_c              : {self.V_c.cgs:.2e}\n"
        f"z                : {self.z:.3f}\n"
        f"d_L              : {self.d_L.cgs:.2e}\n"
        f"delta_D          : {self.delta_D:.2f}\n"
        f"Gamma            : {self.Gamma:.2f}\n"
        f"Beta             : {self.Beta:.4f}\n"
        f"theta_s          : {self.theta_s:.2f}\n"
        f"A_equi           : {self.A_equi:.2e}\n"
        f"gamma_e_size     : {self.gamma_e_size}\n"
        f"x_size           : {self.x_size}\n"
        f"electron_escape  : {self.electron_escape}\n"
        f"escape_coefficient: {self.escape_coefficient}\n"
        f"\nElectron distribution:\n"
        f"{self._n_e}"
        )
        return resume
