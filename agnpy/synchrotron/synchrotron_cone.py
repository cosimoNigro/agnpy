# module containing the synchrotron radiative process inside a conical jet
import numpy as np
import astropy.units as u
from astropy.constants import e, c, m_e
from ..utils.conversion import B_to_cgs, mec2, nu_obs_to_nu_fluid
from ..radiative_process import RadiativeProcess
from ..utils.synchrotron import Z, tau_to_attenuation
from scipy.integrate import cumulative_trapezoid

e = e.gauss
c=c.cgs
mec2=mec2.cgs

def nu_synch_peak(B, gamma, mass=m_e):
    r"""Critical synchrotron frequency for a relativistic electron.

    Defined as the frequency at which a single electron of Lorentz factor
    :math:`\gamma` and mass :math:`m` in magnetic field :math:`B` emits the
    peak of its synchrotron power (Dermer & Menon 2009, Eq. 7.19):

    .. math::
        \nu_c(\gamma, x) = \frac{3\,e\,B(x)}{4\pi\,m\,c}\,\gamma^2.

    Parameters
    ----------
    B : :class:`~astropy.units.Quantity`
        Magnetic field strength. Converted to cgs Gauss internally.
        Shape must be broadcastable with ``gamma``.
    gamma : array-like or :class:`~astropy.units.Quantity`
        Electron Lorentz factors. Shape must be broadcastable with ``B``.
    mass : :class:`~astropy.constants.Constant`, optional
        Particle mass. Default is :data:`~astropy.constants.m_e` (electron).

    Returns
    -------
    :class:`~astropy.units.Quantity`
        Critical synchrotron frequency in Hz, with shape determined by
        broadcasting ``B`` and ``gamma``.
    """
    B = B_to_cgs(B)
    nu_peak = (3 * e * B / (4 * np.pi * mass * c)) * np.power(gamma, 2)
    return nu_peak.to("Hz")

def eta(nu_fluid, nu_peak):
    r"""Ratio of the observed frequency to the critical synchrotron frequency.

    Defined as the argument of the synchrotron kernel :math:`Z(\eta)`:

    .. math::
        \eta = \frac{\nu}{\nu_c(\gamma, x)}.

    Parameters
    ----------
    nu_fluid : :class:`~astropy.units.Quantity`
        Frequency in the fluid (comoving) frame, in Hz.
        Shape must be broadcastable with ``nu_peak``.
    nu_peak : :class:`~astropy.units.Quantity`
        Critical synchrotron frequency from :func:`nu_synch_peak`, in Hz.
        Shape must be broadcastable with ``nu_fluid``.

    Returns
    -------
    array-like
        Dimensionless ratio :math:`\eta = \nu / \nu_c`, with shape
        determined by broadcasting ``nu_fluid`` and ``nu_peak``.
    """
    eta = nu_fluid / nu_peak
    return eta


class SynchrotronCone(RadiativeProcess):
    r"""Synchrotron radiation from a uniform conical jet.

    Computes the synchrotron spectral energy distribution (SED) by
    integrating the single-electron synchrotron power weighted by the
    electron distribution :math:`N_e(\gamma, x)` over both Lorentz factor
    :math:`\gamma` and jet position :math:`x`.

    Optionally applies synchrotron self-absorption (SSA) by computing the
    fluid-frame opacity :math:`\kappa_\nu(x)` and integrating it along the
    line of sight through the jet following Potter & Cotter (2012), Eq. 29.

    Parameters
    ----------
    emitter : :class:`~agnpy.emission_regions.Cone`
        Conical jet emission region containing the electron distribution,
        magnetic field profile, and geometric parameters.
    ssa : bool, optional
        Whether to apply synchrotron self-absorption. Default is ``False``.
    integrator : callable, optional
        Numerical integration function. Default is :func:`numpy.trapz`.

    Attributes
    ----------
    nu_fluid : :class:`~astropy.units.Quantity`
        Fluid-frame frequencies set during the last call to
        :math:`sed_flux`, shape ``(N_nu,)``.
    Z_eta : :class:`~numpy.ndarray`
        Synchrotron kernel :math:`Z(\eta)` evaluated during the last call
        to :math:`sed_flux`, shape ``(N_x, N_gamma, N_nu)``.
    sed_nossa : :class:`~astropy.units.Quantity`
        Total SED without SSA from the last call to :math:`sed_flux`,
        shape ``(N_nu,)``, units ``erg cm⁻² s⁻¹``.
    sed_nossa_x : :class:`~astropy.units.Quantity`
        Spatially resolved SED per unit jet length without SSA,
        shape ``(N_x, N_nu)``, units ``erg cm⁻³ s⁻¹``.
    sed_ssa : :class:`~astropy.units.Quantity`
        SSA-attenuated total SED (set only when ``ssa=True``),
        shape ``(N_nu,)``, units ``erg cm⁻² s⁻¹``.
    sed_ssa_x : :class:`~astropy.units.Quantity`
        Spatially resolved SSA-attenuated SED (set only when ``ssa=True``),
        shape ``(N_x, N_nu)``, units ``erg cm⁻³ s⁻¹``.
    kappa_x_nu : :class:`~astropy.units.Quantity`
        SSA absorption coefficient (set only when ``ssa=True``),
        shape ``(N_x, N_nu)``, units ``cm⁻¹``.
    attenuation : :class:`~numpy.ndarray`
        SSA attenuation factor :math:`e^{-\tau}` (set only when ``ssa=True``),
        shape ``(N_x, N_nu)``.
    """

    def __init__(self, emitter, ssa=False, integrator=np.trapz):
        self.emitter = emitter
        self.ssa = ssa
        self.integrator = integrator
    
    @staticmethod
    def evaluate_sed_flux(
        nu_fluid,
        B_x,
        x,
        gamma_e,
        N_e_xg,
        integrator=np.trapz,
    ):
        r"""Compute the fluid-frame synchrotron luminosity for a conical jet.

        Evaluates Eq. 7.44 of Dermer & Menon (2009) by integrating the
        single-electron synchrotron power weighted by the electron
        distribution over :math:`\gamma` and then over :math:`x`:

        .. math::
            P_\nu^{\rm fluid} = \int dx\,\frac{\sqrt{3}\,e^3\,B(x)}{m_e c^2}
            \int d\gamma\,N_e(\gamma, x)\,Z\!\left(\frac{\nu}{\nu_c(\gamma,x)}\right).

        All array arguments should be pre-shaped for broadcasting before
        calling this method (see :meth:`sed_flux`).

        Parameters
        ----------
        nu_fluid : :class:`~astropy.units.Quantity`
            Frequencies in the fluid (comoving) frame, shape ``(1, 1, N_nu)``.
        B_x : :class:`~astropy.units.Quantity`
            Magnetic field profile along the jet axis, shape ``(N_x, 1, 1)``.
        x : :class:`~astropy.units.Quantity`
            Positions along the jet axis in cm, shape ``(N_x,)``.
        gamma_e : array-like
            Electron Lorentz factors, shape ``(1, N_gamma, 1)``.
        N_e_xg : :class:`~astropy.units.Quantity`
            Electron line density (electrons per cm of jet length) as a
            function of position and Lorentz factor, shape ``(N_x, N_gamma, 1)``.
        integrator : callable, optional
            Numerical integration function. Default is :func:`numpy.trapz`.

        Returns
        -------
        emission_x_nu : :class:`~astropy.units.Quantity`
            Synchrotron emissivity per unit jet length integrated over
            :math:`\gamma`, before the :math:`x`-integration.
            Shape ``(N_x, N_nu)``, units ``erg s⁻¹ Hz⁻¹ cm⁻¹``.
        Z_eta : :class:`~numpy.ndarray`
            Synchrotron kernel :math:`Z(\eta)` evaluated at all
            combinations of :math:`(x, \gamma, \nu)`.
            Shape ``(N_x, N_gamma, N_nu)``.
        P_synch : :class:`~astropy.units.Quantity`
            Total fluid-frame synchrotron luminosity integrated over
            :math:`\gamma` and :math:`x`.
            Shape ``(N_nu,)``, units ``erg s⁻¹ Hz⁻¹``.
        """
        nu_peak = nu_synch_peak(B_x, gamma_e)
        eta_ = eta(nu_fluid, nu_peak)
        Z_eta = Z(eta_)
        gamma_integral = integrator(B_x * N_e_xg * Z_eta, gamma_e, axis=1)
        prefactor = (np.sqrt(3) * (e) ** 3 / (mec2)).cgs
        emission = prefactor * gamma_integral
        P_synch = integrator(emission, x, axis=0)
        return emission, Z_eta, P_synch.to("erg Hz-1 s-1")


    def sed_flux(self, nu_obs,ssa=False):
        r"""Compute the observed synchrotron SED :math:`\nu F_\nu`.

        Transforms the fluid-frame luminosity to the observer frame using
        the Doppler factor and luminosity distance, optionally applying
        SSA attenuation via :meth:`evaluate_attenuation`.

        The observer-frame SED is:

        .. math::
            \nu F_\nu = \frac{\delta_D^3}{4\pi d_L^2}\,\nu_{\rm obs}\,
            \int dx\,\frac{\sqrt{3}\,e^3\,B(x)}{m_e c^2}
            \int d\gamma\,N_e(\gamma,x)\,Z(\eta)\,
            \cdot \begin{cases} 1 & \text{(no SSA)} \\
            e^{-\tau(x,\nu)} & \text{(SSA)} \end{cases}

        where :math:`\delta_D^3` accounts for one power of
        :math:`\delta_D` from the frequency transformation and two from
        the solid-angle and time-dilation boost of the luminosity.

        Side effects: sets :attr:`nu_fluid`, :attr:`Z_eta`,
        :attr:`sed_nossa`, :attr:`sed_nossa_x`, and (when ``ssa=True``)
        :attr:`sed_ssa`, :attr:`sed_ssa_x`, :attr:`kappa_x_nu`,
        :attr:`attenuation`.

        Parameters
        ----------
        nu_obs : :class:`~astropy.units.Quantity`
            Observed frequencies in Hz, shape ``(N_nu,)``.
        ssa : bool, optional
            Whether to apply synchrotron self-absorption. Default is
            ``False``. Overrides the instance-level ``self.ssa``.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Observed SED :math:`\nu F_\nu` in ``erg cm⁻² s⁻¹``,
            shape ``(N_nu,)``.
        """
        self.nu_fluid = nu_obs_to_nu_fluid(nu_obs,self.emitter.z,self.emitter.delta_D)
        B_x_reshaped = self.emitter.B_x[:,None,None]
        gamma_e_reshaped = self.emitter.gamma_e[None,:,None]
        nu_fluid_reshaped =  self.nu_fluid[None,None,:]
        N_e_xg_reshaped = self.emitter.N_e_xg[:, :, None]
        self.L_x_nu_fluid, self.Z_eta, L_nu_fluid = self.evaluate_sed_flux(
            nu_fluid_reshaped,
            B_x_reshaped,
            self.emitter.x,
            gamma_e_reshaped,
            N_e_xg_reshaped,
            integrator=np.trapz,
        )
        self.sed_nossa  = L_nu_fluid *  self.emitter.delta_D**3 / (4 * np.pi * np.power(self.emitter.d_L, 2)) * nu_obs
        self.sed_nossa_x = self.L_x_nu_fluid * self.emitter.delta_D**3 / (4 * np.pi * np.power(self.emitter.d_L, 2)) * nu_obs
        self.sed = self.sed_nossa
        if ssa:
            self.evaluate_attenuation()
            self.sed_ssa_x = self.sed_nossa_x * self.attenuation
            self.sed_ssa = np.trapz(self.sed_ssa_x,self.emitter.x,axis=0) 
            self.sed = self.sed_ssa
        return self.sed.to("erg cm-2 s-1")
    
    def sed_flux_x(self, ssa=False):
        r"""Return the spatially resolved SED per unit jet length.

        Provides the differential contribution :math:`d(\nu F_\nu)/dx`
        from each position along the jet axis, after the last call to
        :meth:`sed_flux`.

        Parameters
        ----------
        ssa : bool, optional
            If ``True``, return the SSA-attenuated spatially resolved SED
            :attr:`sed_ssa_x`. Requires that :meth:`sed_flux` was
            previously called with ``ssa=True``. Default is ``False``.

        Returns
        -------
        :class:`~astropy.units.Quantity`
            Spatially resolved SED in ``erg cm⁻³ s⁻¹``,
            shape ``(N_x, N_nu)``.
        """
        sed_x = self.sed_nossa_x.to("erg cm-3 s-1 ")
        if ssa:
            sed_x = self.sed_ssa_x
        return sed_x.to("erg cm-3 s-1 ")

    def evaluate_attenuation(self):
        r"""Compute the SSA optical depth and attenuation along the jet.

        Evaluates the fluid-frame SSA absorption coefficient
        :math:`\kappa_\nu(x)` using the Dermer & Menon (2009) formula
        (equation before 7.122):

        .. math::
            \kappa_\nu(x) = -\frac{1}{8\pi m_e \nu^2}
            \int d\gamma\,\gamma^2\,
            \frac{\partial}{\partial\gamma}
            \left[\frac{n_e(\gamma,x)\,P(\nu,\gamma,x)}{\gamma^2}\right],

        where :math:`n_e(\gamma,x) = N_e(\gamma,x) / (\pi R(x)^2)` is the
        volume number density and :math:`P(\nu,\gamma,x)` is the
        single-electron synchrotron power computed from :attr:`Z_eta`.

        The numerical derivative is taken in log-:math:`\gamma` space to
        preserve accuracy on the logarithmically spaced Lorentz factor grid:

        .. math::
            \frac{\partial f}{\partial\gamma}
            = \frac{1}{\gamma}\frac{\partial f}{\partial\ln\gamma}.

        The optical depth from position :math:`x` to the far end of the
        jet (Potter & Cotter 2012, Eq. 29) is then:

        .. math::
            \tau(\nu, x) = \Gamma^2
            \left(\frac{1}{\mu_s} - \beta\right)
            \int_x^L \kappa_\nu(x')\,dx',

        where the geometric factor accounts for the Lorentz transformation
        of the path length from the fluid frame to the observer frame.
        The attenuation at each position is :math:`e^{-\tau(\nu,x)}`.

        Side effects: sets :attr:`kappa_x_nu` and :attr:`attenuation`.

        Returns
        -------
        attenuation : :class:`~numpy.ndarray`
            SSA attenuation factor :math:`e^{-\tau(\nu,x)}`,
            shape ``(N_x, N_nu)``.
        """
        #Reshaping arrays for broadcasting
        n_e_xg_reshaped = self.emitter.N_e_xg[:, :, None] / (np.pi * (self.emitter.R_x[:, None, None]) ** 2)
        gamma_e_reshaped = self.emitter.gamma_e[None, :, None]

        prefactor = 1 / (8 * np.pi * m_e.cgs * self.nu_fluid[None, :] ** 2)
        P_xg_nu = (
            np.sqrt(3)
            * (e) ** 3
            * self.emitter.B_x[:, None, None]
            * self.Z_eta
            / (m_e.cgs * c.cgs**2)
        ).cgs

        integrand = P_xg_nu * np.gradient(
            n_e_xg_reshaped / gamma_e_reshaped**2, self.emitter.gamma_e, axis=1
        )
        integrand *= gamma_e_reshaped **2
        self.kappa_x_nu = -prefactor * np.trapz(integrand, self.emitter.gamma_e, axis=1)
        geom = self.emitter.Gamma**2 * (1 / self.emitter.mu_s - self.emitter.Beta)
        tau_x_nu = - (
        cumulative_trapezoid(self.kappa_x_nu[::-1],self.emitter.x[::-1],axis=0,initial=0,)[::-1])
        tau_x_nu *= geom
        self.attenuation = np.exp(-tau_x_nu)
        return self.attenuation

    def sed_peak_flux(self):
        """provided a grid of frequencies nu, returns the peak flux of the SED"""
        return self.sed.max()

    def sed_peak_nu(self, nu_obs):
        """provided a grid of frequencies nu, returns the frequency at which the
        SED peaks"""
        idx_max = self.sed_flux(nu_obs).argmax()
        return nu_obs[idx_max]
