# module containing the synchrotron radiative process inside a conical jet
import numpy as np
import astropy.units as u
from astropy.constants import e, c, m_e
from ..utils.conversion import B_to_cgs, mec2, nu_obs_to_nu_fluid
from ..radiative_process import RadiativeProcess
from ..utils.synchrotron import Z, tau_to_attenuation

e = e.gauss
c=c.cgs
mec2=mec2.cgs

def nu_synch_peak(B, gamma, mass=m_e):
    """Critical synchrotron frequency (Dermer 2009 Eq. 7.19)."""
    B = B_to_cgs(B)
    nu_peak = (3 * e * B / (4 * np.pi * mass * c)) * np.power(gamma, 2)
    return nu_peak.to("Hz")

def eta(nu_fluid, nu_peak):
    """Calculates ration of frequency to critical synchrotron frequency
    for Conical Jet Model"""
    eta = nu_fluid / nu_peak
    return eta


class SynchrotronCone(RadiativeProcess):
    """
    Synchrotron emission from a conical jet.

    Produces:
    - F_nu       → observed flux density
    - F_nu(x)    → spatially resolved flux
    - nuF_nu      → SED
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
        """
        Compute synchrotron emission for a conical jet.

        Parameters
        ----------
        nu_fluid : array-like
            Frequencies in the comoving (fluid) frame, shape (1,1,N_nu)
        B_x : array-like
            Magnetic field along the jet, shape (N_x,1,1)
        x : array-like
            Spatial coordinate along the jet (cm), shape (N_x,)
        gamma_e : array-like
            Electron Lorentz factors, shape (1,N_gamma,1)
        N_e_xg : array-like
            Electron distribution, shape (N_x,N_gamma,1)

        Returns
        -------
        emission_x_nu : ndarray
            Differential emissivity along the jet (before x-integration),
            shape (N_x, N_nu)
        kernel_Z : ndarray
            Synchrotron kernel Z(η), shape (N_x, N_gamma, N_nu)
        L_nu_fluid : Quantity
            Total luminosity in fluid frame (integrated over x), [erg s⁻¹ Hz⁻¹]

        Notes
        -----
        Implements Eq. 7.44 from Dermer (2009).
        P_mu_fluid = ∫ dx  [ np.sqrt(3) e^3 B(x) / (m_e c^2) ] ∫ d(gamma) N_e(gamma,x) Z(eta = mu/mu_c)
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
        """
        Compute observed flux density F_nu and spatial flux F_nu(x).
        Returns
        -------
        F_nu : Quantity (N_nu)
        """
        self.nu_fluid = nu_obs_to_nu_fluid(nu_obs,self.emitter.z,self.emitter.delta_D)
        B_x_reshaped = self.emitter.B_x[:,None,None]
        gamma_e_reshaped = self.emitter.gamma_e[None,:,None]
        nu_fluid_reshaped =  self.nu_fluid[None,None,:]
        N_e_xg_reshaped = self.emitter.N_e_xg[:, :, None]
        L_x_nu_fluid, self.Z_eta, L_nu_fluid = self.evaluate_sed_flux(
            nu_fluid_reshaped,
            B_x_reshaped,
            self.emitter.x,
            gamma_e_reshaped,
            N_e_xg_reshaped,
            integrator=np.trapz,
        )
        L_nu_obs = L_nu_fluid * self.emitter.delta_D**3
        L_x_nu_obs = L_x_nu_fluid * self.emitter.delta_D**3
        self.F_nu_obs = L_nu_obs / (4 * np.pi * np.power(self.emitter.d_L, 2))
        self.F_x_nu_obs = L_x_nu_obs / (4 * np.pi * np.power(self.emitter.d_L, 2))
        self.sed = self.F_nu_obs * nu_obs
        if ssa:
            self.evaluate_attenuation()
            self.sed *= self.attenuation
        return self.sed.to("erg cm-2 s-1")
    
    def sed_flux_x(self, nu_obs, ssa=False):
        self.sed_x = self.F_x_nu_obs * nu_obs
        if ssa:
            self.evaluate_attenuation()
            self.sed_x *= tau_to_attenuation(self.kappa_x_nu.cgs.value)
        return self.sed_x.to("erg cm-3 s-1")

    def evaluate_attenuation(self):
        r"""
        Compute SSA absorption coefficient kappa_nu(x) for a conical jet (vectorized over frequencies).

        Parameters
        ----------
        nu_arr : ndarray
            Array of frequencies (Hz), shape (N_nu,)
        gamma_e : ndarray
            Electron Lorentz factors, shape (Ngamma,)
        N_e_xg : ndarray
            Electron distribution, shape (Nx, Ngamma)
        P_nu_gamma_x : ndarray
            Single-electron power, shape (Nx, Ngamma, N_nu)
        integrator : function
            Numerical integrator (default np.trapz)

        Returns
        -------
        kappa_nu : ndarray
            SSA opacity along the jet, shape (Nx, N_nu)

            ::math::
        kappa_nu(x) = - frac{1}{8 \pi m_e nu^2} \int d\gamma \, \gamma^2 \, frac{\partial}{\partial \gamma} \left[ frac{N_e(\gamma, x) \, P(nu, \gamma, x)}{\gamma^2} \right]
        """
        n_e_xg_reshaped = self.emitter.N_e_xg[:, :, None] / (
            np.pi * (self.emitter.R_x[:, None, None]) ** 2
        )
        gamma_e_reshaped = self.emitter.gamma_e[None, :, None]
        prefactor = 1 / (8 * np.pi * m_e.cgs * self.nu_fluid[None, :] ** 2)
        P_xg_nu = (
            np.sqrt(3)
            * (e) ** 3
            * self.emitter.B_x[:, None, None]
            * self.Z_eta
            / (m_e.cgs * c.cgs**2)
        ).cgs
        integrand = np.gradient(
            P_xg_nu * n_e_xg_reshaped / gamma_e_reshaped**2, self.emitter.gamma_e, axis=1
        )
        integrand *= gamma_e_reshaped**2
        self.kappa_x_nu = -prefactor * np.trapz(integrand, self.emitter.gamma_e, axis=1)
        tau_nu = np.trapz(self.kappa_x_nu, self.emitter.x, axis=0)
        self.attenuation = tau_to_attenuation(tau_nu)
        return self.attenuation

    def sed_peak_flux(self):
        """provided a grid of frequencies nu, returns the peak flux of the SED"""
        return self.sed.max()

    def sed_peak_nu(self, nu_obs):
        """provided a grid of frequencies nu, returns the frequency at which the
        SED peaks"""
        idx_max = self.sed_flux(nu_obs).argmax()
        return nu_obs[idx_max]
