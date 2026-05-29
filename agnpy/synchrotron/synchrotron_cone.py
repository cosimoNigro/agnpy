import numpy as np
from agnpy.emission_regions import Cone, Blob
import astropy.units as u
from astropy.constants import e, h, c, m_e, sigma_T, mu0
from ..utils.math import axes_reshaper, gamma_e_to_integrate
from ..utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c_e, mec2
from ..radiative_process import RadiativeProcess
e = e.gauss

# ============================================================
# Utility Functions
# ============================================================

def nu_obs_to_nu_fluid(nu_obs, z, delta_D):
    """Convert observed frequency to comoving (fluid) frame."""
    nu_fluid = nu_obs * (1+ z) / delta_D
    return nu_fluid 

def nu_synch_peak(B, gamma, mass=m_e):
    """Critical synchrotron frequency (Dermer 2009 Eq. 7.19)."""
    B = B_to_cgs(B)
    nu_peak = (3 * e * B / (4 * np.pi * mass * c)) * np.power(gamma, 2)
    return nu_peak.to("Hz")

def Z(eta):
    """Synchrotron kernel approximation (Aharonian 2010, Eq. D7)."""
    term_1_num = 1.808 * np.power(eta, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(eta, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(eta, 2 / 3) + 0.347 * np.power(eta, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(eta, 2 / 3) + 0.217 * np.power(eta, 4 / 3)
    return term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-eta)

def tau_to_attenuation(tau):
    """Convert SSA optical depth → attenuation factor.
    Eq. 7.122 in [DermerMenon2009]_."""
    u = 0.5 + np.exp(-tau) / tau - (1 - np.exp(-tau)) / np.power(tau, 2)
    return np.where(tau < 1e-3, 1, 3 * u / tau)

def epsilon_B(B):
    r""":math:`\epsilon_B`, Eq. 7.21 [DermerMenon2009]_"""
    return (B / B_cr).to_value("")

def single_particle_synch_power(B_cgs, epsilon, gamma, mass=m_e):
    """angle-averaged synchrotron power for a single particle of mass m_e,
    to be folded with the electron distribution
    """
    eta = calc_eta(B_cgs, epsilon, gamma, mass)
    prefactor = np.sqrt(3) * np.power(e, 3) * B_cgs / h
    return prefactor * Z(eta)

def calc_eta(B_cgs, epsilon, gamma, mass=m_e):
    """ratio of the frequency to the critical synchrotron frequency from
    Eq. 7.34 in [DermerMenon2009]_, argument of R(x),
    note B has to be in cgs Gauss units"""
    eta = (
        4
        * np.pi
        * epsilon
        * np.power(mass, 2)
        * np.power(c, 3)
        / (3 * e * B_cgs * h * np.power(gamma, 2))
    )
    return eta.value

def eta(nu_fluid, nu_peak):
    """Calculates ration of frequency to critical synchrotron frequency
    for Conical Jet Model"""
    eta = nu_fluid / nu_peak
    return eta

def P_sync(
        nu_fluid,         # Ratio b/w Observed Frequency Array and Critical Frequency 
        B_x,           # magnetic field array along x (Quantity)
        x,           # spatial coordinate array (Quantity)
        gamma_e,       # Electron Lorentz factor array (ndarray)
        N_e_xg,         # electron distribution array shape (N_x, N_gamma)
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
    P_mu_fluid = ∫ dx  [ √3 e^3 B(x) / (m_e c^2) ] ∫ d(gamma) N_e(gamma,x) Z(eta = mu/mu_c)
    """   

    # Calculate Synchrotron Peak
    nu_peak = nu_synch_peak(B_x, gamma_e)
    #Calculate Eta
    eta_ = eta(nu_fluid,nu_peak)
    Z_eta = Z(eta_)
    # --- integrate over gamma ---
    gamma_integral = integrator(
            B_x * N_e_xg * Z_eta,
            gamma_e,
            axis=1
            )                                        
    # --- prefactor √3 e³ / (m c²) ---
    prefactor = (np.sqrt(3) * (e)**3 / (mec2.cgs)).cgs
    emission = prefactor * gamma_integral  
    P_synch = integrator(emission, x, axis=0)
    return nu_peak, emission, Z_eta, P_synch.to("erg Hz-1 s-1")


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
    
    def flux_obs(self, nu_obs):
        """
        Compute observed flux density F_nu and spatial flux F_nu(x).
        Returns
        -------
        F_nu : Quantity (N_nu)
        """
        self.nu_fluid = nu_obs * (1+ self.emitter.z) / self.emitter.delta_D
        nu_fluid_reshaped = self.nu_fluid[None,None,:]
        N_e_xg_reshaped = self.emitter.N_e_xg[:,:,None]
        B_x_reshaped = self.emitter.B_x[:,None,None]                              #(N_x, 1)
        gamma_e_reshaped = self.emitter.gamma_e[None,:,None]               #(1, N_gamma)
        nu_p, L_xv_fluid, self.Z_eta, L_v_fluid = P_sync(nu_fluid_reshaped, 
                     B_x_reshaped, 
                     self.emitter.x, 
                     gamma_e_reshaped, 
                     N_e_xg_reshaped,
                     integrator=np.trapz) #.value * (u.erg/(u.s * u.Hz))
        self.nu_p = nu_p
        L_v_obs = L_v_fluid * self.emitter.delta_D**3
        L_xv_obs = L_xv_fluid * self.emitter.delta_D**3
        self.F_v_obs = L_v_obs / (4 * np.pi * np.power(self.emitter.d_L, 2))
        self.F_xv_obs = L_xv_obs / (4 * np.pi * np.power(self.emitter.d_L, 2))
        return self.F_v_obs
    
    def Flux_xv(self):
        return self.F_xv_obs
    
    def Bessel(self):
        return self.Z_eta
    
    def nu_peak(self):
        return self.nu_p

# ============================================================
# SSA
# ============================================================

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
        n_e_xg_reshaped = self.emitter.N_e_xg[:, :, None]/(np.pi * (self.emitter.R_x[:,None,None])**2)
        gamma_e_reshaped = self.emitter.gamma_e[None,:, None]
        prefactor = 1 / (8 * np.pi * m_e.cgs* self.nu_fluid[None, :]**2)
        P_xgv = (np.sqrt(3) * (e)**3 * self.emitter.B_x[:, None,None] * self.Z_eta / (m_e.cgs * c.cgs**2)).cgs
        integrand = np.gradient(P_xgv* n_e_xg_reshaped  / gamma_e_reshaped**2, self.emitter.gamma_e, axis=1)
        integrand *= gamma_e_reshaped**2  # multiply by γ^2 outside derivative
        # integrate over gamma
        self.kappa_xv = - prefactor * np.trapz(integrand , self.emitter.gamma_e, axis=1) 
        tau_nu = np.trapz(self.kappa_xv,self.emitter.x, axis=0)  # optical depth along the jet
        self.attenuation = tau_to_attenuation(tau_nu)      # reduces flux
        return self.attenuation  # shape (N_nu)

    def sed_flux(self, nu_obs, ssa=False):
        self.sed = self.F_v_obs * nu_obs
        if ssa:
            self.sed *= self.attenuation
        return self.sed.to('erg cm-2 s-1')
    
    def sed_spatial(self, nu_obs, ssa=False):
        self.sed_x = self.F_xv_obs * nu_obs
        if ssa:
            self.sed_x *= tau_to_attenuation(self.kappa_xv.cgs.value)
        return self.sed_x.to('erg cm-3 s-1')
    
    def sed_peak_flux(self):
        """#provided a grid of frequencies nu, returns the peak flux of the SED
        """
        return self.sed.max()

    def sed_peak_nu(self,nu_obs):
        """#provided a grid of frequencies nu, returns the frequency at which the
        #SED peaks"""
        idx_max = self.sed_flux(nu_obs).argmax()
        return nu_obs[idx_max]
    