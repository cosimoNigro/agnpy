 import numpy as np
import astropy.units as u
from astropy.constants import e, h, c, m_e, sigma_T, mu0
from ..utils.math import axes_reshaper, gamma_e_to_integrate
from ..utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c_e, mec2
from ..radiative_process import RadiativeProcess

def Z(eta):
    """Eq. 7.45 in [Dermer2009]_, angle-averaged integrand of the radiated power, the
    approximation of this function, given in Eq. D7 of [Aharonian2010]_, is used.
    """
    term_1_num = 1.808 * np.power(eta, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(eta, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(eta, 2 / 3) + 0.347 * np.power(eta, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(eta, 2 / 3) + 0.217 * np.power(eta, 4 / 3)
    return term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-eta)

class SynchrotronCone(Synchrotron):
    """Class for synchrotron radiation computation

    Parameters
    ----------
    emitter : :class:`~agnpy.emission_region`
        emitting region and electron distribution
    ssa : bool
        whether or not to consider synchrotron self absorption (SSA).
        The absorption factor will be taken into account in
        :func:`~agnpy.synchrotron.Synchrotron.com_sed_emissivity`, in order to be
        propagated to :func:`~agnpy.synchrotron.Synchrotron.sed_luminosity` and
        :func:`~agnpy.synchrotron.Synchrotron.sed_flux`.
    integrator : func
        function to be used for integration (default = `np.trapz`)
	"""

    def __init__(self, emitter, ssa=False, integrator=np.trapz):
        self.emitter = emitter
        self.ssa = ssa
        self.integrator = integrator

    # Can be shifted to utility function?
    def nu_obs_to_nu_fluid(self, nu_obs):
        self.nu_fluid = nu_obs * (1+ self.emitter.z) / self.emitter.delta_D
        return self.nu_fluid 
    
    def flux_obs(self, nu_obs):
        self.nu_fluid = nu_obs * (1+ self.emitter.z) / self.emitter.delta_D
        B_x = self.emitter.B_x[:,None]                 #(N_x, 1)
        gamma_e = self.emitter.gamma_e[None,:]         #(1, N_gamma)
        self.nu_peak = ((3 / 2) *(e.gauss.decompose() * B_x / (2 * np.pi * m_e.cgs * c.cgs)) * np.power(gamma_e, 2)).cgs.to('Hz')
        nu_reshaped = self.nu_fluid[None, None, :].value             # shape (1, 1, N_nu)
        nu_reshaped_c = self.nu_peak[:, :, None].value                 # shape (N_x, N_gamma, 1)
        self.eta = nu_reshaped / nu_reshaped_c
        term_1_num = 1.808 * np.power(self.eta, 1 / 3)
        term_1_denom = np.sqrt(1 + 3.4 * np.power(self.eta, 2 / 3))
        term_2_num = 1 + 2.21 * np.power(self.eta, 2 / 3) + 0.347 * np.power(self.eta, 4 / 3)
        term_2_denom = 1 + 1.353 * np.power(self.eta, 2 / 3) + 0.217 * np.power(self.eta, 4 / 3)
        self.Z_eta = term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-self.eta)
        self.L_v_fluid = self.P_sync(self.Z_eta, 
                     self.emitter.B_x, 
                     self.emitter.x, 
                     self.emitter.gamma_e, 
                     self.emitter.N_e_xg,
                     integrator=np.trapz ).value * (u.erg/(u.s * u.Hz))
        self.L_v_obs = self.L_v_fluid * self.emitter.delta_D**3
        self.F_v_obs = self.L_v_obs / (4 * np.pi * np.power(self.emitter.d_L, 2))
        return self.F_v_obs
    
    def tau_to_attenuation(self, tau):
        """Converts the synchrotron self-absorption optical depth to an attenuation
        Eq. 7.122 in [DermerMenon2009]_."""
        u = 1 / 2 + np.exp(-tau) / tau - (1 - np.exp(-tau)) / np.power(tau, 2)
        return np.where(tau < 1e-3, 1, 3 * u / tau)

    def evaluate_attenuation(self):
        """
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
        \kappa_\nu(x) = - \frac{1}{8 \pi m_e \nu^2} \int d\gamma \, \gamma^2 \, \frac{\partial}{\partial \gamma} \left[ \frac{N_e(\gamma, x) \, P(\nu, \gamma, x)}{\gamma^2} \right]
        """
        N_e_reshaped = self.emitter.N_e_xg[:, :, None]/(np.pi * (self.emitter.R_x[:,None,None])**2)
        prefactor = 1 / (8 * np.pi * m_e.cgs* self.nu_fluid[None, :]**2)
        integrand = np.gradient( N_e_reshaped  / self.emitter.gamma_e[None, :, None]**2, self.emitter.gamma_e, axis=1)
        integrand *= self.emitter.gamma_e[None, :, None]**2  # multiply by γ^2 outside derivative
        P_xgv = (np.sqrt(3) * (e.gauss)**3 * self.emitter.B_x[:, None,None] * self.Z_eta / (m_e.cgs * c.cgs**2)).cgs
        # integrate over gamma
        kappa_xv = - prefactor * np.trapz(integrand * P_xgv, self.emitter.gamma_e, axis=1) 
        tau_nu = np.trapz(kappa_xv, self.emitter.x, axis=0)  # optical depth along the jet
        self.attenuation = self.tau_to_attenuation(tau_nu)      # reduces flux
        return self.attenuation  # shape (N_nu)

    def sed_flux(self, nu_obs, ssa=False):
        self.sed = self.F_v_obs * nu_obs
        if ssa:
            self.sed *= self.attenuation
        return self.sed.to('erg cm-2 s-1')
    
    @staticmethod
    def P_sync(
        Z_eta,         # Ratio b/w Observed Frequency Array and Critical Frequency 
        B_x,           # magnetic field array along x (Quantity)
        x,           # spatial coordinate array (Quantity)
        gamma_e,       # Electron Lorentz factor array (ndarray)
        N_e_xg,         # electron distribution array shape (N_x, N_gamma)
        integrator=np.trapz,
    ):
        """
    Synchrotron Power of a conical jet (erg s-1 Hz-1).

    Implements:
    P_mu_fluid = ∫ dx  [ √3 e^3 B(x) / (m_e c^2) ] ∫ d(gamma) N_e(gamma,x) Z(eta = mu/mu_c)
        """
    # --- compute R(eta) ---
        R_eta = Z_eta                           # (N_x, N_gamma, N_nu)
    # --- reshape electron distribution ---
        n_reshaped = N_e_xg[:, :, None]             # (N_x, N_gamma, 1)
    # --- integrate over gamma ---
        gamma_integral = integrator(
            n_reshaped * R_eta,
            gamma_e,
            axis=1
            )                                        # (N_x, N_nu)
    # --- prefactor √3 e³ B / (m c²) ---
        prefactor = (np.sqrt(3) * (e.gauss)**3 * B_x[:, None] / (mec2.cgs)).cgs
        emission = prefactor * gamma_integral  # (N_x, N_nu) 
        P_synch = integrator(emission, x, axis=0)
        return P_synch.to("erg Hz-1 s-1")
    
    def tot_power_synch(self, nu_obs):
        P_tot = np.trapz(self.P_sync, nu_obs)
        return P_tot.to("erg s-1")

"""
    def sed_peak_flux(self, nu):
"""#provided a grid of frequencies nu, returns the peak flux of the SED
"""
        return self.sed_flux(nu).max()

    def sed_peak_nu(self, nu):
        """#provided a grid of frequencies nu, returns the frequency at which the
        #SED peaks
"""
        idx_max = self.sed_flux(nu).argmax()
        return nu[idx_max]
    
"""