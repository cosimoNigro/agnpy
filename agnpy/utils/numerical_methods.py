from math import gamma
import numpy as np
import astropy.units as u
from astropy.constants import c, sigma_T, m_e
from scipy.sparse import diags
from agnpy.utils.conversion import B_to_cgs
from ..utils.synchrotron import Z, nu_synch_peak
from astropy.constants import e
from ..utils.conversion import mec2

def gamma_loss_synch(gamma, B):
    u_B = ((B_to_cgs(B) ** 2) / (8 * np.pi)).to("erg cm-3")
    prefactor = (4 * c.cgs * sigma_T.cgs) / (3 * m_e.cgs * c.cgs * c.cgs)
    value = prefactor * (u_B) * np.power(gamma, 2)
    return -value.to("s-1")

def gamma_loss_ssc(gamma, u_ph):
    prefactor = (4 * c.cgs * sigma_T.cgs) / (3 * m_e.cgs * c.cgs * c.cgs)
    value = prefactor * (u_ph) * np.power(gamma, 2)
    return -value.to("s-1")

def gamma_loss(gamma, B, u_ph):
    u_B = ((B_to_cgs(B) ** 2) / (8 * np.pi)).to("erg cm-3")
    prefactor = (4 * c.cgs * sigma_T.cgs) / (3 * m_e.cgs * c.cgs * c.cgs)
    value = prefactor * (u_B +u_ph) * np.power(gamma, 2)
    return -value.to("s-1")

def local_synchrotron_energy_density(gamma, N_gamma, B, R, escape_coefficient, n_nu=60):
    """Single-zone, self-consistent synchrotron photon energy density
    produced by N_gamma(gamma) in a zone of field B and radius R,
    trapped for t_esc = escape_coefficient * R / c.
    Returns U_ph in erg/cm^3 (integrated over frequency).
    """
    nu_pk_max = nu_synch_peak(B, gamma[-1])
    nu_pk_min = nu_synch_peak(B, gamma[0])
    nu = np.logspace(
        np.log10(nu_pk_min.to_value("Hz")) - 2,
        np.log10(nu_pk_max.to_value("Hz")) + 2,
        n_nu,
    ) * u.Hz

    nu_peak = nu_synch_peak(B, gamma)[:, None]           #  ( N_gamma, 1 )
    eta_ = nu[None, :] / nu_peak                          # ( N_gamma, N_nu)
    Z_eta = Z(eta_)

    prefactor = (np.sqrt(3) * e.gauss**3 / mec2.cgs).cgs
    L_nu = prefactor * B_to_cgs(B) * np.trapz(N_gamma[:, None] * Z_eta, gamma, axis=0)  # (,N_nu) erg/s/Hz/cm

    t_esc = escape_coefficient * R / c.cgs
    u_nu = (L_nu * t_esc / (np.pi * R**2)).to("erg cm-3 Hz-1")

    return np.trapz(u_nu, nu).to("erg cm-3")    

class ChangCooperSolver:
    def __init__(
        self, gamma_e, x, R_0, B_0, theta_open, n_e, electron_escape, escape_coefficient
    ):
        """class implementing the Chang and Cooper scheme in Chaiberge and Ghisellini (1998):

        Parameters
        ----------
        gamma_min : float
            minimum Lorentz factor of the electorn distribution
        gamma_max : float
            maximum Lorentz factor of the electron distribution
        n_gamma : int
            number of steps in the Lorentz factor grid
        t_max_ev : `~astropy.Units.Quantity`
            maximum time of the evolution
        n_t : int
            number of time steps
        t_max_inj : `~astropy.Units.Quantity`
            maximum time of injection
        t_esc : `~astropy.Units.Quantity`
            escape time
        injection_dict : dict
            dictionary with injection specifications (spectrum and maximum injection time)
        """
        self.R_0 = R_0
        self.B_0 = B_0
        self.theta_open = theta_open
        self.electron_escape = electron_escape
        self.escape_coefficient = escape_coefficient
        _gamma = gamma_e
        # $\gamma_{i \pm 1/2}$ in Chiaberge et al. (1998)
        self.gamma_midpts = _gamma[::2]
        # $\delta gamma_i$ in Chiaberge et al. (1998)
        self.delta_gamma = self.gamma_midpts[:-1] - self.gamma_midpts[1:]
        # $\gamma_i$ in Chiaberge et al. (1998)
        self.gamma = _gamma[1:-1:2]
        self.n_e = n_e
        self.x = x.to("cm")
        self.t = x / c.cgs

    def run(self):
        """solve the temporal evolution, return the result at each step"""
        solutions = dict()
        N_prev = self.n_e
        self.t_cool = []
        self.t_cool_synch = []
        self.t_cool_ssc = []
        N_e_tg_list = [N_prev]  
        for i in range(len(self.t) - 1):
            if self.electron_escape:
                coeff = self.escape_coefficient
            else:
                coeff = np.inf
            elapsed_time = self.t[i]
            dt = (self.t[i + 1] - self.t[i]).to("s")
            R_t = self.R_0 + (self.t[i] * c.cgs) * np.tan(self.theta_open)
            B_t = self.B_0 * (self.R_0 /R_t)
            t_esc = coeff * R_t / c.cgs
            escape_term = (dt / t_esc).decompose().value
            U_ph_t = local_synchrotron_energy_density(self.gamma, N_prev, B_t, R_t, self.escape_coefficient)
            loss_rate = gamma_loss(self.gamma_midpts, B_t, U_ph_t)   # full array, matches gamma_midpts
            self.t_cool.append((- self.gamma / gamma_loss(self.gamma, B_t, U_ph_t)).to('s'))
            self.t_cool_synch.append((- self.gamma / gamma_loss_synch(self.gamma, B_t)).to('s'))
            self.t_cool_ssc.append((- self.gamma / gamma_loss_ssc(self.gamma, U_ph_t)).to('s'))
            loss_term = (dt * loss_rate[1:]).decompose().value / self.delta_gamma
            V2 = 1 + loss_term + escape_term
            V3 = -(dt * loss_rate[1:]).decompose().value / self.delta_gamma
            cc_matrix = diags([V2, V3], offsets=[0, 1]).toarray()
            N_next = np.linalg.solve(cc_matrix, N_prev)
            solutions[f"{elapsed_time}"] = N_next 
            N_prev = N_next              
        for key in solutions:
            N_e_tg_list.append(solutions[key])   
        N_e_xg = u.Quantity(N_e_tg_list)     
        return N_e_xg
    
    def synch_cooling_time(self):
        return u.Quantity(self.t_cool_synch)
    
    def ssc_cooling_time(self):
        return u.Quantity(self.t_cool_ssc)

    def cooling_time(self):
        return u.Quantity(self.t_cool)