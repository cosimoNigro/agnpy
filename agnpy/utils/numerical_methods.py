import numpy as np
import astropy.units as u
from astropy.constants import c, sigma_T, m_e
from scipy.sparse import diags
from agnpy.utils.conversion import B_to_cgs


def gamma_loss(gamma, B):
    u_B = ((B_to_cgs(B) ** 2) / (8 * np.pi)).to("erg cm-3")
    prefactor = (4 * c.cgs * sigma_T.cgs) / (3 * m_e.cgs * c.cgs * c.cgs)
    value = prefactor * (u_B) * np.power(gamma, 2)
    return -value.to("s-1")

def cooling_time( gamma, B):
    gamma_loss_rate = gamma_loss(gamma,B)
    t_cool = - gamma / gamma_loss_rate
    return t_cool.to('s')

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
            loss_term = (
                dt * gamma_loss(self.gamma_midpts, B_t)[1:]
            ).decompose().value / self.delta_gamma
            V2 = 1 + loss_term + escape_term
            V3 = (
                -(dt * gamma_loss(self.gamma_midpts, B_t)[1:]).decompose().value
                / self.delta_gamma
            )
            cc_matrix = diags([V2, V3], offsets=[0, 1]).toarray()
            N_next = np.linalg.solve(cc_matrix, N_prev)
            solutions[f"{elapsed_time}"] = N_next 
            N_prev = N_next              
        for key in solutions:
            N_e_tg_list.append(solutions[key])   
        N_e_xg = u.Quantity(N_e_tg_list)     
        return N_e_xg
