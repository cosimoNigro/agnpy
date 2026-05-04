import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from astropy.constants import c, sigma_T, m_e, e, mu0
from scipy.sparse import diags
from matplotlib import cycler, rcParams
from agnpy.utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c_e, mec2

def gamma_loss(gamma):
    B = 0.363 * u.G
    u_B = ((B_to_cgs(B)**2)/(8*np.pi)).to('erg cm-3')
    prefactor = (4 * c.cgs * sigma_T.cgs) / (3 * m_e.cgs * c.cgs * c.cgs )
    value = prefactor * (u_B) * np.power(gamma, 2)
    return -value.to("s-1")

class ChangCooperSolver:
    def __init__(
        self,
        gamma_e, 
        x,
        R_o,
        theta_open,
        n_e

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
        self.R_o = R_o
        self.theta_open = theta_open
        _gamma = gamma_e
        # $\gamma_{i \pm 1/2}$ in Chiaberge et al. (1998)
        self.gamma_midpts = _gamma[::2]
        # $\delta gamma_i$ in Chiaberge et al. (1998)
        self.delta_gamma = self.gamma_midpts[:-1] - self.gamma_midpts[1:]
        # $\gamma_i$ in Chiaberge et al. (1998)
        self.gamma = _gamma[1:-1:2]
        self.n_e = n_e
        self.x = x.to('cm')
        self.t = x/c.cgs

    @property
    def gamma_dot_midpts(self):
        return gamma_loss(self.gamma_midpts)

    def run(self):
        """solve the temporal evolution, return the result at each step"""
        solutions = dict()
        N_prev = self.n_e(self.gamma)
        for i in range(len(self.t) - 1):
            elapsed_time = self.t[i]
            dt = (self.t[i+1] - self.t[i]).to('s')
            loss_term = (dt * self.gamma_dot_midpts[:-1]).decompose().value / self.delta_gamma
            V2 = 1 + loss_term
            V3 = -(dt * self.gamma_dot_midpts[1:]).decompose().value / self.delta_gamma
            cc_matrix = diags([V2, V3], offsets=[0, 1]).toarray()
            N_next = np.linalg.solve(cc_matrix, N_prev)
            R_next = self.R_o + c.cgs*self.t[i+1]*np.tan(self.theta_open)
            area_slice = np.pi * R_next**2
            solutions[f"{elapsed_time}"] = N_next * u.Unit("cm-3") *area_slice
            N_prev = N_next
        N_e_tg_list = []
        for key in solutions:
            N_e_tg_list.append(solutions[key])
        N_e_xg = u.Quantity(N_e_tg_list)
        return N_e_xg
    
