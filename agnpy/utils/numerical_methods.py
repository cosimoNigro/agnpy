import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from astropy.constants import c, sigma_T, m_e, e, mu0
from scipy.sparse import diags
from matplotlib import cycler, rcParams
from agnpy.utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c_e, mec2


def gamma_loss(gamma, B):
    u_B = ((B_to_cgs(B) ** 2) / (8 * np.pi)).to("erg cm-3")
    prefactor = (4 * c.cgs * sigma_T.cgs) / (3 * m_e.cgs * c.cgs * c.cgs)
    value = prefactor * (u_B) * np.power(gamma, 2)
    return -value.to("s-1")


class ChangCooperSolver:
    def __init__(
        self, gamma_e, x, R_o, B_o, theta_open, n_e, electron_escape, escape_coefficient
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
        self.B_o = B_o
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
        N_prev = self.n_e
        N_e_tg = [self.n_e]
        for i in range(len(self.t) - 1):
            if self.electron_escape:
                coeff = self.escape_coefficient
            else:
                coeff = np.inf
            dt = (self.t[i + 1] - self.t[i]).to("s")
            B_t = self.B_o * (
                self.R_o / (self.R_o + (self.t[i] * c.cgs) * np.tan(self.theta_open))
            )
            R_t = self.R_o + (self.t[i] * c.cgs) * np.tan(self.theta_open)
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
            N_e_tg.append(N_next)
        N_e_xg = u.Quantity(N_e_tg)
        return N_e_xg
