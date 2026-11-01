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

def local_synchrotron_energy_density(gamma, N_gamma, B, R, n_nu=200):
    """Single-zone, self-consistent synchrotron photon energy density
    produced by N_gamma(gamma) in a zone of field B and radius R,
    trapped for t_stay = 2 * R / c.
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

    t_stay = 2 * R / c.cgs
    u_nu = (L_nu * t_stay / (np.pi * R**2 )).to("erg cm-3 Hz-1")

    return np.trapz(u_nu, nu).to("erg cm-3")    
    
class ChangCooperSolver:
    """class implementing the Chang and Cooper scheme in Chaiberge and Ghisellini (1998):"""   

    def __init__(self, emitter):
        self.emitter = emitter
        #Extend the gamma grid to lower values to allow cooled electrons to accumlate
        self.gamma_min = 5 
        self._gamma = np.logspace(np.log10(self.gamma_min),np.log10(self.emitter._n_e.gamma_max), self.emitter.gamma_e_size)
        # $\gamma_{i \pm 1/2}$ in Chiaberge et al. (1998)
        self.gamma_edges = self._gamma[::2]
        # $\delta gamma_i$ in Chiaberge et al. (1998)
        self.delta_gamma = self.gamma_edges[:-1] - self.gamma_edges[1:]
        # $\gamma_i$ in Chiaberge et al. (1998)
        self.gamma_center = self._gamma[1:-1:2]
        self.t_max_ev = (self.emitter.L/c).to("s") 
        self.x_res = self.emitter.x_resolution
        self.x0 = self.emitter.x_min
        self.t0 = (self.x0 / c.cgs).to('s')

    def run(self):
        solutions = dict()
        N_prev = self.emitter._n_e(self.gamma_center)* np.pi* self.emitter.R_0**2
        self.t_radiative_loss = []
        self.t_synch_loss = []
        self.t_ssc_loss = []
        self.t_adiabatic_loss = []
        self.t_cooling_loss = []
        N_e_tg_list = [N_prev]  
        elapsed_time = self.t0
        B = self.emitter.B_0
        R = self.emitter.R_0
        self.x = [self.x0]
        self.R_x = [self.emitter.R_0]
        self.B_x = [self.emitter.B_0]

        while elapsed_time < self.t_max_ev:
            U_ph = local_synchrotron_energy_density(self.gamma_center, N_prev, B, R)
            #Calculate losses
            gamma_dot_synch = gamma_loss_synch(self.gamma_center, B)
            gamma_dot_ssc = gamma_loss_ssc(self.gamma_center, U_ph)

            #Calculate time-scales
            self.t_synch_cooling = - self.gamma_center /gamma_dot_synch
            self.t_synch_loss.append(self.t_synch_cooling)
            self.t_ssc_cooling = -self.gamma_center /gamma_dot_ssc
            self.t_ssc_loss.append(self.t_ssc_cooling)
            self.t_radiative_cooling = self.t_synch_cooling*self.t_ssc_cooling/(self.t_ssc_cooling + self.t_synch_cooling)
            self.t_radiative_loss.append(self.t_radiative_cooling)
        
            if self.emitter.adiabatic_cooling:
                self.t_adiabatic = (3*R)/(2*c.cgs*np.tan(self.emitter.theta_open))
                self.t_cooling = self.t_radiative_cooling*self.t_adiabatic/(self.t_radiative_cooling+self.t_adiabatic)
                adiabatic_loss_lower_bin = (- 2 * c.cgs * np.tan(self.emitter.theta_open) * self.gamma_edges[:-1] / (3 * R))
                adiabatic_loss_higher_bin = (- 2 * c.cgs * np.tan(self.emitter.theta_open) * self.gamma_edges[1:] / (3 * R))
                self.t_adiabatic_loss.append(self.t_adiabatic)
            else:
                self.t_cooling = self.t_radiative_cooling
                adiabatic_loss_lower_bin = 0
                adiabatic_loss_higher_bin = 0
            self.t_cooling_loss.append(self.t_cooling)


            loss_rate_lower_bin = gamma_loss(
                self.gamma_edges[:-1], B, U_ph
                ) + adiabatic_loss_lower_bin
            
            loss_rate_lower_bin[0] = 0.0 #Imposing zero flux condition on the gamma minimum
            
            loss_rate_upper_bin = gamma_loss(
                self.gamma_edges[1:], B, U_ph
                ) + adiabatic_loss_higher_bin
            
            loss_rate_upper_bin[-1] = 0.0 #Imposing zero incoming flux from higher gammas. 
  
            dt = self.x_res * np.min(self.t_cooling) #Constraining the time-step based on the minimum cooling time of the electrons.  
            dt = min(dt,self.t_max_ev - elapsed_time)
            dx = c.cgs * dt

            escape_term = (
                dt / (self.emitter.escape_coefficient * (R + dx * np.tan(self.emitter.theta_open))/ c.cgs))
            
            V2 = 1 + ((dt * loss_rate_lower_bin).decompose().value / self.delta_gamma) + escape_term 
            V3 = -(dt * loss_rate_upper_bin).decompose().value / self.delta_gamma

            cc_matrix = diags([V2, V3], offsets=[0, 1]).toarray()
            N_next = np.linalg.solve(cc_matrix, N_prev)
        
            
            solutions[f"{elapsed_time}"] = N_next 

            self.x.append(self.x[-1] + dx)
            self.R_x.append(R + dx * np.tan(self.emitter.theta_open))
            B = B * R / self.R_x[-1]
            self.B_x.append(B)
            R = R + dx * np.tan(self.emitter.theta_open)

            elapsed_time = elapsed_time + dt
            N_prev = N_next      
        for key in solutions:
            N_e_tg_list.append(solutions[key])   
        N_e_xg = u.Quantity(N_e_tg_list)     
        self.x = u.Quantity(self.x)
        self.R_x = u.Quantity(self.R_x)
        self.B_x = u.Quantity(self.B_x)
        # Convert timescale lists into Astropy Quantity arrays
        self.t_radiative_loss = u.Quantity(self.t_radiative_loss)
        self.t_synch_loss = u.Quantity(self.t_synch_loss)
        self.t_ssc_loss = u.Quantity(self.t_ssc_loss)
        if self.emitter.adiabatic_cooling:
            self.t_adiabatic_loss = u.Quantity(self.t_adiabatic_loss)
        else:
            self.t_adiabatic_loss = np.array([]) * u.s
        self.t_cooling_loss = u.Quantity(self.t_cooling_loss)
        return N_e_xg
    
    def calculate_spatial_grid_quantities(self):
        return self.x, self.R_x, self.B_x
    
    def calculate_timescales(self):
        return self.t_radiative_loss, self.t_synch_loss, self.t_ssc_loss, self.t_adiabatic_loss, self.t_cooling_loss

