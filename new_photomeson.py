import astropy.units as u
from astropy.constants import m_e, m_p, c, e, h
from astropy.coordinates import Distance
from scipy.integrate import quad, dblquad, nquad, simps, trapz
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from agnpy.utils.math import axes_reshaper, gamma_p_to_integrate, eta_range
from agnpy.utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c_e
from agnpy.spectra import PowerLaw
from agnpy.photo_meson.kelners import phi_photon, phi_electron

print ('asd')
eta = 1
y_limit = 5
y = 10

phi_photon(eta, y_limit/10**y )
__all__ = ['PhotoMesonProduction']

mpc2 = (m_p * c ** 2).to('eV')
mec2 = (m_e * c ** 2).to('eV')

# 1) FUNCTIONS FOR INTEGRATION: fp, fph, PHI

n_p = PowerLaw(k=3000 * u.Unit("cm-3"),
        p=2.0,
        gamma_min=10,
        gamma_max=1e10,
        mass=m_p,
)

def soft_photon_dist(epsilon):
    T = 2.7 *u.K
    kT = (k_B * T).to('eV').value
    c1 = c.to('cm s-1').value
    h1 = h.to('eV s').value
    norm = 8*np.pi/(h1**3*c1**3)
    num = (mpc2.value *epsilon) ** 2
    denom = np.exp(mpc2.value * epsilon / kT) - 1
    return norm * (num / denom) * u.Unit('cm-3')

def H(eta,y):

    return (1/ (10**y)**2 *
        particle_distribution(10**y).value * \
        soft_photon_dist((eta /  (4*10**y))).value*\
        phi_photon(eta, y_limit/10**y )
        * 10**y * np.log(10)
    )

# Fixed value of the photon produced: E = 0.5 * E_star
E_star = 3 * 1e20 * u.eV
epsilon_limit = E_star / mpc2
y_max = 15

gamma = np.logspace(1,8,200)
y = np.log10(gamma)
eta = np.linspace(1,30,200)

for i in eta:
    H_integrand = 1/ (10**y)**2 * n_p(10**y).value * soft_photon_dist((eta /  (4*10**y))).value *  phi_photon(_eta, 10**y_limit/10**y) * (10**y) * np.log(10)
    H = (1 / 4) * (mpc2.value) * integrator(H_integrand, 10**y, axis=0)

dNdE = integrator(H, _eta, axis=0)
