import numpy as np

ME = 9.10938356e-28
C = 2.99792458e10
E = 4.80320425e-10
H = 6.62607004e-27
SIGMA_T = 6.65245872e-25

def R(x):
    term_1_num = 1.808 * np.power(x, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(x, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(x, 2 / 3) + 0.347 * np.power(x, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(x, 2 / 3) + 0.217 * np.power(x, 4 / 3)
    value = term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-x)
    return value

def com_sed_emissivity(epsilon, gamma, N_e, B):
    prefactor = np.sqrt(3) * epsilon * np.power(E, 3) * B / H
    _gamma = gamma.reshape(gamma.size, 1)
    _N_e = N_e.reshape(N_e.size, 1)
    _epsilon = epsilon.reshape(1, epsilon.size)
    x_num = 4 * np.pi * _epsilon * np.power(ME, 2) * np.power(C, 3)
    x_denom = 3 * E * B * H * np.power(_gamma, 2)
    x = x_num / x_denom
    integrand = _N_e * R(x)
    return prefactor * np.trapz(integrand, gamma, axis=0)