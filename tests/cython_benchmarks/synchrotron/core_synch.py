import numpy as np
import numba

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


def F_c(q, gamma_e):
    term_1 = 2 * q * np.log(q)
    term_2 = (1 + 2 * q) * (1 - q)
    term_3 = 1 / 2 * np.power(gamma_e * q, 2) / (1 + gamma_e * q) * (1 - q)
    return term_1 + term_2 + term_3


def simple_kernel(gamma, epsilon, epsilon_s):
    gamma_e = 4 * gamma * epsilon
    q = (epsilon_s / gamma) / (gamma_e * (1 - epsilon_s / gamma))
    q_min = 1 / (4 * np.power(gamma, 2))
    values = F_c(q, gamma_e)
    # apply the Heaviside function for q in (6.74)
    condition = (q_min <= q) * (q <= 1)
    values[~condition] = 0
    return values


def ssc_sed_emissivity(epsilon_syn, com_syn_emissivity, epsilon, gamma, N_e, B, R_b):
    _gamma = gamma.reshape(gamma.size, 1, 1)
    _N_e = N_e.reshape(N_e.size, 1, 1)
    _epsilon_syn = epsilon_syn.reshape(1, epsilon_syn.size, 1)
    _com_syn_emissivity = com_syn_emissivity.reshape(1, com_syn_emissivity.size, 1)
    _epsilon = epsilon.reshape(1, 1, epsilon.size)
    _kernel = simple_kernel(_gamma, _epsilon_syn, _epsilon)

    integrand_epsilon = _com_syn_emissivity / np.power(_epsilon_syn, 3)
    integrand_gamma = _N_e / np.power(_gamma, 2) * _kernel
    integrand = integrand_epsilon * integrand_gamma

    integral_gamma = np.trapz(integrand, gamma, axis=0)
    integral_epsilon = np.trapz(integral_gamma, epsilon_syn, axis=0)

    prefactor_num = 9 * SIGMA_T * np.power(epsilon, 2)
    prefactor_denom = 16 * np.pi * np.power(R_b, 2)
    return prefactor_num / prefactor_denom * integral_epsilon
