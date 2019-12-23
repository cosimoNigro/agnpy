#cython: cdivision=True
from libc.math cimport pow, sqrt, exp, M_PI, log
import numpy as np
cimport cython

cdef double ME = 9.10938356e-28
cdef double C = 2.99792458e10
cdef double E = 4.80320425e-10
cdef double H = 6.62607004e-27
cdef double SIGMA_T = 6.65245872e-25

cdef double F_c(double q, double gamma_e):
    cdef double term_1 = 2 * q * log(q)
    cdef double term_2 = (1 + 2 * q) * (1 - q)
    cdef double term_3 = 1. / 2 * pow(gamma_e * q, 2) / (1 + gamma_e * q) * (1 - q)
    return term_1 + term_2 + term_3


cdef double simple_kernel(double gamma, double epsilon, double epsilon_s):
    cdef double gamma_e = 4 * gamma * epsilon
    cdef double q = (epsilon_s / gamma) / (gamma_e * (1 - epsilon_s / gamma))
    cdef double q_min = 1. / (4 * pow(gamma, 2))
    if (q_min <= q <= 1):
        return F_c(q, gamma_e)
    else:
        return 0

@cython.boundscheck(False)
def ssc_sed_emissivity(
    double[:] epsilon_syn, 
    double[:] syn_emissivity, # epsilon' * J'_{syn}(epsilon')
    double[:] epsilon, 
    double[:] gamma, 
    double[:] N_e, 
    double B, 
    double R_b
):
    """epsilon' * J'_{SSC}(epsilon')"""
    cdef Py_ssize_t epsilon_syn_size = epsilon_syn.shape[0]
    cdef Py_ssize_t epsilon_size = epsilon.shape[0]
    cdef Py_ssize_t gamma_size = gamma.shape[0]

    cdef double[:] prefactor = np.empty(epsilon_size, dtype=np.float64)
    cdef double[:, :, :] integrand = np.empty((gamma_size, epsilon_syn_size, epsilon_size), dtype=np.float64)
    cdef int i, j, k
    cdef double integrand_epsilon_syn, integrand_gamma
    
    for k in range(epsilon_size):
        prefactor[k] = 9 * SIGMA_T * pow(epsilon[k], 2) / (16 * M_PI * pow(R_b, 2)) 
        for j in range(epsilon_syn_size):
            for i in range(gamma_size):
                integrand_epsilon_syn = syn_emissivity[j] / pow(epsilon_syn[j], 3)
                integrand_gamma = N_e[i] / pow(gamma[i], 2) * simple_kernel(gamma[i], epsilon_syn[j], epsilon[k])
                integrand[i][j][k] = integrand_epsilon_syn * integrand_gamma

    integral_gamma = np.trapz(integrand, gamma, axis=0)
    integral_epsilon = np.trapz(integral_gamma, epsilon_syn, axis=0)
    return prefactor * integral_epsilon