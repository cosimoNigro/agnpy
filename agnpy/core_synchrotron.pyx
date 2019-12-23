#cython: cdivision=True
from libc.math cimport pow, sqrt, exp, M_PI, log
import numpy as np
cimport cython

cdef double ME = 9.10938356e-28
cdef double C = 2.99792458e10
cdef double E = 4.80320425e-10
cdef double H = 6.62607004e-27
cdef double SIGMA_T = 6.65245872e-25
 

cdef double R(double x):
    cdef double term_1_num = 1.808 * pow(x, 1. / 3)
    cdef double term_1_denom = sqrt(1 + 3.4 * pow(x, 2. / 3))
    cdef double term_2_num = 1 + 2.21 * pow(x, 2. / 3) + 0.347 * pow(x, 4. / 3)
    cdef double term_2_denom = 1 + 1.353 * pow(x, 2. / 3) + 0.217 * pow(x, 4. / 3)
    cdef double value = term_1_num / term_1_denom * term_2_num / term_2_denom * exp(-x)
    return value


@cython.boundscheck(False)
def com_sed_emissivity(
    double[:] epsilon, 
    double[:] gamma,
    double[:] N_e, 
    double B
):
    """epsilon' * J'_{syn}(epsilon')"""
    cdef Py_ssize_t gamma_size = gamma.shape[0]
    cdef Py_ssize_t epsilon_size = epsilon.shape[0]

    # memoryview
    cdef double[:] prefactor = np.empty(epsilon_size, dtype=np.float64)
    cdef double[:, :] integrand = np.empty((gamma_size, epsilon_size), dtype=np.float64) 

    cdef int i, j
    cdef double x_num, x_denum, x

    for j in range(epsilon_size):
        prefactor[j] = sqrt(3.) * epsilon[j] * pow(E, 3) * B / H
        for i in range(epsilon_size):
            x_num = 4 * M_PI * epsilon[j] * pow(ME, 2) * pow(C, 3)
            x_denom = 3 * E * B * H * pow(gamma[i], 2)
            x = x_num / x_denom
            integrand[i][j] = N_e[i] * R(x)

    return prefactor * np.trapz(integrand, gamma, axis=0)
