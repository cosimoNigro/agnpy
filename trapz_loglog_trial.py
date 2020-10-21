import warnings
import numpy as np
import astropy.units as u
from astropy.constants import h, e, m_e, c, sigma_T, M_sun, G
from agnpy.emission_regions import Blob
from agnpy.synchrotron import R, epsilon_equivalency
from agnpy.compton import compton_kernel
from agnpy.targets import PointSourceBehindJet, SSDisk
from agnpy.utils.math import trapz_loglog, power
import matplotlib.pyplot as plt

e = e.gauss

# a simple test with a straight line in log-log scale
print("-> test with line in log-log space")


def line_loglog(x, m, n):
    """a straight line in loglog-space"""
    return x ** m * np.e ** n


def integral_line_loglog(x_min, x_max, m, n):
    """analytical integral of the line in log-log space"""
    f_low = line_loglog(x_min, m + 1, n) / (m + 1)
    f_up = line_loglog(x_max, m + 1, n) / (m + 1)
    return f_up - f_low


m = 1.5
n = -2.0
x = np.logspace(2, 5)
y = line_loglog(x, m, n)

analytical_integral = integral_line_loglog(x[0], x[-1], m, n)
trapz_loglog_integral = trapz_loglog(y, x)
np_trapz_integral = np.trapz(y, x)

print(f"analyitcal integral: {analytical_integral:e}")
print(f"trapz_loglog integral: {analytical_integral:e}")
print(f"np.trapz integral: {analytical_integral:e}")

# a test with sycnhrotron radiation
print("-> test with synchrotron radiation")

blob = Blob()
nu = np.logspace(9, 20, 20) * u.Hz

# check the blob
print(blob)


def sed_synch(nu, blob, integration):
    """compute the synchrotron SED"""
    epsilon = nu.to("", equivalencies=epsilon_equivalency)
    # correct epsilon to the jet comoving frame
    epsilon_prime = (1 + blob.z) * epsilon / blob.delta_D
    # electrond distribution lorentz factor
    gamma = blob.gamma
    N_e = blob.N_e(gamma)
    prefactor = np.sqrt(3) * epsilon * np.power(e, 3) * blob.B_cgs / h
    # for multidimensional integration
    # axis 0: electrons gamma
    # axis 1: photons epsilon
    # arrays starting with _ are multidimensional and used for integration
    _gamma = np.reshape(gamma, (gamma.size, 1))
    _N_e = np.reshape(N_e, (N_e.size, 1))
    _epsilon = np.reshape(epsilon, (1, epsilon.size))
    x_num = 4 * np.pi * _epsilon * np.power(m_e, 2) * np.power(c, 3)
    x_denom = 3 * e * blob.B_cgs * h * np.power(_gamma, 2)
    x = (x_num / x_denom).to_value("")
    integrand = _N_e * R(x)
    integral = integration(integrand, gamma, axis=0)
    emissivity = (prefactor * integral).to("erg s-1")
    sed_conversion = np.power(blob.delta_D, 4) / (4 * np.pi * np.power(blob.d_L, 2))
    return (sed_conversion * emissivity).to("erg cm-2 s-1")


sed_trapz = sed_synch(nu, blob, np.trapz)
sed_trapz_loglog = sed_synch(nu, blob, trapz_loglog)
plt.loglog(nu, sed_trapz, marker="o")
plt.loglog(nu, sed_trapz_loglog, ls="--", marker=".")
plt.show()


# a test with external Compton on point like source
print("-> test with EC on point like source")


def sed_flux_point_source(nu, blob, target, r, integrate):
    """SED flux for EC on a point like source behind the jet

    Parameters
    ----------
    nu : `~astropy.units.Quantity`
        array of frequencies, in Hz, to compute the sed, **note** these are 
        observed frequencies (observer frame).
    """
    # define the dimensionless energy
    epsilon_s = nu.to("", equivalencies=epsilon_equivalency)
    # transform to BH frame
    epsilon_s *= 1 + blob.z
    # for multidimensional integration
    # axis 0: gamma
    # axis 1: epsilon_s
    # arrays starting with _ are multidimensional and used for integration
    gamma = blob.gamma_to_integrate
    transformed_N_e = blob.N_e(gamma / blob.delta_D).value
    _gamma = np.reshape(gamma, (gamma.size, 1))
    _N_e = np.reshape(transformed_N_e, (transformed_N_e.size, 1))
    _epsilon_s = np.reshape(epsilon_s, (1, epsilon_s.size))
    # define integrating function
    # notice once the value of mu = 1, phi can assume any value, we put 0
    # convenience
    _kernel = compton_kernel(_gamma, _epsilon_s, target.epsilon_0, blob.mu_s, 1, 0)
    _integrand = np.power(_gamma, -2) * _N_e * _kernel
    integral_gamma = integrate(_integrand, gamma, axis=0)
    prefactor_num = (
        3 * sigma_T * target.L_0 * np.power(epsilon_s, 2) * np.power(blob.delta_D, 3)
    )
    prefactor_denom = (
        np.power(2, 7)
        * np.power(np.pi, 2)
        * np.power(blob.d_L, 2)
        * np.power(r, 2)
        * np.power(target.epsilon_0, 2)
    )
    sed = prefactor_num / prefactor_denom * integral_gamma
    return sed.to("erg cm-2 s-1")


# target and distance
r = 1e16 * u.cm
L_0 = 2e46 * u.Unit("erg s-1")
epsilon_0 = 1e-3
ps = PointSourceBehindJet(L_0, epsilon_0)

nu = np.logspace(20, 30) * u.Hz

# increase the size of the gamma grid
blob.set_gamma_size(500)
sed_trapz = sed_flux_point_source(nu, blob, ps, r, np.trapz)
sed_trapz_loglog = sed_flux_point_source(nu, blob, ps, r, trapz_loglog)
plt.loglog(nu, sed_trapz, marker="o")
plt.loglog(nu, sed_trapz_loglog, ls="--", marker=".")
plt.show()

# a test with external Compton on point like source
print("-> test with EC on disk")
# let us adopt the same disk parameters of Finke 2016
M_BH = 1.2 * 1e9 * M_sun.cgs
L_disk = 2 * 1e46 * u.Unit("erg s-1")
eta = 1 / 12
R_in = 6
R_out = 200
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)


def sed_flux_disk(nu, blob, target, r, integrate):
    """SED flux for EC on SS Disk

    Parameters
    ----------
    nu : `~astropy.units.Quantity`
        array of frequencies, in Hz, to compute the sed, **note** these are 
        observed frequencies (observer frame).
    """
    # define the dimensionless energy
    epsilon_s = nu.to("", equivalencies=epsilon_equivalency)
    # transform to BH frame
    epsilon_s *= 1 + blob.z
    # for multidimensional integration
    # axis 0: gamma
    # axis 1: mu
    # axis 2: phi
    # axis 3: epsilon_s
    # arrays starting with _ are multidimensional and used for integration
    # distance from the disk in gravitational radius units
    r_tilde = (r / target.R_g).to_value("")

    gamma = blob.gamma_to_integrate
    transformed_N_e = blob.N_e(gamma / blob.delta_D).value
    mu = target.mu_from_r_tilde(r_tilde)
    phi = np.linspace(0, 2 * np.pi, 50)

    _gamma = np.reshape(gamma, (gamma.size, 1, 1, 1))
    _N_e = np.reshape(transformed_N_e, (transformed_N_e.size, 1, 1, 1))
    _mu = np.reshape(mu, (1, mu.size, 1, 1))
    _phi = np.reshape(phi, (1, 1, phi.size, 1))
    _epsilon_s = np.reshape(epsilon_s, (1, 1, 1, epsilon_s.size))
    _epsilon = target.epsilon_mu(_mu, r_tilde)
    # define integrating function
    _kernel = compton_kernel(_gamma, _epsilon_s, _epsilon, blob.mu_s, _mu, _phi)
    _integrand_mu_num = target.phi_disk_mu(_mu, r_tilde)
    _integrand_mu_denum = (
        np.power(_epsilon, 2) * _mu * np.power(np.power(_mu, -2) - 1, 3 / 2)
    )
    _integrand = (
        _integrand_mu_num / _integrand_mu_denum * np.power(_gamma, -2) * _N_e * _kernel
    )
    integral_gamma = integrate(_integrand, gamma, axis=0)
    integral_mu = np.trapz(integral_gamma, mu, axis=0)
    integral_phi = np.trapz(integral_mu, phi, axis=0)
    prefactor_num = (
        9
        * sigma_T
        * G
        * target.M_BH
        * target.m_dot
        * np.power(epsilon_s, 2)
        * np.power(blob.delta_D, 3)
    )
    prefactor_denom = (
        np.power(2, 9) * np.power(np.pi, 3) * np.power(blob.d_L, 2) * np.power(r, 3)
    )
    sed = prefactor_num / prefactor_denom * integral_phi
    return sed.to("erg cm-2 s-1")


nu = np.logspace(20, 30) * u.Hz

# increase the size of the gamma grid
blob.set_gamma_size(100)
sed_trapz = sed_flux_disk(nu, blob, disk, r, np.trapz)
sed_trapz_loglog = sed_flux_disk(nu, blob, disk, r, trapz_loglog)
plt.loglog(nu, sed_trapz, marker="o")
plt.loglog(nu, sed_trapz_loglog, ls="--", marker=".")
plt.show()
