# module containing the synchrotron radiative process
import numpy as np
import astropy.units as u
from astropy.constants import e, h, c, m_e, m_p, sigma_T, G

import synchrotron_new as newsyn

#__all__ = ["R_proton", "nu_synch_peak_proton", "ProtonSynchrotron"]
__all__ = ["ProtonSynchrotron2"]


e = e.gauss
B_cr = 4.414e13 * u.G  # critical magnetic field

# Necessary functions from utils:


# ~~~~~~~ math.py ~~~~~~~~

# default arrays to be used for integration
gamma_e_to_integrate = np.logspace(1, 9, 200)
nu_to_integrate = np.logspace(5, 30, 200) * u.Hz  # used for SSC
mu_to_integrate = np.linspace(-1, 1, 100)
phi_to_integrate = np.linspace(0, 2 * np.pi, 50)

# minimum relative distance to the absorber (to avoid infinite integrals)
min_rel_distance = 1.0e-4

# type of float to be used for math operation
numpy_type = np.float64
# smallest positive float
ftiny = np.finfo(numpy_type).tiny
# largest positive float
fmax = np.finfo(numpy_type).max


def axes_reshaper(*args):
    """reshape 1-dimensional arrays of different lengths in order for them to be
    broadcastable in multi-dimensional operations

    the rearrangement scheme for a list of n arrays is the following:
    `args[0]` is reshaped as `(args[0].size, 1, 1, ..., 1)` -> axis 0
    `args[1]` is reshaped as `(1, args[1].size, 1, ..., 1)` -> axis 1
    `args[2]` is reshaped as `(1, 1, args[2].size ..., 1)` -> axis 2
        .
        .
        .
    `args[n-1]` is reshaped as `(1, 1, 1, ..., args[n-1].size)` -> axis n-1
    Parameters
    ----------
    args: 1-dimensional `~numpy.ndarray`s to be reshaped
    """
    n = len(args)
    dim = (1,) * n
    reshaped_arrays = []
    for i, arg in enumerate(args):
        reshaped_dim = list(dim)  # the tuple is copied in the list
        reshaped_dim[i] = arg.size
        reshaped_array = np.reshape(arg, reshaped_dim)
        reshaped_arrays.append(reshaped_array)
    return reshaped_arrays


def log(x):
    """clipped log to avoid RuntimeWarning: divide by zero encountered in log"""
    values = np.clip(x, ftiny, fmax)
    return np.log(values)


def trapz_loglog(y, x, axis=0):
    """
    Integrate a function approximating its discrete intervals as power-laws
    (straight lines in log-log space), largely copied from naima

    Parameters
    ----------
    y : array_like
        array to integrate
    x : array_like, optional
        independent variable to integrate over
    axis : int, optional
        along which axis the integration has to be performed

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.0
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.0

    slice_low = [slice(None)] * y.ndim
    slice_up = [slice(None)] * y.ndim
    # multi-dimensional equivalent of x_low = x[:-1]
    slice_low[axis] = slice(None, -1)
    # multi-dimensional equivalent of x_up = x[1:]
    slice_up[axis] = slice(1, None)

    # reshape x to be broadcastable with y
    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    x_low = x[tuple(slice_low)]
    x_up = x[tuple(slice_up)]
    y_low = y[tuple(slice_low)]
    y_up = y[tuple(slice_up)]

    # slope in the given logarithmic bin
    m = (log(y_low) - log(y_up)) / (log(x_low) - log(x_up))

    vals = np.where(
        np.abs(m + 1) > 1e-10,
        y_low / (m + 1) * (x_up * np.power(x_up / x_low, m) - x_low),
        x_low * y_low * log(x_up / x_low),
    )

    tozero = (
        (y[tuple(slice_low)] == 0.0)
        + (y[tuple(slice_up)] == 0.0)
        + (x[tuple(slice_low)] == x[tuple(slice_up)])
    )
    vals[tozero] = 0.0

    return np.add.reduce(vals, axis) * x_unit * y_unit

# ~~~~~~~ conversion.py ~~~~~~~~

mec2 = m_e.to("erg", equivalencies=u.mass_energy())
mpc2 = m_p.to("erg", equivalencies=u.mass_energy())
lambda_c = (h / (m_e * c)).to("cm")  # Compton wavelength
# equivalency for decomposing Gauss in Gaussian-cgs units (not available in astropy)
Gauss_cgs_unit = "cm(-1/2) g(1/2) s-1"
Gauss_cgs_equivalency = [(u.G, u.Unit(Gauss_cgs_unit), lambda x: x, lambda x: x)]
# equivalency to transform frequencies to energies in electron rest mass units
epsilon_equivalency = [
    (u.Hz, u.Unit(""), lambda x: h.cgs * x / mec2, lambda x: x * mec2 / h.cgs)
]


def nu_to_epsilon_prime(nu, z=0, delta_D=1):
    """convert the frequency to a dimensionless energy in another reference
    frame with redshift z and moving with doppler factor delta_D"""
    epsilon = nu.to("", equivalencies=epsilon_equivalency)
    return (1 + z) * epsilon / delta_D


def B_to_cgs(B):
    """convert a magnetic field to CGS units"""
    return B.to(Gauss_cgs_unit, equivalencies=Gauss_cgs_equivalency)


def to_R_g_units(r, M):
    """convert a distance in graviational radii units"""
    R_g = G * M / c ** 2
    return (r / R_g).to("")



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



'''
def R_proton(x):
    """Eq. 7.45 in [Dermer2009]_, angle-averaged integrand of the radiated power, the
    approximation of this function, given in Eq. D7 of [Aharonian2010]_, is used.
    """
    term_1_num = 1.808 * np.power(x, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(x, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(x, 2 / 3) + 0.347 * np.power(x, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(x, 2 / 3) + 0.217 * np.power(x, 4 / 3)
    return term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-x)


def nu_synch_peak_proton(B, gamma):
    """observed peak frequency for monoenergetic electrons
    Eq. 7.19 in [DermerMenon2009]_"""
    B = B_to_cgs(B)
    nu_peak = (e * B / (2 * np.pi * m_p * c)) * np.power(gamma, 2)
    return nu_peak.to("Hz")


def calc_x(B_cgs, epsilon, gamma):
    """ratio of the frequency to the critical synchrotron frequency from
    Eq. 7.34 in [DermerMenon2009]_, argument of R(x),
    note B has to be in cgs Gauss units"""
    x = (
        4
        * np.pi
        * epsilon
        * np.power(m_p, 2)
        * np.power(c, 3)
        / (3 * e * B_cgs * h * np.power(gamma, 2))
    )
    return x.to_value("")


def epsilon_B(B):
    r""":math:`\epsilon_B`, Eq. 7.21 [DermerMenon2009]_"""
    return (B / B_cr).to_value("")


def single_electron_synch_power(B_cgs, epsilon, gamma):
    """angle-averaged synchrotron power for a single electron,
    to be folded with the electron distribution
    """
    x = calc_x(B_cgs, epsilon, gamma)
    prefactor = np.sqrt(3) * np.power(e, 3) * B_cgs / h
    return prefactor * R_proton(x)


def tau_to_attenuation(tau):
    """Converts the synchrotron self-absorption optical depth to an attenuation
    Eq. 7.122 in [DermerMenon2009]_."""
    u = 1 / 2 + np.exp(-tau) / tau - (1 - np.exp(-tau)) / np.power(tau, 2)
    return np.where(tau < 1e-3, 1, 3 * u / tau)
'''

class ProtonSynchrotron2:
    """Class for synchrotron radiation computation

    Parameters
    ----------
    blob : :class:`~agnpy.emission_region.Blob`
        emitting region and electron distribution
    ssa : bool
        whether or not to consider synchrotron self absorption (SSA).
        The absorption factor will be taken into account in
        :func:`~agnpy.synchrotron.Synchrotron.com_sed_emissivity`, in order to be
        propagated to :func:`~agnpy.synchrotron.Synchrotron.sed_luminosity` and
        :func:`~agnpy.synchrotron.Synchrotron.sed_flux`.
    integrator : func
        function to be used for integration (default = `np.trapz`)
	"""

    def __init__(self, blob, ssa=False, integrator=np.trapz):
        self.blob = blob
        self.ssa = ssa
        self.integrator = integrator

    # @staticmethod
    # def evaluate_tau_ssa(
    #     nu,
    #     z,
    #     d_L,
    #     delta_D,
    #     B,
    #     R_b,
    #     n_e,
    #     *args,
    #     integrator=np.trapz,
    #     gamma=gamma_e_to_integrate,
    # ):
    #     """Computes the syncrotron self-absorption opacity for a general set
    #     of model parameters, see :func:`~agnpy:sycnhrotron.Synchrotron.evaluate_sed_flux`
    #     for parameters defintion. Eq. before 7.122 in [DermerMenon2009]_."""
    #     # conversions
    #     epsilon = nu_to_epsilon_prime(nu, z, delta_D)
    #     B_cgs = B_to_cgs(B)
    #     # multidimensional integration
    #     _gamma, _epsilon = axes_reshaper(gamma, epsilon)
    #     SSA_integrand = n_e.evaluate_SSA_integrand(_gamma, *args)
    #     integrand = SSA_integrand * single_electron_synch_power(B_cgs, _epsilon, _gamma)
    #     integral = integrator(integrand, gamma, axis=0)
    #     prefactor_k_epsilon = (
    #         -1 / (8 * np.pi * m_e * np.power(epsilon, 2)) * np.power(lambda_c / c, 3)
    #     )
    #     k_epsilon = (prefactor_k_epsilon * integral).to("cm-1")
    #     return (2 * k_epsilon * R_b).to_value("")

    @staticmethod
    def evaluate_sed_flux(
        nu,
        z,
        d_L,
        delta_D,
        B,
        R_b,
        n_p,
        *args,
        ssa=False,
        integrator=np.trapz,
        gamma=gamma_e_to_integrate,
    ):
        r"""Evaluates the flux SED (:math:`\nu F_{\nu}`) due to synchrotron radiation,
        for a general set of model parameters. Eq. 21 in [Finke2008]_.

        **Note** parameters after \*args need to be passed with a keyword

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        d_L : :class:`~astropy.units.Quantity`
            luminosity distance of the source
        delta_D : float
            Doppler factor of the relativistic outflow
        B : :class:`~astropy.units.Quantity`
            magnetic field in the blob
        R_b : :class:`~astropy.units.Quantity`
            size of the emitting region (spherical blob assumed)
        n_e : :class:`~agnpy.spectra.ElectronDistribution`
            electron energy distribution
        \*args
            parameters of the electron energy distribution (k_e, p, ...)
        ssa : bool
            whether to consider or not the self-absorption, default false
        integrator : func
            which function to use for integration, default `numpy.trapz`
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to integrate the electron
            distribution

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        # conversions
        epsilon = nu_to_epsilon_prime(nu, z, delta_D)
        B_cgs = B_to_cgs(B)
        # reshape for multidimensional integration
        _gamma, _epsilon = axes_reshaper(gamma, epsilon)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_p = V_b * n_p.evaluate(_gamma, *args)
        # fold the electron distribution with the synchrotron power
        integrand = N_p * newsyn.single_particle_synch_power(B_cgs, _epsilon, _gamma, mass=m_p)
        emissivity = integrator(integrand, gamma, axis=0)
        prefactor = np.power(delta_D, 4) / (4 * np.pi * np.power(d_L, 2))
        sed = (prefactor * epsilon * emissivity).to("erg cm-2 s-1")

        # if ssa:
        #     tau = Synchrotron.evaluate_tau_ssa(
        #         nu,
        #         z,
        #         d_L,
        #         delta_D,
        #         B,
        #         R_b,
        #         n_e,
        #         *args,
        #         integrator=integrator,
        #         gamma=gamma,
        #     )
        #     attenuation = tau_to_attenuation(tau)
        #     sed *= attenuation

        return sed

    @staticmethod
    def evaluate_sed_flux_delta_approx(nu, z, d_L, delta_D, B, R_b, n_p, *args):
        """Synchrotron flux SED using the delta approximation for the
        synchrotron radiation Eq. 7.70 [DermerMenon2009]_."""
        epsilon_prime = nu_to_epsilon_prime(nu, z, delta_D)
        gamma_s = np.sqrt(epsilon_prime / newsyn.epsilon_B(B))
        B_cgs = B_to_cgs(B)
        U_B = np.power(B_cgs, 2) / (8 * np.pi)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_p = V_b * n_p.evaluate(gamma_s, *args)
        prefactor = (
            np.power(delta_D, 4) * c * sigma_T * U_B / (6 * np.pi * np.power(d_L, 2))
        )
        value = prefactor * np.power(gamma_s, 3) * N_p
        return value.to("erg cm-2 s-1")

    def sed_flux(self, nu):
        r"""Evaluates the synchrotron flux SED for a Synchrotron object built
        from a Blob."""
        return self.evaluate_sed_flux(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.B,
            self.blob.R_b,
            self.blob.n_p,
            *self.blob.n_p.parameters,
            ssa=self.ssa,
            integrator=self.integrator,
            gamma=self.blob.gamma_p,
        )

    def sed_flux_delta_approx(self, nu):
        """Evaluates the synchrotron flux SED using the delta approximation for
        a Synchrotron object built from a blob."""
        return self.evaluate_sed_flux_delta_approx(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.B,
            self.blob.R_b,
            self.blob.n_p,
            *self.blob.n_p.parameters,
        )

    def sed_luminosity(self, nu):
        r"""Evaluates the synchrotron luminosity SED
        :math:`\nu L_{\nu} \, [\mathrm{erg}\,\mathrm{s}^{-1}]`
        for a a Synchrotron object built from a blob."""
        sphere = 4 * np.pi * np.power(self.blob.d_L, 2)
        return (sphere * self.sed_flux(nu)).to("erg s-1")

    def sed_peak_flux(self, nu):
        """provided a grid of frequencies nu, returns the peak flux of the SED
        """
        return self.sed_flux(nu).max()

    def sed_peak_nu(self, nu):
        """provided a grid of frequencies nu, returns the frequency at which the
        SED peaks
        """
        idx_max = self.sed_flux(nu).argmax()
        return nu[idx_max]
