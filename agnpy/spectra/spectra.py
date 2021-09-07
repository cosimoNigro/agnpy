# module containing the electron spectra
import numpy as np
import astropy.units as u
from ..utils.math import trapz_loglog
from ..utils.conversion import mec2

__all__ = [
    "ElectronDistribution",
    "PowerLaw",
    "ExpCutoffPowerLaw",
    "BrokenPowerLaw",
    "LogParabola",
]


class ElectronDistribution:
    """Base class grouping common functionalities to be used by all electron 
    distributions. Choose the function to be used for integration. The
    default is :class:`~numpy.trapz`"""

    def __init__(self, integrator=np.trapz):
        self.integrator = integrator

    @staticmethod
    def general_integral(
        self, gamma_low, gamma_up, gamma_power=0, integrator=np.trapz, **kwargs
    ):
        """integral of the electron distribution over the range gamma_low, 
        gamma_up for a general set of parameters

        Parameters
        ----------
        gamma_low : float
            lower integration limit
        gamma_up : float
            higher integration limit
        gamma_power : int
            power of gamma to raise the electron distribution before integration
        integrator: func
            function to be used for integration, default is :class:`~numpy.trapz`
        kwargs : dict
            parameters of the electron distribution
        """
        gamma = np.logspace(np.log10(gamma_low), np.log10(gamma_up), 200)
        values = self.evaluate(gamma, **kwargs)
        values *= np.power(gamma, gamma_power)
        return integrator(values, gamma, axis=0)

    def integral(self, gamma_low, gamma_up, gamma_power=0):
        """integral of **this particular** electron distribution over the range 
        gamma_low, gamma_up

        Parameters
        ----------
        gamma_low : float
            lower integration limit
        gamma_up : float
            higher integration limit
        """
        gamma = np.logspace(np.log10(gamma_low), np.log10(gamma_up), 200)
        values = self.__call__(gamma)
        values *= np.power(gamma, gamma_power)
        return self.integrator(values, gamma, axis=0)

    @classmethod
    def from_normalised_density(cls, n_e_tot, **kwargs):
        r"""sets the normalisation :math:`k_e` from the total particle density 
        :math:`n_{e,\,tot}`"""
        # use gamma_min and gamma_max of the electron distribution as
        # integration limits
        if "gamma_min" in kwargs:
            gamma_min = kwargs.get("gamma_min")
        if "gamma_max" in kwargs:
            gamma_max = kwargs.get("gamma_max")
        k_e = n_e_tot / cls.general_integral(
            cls, gamma_low=gamma_min, gamma_up=gamma_max, gamma_power=0, k_e=1, **kwargs
        )
        return cls(k_e.to("cm-3"), **kwargs)

    @classmethod
    def from_normalised_energy_density(cls, u_e, **kwargs):
        r"""sets the normalisation :math:`k_e` from the total energy density 
        :math:`u_e`, Eq. 6.64 in [DermerMenon2009]_"""
        # use gamma_min and gamma_max of the electron distribution as
        # integration limits
        if "gamma_min" in kwargs:
            gamma_min = kwargs.get("gamma_min")
        if "gamma_max" in kwargs:
            gamma_max = kwargs.get("gamma_max")
        integral = cls.general_integral(
            cls, gamma_low=gamma_min, gamma_up=gamma_max, gamma_power=1, k_e=1, **kwargs
        )
        k_e = u_e / (mec2 * integral)
        return cls(k_e.to("cm-3"), **kwargs)

    @classmethod
    def from_norm_at_gamma_1(cls, norm, **kwargs):
        r"""sets :math:`k_e` such that `norm` = :math:`n_e(\gamma=1)`."""
        k_e = norm.to("cm-3") / cls.evaluate(1, 1, **kwargs)
        return cls(k_e, **kwargs)

    @classmethod
    def from_total_energy(cls, W_e, V_b, **kwargs):
        r"""sets :math:`k_e` from the total energy `W_e`, given a volume
        V_b of the emission region."""
        u_e = W_e / V_b
        return cls.from_normalised_energy_density(u_e, **kwargs)


class PowerLaw(ElectronDistribution):
    r"""Class for power-law particle spectrum. 
    When called, the particle density :math:`n_e(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n_e(\gamma') = k_e \, \gamma'^{-p} \, H(\gamma'; \gamma'_{\rm min}, \gamma'_{\rm max}) 

    Parameters
    ----------
    k_e : :class:`~astropy.units.Quantity`
        spectral normalisation
    p : float
        spectral index, note it is positive by definition, will change sign in the function
    gamma_min : float
        minimum Lorentz factor of the electron distribution
    gamma_max : float
        maximum Lorentz factor of the electron distribution
    integrator: func
        function to be used for integration, default is :class:`~numpy.trapz`
    """

    def __init__(
        self,
        k_e=1e-13 * u.Unit("cm-3"),
        p=2.1,
        gamma_min=10,
        gamma_max=1e5,
        integrator=np.trapz,
    ):
        super().__init__(integrator)
        self.k_e = k_e
        self.p = p
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def parameters(self):
        return [self.k_e, self.p, self.gamma_min, self.gamma_max]

    @staticmethod
    def evaluate(gamma, k_e, p, gamma_min, gamma_max):
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max), k_e * gamma ** (-p), 0
        )

    def __call__(self, gamma):
        return self.evaluate(gamma, self.k_e, self.p, self.gamma_min, self.gamma_max)

    @staticmethod
    def evaluate_SSA_integrand(gamma, k_e, p, gamma_min, gamma_max):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        return k_e * np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            -(p + 2) * np.power(gamma, -p - 1),
            0,
        )

    def SSA_integrand(self, gamma):
        return self.evaluate_SSA_integrand(
            gamma, self.k_e, self.p, self.gamma_min, self.gamma_max
        )

    def __str__(self):
        return (
            f"* electron spectrum\n"
            + f" - power law\n"
            + f" - k_e: {self.k_e:.2e}\n"
            + f" - p: {self.p:.2f}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}"
        )


class BrokenPowerLaw(ElectronDistribution):
    r"""Class for broken power-law particle spectrum.
    When called, the particle density :math:`n_e(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n_e(\gamma') = k_e \left[
        \left(\frac{\gamma'}{\gamma'_b}\right)^{-p_1} \, H(\gamma'; \gamma'_{\rm min}, \gamma'_b) +
        \left(\frac{\gamma'}{\gamma'_b}\right)^{-p_2} \, H(\gamma'; \gamma'_{b}, \gamma'_{\rm max}) 
        \right]

    Parameters
    ----------
    k_e : :class:`~astropy.units.Quantity`
        spectral normalisation
    p1 : float
        spectral index before the break (positive by definition)
    p2 : float
        spectral index after the break (positive by definition)   
    gamma_b : float
        Lorentz factor at which the change in spectral index is occurring 
    gamma_min : float
        minimum Lorentz factor of the electron distribution
    gamma_max : float
        maximum Lorentz factor of the electron distribution
    integrator: func
        function to be used for integration, default is :class:`~numpy.trapz`
    """

    def __init__(
        self,
        k_e=1e-13 * u.Unit("cm-3"),
        p1=2.0,
        p2=3.0,
        gamma_b=1e3,
        gamma_min=10,
        gamma_max=1e7,
        integrator=np.trapz,
    ):
        super().__init__(integrator)
        self.k_e = k_e
        self.p1 = p1
        self.p2 = p2
        self.gamma_b = gamma_b
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def parameters(self):
        return [
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        ]

    @staticmethod
    def evaluate(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
        index = np.where(gamma <= gamma_b, p1, p2)
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            k_e * (gamma / gamma_b) ** (-index),
            0,
        )

    def __call__(self, gamma):
        return self.evaluate(
            gamma,
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )

    @staticmethod
    def evaluate_SSA_integrand(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        index = np.where(gamma <= gamma_b, p1, p2)
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            k_e * -(index + 2) / gamma * (gamma / gamma_b) ** (-index),
            0,
        )

    def SSA_integrand(self, gamma):
        return self.evaluate_SSA_integrand(
            gamma,
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )

    def __str__(self):
        return (
            f"* electron spectrum\n"
            + f" - broken power law\n"
            + f" - k_e: {self.k_e:.2e}\n"
            + f" - p1: {self.p1:.2f}\n"
            + f" - p2: {self.p2:.2f}\n"
            + f" - gamma_b: {self.gamma_b:.2e}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}"
        )


class LogParabola(ElectronDistribution):
    r"""Class for log-parabolic particle spectrum. Built on :class:`~astropy.modeling.Fittable1DModel`.
    When called, the particle density :math:`n_e(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n_e(\gamma') = k_e \, \left(\frac{\gamma'}{\gamma'_0}\right)^{-(p + q \log_{10}(\gamma' / \gamma'_0))}  

    Parameters
    ----------
    k_e : :class:`~astropy.units.Quantity`
        spectral normalisation
    p : float
        spectral index, note it is positive by definition, will change sign in the function
    q : float
        spectral curvature, note it is positive by definition, will change sign in the function
    gamma_0 : float
        reference Lorentz factor
    gamma_min : float
        minimum Lorentz factor of the electron distribution
    gamma_max : float
        maximum Lorentz factor of the electron distribution
    integrator: func
        function to be used for integration, default is :class:`~numpy.trapz`
    """

    def __init__(
        self,
        k_e=1e-13 * u.Unit("cm-3"),
        p=2.0,
        q=0.1,
        gamma_0=1e3,
        gamma_min=10,
        gamma_max=1e7,
        integrator=np.trapz,
    ):
        super().__init__(integrator)
        self.k_e = k_e
        self.p = p
        self.q = q
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def parameters(self):
        return [self.k_e, self.p, self.q, self.gamma_0, self.gamma_min, self.gamma_max]

    @staticmethod
    def evaluate(gamma, k_e, p, q, gamma_0, gamma_min, gamma_max):
        gamma_ratio = gamma / gamma_0
        index = -p - q * np.log10(gamma_ratio)
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max), k_e * gamma_ratio ** index, 0,
        )

    def __call__(self, gamma):
        return self.evaluate(
            gamma,
            self.k_e,
            self.p,
            self.q,
            self.gamma_0,
            self.gamma_min,
            self.gamma_max,
        )

    @staticmethod
    def evaluate_SSA_integrand(gamma, k_e, p, q, gamma_0, gamma_min, gamma_max):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        prefactor = -(p + 2 * q * np.log10(gamma / gamma_0) + 2) / gamma
        return prefactor * LogParabola.evaluate(
            gamma, k_e, p, q, gamma_0, gamma_min, gamma_max
        )

    def SSA_integrand(self, gamma):
        return self.evaluate_SSA_integrand(
            gamma,
            self.k_e,
            self.p,
            self.q,
            self.gamma_0,
            self.gamma_min,
            self.gamma_max,
        )

    def __str__(self):
        return (
            f"* electron spectrum\n"
            + f" - log parabola\n"
            + f" - k_e: {self.k_e:.2e}\n"
            + f" - p: {self.p:.2f}\n"
            + f" - q: {self.q:.2f}\n"
            + f" - gamma_0: {self.gamma_0:.2e}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}"
        )


class ExpCutoffPowerLaw(ElectronDistribution):
    r"""Class for power-law with an exponetial cutoff particle spectrum. 
    When called, the particle density :math:`n_e(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n_e(\gamma') = k_e \, \gamma'^{-p} exp(-\gamma'/\gamma_c) \, H(\gamma'; \gamma'_{\rm min}, \gamma'_{\rm max}) 

    Parameters
    ----------
    k_e : :class:`~astropy.units.Quantity`
        spectral normalisation
    p : float
        spectral index, note it is positive by definition, will change sign in the function
    gamma_c : float
        cutoff Lorentz factor of the electron distribution
    gamma_min : float
        minimum Lorentz factor of the electron distribution
    gamma_max : float
        maximum Lorentz factor of the electron distribution
    integrator: func
        function to be used for integration, default is :class:`~numpy.trapz`
    """

    def __init__(
        self,
        k_e=1e-13 * u.Unit("cm-3"),
        p=2.1,
        gamma_c=1e3,
        gamma_min=10,
        gamma_max=1e5,
        integrator=np.trapz,
    ):
        super().__init__(integrator)
        self.k_e = k_e
        self.p = p
        self.gamma_c = gamma_c
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def parameters(self):
        return [self.k_e, self.p, self.gamma_c, self.gamma_min, self.gamma_max]

    @staticmethod
    def evaluate(gamma, k_e, p, gamma_c, gamma_min, gamma_max):
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            k_e * gamma ** (-p) * np.exp(-gamma / gamma_c),
            0,
        )

    def __call__(self, gamma):
        return self.evaluate(
            gamma, self.k_e, self.p, self.gamma_c, self.gamma_min, self.gamma_max
        )

    @staticmethod
    def evaluate_SSA_integrand(gamma, k_e, p, gamma_c, gamma_min, gamma_max):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        prefactor = -(p + 2) / gamma + (-1 / gamma_c)

        return prefactor * ExpCutoffPowerLaw.evaluate(
            gamma, k_e, p, gamma_c, gamma_min, gamma_max
        )

    def SSA_integrand(self, gamma):
        return self.evaluate_SSA_integrand(
            gamma, self.k_e, self.p, self.gamma_c, self.gamma_min, self.gamma_max
        )

    def __str__(self):
        return (
            f"* electron spectrum\n"
            + f" - power law\n"
            + f" - k_e: {self.k_e:.2e}\n"
            + f" - p: {self.p:.2f}\n"
            + f" - gamma_c: {self.gamma_c:.2f}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}"
        )
