import numpy as np
import astropy.units as u
from astropy.constants import m_e
from astropy.modeling import Parameter, Fittable1DModel


mec2 = m_e.to("erg", equivalencies=u.mass_energy())


__all__ = [
    "ElectronDistribution",
    "PowerLaw",
    "BrokenPowerLaw",
    "LogParabola",
]


class ElectronDistribution(Fittable1DModel):
    """Base class grouping common functions to be used by all electron 
    distributions."""

    def general_integral(self, gamma_low, gamma_up, gamma_power=0, **kwargs):
        """integral of __any__ electron distribution over the range gamma_low, 
        gamma_up

        Parameters
        ----------
        gamma_low : float
            lower integration limit
        gamma_up : float
            higher integration limit
        gamma_power : int
            power of gamma to raise the electron distribution before integration
        kwargs : dict
            parameters of the electron distribution
        """
        gamma = np.logspace(np.log10(gamma_low), np.log10(gamma_up), 200)
        values = self.evaluate(gamma, **kwargs)
        values *= np.power(gamma, gamma_power)
        return np.trapz(values, gamma, axis=0)

    def integral(self, gamma_low, gamma_up, gamma_power=0):
        """integral of __this__ electron distribution over the range gamma_low, 
        gamma_up

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
        return np.trapz(values, gamma, axis=0)

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


class PowerLaw(ElectronDistribution):
    r"""Class for power-law particle spectrum. Built on :class:`~astropy.modeling.Fittable1DModel`.
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
    """
    k_e = Parameter(default=1e-13 * u.Unit("cm-3"), min=1e-50 * u.Unit("cm-3"))
    p = Parameter(default=2.0, min=0, max=5)
    gamma_min = Parameter(default=10)
    gamma_max = Parameter(default=1e7)

    @staticmethod
    def evaluate(gamma, k_e, p, gamma_min, gamma_max):
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max), k_e * gamma ** (-p), 0
        )

    def __str__(self):
        return (
            f"* electron spectrum\n"
            + f" - power law\n"
            + f" - k_e: {self.k_e.quantity:.2e}\n"
            + f" - p: {self.p.value:.2f}\n"
            + f" - gamma_min: {self.gamma_min.value:.2e}\n"
            + f" - gamma_max: {self.gamma_max.value:.2e}\n"
        )

    def SSA_integrand(self, gamma):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        return self.k_e * np.where(
            (self.gamma_min <= gamma) * (gamma <= self.gamma_max),
            -(self.p + 2) * np.power(gamma, -self.p - 1),
            0,
        )


class BrokenPowerLaw(ElectronDistribution):
    r"""Class for broken power-law particle spectrum. Built on :class:`~astropy.modeling.Fittable1DModel`.
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
    """
    k_e = Parameter(default=1e-13 * u.Unit("cm-3"), min=1e-50 * u.Unit("cm-3"))
    p1 = Parameter(default=2.0, min=0, max=5)
    p2 = Parameter(default=3.0, min=0, max=5)
    gamma_b = Parameter(default=1e3, min=10, max=1e6)
    gamma_min = Parameter(default=10)
    gamma_max = Parameter(default=1e7)

    @staticmethod
    def evaluate(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
        index = np.where(gamma <= gamma_b, p1, p2)
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            k_e * (gamma / gamma_b) ** (-index),
            0,
        )

    def __str__(self):
        return (
            f"* electron spectrum\n"
            + f" - broken power law\n"
            + f" - k_e: {self.k_e.quantity:.2e}\n"
            + f" - p1: {self.p1.value:.2f}\n"
            + f" - p2: {self.p2.value:.2f}\n"
            + f" - gamma_b: {self.gamma_b.value:.2e}\n"
            + f" - gamma_min: {self.gamma_min.value:.2e}\n"
            + f" - gamma_max: {self.gamma_max.value:.2e}\n"
        )

    def SSA_integrand(self, gamma):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        index = np.where(gamma <= self.gamma_b, self.p1, self.p2)
        return np.where(
            (self.gamma_min <= gamma) * (gamma <= self.gamma_max),
            self.k_e * -(index + 2) / gamma * (gamma / self.gamma_b) ** (-index),
            0,
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
    """
    k_e = Parameter(default=1e-13 * u.Unit("cm-3"), min=1e-50 * u.Unit("cm-3"))
    p = Parameter(default=2.0, min=0, max=5)
    q = Parameter(default=0.1, min=0, max=5)
    gamma_0 = Parameter(default=1e3)
    gamma_min = Parameter(default=10)
    gamma_max = Parameter(default=1e7)

    @staticmethod
    def evaluate(gamma, k_e, p, q, gamma_0, gamma_min, gamma_max):
        gamma_ratio = gamma / gamma_0
        index = -p - q * np.log10(gamma_ratio)
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max), k_e * gamma_ratio ** index, 0,
        )

    def __str__(self):
        return (
            f"* electron spectrum\n"
            + f" - log parabola\n"
            + f" - k_e: {self.k_e.quantity:.2e}\n"
            + f" - p: {self.p.value:.2f}\n"
            + f" - q: {self.q.value:.2f}\n"
            + f" - gamma_0: {self.gamma_0.value:.2e}\n"
            + f" - gamma_min: {self.gamma_min.value:.2e}\n"
            + f" - gamma_max: {self.gamma_max.value:.2e}\n"
        )

    def SSA_integrand(self, gamma):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        prefactor = -(self.p + 2 * self.q * np.log10(gamma / self.gamma_0) + 2) / gamma
        return prefactor * self.__call__(gamma)
