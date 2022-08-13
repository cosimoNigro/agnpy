# module containing the particle spectra
import numpy as np
import astropy.units as u
from astropy.constants import m_e, m_p


__all__ = [
    "ParticleDistribution",
    "PowerLaw",
    "ExpCutoffPowerLaw",
    "BrokenPowerLaw",
    "LogParabola",
]


class ParticleDistribution:
    """Base class grouping common functionalities to be used by all particle
    distributions.

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        particle mass, default is the electron mass
    integrator : function
        function to be used to integrate the particle distribution
    """

    def __init__(self, mass=m_e, integrator=np.trapz):
        self.integrator = integrator
        if mass is m_e:
            self.mass = m_e
            self.type = "electrons"
        elif mass is m_p:
            self.mass = m_p
            self.type = "protons"
        else:
            raise ValueError(
                f"No distribution for particles with mass {mass} is available."
            )
        self.mc2 = self.mass.to("erg", equivalencies=u.mass_energy())

    @staticmethod
    def general_integral(
        self, gamma_low, gamma_up, gamma_power=0, integrator=np.trapz, **kwargs
    ):
        """Integral of the particle distribution over the range gamma_low,
        gamma_up for a general set of parameters.

        Parameters
        ----------
        gamma_low : float
            lower integration limit
        gamma_up : float
            higher integration limit
        gamma_power : int
            power of gamma to raise the particle distribution before integration
        integrator: func
            function to be used for integration, default is :class:`~numpy.trapz`
        kwargs : dict
            parameters of the particle distribution
        """
        gamma = np.logspace(np.log10(gamma_low), np.log10(gamma_up), 200)
        values = self.evaluate(gamma, **kwargs)
        values *= np.power(gamma, gamma_power)
        return integrator(values, gamma, axis=0)

    def integral(self, gamma_low, gamma_up, gamma_power=0):
        """integral of **this particular** particle distribution over the range 
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
    def from_normalised_density(cls, n_tot, mass, **kwargs):
        r"""Set the normalisation of the particle distribution,
        :math:`k [{\rm cm}^{-3}]`, from the total particle density
        :math:`n_{\rm tot} [{\rm cm}^{-3}]`.

        Parameters
        ----------
        n_tot : `~astropy.units.Quantity`
            total particle density (integral of :math:`n(\gamma)`), in cm-3
        mass : `~astropy.units.Quantity`
            particle mass
        """
        # use gamma_min and gamma_max as integration limits
        if "gamma_min" in kwargs:
            gamma_min = kwargs.get("gamma_min")
        if "gamma_max" in kwargs:
            gamma_max = kwargs.get("gamma_max")
        k = n_tot / cls.general_integral(
            cls, gamma_low=gamma_min, gamma_up=gamma_max, gamma_power=0, k=1, **kwargs
        )
        return cls(k=k.to("cm-3"), **kwargs, mass=mass)

    @classmethod
    def from_normalised_energy_density(cls, u_tot, mass, **kwargs):
        r"""Set the normalisation of the particle distribution,
        :math:`k [{\rm cm}^{-3}]`, from the total energy density
        :math:`u_{\rm tot} [{\rm erg}{\rm cm}^{-3}]`, Eq. 6.64 in [DermerMenon2009]_.

        Parameters
        ----------
        u_tot : `~astropy.units.Quantity`
            total energy density (integral of :math:`\gamma n(\gamma)`), in erg cm-3
        mass : `~astropy.units.Quantity`
            particle mass
        """
        # use gamma_min and gamma_max as integration limits
        if "gamma_min" in kwargs:
            gamma_min = kwargs.get("gamma_min")
        if "gamma_max" in kwargs:
            gamma_max = kwargs.get("gamma_max")
        integral = cls.general_integral(
            cls, gamma_low=gamma_min, gamma_up=gamma_max, gamma_power=1, k=1, **kwargs
        )
        mc2 = mass.to("erg", equivalencies=u.mass_energy())
        k = u_tot / (mc2 * integral)
        return cls(k=k.to("cm-3"), **kwargs, mass=mass)

    @classmethod
    def from_norm_at_gamma_1(cls, norm, mass, **kwargs):
        r"""Set the normalisation of the particle distribution,
        :math:`k [{\rm cm}^{-3}]`, such that `norm` = :math:`n(\gamma=1)`.

        Parameters
        ----------
        norm : `~astropy.units.Quantity`
            value :math:`n(\gamma)` should have at :math:`\gamma=1`, in cm-3
        mass : `~astropy.units.Quantity`
            particle mass
        """
        k = norm.to("cm-3") / cls.evaluate(1, 1, **kwargs)
        return cls(k=k.to("cm-3"), **kwargs, mass=mass)

    @classmethod
    def from_total_energy(cls, W, V, mass, **kwargs):
        r"""Set the normalisation of the particle distribution,
        :math:`k [{\rm cm}^{-3}]`, based on the total energy in particles
        :math:`W` = :math:`m c^2 \int {\rm d}\gamma \, \gamma n(\gamma)`.

        Parameters
        ----------
        W : `~astropy.units.Quantity`
            total energy in particles, in erg
        V : `~astropy.units.Quantity`
            volume of the emission region, in cm^3
        mass : `~astropy.units.Quantity`
            particle mass
        """
        u = W / V
        return cls.from_normalised_energy_density(u, mass, **kwargs)


class PowerLaw(ParticleDistribution):
    r"""Class for power-law particle spectrum. 
    When called, the particle density :math:`n(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n(\gamma') = k \, \gamma'^{-p} \, H(\gamma'; \gamma'_{\rm min}, \gamma'_{\rm max}) 

    Parameters
    ----------
    k : :class:`~astropy.units.Quantity`
        spectral normalisation
    p : float
        spectral index, note it is positive by definition, will change sign in the function
    gamma_min : float
        minimum Lorentz factor of the particle distribution
    gamma_max : float
        maximum Lorentz factor of the particle distribution
    mass : `~astropy.units.Quantity`
        particle mass, default is the electron mass
    integrator : func
        function to be used for integration, default is :class:`~numpy.trapz`
    """

    def __init__(
        self,
        k=1e-13 * u.Unit("cm-3"),
        p=2.1,
        gamma_min=10,
        gamma_max=1e5,
        mass=m_e,
        integrator=np.trapz,
    ):
        super().__init__(mass, integrator)
        self.k = k
        self.p = p
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def parameters(self):
        return [self.k, self.p, self.gamma_min, self.gamma_max]

    @staticmethod
    def evaluate(gamma, k, p, gamma_min, gamma_max):
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max), k * gamma ** (-p), 0
        )

    def __call__(self, gamma):
        return self.evaluate(gamma, self.k, self.p, self.gamma_min, self.gamma_max)

    @staticmethod
    def evaluate_SSA_integrand(gamma, k, p, gamma_min, gamma_max):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n(\gamma)}{\gamma'^2}\right)`"""
        return k * np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            -(p + 2) * np.power(gamma, -p - 1),
            0,
        )

    def SSA_integrand(self, gamma):
        return self.evaluate_SSA_integrand(
            gamma, self.k, self.p, self.gamma_min, self.gamma_max
        )

    def __str__(self):
        return (
            f"* {self.type} energy distribution\n"
            + f" - power law\n"
            + f" - k: {self.k:.2e}\n"
            + f" - p: {self.p:.2f}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}"
        )


class BrokenPowerLaw(ParticleDistribution):
    r"""Class for broken power-law particle spectrum.
    When called, the particle density :math:`n(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n(\gamma') = k \left[
        \left(\frac{\gamma'}{\gamma'_b}\right)^{-p_1} \, H(\gamma'; \gamma'_{\rm min}, \gamma'_b) +
        \left(\frac{\gamma'}{\gamma'_b}\right)^{-p_2} \, H(\gamma'; \gamma'_{b}, \gamma'_{\rm max}) 
        \right]

    Parameters
    ----------
    k : :class:`~astropy.units.Quantity`
        spectral normalisation
    p1 : float
        spectral index before the break (positive by definition)
    p2 : float
        spectral index after the break (positive by definition)   
    gamma_b : float
        Lorentz factor at which the change in spectral index is occurring 
    gamma_min : float
        minimum Lorentz factor of the particle distribution
    gamma_max : float
        maximum Lorentz factor of the particle distribution
    mass : `~astropy.units.Quantity`
        particle mass, default is the electron mass
    integrator : func
        function to be used for integration, default is :class:`~numpy.trapz`
    """

    def __init__(
        self,
        k=1e-13 * u.Unit("cm-3"),
        p1=2.0,
        p2=3.0,
        gamma_b=1e3,
        gamma_min=10,
        gamma_max=1e7,
        integrator=np.trapz,
    ):
        super().__init__(integrator)
        self.k = k
        self.p1 = p1
        self.p2 = p2
        self.gamma_b = gamma_b
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def parameters(self):
        return [
            self.k,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        ]

    @staticmethod
    def evaluate(gamma, k, p1, p2, gamma_b, gamma_min, gamma_max):
        index = np.where(gamma <= gamma_b, p1, p2)
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            k * (gamma / gamma_b) ** (-index),
            0,
        )

    def __call__(self, gamma):
        return self.evaluate(
            gamma,
            self.k,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )

    @staticmethod
    def evaluate_SSA_integrand(gamma, k, p1, p2, gamma_b, gamma_min, gamma_max):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        index = np.where(gamma <= gamma_b, p1, p2)
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            k * -(index + 2) / gamma * (gamma / gamma_b) ** (-index),
            0,
        )

    def SSA_integrand(self, gamma):
        return self.evaluate_SSA_integrand(
            gamma,
            self.k,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )

    def __str__(self):
        return (
            f"* {self.type} energy distribution\n"
            + f" - broken power law\n"
            + f" - k: {self.k:.2e}\n"
            + f" - p1: {self.p1:.2f}\n"
            + f" - p2: {self.p2:.2f}\n"
            + f" - gamma_b: {self.gamma_b:.2e}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}\n"
        )


class LogParabola(ParticleDistribution):
    r"""Class for log-parabolic particle spectrum.
    When called, the particle density :math:`n(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n(\gamma') = k \, \left(\frac{\gamma'}{\gamma'_0}\right)^{-(p + q \log_{10}(\gamma' / \gamma'_0))}  

    Parameters
    ----------
    k : :class:`~astropy.units.Quantity`
        spectral normalisation
    p : float
        spectral index, note it is positive by definition, will change sign in the function
    q : float
        spectral curvature, note it is positive by definition, will change sign in the function
    gamma_0 : float
        reference Lorentz factor
    gamma_min : float
        minimum Lorentz factor of the particle distribution
    gamma_max : float
        maximum Lorentz factor of the particle distribution
    mass : `~astropy.units.Quantity`
        particle mass, default is the electron mass
    integrator : func
        function to be used for integration, default is :class:`~numpy.trapz`
    """

    def __init__(
        self,
        k=1e-13 * u.Unit("cm-3"),
        p=2.0,
        q=0.1,
        gamma_0=1e3,
        gamma_min=10,
        gamma_max=1e7,
        integrator=np.trapz,
    ):
        super().__init__(integrator)
        self.k = k
        self.p = p
        self.q = q
        self.gamma_0 = gamma_0
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def parameters(self):
        return [self.k, self.p, self.q, self.gamma_0, self.gamma_min, self.gamma_max]

    @staticmethod
    def evaluate(gamma, k, p, q, gamma_0, gamma_min, gamma_max):
        gamma_ratio = gamma / gamma_0
        index = -p - q * np.log10(gamma_ratio)
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max), k * gamma_ratio ** index, 0,
        )

    def __call__(self, gamma):
        return self.evaluate(
            gamma, self.k, self.p, self.q, self.gamma_0, self.gamma_min, self.gamma_max,
        )

    @staticmethod
    def evaluate_SSA_integrand(gamma, k, p, q, gamma_0, gamma_min, gamma_max):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        prefactor = -(p + 2 * q * np.log10(gamma / gamma_0) + 2) / gamma
        return prefactor * LogParabola.evaluate(
            gamma, k, p, q, gamma_0, gamma_min, gamma_max
        )

    def SSA_integrand(self, gamma):
        return self.evaluate_SSA_integrand(
            gamma, self.k, self.p, self.q, self.gamma_0, self.gamma_min, self.gamma_max,
        )

    def __str__(self):
        return (
            f"* {self.type} energy distribution\n"
            + f" - log parabola\n"
            + f" - k: {self.k:.2e}\n"
            + f" - p: {self.p:.2f}\n"
            + f" - q: {self.q:.2f}\n"
            + f" - gamma_0: {self.gamma_0:.2e}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}\n"
        )


class ExpCutoffPowerLaw(ParticleDistribution):
    r"""Class for power-law with an exponetial cutoff particle spectrum. 
    When called, the particle density :math:`n_e(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n_e(\gamma') = k \, \gamma'^{-p} exp(-\gamma'/\gamma_c) \, H(\gamma'; \gamma'_{\rm min}, \gamma'_{\rm max}) 

    Parameters
    ----------
    k : :class:`~astropy.units.Quantity`
        spectral normalisation
    p : float
        spectral index, note it is positive by definition, will change sign in the function
    gamma_c : float
        cutoff Lorentz factor of the particle distribution
    gamma_min : float
        minimum Lorentz factor of the particle distribution
    gamma_max : float
        maximum Lorentz factor of the particle distribution
    mass : `~astropy.units.Quantity`
        particle mass, default is the electron mass
    integrator : func
        function to be used for integration, default is :class:`~numpy.trapz`
    """

    def __init__(
        self,
        k=1e-13 * u.Unit("cm-3"),
        p=2.1,
        gamma_c=1e3,
        gamma_min=10,
        gamma_max=1e5,
        integrator=np.trapz,
    ):
        super().__init__(integrator)
        self.k = k
        self.p = p
        self.gamma_c = gamma_c
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    @property
    def parameters(self):
        return [self.k, self.p, self.gamma_c, self.gamma_min, self.gamma_max]

    @staticmethod
    def evaluate(gamma, k, p, gamma_c, gamma_min, gamma_max):
        return np.where(
            (gamma_min <= gamma) * (gamma <= gamma_max),
            k * gamma ** (-p) * np.exp(-gamma / gamma_c),
            0,
        )

    def __call__(self, gamma):
        return self.evaluate(
            gamma, self.k, self.p, self.gamma_c, self.gamma_min, self.gamma_max
        )

    @staticmethod
    def evaluate_SSA_integrand(gamma, k, p, gamma_c, gamma_min, gamma_max):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        prefactor = -(p + 2) / gamma + (-1 / gamma_c)

        return prefactor * ExpCutoffPowerLaw.evaluate(
            gamma, k, p, gamma_c, gamma_min, gamma_max
        )

    def SSA_integrand(self, gamma):
        return self.evaluate_SSA_integrand(
            gamma, self.k, self.p, self.gamma_c, self.gamma_min, self.gamma_max
        )

    def __str__(self):
        return (
            f"* {self.type} energy distribution\n"
            + f" - power law\n"
            + f" - k: {self.k:.2e}\n"
            + f" - p: {self.p:.2f}\n"
            + f" - gamma_c: {self.gamma_c:.2f}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}\n"
        )
