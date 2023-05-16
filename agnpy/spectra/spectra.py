# module containing the particle spectra
import numpy as np
import astropy.units as u
from astropy.constants import m_e, m_p
import matplotlib.pyplot as plt


class ParticleDistribution:
    """Base class grouping common functionalities to be used by all particles
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
            self.particle = "electrons"
        elif mass is m_p:
            self.mass = m_p
            self.particle = "protons"
        else:
            raise ValueError(
                f"No distribution for particles with mass {mass} is available."
            )
        self.mc2 = self.mass.to("erg", equivalencies=u.mass_energy())

    @staticmethod
    def integral(
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

    def integrate(self, gamma_low, gamma_up, gamma_power=0):
        """Integral of **this particular** particle distribution over the range
        gamma_low, gamma_up.

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
    def from_total_density(cls, n_tot, mass, **kwargs):
        r"""Set the normalisation of the particle distribution,
        :math:`k [{\rm cm}^{-3}]`, from the total particle density
        :math:`n_{\rm tot} [{\rm cm}^{-3}]`.

        Parameters
        ----------
        n_tot : :class:`~astropy.units.Quantity`
            total particle density (integral of :math:`n(\gamma)`), in cm-3
        mass : :class:`~astropy.units.Quantity`
            particle mass
        """
        # use gamma_min and gamma_max as integration limits
        if "gamma_min" in kwargs:
            gamma_min = kwargs.get("gamma_min")
        if "gamma_max" in kwargs:
            gamma_max = kwargs.get("gamma_max")
        k = n_tot / cls.integral(
            cls, gamma_low=gamma_min, gamma_up=gamma_max, gamma_power=0, k=1, **kwargs
        )
        return cls(k=k.to("cm-3"), **kwargs, mass=mass)

    @classmethod
    def from_total_energy_density(cls, u_tot, mass, **kwargs):
        r"""Set the normalisation of the particle distribution,
        :math:`k [{\rm cm}^{-3}]`, from the total energy density
        :math:`u_{\rm tot} [{\rm erg}{\rm cm}^{-3}]`, Eq. 6.64 in [DermerMenon2009]_.

        Parameters
        ----------
        u_tot : :class:`~astropy.units.Quantity`
            total energy density (integral of :math:`\gamma\,n(\gamma)`), in erg cm-3
        mass : :class:`~astropy.units.Quantity`
            particle mass
        """
        # use gamma_min and gamma_max as integration limits
        if "gamma_min" in kwargs:
            gamma_min = kwargs.get("gamma_min")
        if "gamma_max" in kwargs:
            gamma_max = kwargs.get("gamma_max")
        integral = cls.integral(
            cls, gamma_low=gamma_min, gamma_up=gamma_max, gamma_power=1, k=1, **kwargs
        )
        mc2 = mass.to("erg", equivalencies=u.mass_energy())
        k = u_tot / (mc2 * integral)
        return cls(k=k.to("cm-3"), **kwargs, mass=mass)

    @classmethod
    def from_density_at_gamma_1(cls, n_gamma_1, mass, **kwargs):
        r"""Set the normalisation of the particle distribution,
        :math:`k [{\rm cm}^{-3}]`, such that `norm` = :math:`n(\gamma=1)`.

        Parameters
        ----------
        n_gamma_1 : :class:`~astropy.units.Quantity`
            value :math:`n(\gamma)` should have at :math:`\gamma=1`, in cm-3
        mass : :class:`~astropy.units.Quantity`
            particle mass
        """
        k = n_gamma_1.to("cm-3") / cls.evaluate(1, 1, **kwargs)
        return cls(k=k.to("cm-3"), **kwargs, mass=mass)

    @classmethod
    def from_total_energy(cls, W, V, mass, **kwargs):
        r"""Set the normalisation of the particle distribution,
        :math:`k [{\rm cm}^{-3}]`, based on the total energy in particles
        :math:`W = m c^2 \, \int {\rm d}\gamma \, \gamma \, n(\gamma)`.

        Parameters
        ----------
        W : :class:`~astropy.units.Quantity`
            total energy in particles, in erg
        V : :class:`~astropy.units.Quantity`
            volume of the emission region, in cm^3
        mass : `~astropy.units.Quantity`
            particle mass
        """
        u = W / V
        return cls.from_total_energy_density(u, mass, **kwargs)

    def plot(self, gamma=None, gamma_power=0, ax=None, **kwargs):
        """Plot the particle energy distribution.

        Parameters
        ----------
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factors over which to plot the SED
        gamma_power : float
            power of gamma to raise the electron distribution
        ax : :class:`~matplotlib.axes.Axes`, optional
            Axis
        """
        ax = plt.gca() if ax is None else ax

        gamma = np.logspace(0, 8, 200) if gamma is None else gamma

        ax.loglog(gamma, np.power(gamma, gamma_power) * self.__call__(gamma), **kwargs)
        ax.set_xlabel(r"$\gamma$")

        if gamma_power == 0:
            ax.set_ylabel(r"$n(\gamma)\,/\,{\rm cm}^{-3}$")

        else:
            ax.set_ylabel(
                r"$\gamma^{"
                + str(gamma_power)
                + r"}$"
                + r"$\,n(\gamma)\,/\,{\rm cm}^{-3}$"
            )

        return ax
