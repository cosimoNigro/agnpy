# module containing the particles distributions
import numpy as np
from astropy.constants import m_e, m_p

__all__ = [
    "ParticleDistribution",
    "Electrons",
    "Protons"
]

class ParticleDistribution():
    """Base class grouping common functionalities to be used by all particle
    distributions.

    Parameters
    ----------
    energy_distribution : `~agnpy.spectra.EnergyDistribution`
        distribution of particles as a function of the energy (Lorentz Factor)
    
    """

    def __init__(self, energy_distribution, mass):
        self.energy_distribution = energy_distribution
        self.mass = mass

    @classmethod
    def from_normalised_density(cls, n_tot, **kwargs):
        r"""Set the normalisation :math:`k_{\rm part}` from the total particle
        volume density, :math:`n_{part,\,tot}`."""
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
    def from_normalised_energy_density(cls, u_part, mass, **kwargs):
        r"""Set the normalisation :math:`k_{\rm part}` from the total energy
        density :math:`u_{\rm part}`, Eq. 6.64 in [DermerMenon2009]_"""
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

class Electrons(ParticleDistribution):

    def __init__(self, energy_distribution):
        super().__init__(energy_distribution, m_e)

class Protons(ParticleDistribution):

    def __init__(self, energy_distribution):
        super().__init__(energy_distribution, m_p)
