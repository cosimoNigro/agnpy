import numpy as np
import astropy.units as u
from astropy.constants import m_e


mec2 = m_e.to("erg", equivalencies=u.mass_energy())


__all__ = ["PowerLaw", "BrokenPowerLaw", "BrokenPowerLaw2"]


# the following functions describe the dependency on the Lorentz factor
# of the electron distributions, they do not depend on units
def _power_law(gamma, p, gamma_min, gamma_max):
    """simple power law"""
    pwl = np.power(gamma, -p)
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return pwl


def _power_law_integral(p, gamma_min, gamma_max):
    """analytical integral of the simple power law"""
    if np.isclose(p, 1.0):
        return np.log(gamma_max / gamma_min)
    else:
        return (np.power(gamma_max, 1 - p) - np.power(gamma_min, 1 - p)) / (1 - p)


def _power_law_times_gamma_integral(p, gamma_min, gamma_max):
    """analytical integral of the simple power law multiplied by gamma"""
    if np.isclose(p, 2.0):
        return np.log(gamma_max / gamma_min)
    else:
        return (np.power(gamma_max, 2 - p) - np.power(gamma_min, 2 - p)) / (2 - p)


def _power_law_ssa_integrand(gamma, p, gamma_min, gamma_max):
    """analytical form of the SSA integrand"""
    pwl = np.power(gamma, -p - 1)
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return (-p - 2) * pwl


def _broken_power_law(gamma, p1, p2, gamma_b, gamma_min, gamma_max):
    """power law with two spectral indexes"""
    pwl = np.power(gamma / gamma_b, -p1)
    # compute power law with the second spectral index
    p2_condition = gamma > gamma_b
    pwl[p2_condition] = np.power(gamma[p2_condition] / gamma_b, -p2)
    # return zero outside minimum and maximum Lorentz factor values
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return pwl


def _broken_power_law_integral(p1, p2, gamma_b, gamma_min, gamma_max):
    """analytical integral of the power law with two spectral indexes"""
    if np.allclose(p1, 1.0):
        term_1 = gamma_b * np.log(gamma_b / gamma_min)
    else:
        term_1 = gamma_b * (1 - np.power(gamma_min / gamma_b, 1 - p1)) / (1 - p1)
    if np.allclose(p2, 1.0):
        term_2 = gamma_b * np.log(gamma_max / gamma_b)
    else:
        term_2 = gamma_b * (np.power(gamma_max / gamma_b, 1 - p2) - 1) / (1 - p2)
    return term_1 + term_2


def _broken_power_law_times_gamma_integral(p1, p2, gamma_b, gamma_min, gamma_max):
    """analytical integral of the power law with two spectral indexes multiplied 
    by gamma"""
    if np.allclose(p1, 2.0):
        term_1 = np.power(gamma_b, 2) * np.log(gamma_b / gamma_min)
    else:
        term_1 = (
            np.power(gamma_b, 2)
            * (1 - np.power(gamma_min / gamma_b, 2 - p1))
            / (2 - p1)
        )
    if np.allclose(p2, 2.0):
        term_2 = np.power(gamma_b, 2) * np.log(gamma_max / gamma_b)
    else:
        term_2 = (
            np.power(gamma_b, 2)
            * (np.power(gamma_max / gamma_b, 2 - p2) - 1)
            / (2 - p2)
        )
    return term_1 + term_2


def _broken_power_law_ssa_integrand(gamma, p1, p2, gamma_b, gamma_min, gamma_max):
    """analytical form of the SSA integrand"""
    pwl = np.power(gamma / gamma_b, -p1 - 1)
    pwl_prefactor = (-p1 - 2) / gamma
    # compute power law with the second spectral index
    p2_condition = gamma > gamma_b
    pwl[p2_condition] = np.power(gamma[p2_condition] / gamma_b, -p2 - 1)
    pwl_prefactor[p2_condition] = (-p2 - 2) / gamma[p2_condition]
    # return zero outside minimum and maximum Lorentz factor values
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return pwl_prefactor * pwl


def _broken_power_law_2(gamma, p1, p2, gamma_b, gamma_min, gamma_max):
    """Tavecchio's Broken Power Law
    https://ui.adsabs.harvard.edu/abs/1998ApJ...509..608T/abstract"""
    pwl = np.power(gamma, -p1)
    p2_condition = gamma > gamma_b
    pwl[p2_condition] = np.power(gamma_b, p2 - p1) * np.power(gamma[p2_condition], -p2)
    # return zero outside minimum and maximum Lorentz factor values
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return pwl


def _broken_power_law_2_integral(p1, p2, gamma_b, gamma_min, gamma_max):
    """analytical integral of Tavecchio's broken power law"""
    term_1 = _power_law_integral(p1, gamma_min, gamma_b)
    term_2 = np.power(gamma_b, p2 - p1) * _power_law_integral(p2, gamma_b, gamma_max)
    return term_1 + term_2


def _broken_power_law_2_times_gamma_integral(p1, p2, gamma_b, gamma_min, gamma_max):
    """analytical integral of Tavecchio's broken power law multiplied by gamma"""
    term_1 = _power_law_times_gamma_integral(p1, gamma_min, gamma_b)
    term_2 = np.power(gamma_b, p2 - p1) * _power_law_times_gamma_integral(
        p2, gamma_b, gamma_max
    )
    return term_1 + term_2


def _broken_power_law_2_ssa_integrand(gamma, p1, p2, gamma_b, gamma_min, gamma_max):
    """analytical form of the SSA integrand"""
    pwl = (-p1 - 2) * np.power(gamma, -p1 - 1)
    # compute power law with the second spectral index
    p2_condition = gamma > gamma_b
    pwl[p2_condition] = (
        (-p2 - 2) * np.power(gamma_b, p2 - p1) * np.power(gamma[p2_condition], -p2 - 1)
    )
    # return zero outside minimum and maximum Lorentz factor values
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return pwl


class PowerLaw:
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
    """

    def __init__(self, k_e=1e-13 * u.Unit("cm-3"), p=2.0, gamma_min=10, gamma_max=1e5):
        self.k_e = k_e
        self.p = p
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, gamma):
        return self.k_e * _power_law(gamma, self.p, self.gamma_min, self.gamma_max)

    def __str__(self):
        return (
            f"* electron spectrum\n"
            + f" - power law\n"
            + f" - k_e: {self.k_e:.2e}\n"
            + f" - p: {self.p:.2f}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}\n"
        )

    @classmethod
    def from_normalised_density(cls, n_e_tot, p, gamma_min, gamma_max):
        r"""sets the normalisation :math:`k_e` from the total particle density 
        :math:`n_{e,\,tot}`"""
        k_e = n_e_tot / _power_law_integral(p, gamma_min, gamma_max)
        return cls(k_e.to("cm-3"), p, gamma_min, gamma_max)

    @classmethod
    def from_normalised_u_e(cls, u_e, p, gamma_min, gamma_max):
        r"""sets the normalisation :math:`k_e` from the total energy density 
        :math:`u_e`, Eq. 6.64 in [DermerMenon2009]_"""
        k_e = u_e / (mec2 * _power_law_times_gamma_integral(p, gamma_min, gamma_max))
        return cls(k_e.to("cm-3"), p, gamma_min, gamma_max)

    @classmethod
    def from_norm_at_gamma_1(cls, norm, p, gamma_min, gamma_max):
        r"""sets :math:`k_e` such that `norm` = :math:`n_e(\gamma=1)`."""
        return cls(norm.to("cm-3"), p, gamma_min, gamma_max)

    def SSA_integrand(self, gamma):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        return self.k_e * _power_law_ssa_integrand(
            gamma, self.p, self.gamma_min, self.gamma_max
        )


class BrokenPowerLaw:
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
    """

    def __init__(
        self,
        k_e=1e-13 * u.Unit("cm-3"),
        p1=2.0,
        p2=3.0,
        gamma_b=1e3,
        gamma_min=10,
        gamma_max=1e5,
    ):
        self.k_e = k_e
        self.p1 = p1
        self.p2 = p2
        self.gamma_b = gamma_b
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, gamma):
        return self.k_e * _broken_power_law(
            gamma, self.p1, self.p2, self.gamma_b, self.gamma_min, self.gamma_max,
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
            + f" - gamma_max: {self.gamma_max:.2e}\n"
        )

    @classmethod
    def from_normalised_density(cls, n_e_tot, p1, p2, gamma_b, gamma_min, gamma_max):
        r"""sets the normalisation :math:`k_e` from the total particle density 
        :math:`n_{e,\,tot}`"""
        k_e = n_e_tot / _broken_power_law_integral(
            p1, p2, gamma_b, gamma_min, gamma_max
        )
        return cls(k_e.to("cm-3"), p1, p2, gamma_b, gamma_min, gamma_max)

    @classmethod
    def from_normalised_u_e(cls, u_e, p1, p2, gamma_b, gamma_min, gamma_max):
        r"""sets the normalisation :math:`k_e` from the total energy density 
        :math:`u_e`, Eq. 6.64 in [DermerMenon2009]_"""
        k_e = u_e / (
            mec2
            * _broken_power_law_times_gamma_integral(
                p1, p2, gamma_b, gamma_min, gamma_max
            )
        )
        return cls(k_e.to("cm-3"), p1, p2, gamma_b, gamma_min, gamma_max)

    @classmethod
    def from_norm_at_gamma_1(cls, norm, p1, p2, gamma_b, gamma_min, gamma_max):
        r"""sets :math:`k_e` such that `norm` = :math:`n_e(\gamma=1)`."""
        k_e = norm.to("cm-3") * np.power(gamma_b, -p1)
        print(
            f"normalising broken power-law to value {norm:.2e} at gamma = 1, and {k_e: .2e} at gamma_b = {gamma_b:.2e}"
        )
        return cls(k_e, p1, p2, gamma_b, gamma_min, gamma_max)

    def SSA_integrand(self, gamma):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        return self.k_e * _broken_power_law_ssa_integrand(
            gamma, self.p1, self.p2, self.gamma_b, self.gamma_min, self.gamma_max,
        )


class BrokenPowerLaw2:
    r"""Broken power law as in Eq. 1 of [Tavecchio1998]_.
    When called, the particle density :math:`n_e(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n_e(\gamma') = k_e \left[
        \gamma'^{-p_1} \, H(\gamma'; \gamma'_{\rm min}, \gamma'_b) +
        \gamma'^{(p_2 - p_1)}_b \, \gamma'^{-p_2} \, H(\gamma'; \gamma'_{b}, \gamma'_{\rm max}) 
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

    def __init__(
        self,
        k_e=1e-13 * u.Unit("cm-3"),
        p1=2.0,
        p2=3.0,
        gamma_b=1e3,
        gamma_min=10,
        gamma_max=1e5,
    ):
        self.k_e = k_e
        self.p1 = p1
        self.p2 = p2
        self.gamma_b = gamma_b
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, gamma):
        return self.k_e * _broken_power_law_2(
            gamma, self.p1, self.p2, self.gamma_b, self.gamma_min, self.gamma_max,
        )

    def __str__(self):
        return (
            f"* electron spectrum\n"
            + f" - broken power law 2\n"
            + f" - k_e: {self.k_e:.2e}\n"
            + f" - p1: {self.p1:.2f}\n"
            + f" - p2: {self.p2:.2f}\n"
            + f" - gamma_b: {self.gamma_b:.2e}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}\n"
        )

    @classmethod
    def from_normalised_density(cls, n_e_tot, p1, p2, gamma_b, gamma_min, gamma_max):
        r"""sets the normalisation :math:`k_e` from the total particle density 
        :math:`n_{e,\,tot}`"""
        k_e = n_e_tot / _broken_power_law_2_integral(
            p1, p2, gamma_b, gamma_min, gamma_max
        )
        return cls(k_e.to("cm-3"), p1, p2, gamma_b, gamma_min, gamma_max)

    @classmethod
    def from_normalised_u_e(cls, u_e, p1, p2, gamma_b, gamma_min, gamma_max):
        r"""sets the normalisation :math:`k_e` from the total energy density 
        :math:`u_e`, Eq. 6.64 in [DermerMenon2009]_"""
        k_e = u_e / (
            mec2
            * _broken_power_law_2_times_gamma_integral(
                p1, p2, gamma_b, gamma_min, gamma_max
            )
        )
        return cls(k_e.to("cm-3"), p1, p2, gamma_b, gamma_min, gamma_max)

    @classmethod
    def from_norm_at_gamma_1(cls, norm, p1, p2, gamma_b, gamma_min, gamma_max):
        r"""sets :math:`k_e` such that `norm` = :math:`n_e(\gamma=1)`."""
        return cls(norm.to("cm-3"), p1, p2, gamma_b, gamma_min, gamma_max)

    def SSA_integrand(self, gamma):
        r"""(analytical) integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \frac{d}{d \gamma'} \left(\frac{n_e(\gamma)}{\gamma'^2}\right)`"""
        return self.k_e * _broken_power_law_2_ssa_integrand(
            gamma, self.p1, self.p2, self.gamma_b, self.gamma_min, self.gamma_max,
        )
