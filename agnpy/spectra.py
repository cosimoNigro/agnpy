import numpy as np
import astropy.units as u
import astropy.constants as const


MEC2 = (const.m_e * const.c * const.c).cgs


__all__ = ["PowerLaw", "BrokenPowerLaw", "BrokenPowerLaw2"]


# in the following functions k_e is supposed to have dimensions cm-3
def _power_law(gamma, k_e, p, gamma_min, gamma_max):
    """simple power law"""
    pwl = np.power(gamma, -p)
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return k_e * pwl


def _power_law_ssa_integrand(gamma, k_e, p, gamma_min, gamma_max):
    """analytical form of the SSA integrand"""
    pwl = np.power(gamma, -p - 1)
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return (-p - 2) * k_e * pwl


def _broken_power_law(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
    """power law with two spectral indices"""
    pwl = np.power(gamma / gamma_b, -p1)
    # compute power law with the second spectral index
    p2_condition = gamma > gamma_b
    pwl[p2_condition] = np.power(gamma[p2_condition] / gamma_b, -p2)
    # return zero outside minimum and maximum Lorentz factor values
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return k_e * pwl


def _broken_power_law_ssa_integrand(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
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
    return k_e * pwl_prefactor * pwl


def _broken_power_law_2(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
    """Tavecchio's Broken Power Law
    https://ui.adsabs.harvard.edu/abs/1998ApJ...509..608T/abstract"""
    pwl = np.power(gamma, -p1)
    p2_condition = gamma > gamma_b
    pwl[p2_condition] = np.power(gamma_b, p2 - p1) * np.power(gamma[p2_condition], -p2)
    # return zero outside minimum and maximum Lorentz factor values
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return k_e * pwl


def _broken_power_law_2_ssa_integrand(
    gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max
):
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
    return k_e * pwl


class PowerLaw:
    """Class for power-law particle spectrum. 
    When called, the particle density :math:`n_e(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n_e(\gamma') = k_e \, \gamma'^{-p} \, H(\gamma'; \gamma'_{min}, \gamma'_{max}) 

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
        return _power_law(gamma, self.k_e, self.p, self.gamma_min, self.gamma_max)

    def __str__(self):
        summary = (
            f"* electron spectrum\n"
            + f" - power law\n"
            + f" - k_e: {self.k_e:.2e}\n"
            + f" - p: {self.p:.2f}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}\n"
        )
        return summary

    @classmethod
    def from_normalised_u_e(cls, u_e, p, gamma_min, gamma_max):
        """set the normalisation :math:`k_e` from the total energy density 
        :math:`u_e`, Eq. 6.64 in [DermerMenon2009]_"""
        # avoid and exact value of 2 for the index that will make the analytical
        # simplification diverge
        if np.isclose(p, 2.0):
            p += 1e-3
        k_e_num = (p - 2) * u_e
        k_e_denum = MEC2 * (np.power(gamma_min, 2 - p) - np.power(gamma_max, 2 - p))
        k_e = (k_e_num / k_e_denum).to("cm-3")
        print(f"normalising power-law to total energy density u_e: {u_e:.2e}")
        return cls(k_e, p, gamma_min, gamma_max)

    @classmethod
    def from_normalised_density(cls, norm, p, gamma_min, gamma_max):
        """set the normalisation :math:`k_e` from the total particle density 
        :math:`n_{e,\,tot}`"""
        # avoid and exact value of 1 for the index that will make the analytical
        # simplification diverge
        if np.isclose(p, 1.0):
            p += 1e-3
        k_e_num = (p - 1) * norm
        k_e_denum = np.power(gamma_min, 1 - p) - np.power(gamma_max, 1 - p)
        k_e = (k_e_num / k_e_denum).to("cm-3")
        print(f"normalising power-law to total particle density: {norm:.2e}")
        return cls(k_e, p, gamma_min, gamma_max)

    @classmethod
    def from_norm_at_gamma_1(cls, norm, p, gamma_min, gamma_max):
        """sets :math:`k_e` such that `norm` = :math:`n_e(\gamma=1)`."""
        print(f"normalising power-law to value {norm:.2e} at gamma = 1")
        return cls(norm.to("cm-3"), p, gamma_min, gamma_max)

    def SSA_integrand(self, gamma):
        """integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \\frac{d}{d \gamma'} \left(\\frac{n_e}{\gamma'^2}\\right)`"""
        return _power_law_ssa_integrand(
            gamma, self.k_e, self.p, self.gamma_min, self.gamma_max
        )


class BrokenPowerLaw:
    """Class for broken power-law particle spectrum
    When called, the particle density :math:`n_e(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n_e(\gamma') = k_e \left[
        \left(\\frac{\gamma'}{\gamma_b}\\right)^{-p_1} \, H(\gamma'; \gamma'_{min}, \gamma'_b) +
        \left(\\frac{\gamma'}{\gamma_b}\\right)^{-p_2} \, H(\gamma'; \gamma'_{b}, \gamma'_{max}) 
        \\right]

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
        return _broken_power_law(
            gamma,
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )

    def __str__(self):
        summary = (
            f"* electron spectrum\n"
            + f" - broken power law\n"
            + f" - k_e: {self.k_e:.2e}\n"
            + f" - p1: {self.p1:.2f}\n"
            + f" - p2: {self.p2:.2f}\n"
            + f" - gamma_b: {self.gamma_b:.2e}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}\n"
        )
        return summary

    @classmethod
    def from_normalised_u_e(cls, u_e, p1, p2, gamma_b, gamma_min, gamma_max):
        """set the normalisation :math:`k_e` from the total energy density 
        :math:`u_e`, Eq. 6.64 in [DermerMenon2009]_"""
        if np.isclose(p1, 2.0):
            p1 += 1e-3
        if np.isclose(p2, 2.0):
            p2 += 1e-3
        denum_prefactor = MEC2 * np.power(gamma_b, 2)
        denum_term_1 = (1 - np.power(gamma_min / gamma_b, 2 - p1)) / (2 - p1)
        denum_term_2 = (np.power(gamma_max / gamma_b, 2 - p2) - 1) / (2 - p2)
        k_e = (u_e / (denum_prefactor * (denum_term_1 + denum_term_2))).to("cm-3")
        print(f"normalising broken power-law to total energy density u_e: {u_e:.2e}")
        return cls(k_e, p1, p2, gamma_b, gamma_min, gamma_max)

    @classmethod
    def from_normalised_density(cls, norm, p1, p2, gamma_b, gamma_min, gamma_max):
        """set the normalisation :math:`k_e` from the total particle density 
        :math:`n_{e,\,tot}`"""
        if np.isclose(p1, 1.0):
            p1 += 1e-3
        if np.isclose(p2, 1.0):
            p2 += 1e-3
        k_e_denum_1 = (gamma_min * np.power(gamma_min / gamma_b, -p1) - gamma_b) / (
            p1 - 1
        )
        k_e_denum_2 = (gamma_b - gamma_max * np.power(gamma_max / gamma_b, -p2)) / (
            p2 - 1
        )
        k_e = (norm / (k_e_denum_1 + k_e_denum_2)).to("cm-3")
        print(f"normalising broken power-law to total particle density: {norm:.2e}")
        return cls(k_e, p1, p2, gamma_b, gamma_min, gamma_max)

    @classmethod
    def from_norm_at_gamma_1(cls, norm, p1, p2, gamma_b, gamma_min, gamma_max):
        """sets :math:`k_e` such that `norm` = :math:`n_e(\gamma=1)`."""
        k_e = norm.to("cm-3") * np.power(gamma_b, -p1)
        print(
            f"normalising broken power-law to value {k_e:.2e} at gamma = 1, and {norm: .2e} at gamma = gamma_b = {gamma_b:.2e}"
        )
        return cls(k_e, p1, p2, gamma_b, gamma_min, gamma_max)

    def SSA_integrand(self, gamma):
        """integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \\frac{d}{d \gamma'} \left(\\frac{n_e}{\gamma'^2}\\right)`"""
        return _broken_power_law_ssa_integrand(
            gamma,
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )


class BrokenPowerLaw2:
    """Broken power law as in Eq. 1 of [Tavecchio1998]_.
    When called, the particle density :math:`n_e(\gamma)` in :math:`\mathrm{cm}^{-3}` is returned.

    .. math::
        n_e(\gamma') = k_e \left[
        \gamma'^{-p_1} \, H(\gamma'; \gamma'_{min}, \gamma'_b) +
        \gamma'^{(p_2 - p_1)}_b \, \gamma'^{-p_2} \, H(\gamma'; \gamma'_{b}, \gamma'_{max}) 
        \\right]

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
        return _broken_power_law_2(
            gamma,
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )

    def __str__(self):
        summary = (
            f"* electron spectrum\n"
            + f" - broken power law 2\n"
            + f" - k_e: {self.k_e:.2e}\n"
            + f" - p1: {self.p1:.2f}\n"
            + f" - p2: {self.p2:.2f}\n"
            + f" - gamma_b: {self.gamma_b:.2e}\n"
            + f" - gamma_min: {self.gamma_min:.2e}\n"
            + f" - gamma_max: {self.gamma_max:.2e}\n"
        )
        return summary

    @classmethod
    def from_normalised_u_e(cls, u_e, p1, p2, gamma_b, gamma_min, gamma_max):
        """set the normalisation :math:`k_e` from the total energy density 
        :math:`u_e`, Eq. 6.64 in [DermerMenon2009]_"""
        if np.isclose(p1, 2.0):
            p1 += 1e-3
        if np.isclose(p2, 2.0):
            p2 += 1e-3
        k_e_denum_1 = (np.power(gamma_b, 2 - p1) - np.power(gamma_min, 2 - p1)) / (
            2 - p1
        )
        k_e_denum_2 = (
            np.power(gamma_b, p2 - p1)
            * (np.power(gamma_max, 2 - p2) - np.power(gamma_b, 2 - p2))
            / (2 - p2)
        )
        k_e = (u_e / (MEC2 * (k_e_denum_1 + k_e_denum_2))).to("cm-3")
        print(f"normalising broken power-law 2 to total energy density: {u_e:.2e}")
        return cls(k_e, p1, p2, gamma_b, gamma_min, gamma_max)

    @classmethod
    def from_normalised_density(cls, norm, p1, p2, gamma_b, gamma_min, gamma_max):
        """set the normalisation :math:`k_e` from the total particle density 
        :math:`n_{e,\,tot}`"""
        if np.isclose(p1, 1.0):
            p1 += 1e-3
        if np.isclose(p2, 1.0):
            p2 += 1e-3
        k_e_denum_1 = (np.power(gamma_b, 1 - p1) - np.power(gamma_min, 1 - p1)) / (
            1 - p1
        )
        k_e_denum_2 = (
            np.power(gamma_b, p2 - p1)
            * (np.power(gamma_max, 1 - p2) - np.power(gamma_b, 1 - p2))
            / (1 - p2)
        )
        k_e = (norm / (k_e_denum_1 + k_e_denum_2)).to("cm-3")
        print(f"normalising broken power-law 2 to total particle density: {norm:.2e}")
        return cls(k_e, p1, p2, gamma_b, gamma_min, gamma_max)

    @classmethod
    def from_norm_at_gamma_1(cls, norm, p1, p2, gamma_b, gamma_min, gamma_max):
        """sets :math:`k_e` such that `spectrum_norm` = :math:`n_e(\gamma=1)`."""
        print(f"normalising broken power-law 2 to value {norm:.2e} at gamma = 1")
        return cls(norm.to("cm-3"), p1, p2, gamma_b, gamma_min, gamma_max)

    def SSA_integrand(self, gamma):
        """integrand for the synchrotron self-absorption:
        :math:`\gamma'^2 \\frac{d}{d \gamma'} \left(\\frac{n_e}{\gamma'^2}\\right)`"""
        return _broken_power_law_2_ssa_integrand(
            gamma,
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )
