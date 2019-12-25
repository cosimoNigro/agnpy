"""basic spectra for the electrons distributions"""
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
    """anayltical form of \gamma^2 d / d \gamma (n_e / \gamma^2)"""
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
    """analytical form of \gamma^2 d / d \gamma (n_e / \gamma^2)"""
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


def _broken_power_law2(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
    """Tavecchio's Broken Power Law
    https://ui.adsabs.harvard.edu/abs/1998ApJ...509..608T/abstract"""
    pwl = np.power(gamma, -p1)
    p2_condition = gamma > gamma_b
    pwl[p2_condition] = np.power(gamma_b, p2 - p1) * np.power(gamma[p2_condition], -p2)
    # return zero outside minimum and maximum Lorentz factor values
    null_condition = (gamma_min <= gamma) * (gamma <= gamma_max)
    pwl[~null_condition] = 0
    return k_e * pwl


def _broken_power_law2_ssa_integrand(gamma, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
    """analytical form \gamma^2 d / d \gamma (n_e / \gamma^2)"""
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
    """Class for power law spectrum initialization"""

    def __init__(self, k_e, p, gamma_min, gamma_max):
        self.k_e = k_e
        self.p = p
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, gamma):
        return _power_law(gamma, self.k_e, self.p, self.gamma_min, self.gamma_max)

    @classmethod
    def from_normalised_u_e(cls, u_e, p, gamma_min, gamma_max):
        """set k_e in order to normalise to total energy density u_e
        normalization of the electron distribution Eq. 6.64 in [1]"""
        k_e_num = (p - 2) * u_e
        k_e_denum = MEC2 * (np.power(gamma_min, 2 - p) - np.power(gamma_max, 2 - p))
        k_e = (k_e_num / k_e_denum).to("cm-3")
        print(f"normalising power-law to total energy density u_e: {u_e:.2e}")
        return cls(k_e, p, gamma_min, gamma_max)

    @classmethod
    def from_normalised_density(cls, norm, p, gamma_min, gamma_max):
        """set k_e in order to normalise the total particle density"""
        k_e_num = (p - 1) * norm
        k_e_denum = np.power(gamma_min, 1 - p) - np.power(gamma_max, 1 - p)
        k_e = (k_e_num / k_e_denum).to("cm-3")
        print(f"normalising power-law to total particle density: {norm:.2e}")
        return cls(k_e, p, gamma_min, gamma_max)

    def SSA_integrand(self, gamma):
        """integrand for the synchrotron self absorption"""
        return _power_law_ssa_integrand(
            gamma, self.k_e, self.p, self.gamma_min, self.gamma_max
        )


class BrokenPowerLaw:
    """Class for two-indexes power law spectrum initialization"""

    def __init__(self, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
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

    @classmethod
    def from_normalised_u_e(cls, u_e, p1, p2, gamma_b, gamma_min, gamma_max):
        """set k_e in order to normalise to total energy density u_e"""
        denum_prefactor = MEC2 * np.power(gamma_b, 2)
        denum_term_1 = (1 - np.power(gamma_min / gamma_b, 2 - p1)) / (2 - p1)
        denum_term_2 = (np.power(gamma_max / gamma_b, 2 - p2) - 1) / (2 - p2)
        k_e = (u_e / (denum_prefactor * (denum_term_1 + denum_term_2))).to("cm-3")
        print(f"normalising broken power-law to total energy density u_e: {u_e:.2e}")
        return cls(k_e, p1, p2, gamma_b, gamma_min, gamma_max)

    @classmethod
    def from_normalised_density(cls, norm, p1, p2, gamma_b, gamma_min, gamma_max):
        """set k_e in order to normalise the total particle density"""
        k_e_denum_1 = (gamma_min * np.power(gamma_min / gamma_b, -p1) - gamma_b) / (
            p1 - 1
        )
        k_e_denum_2 = (gamma_b - gamma_max * np.power(gamma_max / gamma_b, -p2)) / (
            p2 - 1
        )
        k_e = (norm / (k_e_denum_1 + k_e_denum_2)).to("cm-3")
        print(f"normalising broken power-law to total particle density: {norm:.2e}")
        return cls(k_e, p1, p2, gamma_b, gamma_min, gamma_max)

    def SSA_integrand(self, gamma):
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
    """Smoothly broken power law as in Tavecchio et al. (1998)
    https://ui.adsabs.harvard.edu/#abs/1998ApJ...509..608T/abstract"""

    def __init__(self, k_e, p1, p2, gamma_b, gamma_min, gamma_max):
        self.k_e = k_e
        self.p1 = p1
        self.p2 = p2
        self.gamma_b = gamma_b
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, gamma):
        return _broken_power_law2(
            gamma,
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )

    @classmethod
    def from_normalised_u_e(cls, u_e, p1, p2, gamma_b, gamma_min, gamma_max):
        """set k_e in order to normalise the total particle density"""
        k_e_denum_1 = (np.power(gamma_b, 2 - p1) - np.power(gamma_min, 2 - p1)) / (
            2 - p1
        )
        k_e_denum_2 = (
            np.power(gamma_b, p2 - p1)
            * (np.power(gamma_max, 2 - p1) - np.power(gamma_b, 2 - p1))
            / (2 - p2)
        )
        k_e = (u_e / (k_e_denum_1 + k_e_denum_2)).to("cm-3")
        print(f"normalising smooth broken power-law to total energy density: {u_e:.2e}")
        return cls(k_e, p1, p2, gamma_b, gamma_min, gamma_max)

    @classmethod
    def from_normalised_density(cls, norm, p1, p2, gamma_b, gamma_min, gamma_max):
        """set k_e in order to normalise the total particle density"""
        k_e_denum_1 = (np.power(gamma_b, 1 - p1) - np.power(gamma_min, 1 - p1)) / (
            1 - p1
        )
        k_e_denum_2 = (
            np.power(gamma_b, p2 - p1)
            * (np.power(gamma_max, 1 - p2) - np.power(gamma_b, 1 - p2))
            / (1 - p2)
        )
        k_e = (norm / (k_e_denum_1 + k_e_denum_2)).to("cm-3")
        print(
            f"normalising smooth broken power-law to total particle density: {norm:.2e}"
        )
        return cls(k_e, p1, p2, gamma_b, gamma_min, gamma_max)

    def SSA_integrand(self, gamma):
        """integrand for the synchrotron self absorption"""
        return _broken_power_law2_ssa_integrand(
            gamma,
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )
