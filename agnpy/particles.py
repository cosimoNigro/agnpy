import numpy as np
import astropy.constants as const
from astropy.coordinates import Distance
import astropy.units as u
from .utils import (
    _power_law,
    _broken_power_law,
    _power_law_ssa_integrand,
    _broken_power_law_ssa_integrand,
    _broken_power_law2,
    _broken_power_law2_ssa_integrand,
)

C = const.c.cgs.value
MEC2 = (const.m_e * const.c * const.c).cgs.value

__all__ = ["Blob", "PowerLaw", "BrokenPowerLaw", "BrokenPowerLaw2"]


class Blob:
    """class to represent the emitting region and the distribution of electrons
    it contains.

    Parameters
    ----------
    R_b : `~astropy.units.Quantity`
        size of the emitting region
    z : float
        redshift of the source
    delta_D : float
        Doppler factor of the relativistic outflow
    Gamma :  float
        Lorentz factor of the relativistic outflow
    B : `~astropy.units.Quantity`
        magnetic field in the blob (Gauss)
    spectrum_norm : `~astropy.units.Quantity`
        normalization of the electron spectra, can be, following
        the notation in [1]:
        - k_e : power law spectrum normalization in cm-3
        - u_e : total energy density in non thermal electrons in erg cm-3
        - W_e : total non thermal electron energy in erg
    spectrum_dict : dictionary
        dictionary containing type and spectral shape information, e.g.:
        type : "PowerLaw"
        parameters :
            p : 2
            gamma_min : 1e2
            gamma_max : 1e5
    Reference
    ---------
    [1] : Dermer, Menon; High Energy Radiation From Black Holes;
    Princeton Series in Astrophysics

    N.B.:
    All these quantities are defined in the comoving frame so they are actually
    primed quantities, when referring the notation in [1]
    """

    def __init__(
        self, R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict, gamma_size=200
    ):
        self.R_b = R_b.to("cm").value
        self.z = z
        self.d_L = Distance(z=self.z).cgs.value
        self.V_b = 4 / 3 * np.pi * np.power(self.R_b, 3)  # cm-3
        self.delta_D = delta_D
        self.Gamma = Gamma
        self.Beta = np.sqrt(1 - 1 / np.power(self.Gamma, 2))
        # viewing angle
        self.mu_s = (1 - 1 / (self.Gamma * self.delta_D)) / self.Beta
        self.B = B.to("G").value
        self.spectrum_norm = spectrum_norm
        self.spectrum_dict = spectrum_dict
        # size of the electron Lorentz factor grid
        self.gamma_size = gamma_size
        self.gamma_min = self.spectrum_dict["parameters"]["gamma_min"]
        self.gamma_max = self.spectrum_dict["parameters"]["gamma_max"]
        # grid of Lorentz factor for the integration in the blob comoving frame
        self.gamma = np.logspace(
            np.log10(self.gamma_min), np.log10(self.gamma_max), self.gamma_size
        )
        # grid of Lorentz factors for integration in the external frame
        self.gamma_to_integrate = np.logspace(1, 9, self.gamma_size)

        if self.spectrum_dict["type"] == "PowerLaw":
            if self.spectrum_norm.unit == u.Unit("cm-3"):
                # the normalization is directly k_e the power law normalization
                self.n_e = PowerLaw.from_normalized_density(
                    self.spectrum_norm.value, **self.spectrum_dict["parameters"]
                )
            if self.spectrum_norm.unit == u.Unit("erg cm-3"):
                # the normalization is u_e, the energy density, in this case
                self.n_e = PowerLaw.from_normalized_u_e(
                    self.spectrum_norm.value, **self.spectrum_dict["parameters"]
                )
            if self.spectrum_norm.unit == u.Unit("erg"):
                # the normalization W_e, the total energy, in this case
                _u_e = self.spectrum_norm.value / self.V_b
                self.n_e = PowerLaw.from_normalized_u_e(
                    _u_e, **self.spectrum_dict["parameters"]
                )

        if self.spectrum_dict["type"] == "BrokenPowerLaw":
            if self.spectrum_norm.unit == u.Unit("cm-3"):
                # the normalization is directly k_e in this case
                self.n_e = BrokenPowerLaw.from_normalized_density(
                    self.spectrum_norm.value, **self.spectrum_dict["parameters"]
                )
            if self.spectrum_norm.unit == u.Unit("erg cm-3"):
                # the normalization is u_e, the energy density, in this case
                self.n_e = BrokenPowerLaw.from_normalized_u_e(
                    self.spectrum_norm.value, **spectrum_dict["parameters"]
                )
            if self.spectrum_norm.unit == u.Unit("erg"):
                # the normalization is directly W_e, the total energy, in this case
                _u_e = self.spectrum_norm.value / self.V_b
                self.n_e = BrokenPowerLaw.from_normalized_u_e(
                    _u_e, **self.spectrum_dict["parameters"]
                )

        if self.spectrum_dict["type"] == "BrokenPowerLaw2":
            if self.spectrum_norm.unit == u.Unit("cm-3"):
                # the normalization is directly k_e in this case
                self.n_e = BrokenPowerLaw2.from_normalized_density(
                    self.spectrum_norm.value, **self.spectrum_dict["parameters"]
                )
            else:
                print("only normalization from total density from this class!")

    def set_gamma_size(self, gamma_size):
        """change size of electron Lorentz factor grid, update gamma grids"""
        self.gamma_size = gamma_size
        self.gamma = np.logspace(
            np.log10(self.gamma_min), np.log10(self.gamma_max), self.gamma_size
        )
        self.gamma_to_integrate = np.logspace(1, 9, self.gamma_size)

    def N_e(self, gamma):
        """N_e represents the particle number"""
        return self.V_b * self.n_e(gamma)

    @property
    def norm(self):
        return np.trapz(self.n_e(self.gamma), self.gamma) * u.Unit("cm-3")

    @property
    def u_e(self):
        """numerical check that u_e si correctly computed"""
        return (
            MEC2
            * np.trapz(self.gamma * self.n_e(self.gamma), self.gamma)
            * u.Unit("erg cm-3")
        )

    @property
    def W_e(self):
        """numerical check that W_e is correctly computed"""
        return (
            MEC2
            * np.trapz(self.gamma * self.N_e(self.gamma), self.gamma)
            * u.Unit("erg")
        )


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
    def from_normalized_u_e(cls, u_e, p, gamma_min, gamma_max):
        """set k_e in order to normalize to total energy density u_e
        """
        # normalization of the electron distribution Eq. 6.64 in [1]
        k_e_num = (p - 2) * u_e
        k_e_denom = MEC2 * (np.power(gamma_min, 2 - p) - np.power(gamma_max, 2 - p))
        k_e = k_e_num / k_e_denom
        print(f"normalizing power-law to total energy density u_e: {u_e:.2e} erg cm-3")
        return cls(k_e, p, gamma_min, gamma_max)

    @classmethod
    def from_normalized_density(cls, norm, p, gamma_min, gamma_max):
        """set k_e in order to normalize the total particle density
        """
        k_e_num = (p - 1) * norm
        k_e_denom = np.power(gamma_min, 1 - p) - np.power(gamma_max, 1 - p)
        k_e = k_e_num / k_e_denom
        print(f"normalizing power-law to total particle density: {norm:.2e} cm-3")
        return cls(k_e, p, gamma_min, gamma_max)

    def SSA_integrand(self, gamma):
        """integrand for the synchrotron self absorption
        """
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
    def from_normalized_u_e(cls, u_e, p1, p2, gamma_b, gamma_min, gamma_max):
        """set k_e in order to normalize to total energy density u_e
        """
        denom_prefactor = MEC2 * np.power(gamma_b, 2)
        denom_term_1 = (1 - np.power(gamma_min / gamma_b, 2 - p1)) / (2 - p1)
        denom_term_2 = (np.power(gamma_max / gamma_b, 2 - p2) - 1) / (2 - p2)
        k_e = u_e / (denom_prefactor * (denom_term_1 + denom_term_2))
        print(f"normalizing power-law to total energy density u_e: {u_e} erg cm-3")
        return cls(k_e, p1, p2, gamma_b, gamma_min, gamma_max)

    @classmethod
    def from_normalized_density(cls, norm, p1, p2, gamma_b, gamma_min, gamma_max):
        """set k_e in order to normalize the total particle density
        """
        k_e_denom_1 = (gamma_min * np.power(gamma_min / gamma_b, -p1) - gamma_b) / (
            p1 - 1
        )
        k_e_denom_2 = (gamma_b - gamma_max * np.power(gamma_max / gamma_b, -p2)) / (
            p2 - 1
        )
        k_e = norm / (k_e_denom_1 + k_e_denom_2)
        print(f"normalizing power-law to total particle density: {norm:.2e} cm-3")
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
    """Smoothly broken power law as in Tavecchio et al 1998
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
    def from_normalized_density(cls, norm, p1, p2, gamma_b, gamma_min, gamma_max):
        """set k_e in order to normalize the total particle density
        """
        k_e_denom_1 = (np.power(gamma_min, 1 - p1) - np.power(gamma_b, 1 - p1)) / (
            p1 - 1
        )
        k_e_denom_2 = (
            np.power(gamma_b, p2 - p1)
            * (np.power(gamma_b, 1 - p2) - np.power(gamma_max, 1 - p2))
            / (p2 - 1)
        )
        k_e = norm / (k_e_denom_1 + k_e_denom_2)
        print(f"normalizing power-law to total particle density: {norm:.2e} cm-3")
        return cls(k_e, p1, p2, gamma_b, gamma_min, gamma_max)

    def SSA_integrand(self, gamma):
        """integrand for the synchrotron self absorption
        """
        return _broken_power_law2_ssa_integrand(
            gamma,
            self.k_e,
            self.p1,
            self.p2,
            self.gamma_b,
            self.gamma_min,
            self.gamma_max,
        )
