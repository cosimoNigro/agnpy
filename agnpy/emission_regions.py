import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Distance
from .spectra import PowerLaw, BrokenPowerLaw, BrokenPowerLaw2


MEC2 = (const.m_e * const.c * const.c).cgs


__all__ = ["Blob"]


class Blob:
    """Simple spherical emission region

    Parameters
    ----------
    R_b : `~astropy.units.Quantity`
        radius of the blob
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
        self.R_b = R_b.to("cm")
        self.z = z
        self.d_L = Distance(z=self.z).cgs
        self.V_b = 4 / 3 * np.pi * np.power(self.R_b, 3)
        self.delta_D = delta_D
        self.Gamma = Gamma
        self.Beta = np.sqrt(1 - 1 / np.power(self.Gamma, 2))
        # viewing angle
        self.mu_s = (1 - 1 / (self.Gamma * self.delta_D)) / self.Beta
        self.B = B.to("G")
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
        # assign the spectral density
        if spectrum_dict["type"] == "PowerLaw":
            _model = PowerLaw
        if spectrum_dict["type"] == "BrokenPowerLaw":
            _model = BrokenPowerLaw
        if spectrum_dict["type"] == "BrokenPowerLaw2":
            _model = BrokenPowerLaw2
        if spectrum_norm.unit == u.Unit("cm-3"):
            self.n_e = _model.from_normalised_density(
                spectrum_norm, **spectrum_dict["parameters"]
            )
        if spectrum_norm.unit == u.Unit("erg cm-3"):
            self.n_e = _model.from_normalised_u_e(u_e, **spectrum_dict["parameters"])
        if spectrum_norm.unit == u.Unit("erg"):
            u_e = (spectrum_norm / self.V_b).to("erg cm-3")
            self.n_e = _model.from_normalised_u_e(u_e, **spectrum_dict["parameters"])

    def __str__(self):
        """printable summary of the blob"""
        summary = (
            "* spherical emission region\n"
            + f" - R_b (radius of the blob): {self.R_b:.2e}\n"
            + f" - V_b (volume of the blob): {self.V_b:.2e}\n"
            + f" - z (source redshift): {self.z:.2f}\n"
            + f" - d_L (source luminosity distance):{self.d_L:.2e}\n"
            + f" - delta_D (blob Doppler factor): {self.delta_D:.2e}\n"
            + f" - Gamma (blob Lorentz factor): {self.delta_D:.2e}\n"
            + f" - Beta (blob relativistic velocity): {self.Beta:.2e}\n"
            + f" - mu_s (cosine of the jet viewing angle): {self.mu_s:.2e}\n"
            + f" - B (magnetic field tangled to the jet): {self.B:.2e}\n"
            + f" - electron spectra:\n"
            + f"  |- normalisation: {self.spectrum_norm:.2e}\n"
            + f"  |- spectral function: {self.spectrum_dict['type']}\n"
            + f"  |- gamma_min (minimum Lorentz factor): {self.gamma_min:.2e}\n"
            + f"  '- gamma_max (maximum Lorentz factor): {self.gamma_max:.2e}\n"
        )
        return summary

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
        return np.trapz(self.n_e(self.gamma), self.gamma)

    @property
    def u_e(self):
        """numerical check that u_e si correctly computed"""
        return MEC2 * np.trapz(self.gamma * self.n_e(self.gamma), self.gamma)

    @property
    def W_e(self):
        """numerical check that W_e is correctly computed"""
        return MEC2 * np.trapz(self.gamma * self.N_e(self.gamma), self.gamma)
