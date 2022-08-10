"""This module describes the emission regions responsible for the
acceleration of particles to relativistic energies. Beside physical quantities
related to the emission itself it contains the electrons energy distributions"""
import numpy as np
import astropy.units as u
from astropy.constants import c, sigma_T
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
from .. import spectra
from ..utils.conversion import mec2, B_to_cgs
from ..utils.plot import plot_eed

__all__ = ["Blob"]


def init_spectrum_norm_dict(norm, spectrum_dict, norm_type="integral", V_b=None):
    """initialize a spectrum from a normalisation and a spectrum dictionary of
    the following type

    .. code-block:: python

        spectrum_dict = {
            "type": "PowerLaw",
            "parameters": {
                "p": 2.8,
                "gamma_min": 1e2,
                "gamma_max": 1e7
            }
        }

    Parameters
    ----------
    norm : :class:`~astropy.units.Quantity`
        normalisation of the spectrum, might be in cm-3, erg cm-3 or erg,
        if erg are specified then V_b has to be provided
    spectrum_dict : dictionary
        dictionary with the spectrum type and parameters
    norm_type : ["integral", "differential", "gamma=1"]
        normalisation type
    V_b : :class:`~astropy.units.Quantity`
        volume of the emission region, to be provided only if the normalisation
        is in erg
    """
    model = getattr(spectra, spectrum_dict["type"])

    if norm.unit in (u.Unit("erg"), u.Unit("erg cm-3")) and norm_type != "integral":
        raise NameError(
            "Normalisation different than 'integral' available only for 'spectrum_norm' in cm-3"
        )

    # check the units of the normalisation
    # cm-3 is the only one allowing more than one normalisation type
    if norm.unit == u.Unit("cm-3"):
        if norm_type == "differential":
            final_model = model(norm, **spectrum_dict["parameters"])
        elif norm_type == "gamma=1":
            final_model = model.from_norm_at_gamma_1(
                norm, **spectrum_dict["parameters"]
            )
        elif norm_type == "integral":
            final_model = model.from_normalised_density(
                norm, **spectrum_dict["parameters"]
            )

    elif norm.unit == u.Unit("erg cm-3"):
        final_model = model.from_normalised_energy_density(
            norm, **spectrum_dict["parameters"]
        )

    elif norm.unit == u.Unit("erg"):
        if V_b is None:
            raise ValueError(
                "if normalisation in erg provided, the volume V_b must be specified"
            )
        final_model = model.from_total_energy(norm, V_b, **spectrum_dict["parameters"])

    return final_model


class Blob:
    r"""Simple spherical emission region.

    **Note:** all these quantities are defined in the comoving frame so they are actually
    primed quantities, when referring the notation in [DermerMenon2009]_.

    Parameters
    ----------
    R_b : :class:`~astropy.units.Quantity`
        radius of the blob
    z : float
        redshift of the source
    delta_D : float
        Doppler factor of the relativistic outflow
    Gamma : float
        Lorentz factor of the relativistic outflow
    B : :class:`~astropy.units.Quantity`
        magnetic field in the blob (Gauss)
    xi : float
        acceleration coefficient :math:`\xi` for first-order Fermi acceleration
        :math:`(\mathrm{d}E/\mathrm{d}t \propto v \approx c)`
        used to compute limits on the maximum Lorentz factor via
        :math:`(\mathrm{d}E/\mathrm{d}t)_{\mathrm{acc}} = \xi c E / R_L`

    spectrum_norm : :class:`~astropy.units.Quantity`
        normalisation of the electron spectra, by default can be, following
        the notation in [DermerMenon2009]_:

            - :math:`n_{e,\,tot}`: total electrons density, in :math:`\mathrm{cm}^{-3}`
            - :math:`u_e` : total electrons energy density, in :math:`\mathrm{erg}\,\mathrm{cm}^{-3}`
            - :math:`W_e` : total energy in electrons, in :math:`\mathrm{erg}`

        see `spectrum_norm_type` for more details on the normalisation

    spectrum_dict : dictionary
        dictionary containing type and spectral shape information, e.g.:

        .. code-block:: python

            spectrum_dict = {
                "type": "PowerLaw",
                "parameters": {
                    "p": 2.8,
                    "gamma_min": 1e2,
                    "gamma_max": 1e7
                }
            }

    spectrum_norm_type : ["integral", "differential", "gamma=1"]
        only with a normalisation in "cm-3" one can select among three types:

        * ``"integral"``: (default) the spectrum is set such that :math:`n_{e,\,tot}` equals the value provided by ``spectrum_norm``;

        * ``"differential"``: the spectrum is set such that :math:`k_e` equals the value provided by ``spectrum_norm``;

        * ``"gamma=1"``: the spectrum is set such that :math:`n_e(\gamma=1)` equals the value provided by ``spectrum_norm``.

    gamma_size : int
        size of the array of electrons Lorentz factors
    """

    def __init__(
        self,
        R_b=1e16 * u.cm,
        z=0.069,
        delta_D=10,
        Gamma=10,
        B=1 * u.G,
        spectrum_norm=1e48 * u.Unit("erg"),
        spectrum_dict={
            "type": "PowerLaw",
            "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e7},
        },
        spectrum_norm_type="integral",
        xi=1.0,
        gamma_size=200,
    ):
        self.R_b = R_b.to("cm")
        self.z = z
        self.d_L = Distance(z=self.z).cgs
        self.V_b = 4 / 3 * np.pi * np.power(self.R_b, 3)
        self.delta_D = delta_D
        self.Gamma = Gamma
        self.Beta = np.sqrt(1 - 1 / np.power(self.Gamma, 2))
        self.t_var = (((1 + self.z) * self.R_b) / (c * self.delta_D)).to("d")
        # viewing angle
        self.mu_s = (1 - 1 / (self.Gamma * self.delta_D)) / self.Beta
        self.theta_s = (np.arccos(self.mu_s) * u.rad).to("deg")
        self.B = B
        # B decomposed in Gaussian-cgs units
        self.B_cgs = B_to_cgs(B)
        self.spectrum_norm = spectrum_norm
        self.spectrum_norm_type = spectrum_norm_type
        self.spectrum_dict = spectrum_dict
        self.xi = xi
        # default grid of Lorentz factors for integration in the external frame
        self.gamma_to_integrate = np.logspace(1, 9, gamma_size)
        # model for the electron density
        self.set_spectrum(
            self.spectrum_norm, self.spectrum_dict, self.spectrum_norm_type, gamma_size
        )

    def set_gamma_size(self, gamma_size):
        """change size of the array of electrons Lorentz factors"""
        self.gamma_size = gamma_size
        self.gamma = np.logspace(
            np.log10(self.gamma_min), np.log10(self.gamma_max), self.gamma_size
        )
        self.gamma_to_integrate = np.logspace(1, 9, self.gamma_size)

    def set_gamma(self, gamma_min, gamma_max, gamma_size):
        """set the array of Lorentz factors to be used for integration in the
        frame comoving with the blob"""
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_size = gamma_size
        # grid of Lorentz factor for the integration in the blob comoving frame
        self.gamma = np.logspace(
            np.log10(self.gamma_min), np.log10(self.gamma_max), self.gamma_size
        )

    def set_spectrum(
        self, spectrum_norm, spectrum_dict, spectrum_norm_type, gamma_size=200
    ):
        r"""set the spectrum :math:`n_e` for the electrons accelerated in the
        blob, reset also the array of Lorentz factor given the `gamma_min` and
        `gamma_max` in the parameters dictionary"""
        self.set_gamma(
            spectrum_dict["parameters"]["gamma_min"],
            spectrum_dict["parameters"]["gamma_max"],
            gamma_size,
        )
        self.n_e = init_spectrum_norm_dict(
            spectrum_norm, spectrum_dict, spectrum_norm_type, self.V_b
        )

    def __str__(self):
        """printable summary of the blob"""
        return (
            "* spherical emission region\n"
            + f" - R_b (radius of the blob): {self.R_b.cgs:.2e}\n"
            + f" - t_var (variability time scale): {self.t_var:.2e}\n"
            + f" - V_b (volume of the blob): {self.V_b.cgs:.2e}\n"
            + f" - z (source redshift): {self.z:.2f}\n"
            + f" - d_L (source luminosity distance):{self.d_L.cgs:.2e}\n"
            + f" - delta_D (blob Doppler factor): {self.delta_D:.2e}\n"
            + f" - Gamma (blob Lorentz factor): {self.Gamma:.2e}\n"
            + f" - Beta (blob relativistic velocity): {self.Beta:.2e}\n"
            + f" - theta_s (jet viewing angle): {self.theta_s:.2e}\n"
            + f" - B (magnetic field tangled to the jet): {self.B:.2e}\n"
            + f" - xi (coefficient for 1st order Fermi acceleration) : {self.xi:.2e}\n"
            + str(self.n_e)
        )

    def set_delta_D(self, Gamma, theta_s):
        """set the viewing angle and the Lorentz factor of the outflow to
        obtain a specific Doppler factor

        Parameters
        ----------
        Gamma : float
            Lorentz factor of the relativistic outflow
        theta_s : :class:`~astropy.units.Quantity`
            viewing angle of the jet
        """
        mu_s = np.cos(theta_s.to("rad").value)
        Beta = np.sqrt(1 - 1 / np.power(Gamma, 2))
        delta_D = 1 / (Gamma * (1 - Beta * mu_s))

        self.theta_s = theta_s
        self.mu_s = mu_s
        self.Gamma = Gamma
        self.Beta = Beta
        self.delta_D = delta_D

    def N_e(self, gamma):
        r"""number of electrons as a function of the Lorentz factor,
        :math:`N_e(\gamma') = V_b\,n_e(\gamma')`"""
        return self.V_b * self.n_e(gamma)

    @property
    def n_e_tot(self):
        r"""total electrons density

        .. math::
            n_{e,\,tot} = \int^{\gamma'_{\rm max}}_{\gamma'_{\rm min}} {\rm d}\gamma' n_e(\gamma')
        """
        return np.trapz(self.n_e(self.gamma), self.gamma)

    @property
    def N_e_tot(self):
        r"""total number of electrons

        .. math::
            N_{e,\,tot} = \int^{\gamma'_{\rm max}}_{\gamma'_{\rm min}} {\rm d}\gamma' N_e(\gamma')
        """
        return np.trapz(self.N_e(self.gamma), self.gamma)

    @property
    def u_e(self):
        r"""total energy density in non-thermal electrons

        .. math::
            u_{e} = m_e c^2\,\int^{\gamma'_{\rm max}}_{\gamma'_{\rm min}} {\rm d}\gamma' \gamma' n_e(\gamma')
        """
        return mec2 * np.trapz(self.gamma * self.n_e(self.gamma), self.gamma)

    @property
    def W_e(self):
        r"""total energy in non-thermal electrons

        .. math::
            W_{e} = m_e c^2\,\int^{\gamma'_{\rm max}}_{\gamma'_{\rm min}} {\rm d}\gamma' \gamma' N_e(\gamma')
        """
        return mec2 * np.trapz(self.gamma * self.N_e(self.gamma), self.gamma)

    @property
    def U_B(self):
        r"""energy density of magnetic field

        .. math::
            U_B = B^2 / (8 \pi)
        """
        U_B = np.power(self.B_cgs, 2) / (8 * np.pi)
        return U_B.to("erg cm-3")

    @property
    def k_eq(self):
        """equipartition parameter: ratio between totoal electron energy density
        magnetic field energy density, Eq. 7.75 of [DermerMenon2009]_"""
        return (self.u_e / self.U_B).to_value("")

    @property
    def P_jet_e(self):
        r"""jet power in electrons

        .. math::
            P_{\mathrm{jet},\,e} = 2 \pi R_b^2 \beta \Gamma^2 c u_e
        """
        prefactor = (
            2 * np.pi * np.power(self.R_b, 2) * self.Beta * np.power(self.Gamma, 2) * c
        )
        return (prefactor * self.u_e).to("erg s-1")

    @property
    def P_jet_B(self):
        r"""jet power in magnetic field

        .. math::
            P_{\mathrm{jet},\,B} = 2 \pi R_b^2 \beta \Gamma^2 c \frac{B^2}{8\pi}
        """
        prefactor = (
            2 * np.pi * np.power(self.R_b, 2) * self.Beta * np.power(self.Gamma, 2) * c
        )
        return (prefactor * self.U_B).to("erg s-1")

    @property
    def u_ph_synch(self):
        r"""energy density of the synchrotron photons energy losses are:

        .. math::
            (\mathrm{d}E/\mathrm{d}t)_{\mathrm{synch}} = 4 / 3 \sigma_T c U_B \gamma^2

        the radiation stays an average time of :math:`(3/4) (R_b/c)`
        (the factor of 3/4 cames from averaging over a sphere),
        so an e- with Lorentz factor :math:`\gamma` produces:

        .. math::
            0.75\,(\mathrm{d}E/\mathrm{d}t)_{\mathrm{synch}}\,(R_b/c)\,/\,V_b

        of radiation. We need to integrate over the electron spectrum  (and multiply back by V_b)

        .. math::
            0.75\,\int n_e(\gamma) (\mathrm{d}E/\mathrm{d}t)_{\mathrm{synch}} R_b \mathrm{d}\gamma

        so

        .. math::
            u_{\mathrm{synch}} = \sigma_T  U_B  R_b  \int n_e(\gamma) \, \gamma^2 \mathrm{d}\gamma

        WARNING: this does not take into account SSA!
        """
        u_ph = (
            sigma_T.cgs
            * self.U_B
            * self.R_b
            * np.trapz(np.power(self.gamma, 2) * self.n_e(self.gamma), self.gamma)
        )
        return u_ph.to("erg cm-3")

    def plot_n_e(self, ax=None, gamma_power=0, **kwargs):
        """plot the  electron distribution

        Parameters
        ----------
        ax : :class:`~matplotlib.axes.Axes`, optional
            Axis
        gamma_power : float
            power of gamma to raise the electron distribution
        """
        plot_eed(self.gamma, self.n_e, gamma_power, ax, **kwargs)
