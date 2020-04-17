import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
from . import spectra


MEC2 = (const.m_e * const.c * const.c).cgs


__all__ = ["Blob"]


class Blob:
    """Simple spherical emission region.

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
        used in the functions computing energy limits 
        acceleration coefficient :math:`dE/dt = \ksi  E  c / R_L`
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

    spectrum_norm_type : `["integral", "differential", "gamma=1"]`
        select the type of normalisation: `"integral"` is the default; 
        `"differential"` assigns `spectrum_norm` directly to :math:`k_e`; 
        `"gamma=1"` sets :math:`k_e` such that `spectrum_norm` = :math:`n_e(\gamma=1)`.

    gamma_size : int
        size of the array of electrons Lorentz factors
    """

    def __init__(
        self,
        R_b,
        z,
        delta_D,
        Gamma,
        B,
        spectrum_norm,
        spectrum_dict,
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
        # viewing angle
        self.mu_s = (1 - 1 / (self.Gamma * self.delta_D)) / self.Beta
        self.theta_s = (np.arccos(self.mu_s) * u.rad).to("deg")
        self.B = B.to("G")
        self.spectrum_norm = spectrum_norm
        self.spectrum_norm_type = spectrum_norm_type
        self.spectrum_dict = spectrum_dict
        self.xi=xi
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
        self.set_n_e(self.spectrum_norm, self.spectrum_dict, self.spectrum_norm_type)

    def set_n_e(self, spectrum_norm, spectrum_dict, spectrum_norm_type):
        """set the spectrum :math:`n_e` for the blob"""
        print(f"* normalising {spectrum_dict['type']} in {spectrum_norm_type} mode")
        model_dict = {
            "PowerLaw": spectra.PowerLaw,
            "BrokenPowerLaw": spectra.BrokenPowerLaw,
            "BrokenPowerLaw2": spectra.BrokenPowerLaw2,
        }
        spectrum_type = spectrum_dict["type"]

        if spectrum_norm_type != "integral" and spectrum_norm.unit in (
            u.Unit("erg"),
            u.Unit("erg cm-3"),
        ):
            raise TypeError(
                "Normalisations different than 'integral' available only for 'spectrum_norm' in cm-3"
            )

        # check the units of the normalisation
        # cm-3 is the only one allowing more than one normalisation type
        if spectrum_norm.unit == u.Unit("cm-3"):

            if spectrum_norm_type == "integral":
                self.n_e = model_dict[spectrum_type].from_normalised_density(
                    spectrum_norm, **spectrum_dict["parameters"]
                )
            elif spectrum_norm_type == "differential":
                self.n_e = model_dict[spectrum_type](
                    spectrum_norm, **spectrum_dict["parameters"]
                )
                print(f"setting k_e directly to {spectrum_norm:.2e}")
            elif spectrum_norm_type == "gamma=1":
                self.n_e = model_dict[spectrum_type].from_norm_at_gamma_1(
                    spectrum_norm, **spectrum_dict["parameters"]
                )

        elif spectrum_norm.unit == u.Unit("erg cm-3"):
            self.n_e = model_dict[spectrum_type].from_normalised_u_e(
                spectrum_norm, **spectrum_dict["parameters"]
            )

        elif spectrum_norm.unit == u.Unit("erg"):
            u_e = (spectrum_norm / self.V_b).to("erg cm-3")
            self.n_e = model_dict[spectrum_type].from_normalised_u_e(
                u_e, **spectrum_dict["parameters"]
            )

    def __str__(self):
        """printable summary of the blob"""
        summary = (
            "* spherical emission region\n"
            + f" - R_b (radius of the blob): {self.R_b:.2e}\n"
            + f" - V_b (volume of the blob): {self.V_b:.2e}\n"
            + f" - z (source redshift): {self.z:.2f}\n"
            + f" - d_L (source luminosity distance):{self.d_L:.2e}\n"
            + f" - delta_D (blob Doppler factor): {self.delta_D:.2e}\n"
            + f" - Gamma (blob Lorentz factor): {self.Gamma:.2e}\n"
            + f" - Beta (blob relativistic velocity): {self.Beta:.2e}\n"
            + f" - theta_s (jet viewing angle): {self.theta_s:.2e}\n"
            + f" - B (magnetic field tangled to the jet): {self.B:.2e}\n"
            + str(self.n_e)
        )
        return summary

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
        Beta = np.sqrt(1 - 1 / np.power(self.Gamma, 2))
        delta_D = 1 / (Gamma * (1 - Beta * mu_s))

        self.theta_s = theta_s
        self.mu_s = mu_s
        self.Gamma = Gamma
        self.Beta = Beta
        self.delta_D = delta_D

    def set_gamma_size(self, gamma_size):
        """change size of the array of electrons Lorentz factors"""
        self.gamma_size = gamma_size
        self.gamma = np.logspace(
            np.log10(self.gamma_min), np.log10(self.gamma_max), self.gamma_size
        )
        self.gamma_to_integrate = np.logspace(1, 9, self.gamma_size)

    def N_e(self, gamma):
        """number of electrons as a function of the Lorentz factor, 
        :math:`N_e(\gamma') = V_b\,n_e(\gamma')`"""
        return self.V_b * self.n_e(gamma)

    @property
    def n_e_tot(self):
        """total electrons density

        .. math::
            n_{e,\,tot} = \int^{\gamma'_{max}}_{\gamma'_{min}} d\gamma' \, n_e(\gamma')
        """
        return np.trapz(self.n_e(self.gamma), self.gamma)

    @property
    def N_e_tot(self):
        """total electrons number

        .. math::
            N_{e,\,tot} = \int^{\gamma'_{max}}_{\gamma'_{min}} d\gamma' \, N_e(\gamma')
        """
        return np.trapz(self.N_e(self.gamma), self.gamma)

    @property
    def u_e(self):
        """total electrons energy density

        .. math::
            u_{e} = m_e\,c^2\,\int^{\gamma'_{max}}_{\gamma'_{min}} d\gamma' \,  \gamma' \, n_e(\gamma')
        """
        return MEC2 * np.trapz(self.gamma * self.n_e(self.gamma), self.gamma)

    @property
    def W_e(self):
        """total energy in non-thermal electrons

        .. math::
            W_{e} = m_e\,c^2\,\int^{\gamma'_{max}}_{\gamma'_{min}} d\gamma' \,  \gamma' \, N_e(\gamma')
        """
        return MEC2 * np.trapz(self.gamma * self.N_e(self.gamma), self.gamma)

    @property
    def P_jet_e(self):
        """jet power in electrons

        .. math::
            P_{jet,\,e} = 2 \pi R_b^2 \\beta \Gamma^2 c u_e
        """
        prefactor = (
            2
            * np.pi
            * np.power(self.R_b, 2)
            * self.Beta
            * np.power(self.Gamma, 2)
            * const.c
        )
        return (prefactor * self.u_e).to("erg s-1")

    @property
    def P_jet_B(self):
        """jet power in magnetic field

        .. math::
            P_{jet,\,B} = 2 \pi R_b^2 \\beta \Gamma^2 c \\frac{B^2}{8\pi}
        """
        prefactor = (
            2
            * np.pi
            * np.power(self.R_b, 2)
            * self.Beta
            * np.power(self.Gamma, 2)
            * const.c
        )
        U_B = np.power(self.B.value, 2) / (8 * np.pi) * u.Unit("erg cm-3")
        return (prefactor * U_B).to("erg s-1")

    @property
    def gamma_max_confined(self):
        """Maximum gamma factor of electrons that have their Larmour radius R_L smaller then R_b. In SI units Larmour radius is (see e.g. https://en.wikipedia.org/wiki/Gyroradius#Relativistic_case)
        .. math::
           R_L=p_{\perp} / q * B \approx = \gamma m_e * c / q * B 

           so:

        .. math::
            \gamma = e * B * R_b / m_e *c 
        """
#        return self.R_b/(const.m_e*const.c / (self.B.si * const.e.si)).to("cm")
        return (self.R_b* self.B.si * const.e.si/(const.m_e*const.c)).to("").value

    @property
    def gamma_max_ballistic(self):
        """Very simple (naive) estimation of maximum gamma factor of electrons 
        comparing acceleration time scale with ballistic time scale. 
        for ballistic limit we assume that blob crosses its (longitudal) radius
        (or in the frame of the blob the jet crosses R_b of the blob) 
        For definition of xi (and T_acc formula) check e.g. https://arxiv.org/abs/1208.6200a eq (2)
        Might be too naive ... 

        .. math::
            dE_{acc}/dt = xi *c * E/ R_L
            T_{acc} = RL/(xi *c)
            T_{bal} = R_b/c
            gamma_{max} = R_b * xi * e*B / (m_e * c^2) # SI UNITS!
        """
        return (self.xi * self.R_b* self.B.si * const.e.si/(const.m_e*const.c)).to("").value

    @property
    def gamma_max_synch(self):
        """Simple estimation of maximum gamma factor of electrons 
        comparing acceleration time scale with synchrotron energy losses. 
        xi and dE_acc like in gamma_max_ballistic
        .. math::
            dE_{acc}/dt = xi *c * E/ R_L
            dE_{synch}/dt = (4/3) * sigmaT *(B^2/(2 mu0)) * \gamma^2 *c # SI!
            gamma = \sqrt{1.5 mu0 \ksi  c e /(\sigma_T B)}
        """
        return np.sqrt(1.5*self.xi*const.c*const.e.si*const.mu0.si/(const.sigma_T*self.B)).to("").value
    

    @property
    def gamma_break_synch(self):
        """Simple estimation of cooling break of electrons 
        comparing synchrotron cooling with dynamic time scale.
        check eq F.1 in https://ui.adsabs.harvard.edu/abs/2020arXiv200107729M/abstract
        original formula:
        .. math::
            gamma_b = 3 \pi m_e c^2 / \sigma_T B^2 R 
        implemented formula with 2 changes:
        3 ==> 6 (in the paper the assume that synchr and IC losses are comparable hence they probably took them twice, 
                 or they compared with 2 * R instead of R)
        B^2 ==> B^2 * 4 pi / mu0  # to go for SI units  
        .. math::
            gamma_b = 3 m_e c^2 mu_0 / 2 \sigma_T B^2 * R 
        """
        return (1.5 *const.c**2 * const.m_e *const.mu0.si/(const.sigma_T*self.B**2*self.R_b)).to("")# .value
    

    def plot_n_e(self):
        plt.loglog(self.gamma, self.n_e(self.gamma))
        plt.xlabel(r"$\gamma$")
        plt.ylabel(r"$n_e(\gamma)\,/\,{\rm cm}^{-3}$")
        plt.show()
