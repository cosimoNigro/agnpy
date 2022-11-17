"""This module describes the emission regions responsible for the
acceleration of particles to relativistic energies. Beside physical quantities
related to the emission itself it contains the electrons energy distributions"""
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from astropy.constants import c, sigma_T, m_e
from ..spectra import PowerLaw
from ..utils.conversion import mec2, mpc2, B_to_cgs


__all__ = ["Blob"]


class Blob:
    r"""Simple spherical emission region.

    **Note:** all these quantities are defined in the comoving frame so they are
    actually primed quantities, when referring the notation in [DermerMenon2009]_.

    All the quantities returned from the base attributes (e.g. volume of the
    emission region and variability time scale are derived from the blob radius)
    will be returned as properties, such that their value will be updated if the
    parameter from which they are derived is updated.

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
    n_e : :class:`~agnpy.spectra.ParticleDistribution`
        electron distribution contained in the blob
    n_p : :class:`~agnpy.spectra.ParticleDistribution`
        proton distribution contained in the blob
    xi : float
        acceleration coefficient :math:`\xi` for first-order Fermi acceleration
        :math:`(\mathrm{d}E/\mathrm{d}t \propto v \approx c)`
        used to compute limits on the maximum Lorentz factor via
        :math:`(\mathrm{d}E/\mathrm{d}t)_{\mathrm{acc}} = \xi c E / R_L`
    gamma_e_size : int
        size of the array of electrons Lorentz factors
    gamma_p_size : int
        size of the array of protons Lorentz factors
    """

    def __init__(
        self,
        R_b=1e16 * u.cm,
        z=0.069,
        delta_D=10,
        Gamma=10,
        B=1 * u.G,
        n_e=PowerLaw(mass=m_e),
        n_p=None,
        xi=1.0,
        gamma_e_size=200,
        gamma_p_size=200,
    ):
        self.R_b = R_b.to("cm")
        self.z = z
        self.delta_D = delta_D
        self.Gamma = Gamma
        self.B = B
        self._n_e = n_e
        self._n_p = n_p
        self.xi = xi

        # we might want to have different array of Lorentz factors for e and p
        self.set_gamma_e(gamma_e_size, self._n_e.gamma_min, self._n_e.gamma_max)
        if self._n_p is not None:
            self.set_gamma_p(gamma_p_size, self._n_p.gamma_min, self._n_p.gamma_max)

    @property
    def V_b(self):
        """Volume of the blob."""
        return 4 / 3 * np.pi * np.power(self.R_b, 3)

    @property
    def t_var(self):
        r"""Variability time scale, defined as
        :math:`t_{\rm var} = \frac{(1 + z) R_{\rm b}}{c \delta_{\rm D}}`.
        """
        return (((1 + self.z) * self.R_b) / (c * self.delta_D)).to("d")

    @property
    def d_L(self):
        """Luminosity distance."""
        return Distance(z=self.z).cgs

    @property
    def Beta(self):
        """Bulk Lorentz factor of the Blob."""
        return np.sqrt(1 - 1 / np.power(self.Gamma, 2))

    @property
    def mu_s(self):
        """Cosine of the viewing angle from the jet axis to the observer."""
        return (1 - 1 / (self.Gamma * self.delta_D)) / self.Beta

    @property
    def theta_s(self):
        """Viewing angle from the jet axis to the observer."""
        return (np.arccos(self.mu_s) * u.rad).to("deg")

    @property
    def B_cgs(self):
        """Magnetic field decomposed in Gaussian-cgs units."""
        return B_to_cgs(self.B)

    def set_delta_D(self, Gamma, theta_s):
        """Set the Doppler factor by specifying the bulk Lorentz factor of the
        outflow and the viewing angle.

        Parameters
        ----------
        Gamma : float
            Lorentz factor of the relativistic outflow
        theta_s : :class:`~astropy.units.Quantity`
            viewing angle of the jet
        """
        self.Gamma = Gamma
        mu_s = np.cos(theta_s.to("rad").value)
        self.delta_D = 1 / (self.Gamma * (1 - self.Beta * mu_s))

    def set_gamma_e(self, gamma_size, gamma_min=1, gamma_max=1e8):
        """Set the array of Lorentz factors for the electrons."""
        self.gamma_e_size = gamma_size
        self.gamma_e_min = gamma_min
        self.gamma_e_max = gamma_max

    def set_gamma_p(self, gamma_size, gamma_min=1, gamma_max=1e8):
        """Set the array of Lorentz factors for the protons."""
        self.gamma_p_size = gamma_size
        self.gamma_p_min = gamma_min
        self.gamma_p_max = gamma_max

    @property
    def gamma_e(self):
        """Array of electrons Lorentz factors, to be used for integration in the
        reference frame comoving with the emission region."""
        return np.logspace(
            np.log10(self.gamma_e_min), np.log10(self.gamma_e_max), self.gamma_e_size
        )

    @property
    def gamma_e_external_frame(self):
        """Array of electrons Lorentz factors, to be used for integration in the
        reference frame external to the emission region."""
        return np.logspace(1, 9, self.gamma_e_size)

    @property
    def gamma_p(self):
        """Array of protons Lorentz factors, to be used for integration in the
        reference frame comoving with the emission region."""
        if self._n_p is not None:
            return np.logspace(
                np.log10(self.gamma_p_min),
                np.log10(self.gamma_p_max),
                self.gamma_p_size,
            )
        else:
            raise AttributeError("No proton distribution initialised for this blob.")

    @property
    def n_e(self):
        """Electron distribution."""
        return self._n_e

    @n_e.setter
    def n_e(self, spectrum):
        """Setter of the electron distribution."""
        self._n_e = spectrum

    @property
    def n_p(self):
        """Proton distribution."""
        if self._n_p is None:
            raise AttributeError("No proton distribution initialised for this blob.")
        else:
            return self._n_p

    @n_p.setter
    def n_p(self, spectrum):
        """Setter of the proton distribution."""
        self._n_p = spectrum
        # set also the array of Lorentz factor of the protons
        self.set_gamma_p(200, self._n_p.gamma_min,  self._n_p.gamma_max)

    def __str__(self):
        """Printable summary of the blob."""
        resume = (
            "* Spherical emission region\n"
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
            + str(self._n_e)
        )
        if self._n_p is not None:
            resume += str(self._n_p)
        return resume

    def N_e(self, gamma):
        r"""Number of electrons as a function of the Lorentz factor,
        :math:`N_{\rm e}(\gamma') = V_{\rm b}\,n_{\rm e}(\gamma')`.

        Parameters
        ----------
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to evaluate the number of electrons
        """
        return self.V_b * self.n_e(gamma)

    def N_p(self, gamma):
        r"""Number of protons as a function of the Lorentz factor,
        :math:`N_{\rm p}(\gamma') = V_{\rm b}\,n_{\rm p}(\gamma')`.

        Parameters
        ----------
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to evaluate the number of electrons
        """
        return self.V_b * self.n_p(gamma)

    @property
    def n_e_tot(self):
        r"""Total density of electrons

        .. math::
            n_{\rm e,\,tot} = \int^{\gamma'_{\rm max}}_{\gamma'_{\rm min}} {\rm d}\gamma' n_{\rm e}(\gamma').
        """
        return np.trapz(self.n_e(self.gamma_e), self.gamma_e)

    @property
    def n_p_tot(self):
        r"""Total density of protons

        .. math::
            n_{\rm p,\,tot} = \int^{\gamma'_{\rm max}}_{\gamma'_{\rm min}} {\rm d}\gamma' n_{\rm p}(\gamma').
        """
        return np.trapz(self.n_p(self.gamma_p), self.gamma_p)

    @property
    def N_e_tot(self):
        r"""Total number of electrons

        .. math::
            N_{\rm e,\,tot} = \int^{\gamma'_{\rm max}}_{\gamma'_{\rm min}} {\rm d}\gamma' N_{\rm e}(\gamma').
        """
        return self.V_b * self.n_e_tot

    @property
    def N_p_tot(self):
        r"""total number of electrons

        .. math::
            N_{\rm p,\,tot} = \int^{\gamma'_{\rm max}}_{\gamma'_{\rm min}} {\rm d}\gamma' N_{\rm p}(\gamma').
        """
        return self.V_b * self.n_p_tot

    @property
    def u_e(self):
        r"""Total energy density of electrons

        .. math::
            u_{\rm e} = m_{\rm e} c^2\,\int^{\gamma'_{\rm max}}_{\gamma'_{\rm min}} {\rm d}\gamma' \gamma' n_{\rm e}(\gamma').
        """
        return mec2 * np.trapz(self.gamma_e * self.n_e(self.gamma_e), self.gamma_e)

    @property
    def u_p(self):
        r"""Total energy density of protons

        .. math::
            u_{\rm p} = m_{\rm p} c^2\,\int^{\gamma'_{\rm max}}_{\gamma'_{\rm min}} {\rm d}\gamma' \gamma' n_{\rm p}(\gamma').
        """
        if self.n_p is None:
            raise AttributeError(
                "The proton density, Blob.n_p, was not initialised for this blob."
            )
        else:
            return mpc2 * np.trapz(self.gamma_p * self.n_p(self.gamma_p), self.gamma_p)

    @property
    def W_e(self):
        r"""Total energy in electrons

        .. math::
            W_{\rm e} = m_{\rm e} c^2\,\int^{\gamma'_{\rm max}}_{\gamma'_{\rm min}} {\rm d}\gamma' \gamma' N_{\rm e}(\gamma').
        """
        return self.V_b * self.u_e

    @property
    def W_p(self):
        r"""Total energy in protons

        .. math::
            W_{\rm p} = m_{\rm p} c^2\,\int^{\gamma'_{\rm max}}_{\gamma'_{\rm min}} {\rm d}\gamma' \gamma' N_{\rm p}(\gamma').
        """
        return self.V_b * self.u_p

    @property
    def U_B(self):
        r"""Energy density of magnetic field

        .. math::
            U_B = B^2 / (8 \pi)
        """
        U_B = np.power(self.B_cgs, 2) / (8 * np.pi)
        return U_B.to("erg cm-3")

    @property
    def k_eq(self):
        """Equipartition parameter: ratio between totoal particle energy density
        and magnetic field energy density, Eq. 7.75 of [DermerMenon2009]_"""
        if self._n_p is None:
            return (self.u_e / self.U_B).to_value("")
        else:
            return ((self.u_e + self.u_p) / self.U_B).to_value("")

    @property
    def P_jet_ke(self):
        r"""Total jet power in kinetic energy of the particles

        .. math::
            P_{{\rm jet},\,{\rm ke}} = 2 \pi R_{\rm b}^2 \beta \Gamma^2 c (u_{\rm e} + u_{\rm p}).
        """
        prefactor = (
            2 * np.pi * np.power(self.R_b, 2) * self.Beta * np.power(self.Gamma, 2) * c
        )
        if self._n_p is None:
            return (prefactor * self.u_e).to("erg s-1")
        else:
            return (prefactor * (self.u_e + self.u_p)).to("erg s-1")

    @property
    def P_jet_B(self):
        r"""Jet power in magnetic field

        .. math::
            P_{\mathrm{jet},\,B} = 2 \pi R_{\rm b}^2 \beta \Gamma^2 c \frac{B^2}{8\pi}.
        """
        prefactor = (
            2 * np.pi * np.power(self.R_b, 2) * self.Beta * np.power(self.Gamma, 2) * c
        )
        return (prefactor * self.U_B).to("erg s-1")

    @property
    def u_ph_synch(self):
        r"""Energy density of the synchrotron photons energy losses are:

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
            * np.trapz(np.power(self.gamma_e, 2) * self.n_e(self.gamma_e), self.gamma_e)
        )
        return u_ph.to("erg cm-3")
