import numpy as np
import astropy.units as u
from astropy.constants import e, sigma_T
from ..utils.conversion import mec2

e = e.gauss

__all__ = ["SpectralConstraints"]


class SpectralConstraints:
    r""" Class to describe the self-consistency constraints on the electron energy distribution

    Parameters
    ----------
    blob : :class: `Blob` emission region
    """

    def __init__(self, blob):
        self.blob = blob

    @property
    def gamma_max_larmor(self):
        r"""maximum Lorentz factor of electrons that have their Larmour radius
        smaller than the blob radius: :math:`R_L < R_b`.
        The Larmor frequency and radius in Gaussian units read

        .. math::

            \omega_L &= \frac{eB}{\gamma m_e c} \\
            R_L &= \frac{v}{\omega_L} = \frac{\gamma m_e v c}{e B} \approx \frac{\gamma m_e c^2}{e B}

        therefore

        .. math::

            R_L < R_b \Rightarrow \gamma_{\mathrm{max}} < \frac{R_b e B}{m_e c^2}
        """
        return (self.blob.R_b * e * self.blob.B_cgs / mec2).to_value("")

    @property
    def gamma_max_ballistic(self):
        r"""Naive estimation of maximum Lorentz factor of electrons comparing
        acceleration time scale with ballistic time scale.
        For the latter we assume that the particles crosses the blob radius.

        .. math::

            (\mathrm{d}E/\mathrm{d}t)_{\mathrm{acc}} &= \xi c E / R_L \\
            T_{\mathrm{acc}} &= E \,/\,(\mathrm{d}E/\mathrm{d}t)_{\mathrm{acc}} = R_L / (\xi c) \\
            T_{\mathrm{bal}} &= R_b / c \\
            T_{\mathrm{acc}} &< T_{\mathrm{bal}}
            \Rightarrow \gamma_{\mathrm{max}} < \frac{\xi  R_b e B}{m_e c^2}
        """
        return self.blob.xi * self.gamma_max_larmor

    @property
    def gamma_max_synch(self):
        r"""Simple estimation of maximum Lorentz factor of electrons
        comparing the acceleration time scale with the synchrotron energy loss

        .. math::
            (\mathrm{d}E/\mathrm{d}t)_{\mathrm{acc}} &= \xi c E / R_L \\
            (\mathrm{d}E/\mathrm{d}t)_{\mathrm{synch}} &= 4 / 3 \sigma_T c U_B \gamma^2 \\
            (\mathrm{d}E/\mathrm{d}t)_{\mathrm{acc}} &= (\mathrm{d}E/\mathrm{d}t)_{\mathrm{synch}}
            \Rightarrow \gamma_{\mathrm{max}} < \sqrt{\frac{6 \pi \xi e}{\sigma_T B}}
        """
        return np.sqrt(
            6 * np.pi * self.blob.xi * e / (sigma_T * self.blob.B_cgs)
        ).to_value("")

    @property
    def gamma_max_SSC(self):
        r"""Simple estimation of maximum Lorentz factor of electrons
        comparing the acceleration time scale with the SSC energy loss (in Thomson range)
        WARNING: the highest energy electrons will most often scatter in Klein-Nishina range instead

        .. math::
            (\mathrm{d}E/\mathrm{d}t)_{\mathrm{acc}} &= \xi c E / R_L \\
            (\mathrm{d}E/\mathrm{d}t)_{\mathrm{SSC}} &= 4 / 3 \sigma_T c U_{\mathrm{synch}} \gamma^2 \\
            (\mathrm{d}E/\mathrm{d}t)_{\mathrm{acc}} &= (\mathrm{d}E/\mathrm{d}t)_{\mathrm{SSC}}
            \Rightarrow \gamma_{\mathrm{max}} < \sqrt{\frac{3 \xi e B }{\sigma_T U_SSC}}
        """
        return np.sqrt(
            3
            * self.blob.xi
            * e
            * self.blob.B_cgs
            / (4 * sigma_T * self.blob.u_ph_synch)
        ).to_value("")

    def gamma_max_EC_DT(self, dt, r=0 * u.cm):
        r"""Simple estimation of maximum Lorentz factor of electrons comparing the acceleration time scale
        with the EC energy loss (in Thomson range, see B&G 1970), like in gamma_max_SSC
        WARNING: assumes Thomson regime

        .. math::
            \gamma_{\mathrm{max}} = \sqrt{\frac{3 \xi e B }{ \sigma_T U'_\mathrm{ext}}}
        """
        return np.sqrt(
            3 * self.blob.xi * e * self.blob.B_cgs / (4 * sigma_T * dt.u(r, self.blob))
        ).to_value("")

    @property
    def gamma_break_synch(self):
        r"""Simple estimation of the cooling break of electrons comparing
        synchrotron cooling time scale with the ballistic time scale:

        .. math::

            T_{\mathrm{synch}} &= E\,/\,(\mathrm{d}E/\mathrm{d}t)_{\mathrm{synch}}
            =  3 m_e c^2 / (4 \sigma_T U_B \gamma) \\
            T_{\mathrm{bal}} &= R_b / c \\
            T_{\mathrm{synch}} &= T_{\mathrm{bal}} \Rightarrow \gamma_b = 6 \pi m_e c^2 / \sigma_T B^2 R_b
        """
        gamma_max = (
            (
                6
                * np.pi
                * mec2
                / (sigma_T * np.power(self.blob.B_cgs, 2) * self.blob.R_b)
            )
            .to("")
            .value
        )
        return gamma_max

    @property
    def gamma_break_SSC(self):
        r"""Simple estimation of the cooling break of electrons comparing
        SSC time scale (see B&G 1970) with the ballistic time scale:
        WARNING: only applicable in Thomson regime

        .. math::
            T_{\mathrm{SSC}} &= E\,/\,(\mathrm{d}E/\mathrm{d}t)_{\mathrm{SSC}}
            =  3 m_e c^2 / (4 \sigma_T U_{\mathrm{SSC}} \gamma) \\
            T_{\mathrm{bal}} &= R_b / c \\
            T_{\mathrm{SSC}} &= T_{\mathrm{bal}} \Rightarrow \gamma_b = 3  m_e c^2 / 4 \sigma_T U_{\mathrm{SSC}} R_b
        """
        return (
            (3 * mec2 / (4 * sigma_T * self.blob.u_ph_synch * self.blob.R_b))
            .to("")
            .value
        )

    def gamma_break_EC_DT(self, dt, r=0 * u.cm):
        r"""Simple estimation of the cooling break of electrons comparing
        EC time scale (see B&G 1970) with the ballistic time scale, like in gamma_break_SSC
        WARNING: assumes Thomson regime

        .. math::
            \gamma_b = 3  m_e c^2 / 4 \sigma_T U'_{\mathrm{ext}} R_b
        """
        #        u_ext=np.power(self.Gamma,2) * np.power(1-mu*self.Beta,2) * dt.xi_dt*dt.L_disk/(4*np.pi*np.power(d,2) * c)
        return (
            (3 * mec2 / (4 * sigma_T * dt.u(r, self.blob) * self.blob.R_b)).to("").value
        )
