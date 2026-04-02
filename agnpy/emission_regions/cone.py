""" This module describes a uniform conical jet emission region
responsible for the acceleration of particles to relativistic energies. 
Besides physical parameters related to the emission region itself, 
it contains the electron energy distributions"""

import numbers
from typing import Iterable

import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from scipy.integrate import cumulative_trapezoid
from astropy.constants import c, sigma_T, m_e

from .. import InterpolatedDistribution, ParticleDistribution
from ..spectra import PowerLaw
from ..utils.conversion import mec2, mpc2, B_to_cgs


class Cone:
    r"""Uniform Conical Jet Model.

    **Note:** We will assume that the jet has an opening angle θ_{opening}, 
    initial radius R0 and length L, all defined in the centre of momentum 
    frame of the fluid.

    Parameters
    ----------
    R_o : :class:`~astropy.units.Quantity`
        Initial radius at the base of the cone 
    L : :class:`~astropy.units.Quantity`
        Length of the Jet
    theta_OPEN : class:`~astropy.units.Quantity`
        Opening angle of the cone
    z : float
        redshift of the source
    delta_D : float
        Doppler factor of the relativistic outflow
    Gamma : float
        Lorentz factor of the relativistic outflow
    B : :class:`~astropy.units.Quantity`
        magnetic field at the base of the cone
    n_e : :class:`~agnpy.spectra.ParticleDistribution`
        electron distribution contained in the blob
    n_p : :class:`~agnpy.spectra.ParticleDistribution`
        proton distribution contained in the blob
    """

## Keeping the default parameters same as defined in Blob (CΟNFIRM FOR NEW PARAMETERS!!!!)
    def __init__(
        self,
        L = 1e18 * u.cm,
        R_o = 1e14 * u.cm, 
        B_o = 1 * u.G,
        theta = 5 * u.deg, 
        z=0.033,
        delta_D=10,
        Gamma=10,
        n_e : ParticleDistribution = PowerLaw(mass=m_e),
        xi=1.0,
        gamma_e_size=200,
        x_size=100,
        cosmology=None
    ):
        if not isinstance(delta_D, numbers.Number) or delta_D <= 0:
            raise ValueError("delta_D must be a positive number")

        self.L = L.to("cm")
        self.R_o = R_o
        self.B_o = B_o
        self.theta = theta.to("rad")
        self.z = z
        # if the luminosity distance is not specified, it will be computed from z
        self.d_L = Distance(z=self.z, cosmology=cosmology).cgs
        self.delta_D = delta_D
        self.Gamma = Gamma
        self._n_e : ParticleDistribution = n_e          
        self.xi = xi
        self.gamma_e_size = gamma_e_size
        self.x_size = x_size
    
    @classmethod
    def from_jet_power(cls, W_j, **kwargs ):
        B = kwargs.get("B_o")
        gamma = kwargs.get("Gamma")
        u_B = ((B_to_cgs(B)**2)/(8*np.pi)).to('erg cm-3')
        R_o_squared = (W_j)/(2*u_B*np.pi*(gamma**2)*c.cgs)
        R_o = np.sqrt(R_o_squared).to('cm')
        return cls( R_o=R_o , **kwargs)

    #@property
    #def R_o(self):
    #    """Radius at base derived from lab frame jet power and equipartition"""
    #    u_B = ((self.B_o**2)/(8*np.pi)).to('erg cm-3')
    #    R_o_squared = (self.W_j)/(2*u_B*np.pi*(self.Gamma**2)*c.cgs)
    #    return np.sqrt(R_o_squared).to('cm')
        
## 1. Jet Structure
    @property
    def V_c(self):
        """Volume of the cone."""
        R_L = self.R_o + self.L * np.tan(self.theta)
        return 1/3 * (np.pi * self.L) * (self.R_o**2 + self.R_o*R_L + R_L**2)
    

    @property
    def R_x(self):
        return self.R_o + self.x * np.tan(self.theta)
    
    @property
    def B_x(self):
        return self.B_o * (self.R_o / self.R_x)

    @property
    def x(self):
        x = np.logspace(np.log10(1), np.log10(self.L.to_value("cm")), self.x_size) * u.cm
        return x
    
    @property
    def gamma_e(self):
        """Array of electrons Lorentz factors, to be used for integration in the
        reference frame comoving with the emission region."""
        return np.logspace(
            np.log10(self._n_e.gamma_min), np.log10(self._n_e.gamma_max), self.gamma_e_size
        #gamma_max is computed through x_i ; based on Fermi 1st order acceleration coefficient
        )
    
    @property
    def gamma_e_external_frame(self):
        """Array of electrons Lorentz factors, to be used for integration in the
        reference frame external to the emission region."""
        return np.logspace(1, 9, self.gamma_e_size)

## 3. Electron Properties: Number Density, Total Number, Energy Density, Total Energy
    @property
    def n_e_base(self):
        """Electron distribution as obtained by agnpy.spectra
        units: cm-3
        """
        return self._n_e(self.gamma_e)
    
    @property
    def N_e_base(self):
        """Number of electrons at the base in a slice of width 1 cm"""
        return self.n_e_base * np.pi * (self.R_o**2) 

    @property
    def norm_equi(self):
        """Modified normalization to convolve with input electron distribution
        to impose equipartition. Obtained by equating Magnetic Energy at the base to 
        Electron Energy. 
        units: cm-3"""
        e_energy_initial = np.trapz(self.gamma_e * self.N_e_base, self.gamma_e)
        K_dash = ( np.pi * self.R_o**2 * (self.B_o ** 2) ) / (8 * np.pi * mec2 * e_energy_initial )
        return K_dash.cgs

    @property
    def N_e_xg(self):
        """Solution to the Electron Evolution Equation as described in Eq 11 in Cotter[2018]
        Returns: the number of electrons per cm at position x along the jet axis"""
        n0 = self.norm_equi * self.N_e_base[None,:] # (1, N_gamma)
        int_B = cumulative_trapezoid((self.B_x)**2, self.x, initial=0) # (N_x, 1)
        cooling = ( sigma_T.cgs * self.gamma_e[None,:] *(int_B[:,None])) / ( 6 * np.pi * mec2.cgs )
        N_e_xg = n0 * np.exp(-cooling.value) 
        return (N_e_xg).cgs

    @property
    def N_e_gamma(self):
        """Number of electrons as a function of gamma across the Jet"""
        N_e_gamma = np.trapz(self.N_e_xg, self.x, axis=0)
        return N_e_gamma
    
    @property
    def N_e_x(self): 
        """Number of electrons per cm at position x along the jet axis"""
        N_e_x = np.trapz( self.N_e_xg, self.gamma_e, axis=1)
        return N_e_x 
    
    @property
    def U_e_x(self):
        r"""Energy of electrons in a slice of width 1 cm """
        U_e_x = mec2.cgs * np.trapz(self.gamma_e[None,:] * self.N_e_xg, self.gamma_e, axis=1)
        return U_e_x
    
    @property
    def U_b_x(self):
        r"""Energy of magnetic field in a slice of width 1 cm as a function of distance 
        from the base of the Jet
        """
        return (np.pi * self.R_x**2 * np.power(self.B_x, 2) / (8 * np.pi))
    
    @property
    def W_e(self):
        r"""Total energy of electrons"""
        W_e = mec2.cgs * np.trapz(self.gamma_e * self.N_e_gamma, self.gamma_e)
        return W_e
    
    @property
    def N_e(self): 
        r"""Total Number of electrons in the Jet,
        :math:`N_{\rm p}(\gamma') = V_{\rm b}\,n_{\rm p}(\gamma')`.

        Parameters
        ----------
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to evaluate the number of electrons
        """
        N_e = np.trapz(self.N_e_x, self.x)
        return N_e
        
    @property
    def Beta(self):
        """Bulk Lorentz factor of the Cone."""
        return np.sqrt(1 - 1 / np.power(self.Gamma, 2))

    @property
    def mu_s(self):
        """Cosine of the viewing angle from the jet axis to the observer."""
        return (1 - 1 / (self.Gamma * self.delta_D)) / self.Beta

    @property
    def theta_s(self):
        """Viewing angle from the jet axis to the observer."""
        return (np.arccos(self.mu_s) * u.rad).to("deg")


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

    def __str__(self):
        """Printable summary of the Conical Jet."""
        resume = (
            "* Conical emission region\n"
            + f" - R_o (Radius at the base of the cone): {self.R_o.cgs:.2e}\n"
            + f" - L (Length of the Jet): {self.L.cgs:.2e}\n"
            + f" - θ (Opening Angle of the Conical Jet): {self.theta:.2e}\n"
            + f" - V_c (Volume of the Cone): {self.V_c.cgs:.2e}\n"
            + f" - z (source redshift): {self.z:.2f}\n"
            + f" - d_L (source luminosity distance):{self.d_L.cgs:.2e}\n"
            + f" - delta_D (blob Doppler factor): {self.delta_D:.2e}\n"
            + f" - Gamma (Bulk Lorentz factor): {self.Gamma:.2e}\n"
            + f" - Beta (Bulk relativistic velocity): {self.Beta:.2e}\n"
            + f" - theta_s (jet viewing angle): {self.theta_s:.2e}\n"
            + f" - B (Magnetic field at the base the jet): {self.B_o:.2e}\n"
            + f" - xi (coefficient for 1st order Fermi acceleration) : {self.xi:.2e}\n"
        )
        
  