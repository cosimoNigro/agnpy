# physical particle energy distributions
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c, sigma_T
from sympy import Symbol, uppergamma, lambdify
from .spectra import ParticleDistribution
from ..utils.conversion import B_to_cgs


# lambdify the uppergamma function of sympy (make it applicable to numpy arrays)
# https://docs.sympy.org/latest/modules/functions/special.html#sympy.functions.special.gamma_functions.uppergamma
s = Symbol("s")
t = Symbol("t")
expr = uppergamma(s, t)
lambda_uppergamma = lambdify((s, t), expr, modules=["numpy", "sympy"])
vecotrised_uppergamma = np.vectorize(lambda_uppergamma) 


def gamma_b_tau_synchrotron_tau_ad(B, R):
    """Break Lorentz factor assuming equal time scales in cooling and adiabatic expansion.
    Eq. 2.26 in [Inoue]_"""
    B = B_to_cgs(B)
    u_B = B**2 / (4 * np.pi)
    return (3 * m_e * c**2 / (4 * sigma_T * u_B * R)).to_value("")


class SteadyStateSynchrotronCooling(ParticleDistribution):
    r"""Solution of the steady state electron differential equation

    .. math::
        N(\gamma') = ...

    Parameters
    ----------
    k : :class:`~astropy.units.Quantity`
        injection rate
    p : float
        spectral index of the injected power law
    B : float
        magnetic field
    tau_ad : :class:`~astropy.units.Quantity`
        adiabatic time scale
    """
    def __init__(
        self,
        q_inj=1e-5 * u.Unit("cm-3 s-1"),
        p=2,
        B=0.1 * u.G,
        R_b=1e17 * u.cm,
        mass=m_e,
        integrator=np.trapz,
    ):
        super().__init__(mass, integrator)
        self.q_inj = q_inj
        self.p = p
        self.B = B
        self.R_b = R_b

    @property
    def parameters(self):
        return [self.q_inj, self.p, self.B, self.R_b]

    @staticmethod
    def evaluate(gamma, q_inj, p, B, R_b):
        """Solution, Eq. 2.26 in [Inoue]"""
        tau_ad = R_b / c
        gamma_b = gamma_b_tau_synchrotron_tau_ad(B, R_b)
        print(gamma_b)
        prefactor = (
            np.exp(gamma / gamma_b)
            * tau_ad.to("s")
            * q_inj.to("cm-3 s-1")
            * gamma_b ** (2 - p)
            / gamma**2
        )
        return prefactor * vecotrised_uppergamma(1 - p, gamma / gamma_b)

    def __call__(self, gamma):
        return self.evaluate(gamma, self.q_inj, self.p, self.B, self.R_b)

    def __str__(self):
        return (
            f"ciao"
        )
