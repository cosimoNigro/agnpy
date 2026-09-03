import numpy as np
from astropy.constants import c
from dataclasses import dataclass
from typing import Callable, Union, Sequence, Literal, NamedTuple, List
from astropy.units import Quantity, UnitConversionError
from numpy._typing import NDArray

@dataclass(frozen=True)
class BinsWithDensities:
    gamma_bins: NDArray
    densities: Quantity

@dataclass(frozen=True)
class FnParams:
    gamma: NDArray
    densities: Quantity
    density_subgroups: NDArray

class TimeEvaluationResult(NamedTuple):
    """
    The state of the blob at one point of a time evolution. Note that `blob_time` is the time on the blob's clock,
     i.e. `TimeEvolution.t0` plus the elapsed simulation time - which means it is an instant, not a duration
     (with the default `t0` of zero it equals the elapsed time, but with an offset `t0` it does not).
    """
    blob_time: Quantity
    gamma: NDArray
    density: Quantity
    density_subgroups: NDArray
    en_chg_rates: dict[str, Quantity]
    rel_inj_rates: dict[str, Quantity]
    abs_inj_rates: dict[str, Quantity]

@dataclass(frozen=True)
class BlobExpansion:
    """
    Describes the expansion of the blob at a constant radius growth rate: R(t) = R_0 + v_exp * t, in the blob frame.

    Parameters
    ----------
    v_exp :
        constant expansion speed dR/dt; must be non-negative and lower than the speed of light
    magnetic_field_index :
        index m of the magnetic field scaling B = B_0 * (R_0 / R)^m;
        m=1 corresponds to conservation of the toroidal field flux, m=2 of the poloidal one,
        m=0 keeps the field constant
    """
    v_exp: Quantity
    magnetic_field_index: float = 1.0

    def __post_init__(self):
        try:
            v_exp = self.v_exp.to("cm s-1")
        except (AttributeError, UnitConversionError):
            raise ValueError("v_exp must be a velocity Quantity")
        if not v_exp.isscalar or v_exp < 0 or v_exp >= c:
            raise ValueError("v_exp must be a scalar velocity in the range [0, c)")
        if not np.isfinite(self.magnetic_field_index) or self.magnetic_field_index < 0:
            raise ValueError("magnetic_field_index must be a finite non-negative number")


class DistributionToSinglePointCollapseError(Exception):
    def __init__(self, gamma_point):
        self.gamma_point = gamma_point
        super().__init__(f"Unsupported state, cannot create InterpolatedDistribution - distribution collapsed to a single gamma point {gamma_point}")

GammaFn = Callable[[FnParams], Quantity]
""" 
An abstract function that, for given gamma values provided in FnParams (and optionally densities), calculates a new
Quantity array that represents energy change rate or injection rate. 
"""

EnergyChangeFn = GammaFn
""" 
A GammaFn function that returns energy change rates, in the blob frame (unit: erg s-1)
"""

InjectionRelFn = GammaFn
""" 
A GammaFn function that returns relative injections rates, in the blob frame (unit: s-1) 
"""

InjectionAbsFn = Callable[[FnParams], Quantity]
"""
A GammaFn function that returns absolute injections rates, in the blob frame (unit: s-1 cm-3)
"""

EnergyChangeFns = Union[EnergyChangeFn, Sequence[EnergyChangeFn], dict[str, EnergyChangeFn]]
InjectionRelFns = Union[InjectionRelFn, Sequence[InjectionRelFn], dict[str, InjectionRelFn]]
InjectionAbsFns = Union[InjectionAbsFn, Sequence[InjectionAbsFn], dict[str, InjectionAbsFn]]
NumericalMethod = Literal["euler", "heun"]
CallbackFnType = Callable[[TimeEvaluationResult], None]
SubgroupsList = List[List[str]]