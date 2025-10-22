from dataclasses import dataclass
from typing import Callable, Union, Sequence, Literal, NamedTuple, List
from astropy.units import Quantity
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
    total_time: Quantity
    gamma: NDArray
    density: Quantity
    density_subgroups: NDArray
    en_chg_rates: dict[str, Quantity]
    abs_inj_rates: dict[str, Quantity]
    rel_inj_rates: dict[str, Quantity]

GammaFn = Callable[[FnParams], Quantity]
""" 
An abstract function that, for given gamma values provided in FnParams (and optionally densities), calculates a new
Quantity array that represents energy change rate or injection rate. 
"""

EnergyChangeFn = GammaFn
""" 
A GammaFn function that returns energy change rates (unit: erg s-1)
"""

InjectionRelFn = GammaFn
""" 
A GammaFn function that returns relative injections rates (unit: s-1) 
"""

InjectionAbsFn = Callable[[FnParams], Quantity]
"""
A GammaFn function that returns absolute injections rates (unit: s-1 cm-3)
"""

EnergyChangeFns = Union[EnergyChangeFn, Sequence[EnergyChangeFn], dict[str, EnergyChangeFn]]
InjectionRelFns = Union[InjectionRelFn, Sequence[InjectionRelFn], dict[str, InjectionRelFn]]
InjectionAbsFns = Union[InjectionAbsFn, Sequence[InjectionAbsFn], dict[str, InjectionAbsFn]]
NumericalMethod = Literal["euler", "heun"]
CallbackFnType = Callable[[TimeEvaluationResult], None]
SubgroupsList = List[List[str]]