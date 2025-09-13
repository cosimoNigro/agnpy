from dataclasses import dataclass
from typing import Callable, Union, Sequence, Literal
from astropy.units import Quantity
from numpy._typing import NDArray

@dataclass(frozen=True)
class CallbackParams:
    total_time: Quantity
    gamma: NDArray
    density: Quantity
    en_chg_rates: dict[str, Quantity]
    abs_inj_rates: dict[str, Quantity]
    rel_inj_rates: dict[str, Quantity]

EnergyChangeFn = Callable[[NDArray], Quantity]
""" 
A function type that takes a single parameter, numpy-array (unitless Lorentz gamma factors), 
and returns a new Quantity-array (with unit of energy per time, same length). It may not change the total density.
"""

InjectionRelFn = Callable[[NDArray], Quantity]
""" 
A function type that takes three parameters, numpy-array (unitless Lorentz gamma factors), Quantity-array (densities), and time, 
and returns a new numpy-array (with unit 1 / time, same length). It might change the total density.
"""

InjectionAbsFn = Callable[[NDArray], Quantity]
"""
A function type that takes one parameter, numpy-array (unitless Lorentz gamma factors),
and returns a new Quantity-array (with unit of density per time, same length). It might change the total density.
"""

EnergyChangeFns = Union[EnergyChangeFn, Sequence[EnergyChangeFn], dict[str, EnergyChangeFn]]
InjectionRelFns = Union[InjectionRelFn, Sequence[InjectionRelFn], dict[str, InjectionRelFn]]
InjectionAbsFns = Union[InjectionAbsFn, Sequence[InjectionAbsFn], dict[str, InjectionAbsFn]]
NumericalMethod = Literal["euler", "heun"]
CallbackFnType = Callable[[CallbackParams], None]