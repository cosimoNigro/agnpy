from typing import Iterable, Callable, Union, Sequence, Literal, Tuple, TypeVar, Generic
from astropy.units import Quantity
from numpy._typing import NDArray

F = TypeVar("F", bound=Callable[..., object])
class LabeledFunction(Generic[F]):
    def __init__(self, func, label):
        self._func = func
        self._label = label

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def __str__(self):
        return self._label

EnergyChangeFnType = Callable[[Quantity], Quantity]
""" 
A function type that takes a single parameter, Quantity-array (unitless Lorentz gamma factors), 
and returns a new Quantity-array (with units of energy per time, same length). It may not change the total density.
"""

InjectionRelFnType = Callable[[Quantity, Quantity, Quantity], Quantity]
""" 
A function type that takes three parameters, Quantity-array (unitless Lorentz gamma factors), Quantity-array (densities),  and time, 
and returns a new Quantity-array (with unitless density scaling coefficients, same length). It might change the total density.
"""

InjectionAbsFnType = Callable[[Quantity], Quantity]
"""
A function type that takes one parameter, Quantity-array (unitless Lorentz gamma factors),
and returns a new Quantity-array (with unit of density per time, same length). It might change the total density.
"""

EnergyChangeFns = Union[EnergyChangeFnType, Sequence[EnergyChangeFnType]]
InjectionRelFns = Union[InjectionRelFnType, Sequence[InjectionRelFnType]]
InjectionAbsFns = Union[InjectionAbsFnType, Sequence[InjectionAbsFnType]]
NumericalMethod = Literal["euler", "heun"]
CallbackFnType = Callable[[Quantity, NDArray, Quantity, dict[str, Quantity], Quantity], None]
"""
A callback function with params: elapsed time; gamma bins; densities; energy rates dict; injection (absolute)
"""