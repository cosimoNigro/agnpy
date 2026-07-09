"""from .gammapy_wrapper import (
    SynchrotronSelfComptonSpectralModel,
    ExternalComptonSpectralModel,
)"""
from .sherpa_wrapper import (
    SynchrotronSelfComptonRegriddableModel1D,
    ExternalComptonRegriddableModel1D,
)

from.sherpa_wrapper import (
    SynchrotronConeRegriddableModel1D
)

class SynchrotronConeModel:
    """Model for Synchrotron Scenario for a Conical Extended Jet"""
    def __new__(cls, n_e, ssa=False, electron_escape=False):
        return SynchrotronConeRegriddableModel1D(n_e, ssa, electron_escape=electron_escape)

class SynchrotronSelfComptonModel:
    """Model for synchrotron self-Compton scenario."""

    def __new__(cls, n_e, ssa=False, backend="gammapy"):
        if backend == "sherpa":
            return SynchrotronSelfComptonRegriddableModel1D(n_e, ssa)
        elif backend == "gammapy":
            return SynchrotronSelfComptonSpectralModel(n_e, ssa)
        else:
            raise ValueError(
                f"{backend} is not an available backend, try gammapy or sherpa"
            )


class ExternalComptonModel:
    """Model for external Compton scenario."""

    def __new__(cls, n_e, targets, ssa=False, backend="gammapy"):
        if backend == "sherpa":
            return ExternalComptonRegriddableModel1D(n_e, targets, ssa)
        elif backend == "gammapy":
            return ExternalComptonSpectralModel(n_e, targets, ssa)
        else:
            raise ValueError(
                f"{backend} is not an available backend, try gammapy or sherpa"
            )
