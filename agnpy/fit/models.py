from .gammapy_wrapper import (
    SynchrotronSelfComptonSpectralModel,
    ExternalComptonSpectralModel,
)
from .sherpa_wrapper import (
    SynchrotronSelfComptonConeRegriddableModel1D,
    SynchrotronSelfComptonBlobRegriddableModel1D,
    ExternalComptonRegriddableModel1D,
)

class SynchrotronSelfComptonConeModel:
    """Model for Synchrotron+SSC Scenario for a Conical Extended Jet"""
    def __new__(cls, n_e, ssa=False, electron_escape=False):
        return SynchrotronSelfComptonConeRegriddableModel1D(n_e, ssa, electron_escape=electron_escape)

class SynchrotronSelfComptonBlobModel:
    """Model for synchrotron self-Compton scenario."""
    def __new__(cls, n_e, ssa=False, backend="gammapy"):
        if backend == "sherpa":
            return SynchrotronSelfComptonBlobRegriddableModel1D(n_e, ssa)
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
