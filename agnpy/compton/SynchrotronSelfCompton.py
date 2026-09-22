from ..radiative_process import RadiativeProcess
from agnpy.emission_regions import Cone, Blob
import numpy as np

class SynchrotronSelfCompton(RadiativeProcess):
    """Class for synchrotron radiation computation

    Parameters
    ----------
    emitter : :class:`~agnpy.emission_region`
        emitting region and electron distribution
    ssa : bool
        whether or not to consider synchrotron self absorption (SSA).
        The absorption factor will be taken into account in
        :func:`~agnpy.synchrotron.Synchrotron.com_sed_emissivity`, in order to be
        propagated to :func:`~agnpy.synchrotron.Synchrotron.sed_luminosity` and
        :func:`~agnpy.synchrotron.Synchrotron.sed_flux`.
    integrator : func
        function to be used for integration (default = `np.trapz`)
    """

    def __init__(self, emitter, ssa=False, integrator=np.trapz):
        self.ssa = ssa
        self.integrator = integrator
        if isinstance(emitter, Cone):
            from .ssc_cone import SynchrotronSelfComptonCone
            self._model = SynchrotronSelfComptonCone(emitter, ssa, integrator)
        elif isinstance(emitter, Blob):
            from .ssc_blob import SynchrotronSelfComptonBlob
            self._model = SynchrotronSelfComptonBlob(emitter, ssa, integrator)

    def __getattr__(self, name):
        return getattr(self._model, name)