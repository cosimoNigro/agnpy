from ..radiative_process import RadiativeProcess
import numpy as np
from agnpy.emission_regions import Cone, Blob


class Synchrotron(RadiativeProcess):
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
            from .synchrotron_cone import SynchrotronCone

            self._model = SynchrotronCone(emitter, ssa, integrator)
        elif isinstance(emitter, Blob):
            from .synchrotron_blob import SynchrotronBlob

            self._model = SynchrotronBlob(emitter, ssa, integrator)

    def __getattr__(self, name):
        return getattr(self._model, name)
