from .synchrotron_cone import *
from .synchrotron_blob import *
from ..radiative_process import RadiativeProcess
import numpy as np
from agnpy.emission_regions import Cone, Blob
import astropy.units as u
from astropy.constants import e, h, c, m_e, sigma_T, mu0
from ..utils.math import axes_reshaper, gamma_e_to_integrate
from ..utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c_e, mec2
from ..radiative_process import RadiativeProcess
e = e.gauss



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
            self._model = SynchrotronCone(emitter, ssa, integrator)
        elif isinstance(emitter, Blob):
            self._model = SynchrotronBlob(emitter, ssa, integrator)

    def __getattr__(self, name):
        return getattr(self._model, name)

