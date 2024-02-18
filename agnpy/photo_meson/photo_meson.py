# class for photomeson production
import numpy as np
from .kernels import PhiKernel
from ..utils.math import axes_reshaper
from ..utils.conversion import mpc2


class PhotoMesonProduction:
    def __init__(self, blob, target, integrator=np.trapz):
        self.blob = blob
        # check that this blob has a proton distribution
        if self.blob._n_p is None:
            raise AttributeError(
                "There is no proton distribution in this emission region"
            )
        self.target = target
        self.integrator = integrator
        self.phi_kernel_gamma = PhiKernel("gamma")
        self.phi_kernel_electron = PhiKernel("electron")
        self.phi_kernel_positron = PhiKernel("positron")
        self.phi_kernel_electron_neutrino = PhiKernel("electron_neutrino")
        self.phi_kernel_electron_antineutrino = PhiKernel("electron_antineutrino")
        self.phi_kernel_muon_neutrino = PhiKernel("muon_neutrino")
        self.phi_kernel_moun_antineutrino = PhiKernel("muon_antineutrino")

    def H(self, eta, E, phi_kernel):
        """Compute the H function in Eq. (70) [KelnerAharonian2008]_.

        Parameters
        ----------
        eta : float
            kinematic variable (:math:`eta = 4 \epsilon \gamma_{\rm p}`)
        E : float
            energy of the secondary
        phi_kernel : `~agnpy.photo_meson.PhiKernel`
            kernel to be used for the integration (depends on the particle)
        """
        # we want this output to be a two-dimensional function, so let us reshape it
        _gamma_p, _eta, _E = axes_reshaper(self.blob.gamma_p, eta, E)
        _x = (_E / (_gamma_p * mpc2)).to_value("")
        # obtain the energy of the target from eta
        _E_ph = _eta * mpc2 / (4 * _gamma_p)
        _H_integrand = (
            mpc2
            / 4
            * self.blob.n_p(_gamma_p)
            * self.target.n_ph(_E_ph)
            * phi_kernel(_eta, _x)
        )
        import IPython

        IPython.embed()
