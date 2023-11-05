import numpy as np
import astropy.units as u
from ..utils.sedintegrable import SedFluxIntegrable


class TestSedIntegrable:

    def test_flatintegral(self):
        """Integrate over flat SED (equal to 1.0 for all frequencies"""
        class FlatSedGenerator(SedFluxIntegrable):
            def sed_flux(self, nu):
                return np.full(fill_value=1.0, shape=nu.shape, dtype=np.float64) * u.erg / (u.cm ** 2 * u.s)

        flux = FlatSedGenerator().integrate_flux(np.logspace(1, 20) * u.Hz)
        assert flux.value == 1e+20
