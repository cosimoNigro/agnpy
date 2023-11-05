import numpy as np
import astropy.units as u


class SedFluxIntegrable:
    def integrate_flux(self, nu):
        r""" Evaluates the SED flux integral over the span of provided frequencies

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the SED
            **note** these are observed frequencies (observer frame)
        """
        sed = self.sed_flux(nu)
        return np.trapz(sed, nu) # should i rather use trapz_loglog from utils.math?
