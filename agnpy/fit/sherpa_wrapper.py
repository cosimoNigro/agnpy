# wrap agnpy SED computation via sherpa's 1D model
import astropy.units as u
from astropy.constants import c
from sherpa.models import model
from ..synchrotron import Synchrotron
from ..compton import SynchrotronSelfCompton, ExternalCompton
from .core import (
    get_spectral_parameters_from_n_e,
    get_emission_region_parameters,
    get_targets_parameters,
)


class SynchrotronSelfComptonRegriddableModel1D(model.RegriddableModel1D):
    def __init__(self, n_e, ssa=False):
        """sherpa wrapper for a source emitting Synchrotron and SSC radiation.

        Parameters
        ----------
        n_e : `~agnpy.spectra.ElectronDistribution`
            electron distribution to be used for this modelling
        ssa : bool
            whether or not to calculate synchrotron self-absorption

        Returns
        -------
        `~sherpa.models.Regriddable1DModel`
        """
        self.name = "ssc"
        self._n_e = n_e
        self.ssa = ssa

        # parameters of the particles energy distribution
        spectral_pars = get_spectral_parameters_from_n_e(
            self._n_e, backend="sherpa", modelname=self.name
        )

        # parameters of the emission region
        emission_region_pars = get_emission_region_parameters(
            "ssc", backend="sherpa", modelname=self.name
        )

        pars_list = [*spectral_pars.values(), *emission_region_pars.values()]

        # each parameter should be declared as an attribute, see
        # https://sherpa.readthedocs.io/en/4.14.0/model_classes/usermodel.html
        pars_attr_list = []

        for par in pars_list:
            setattr(self, par.name, par)
            pars_attr_list.append(getattr(self, par.name))

        model.RegriddableModel1D.__init__(self, self.name, tuple(pars_attr_list))


"""
        def calc(self, pars, x):
            (
                *args,
                z,
                d_L,
                delta_D,
                log10_B,
                t_var,
            ) = pars
            # add units, scale quantities
            x *= u.Hz
            k_e = 10 ** log10_k_e * u.Unit("cm-3")
            gamma_b = 10 ** log10_gamma_b
            gamma_min = 10 ** log10_gamma_min
            gamma_max = 10 ** log10_gamma_max
            B = 10 ** log10_B * u.G
            d_L *= u.cm
            R_b = c.to_value("cm s-1") * t_var * delta_D / (1 + z) * u.cm

            sed_synch = Synchrotron.evaluate_sed_flux(
                nu, z, d_L, delta_D, B, R_b, self._n_e, *args, ssa=self.ssa
            )
            sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
                nu, z, d_L, delta_D, B, R_b, self._n_e, *args, ssa=self.ssa
            )
            sed = sed_synch + sed_ssc
            return sed
"""
