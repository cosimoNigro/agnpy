import astropy.units as u
from ..synchrotron import Synchrotron
from ..compton import SynchrotronSelfCompton
from gammapy.modeling.models import SpectralModel, Parameter, Parameters

class SpectralModelSSC(SpectralModel):

    tag = ["SpectralModelSSC"]

    def __init__(self, blob):
        """Initialise all the parameters from the Blob instance"""

        self.blob = blob

        parameters = []

        # all the parameters of the EED have to be parameters of the fittable model
        # any electron distribution has for attributes the spectral parameters and the integrator
        # we remove the latter
        pars = vars(self.blob.n_e)
        pars.pop("integrator")
        for name in pars.keys():
            parameter = Parameter(name, pars[name])
            parameters.append(parameter)

        # this are the emission region parameters
        z = Parameter("z", blob.z)
        d_L = Parameter("d_L", blob.d_L)
        delta_D = Parameter("delta_D", blob.delta_D)
        B = Parameter("B", blob.B)
        R_b = Parameter("R_b", blob.R_b)
        parameters.extend([z, d_L, delta_D, B, R_b])

        self.default_parameters = Parameters(parameters)
        super().__init__()

    def evaluate(self, energy, **kwargs):
        """evaluate"""
        nu = energy.to("Hz", equivalencies=u.spectral())

        # all the model parameters will be passed as kwargs, sort the source one out
        z = kwargs.pop("z")
        d_L = kwargs.pop("d_L")
        delta_D = kwargs.pop("delta_D")
        B = kwargs.pop("B")
        R_b = kwargs.pop("R_b")

        # expand the remaining kwargs in the spectral parameters
        args = kwargs.values()

        sed_synch = Synchrotron.evaluate_sed_flux(
            nu,
            z,
            d_L,
            delta_D,
            B,
            R_b,
            self.blob.n_e,
            *args
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            nu,
            z,
            d_L,
            delta_D,
            B,
            R_b,
            self.blob.n_e,
            *args
        )

        sed = sed_synch + sed_ssc
        return (sed / energy ** 2).to("1 / (cm2 eV s)")
