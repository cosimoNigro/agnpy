import astropy.units as u
from astropy.coordinates import Distance
from ..synchrotron import Synchrotron
from ..compton import ExternalCompton
from ..compton import SynchrotronSelfCompton
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import SpectralModel


def set_emission_region_pars_scales_ranges(parameters):
    """Properly set the ranges and scales of the emission region parameters."""
    z_min = 0.001
    z_max = 10
    parameters["z"].min = z_min
    parameters["z"].max = z_max
    parameters["z"].frozen = True
    parameters["d_L"].min = Distance(z=z_min).to_value("cm")
    parameters["d_L"].max = Distance(z=z_max).to_value("cm")
    parameters["d_L"].frozen = True
    parameters["delta_D"].min = 1
    parameters["delta_D"].max = 100
    parameters["delta_D"].scale_method = "factor1"
    parameters["B"].min = 1e-4
    parameters["B"].max = 1e3
    parameters["B"].scale_method = "scale10"
    parameters["R_b"].min = 1e12
    parameters["R_b"].max = 1e18
    parameters["R_b"].scale_method = "scale10"


def set_spectrum_pars_scales_ranges(parameters):
    """Properly set the ranges and scales of the particle energy distribution.
    By default minimum and maximum of the energy distribution are frozen."""
    # normalisation
    parameters["k_e"].min = 1
    parameters["k_e"].max = 1e9
    parameters["k_e"].scale_method = "scale10"
    for parameter in parameters:
        # Lorentz factors
        if parameter.name.startswith("gamma"):
            parameter.scale_method = "scale10"
        if parameter.name == "gamma_min":
            parameter.min = 1
            parameter.max = 1e3
            parameter.frozen = True
        if parameter.name == "gamma_max":
            parameter.min = 1e4
            parameter.max = 1e8
            parameter.frozen = True
        # indexes
        if parameter.name.startswith("p"):
            parameter.min = 1
            parameter.max = 5
            parameter.scale_method = "factor1"


class SynchrotronSelfComptonSpectralModel(SpectralModel):

    tag = ["SynchrotronSelfComptonSpectralModel"]

    def __init__(self, blob, ssa=False):
        """Gammapy wrapper for a source emitting Synchrotron and SSC radiation.
        The parameters for the model are initialised from a Blob instance.

        Parameters
        ----------
        blob : `~agnpy.emission_regions.blob`

        Returns
        -------
        """

        self.blob = blob
        self.ssa = ssa

        self.spectral_pars_names = []  # will fetch this from blob.n_e
        self.emission_region_pars_names = ["z", "d_L", "delta_D", "B", "R_b"]

        spectral_pars = []
        emission_region_pars = []

        # all the particle distribution parameters have to be parameters of the
        # fittable model, the normalisation of the electrons, k_e, is the norm
        # of the SpectralModel
        pars = vars(self.blob.n_e)  # EED has in the attributes also the integrator
        pars.pop("integrator")
        for name in pars.keys():
            if name == "k_e":
                parameter = Parameter(name, pars[name], is_norm=True)
            else:
                parameter = Parameter(name, pars[name])
            spectral_pars.append(parameter)
            # create the list with the names of the spectral parameters
            self.spectral_pars_names.append(name)

        # emission region parameters
        for name in self.emission_region_pars_names:
            parameter = Parameter(name, getattr(blob, name))
            emission_region_pars.append(parameter)

        # group the model parameters
        self.default_parameters = Parameters([*spectral_pars, *emission_region_pars])
        # sale them, set min and maxes
        set_spectrum_pars_scales_ranges(self.default_parameters)
        set_emission_region_pars_scales_ranges(self.default_parameters)
        super().__init__()

    @property
    def spectral_parameters(self):
        """Select all the parameters related to the particle distribution."""
        return self.parameters.select(self.spectral_pars_names)

    @property
    def emission_region_parameters(self):
        """Select all the parameters related to the emission region."""
        return self.parameters.select(self.emission_region_pars_names)

    def evaluate(self, energy, **kwargs):
        """evaluate"""
        nu = energy.to("Hz", equivalencies=u.spectral())

        # all the model parameters will be passed as kwargs, by SpectralModel.evaluate()
        # sort the ones related to the source out
        z = kwargs.pop("z")
        d_L = kwargs.pop("d_L")
        delta_D = kwargs.pop("delta_D")
        B = kwargs.pop("B")
        R_b = kwargs.pop("R_b")

        # expand the remaining kwargs in the spectral parameters
        args = kwargs.values()

        sed_synch = Synchrotron.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self.blob.n_e, *args, self.ssa
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self.blob.n_e, *args, self.ssa
        )

        sed = sed_synch + sed_ssc

        # to avoid the problem pointed out in
        # https://github.com/cosimoNigro/agnpy/issues/117
        # we can do here something like
        # sed = sed.reshape(energy.shape)
        # this is done in Gammapy's `NaimaSpectralModel`
        # https://github.com/gammapy/gammapy/blob/master/gammapy/modeling/models/spectral.py#L2119

        # gammapy requires a differential flux in input
        return (sed / energy**2).to("1 / (cm2 eV s)")


class ExternalComptonSpectralModel(SpectralModel):

    tag = ["ExternalComptonSpectralModel"]

    def __init__(self, blob, blr, dt, r, ec_blr=True):
        """Initialise all the parameters from the Blob and the targets instances."""

        self.blob = blob
        self.blr = blr
        self.dt = dt
        self.ec_blr = ec_blr

        # check for the two targets to have the same disk luminosities
        if not u.isclose(self.blr.L_disk, self.dt.L_disk):
            raise ValueError(
                "The BLR and DT provided have the different disk luminosities."
            )

        parameters = []

        # all the parameters of the EED have to be parameters of the fittable model
        # any electron distribution has for attributes the spectral parameters and the integrator
        # we remove the latter
        pars = vars(self.blob.n_e)
        pars.pop("integrator")
        for name in pars.keys():
            parameter = Parameter(name, pars[name])
            parameters.append(parameter)

        # emission region parameters
        z = Parameter("z", blob.z)
        d_L = Parameter("d_L", blob.d_L)
        delta_D = Parameter("delta_D", blob.delta_D)
        B = Parameter("B", blob.B)
        R_b = Parameter("R_b", blob.R_b)
        mu_s = Parameter("mu_s", blob.mu_s)
        # distance of the emission region along the jet
        r = Parameter("r", r)
        parameters.extend([z, d_L, delta_D, B, R_b, mu_s, r])

        # parameters of the targets
        L_disk = Parameter("L_disk", self.blr.L_disk)
        # - BLR
        xi_line = Parameter("xi_line", self.blr.xi_line)
        epsilon_line = Parameter("epsilon_line", self.blr.epsilon_line)
        R_line = Parameter("R_line", self.blr.R_line)
        # - DT
        xi_dt = Parameter("xi_dt", self.dt.xi_dt)
        epsilon_dt = Parameter("epsilon_dt", self.dt.epsilon_dt)
        R_dt = Parameter("R_dt", self.dt.R_dt)
        parameters.extend(
            [L_disk, xi_line, epsilon_line, R_line, xi_dt, epsilon_dt, R_dt]
        )

        self.default_parameters = Parameters(parameters)
        super().__init__()

    def evaluate(self, energy, **kwargs):
        """evaluate"""
        nu = energy.to("Hz", equivalencies=u.spectral())

        # all the model parameters will be passed as kwargs, by SpectralModel.evaluate()
        # sort the ones related to the source out
        z = kwargs.pop("z")
        d_L = kwargs.pop("d_L")
        delta_D = kwargs.pop("delta_D")
        B = kwargs.pop("B")
        R_b = kwargs.pop("R_b")
        mu_s = kwargs.pop("mu_s")
        r = kwargs.pop("r")

        # now sort the target ones out
        L_disk = kwargs.pop("L_disk")
        # - BLR
        xi_line = kwargs.pop("xi_line")
        epsilon_line = kwargs.pop("epsilon_line")
        R_line = kwargs.pop("R_line")
        # - DT
        xi_dt = kwargs.pop("xi_dt")
        epsilon_dt = kwargs.pop("epsilon_dt")
        R_dt = kwargs.pop("R_dt")

        # expand the remaining kwargs in the spectral parameters
        args = kwargs.values()

        sed_synch = Synchrotron.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self.blob.n_e, *args
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self.blob.n_e, *args
        )

        sed_ec_dt = ExternalCompton.evaluate_sed_flux_dt(
            nu,
            z,
            d_L,
            delta_D,
            mu_s,
            R_b,
            L_disk,
            xi_dt,
            epsilon_dt,
            R_dt,
            r,
            self.blob.n_e,
            *args
        )

        sed = sed_synch + sed_ssc + sed_ec_dt

        if self.ec_blr:
            sed_ec_blr = ExternalCompton.evaluate_sed_flux_blr(
                nu,
                z,
                d_L,
                delta_D,
                mu_s,
                R_b,
                L_disk,
                xi_line,
                epsilon_line,
                R_line,
                r,
                self.blob.n_e,
                *args
            )
            sed += sed_ec_blr

        # eventual reshaping
        # we can do here something like
        # sed = sed.reshape(energy.shape)

        return (sed / energy**2).to("1 / (cm2 eV s)")
