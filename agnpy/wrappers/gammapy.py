import copy
import astropy.units as u
from astropy.coordinates import Distance
from ..synchrotron import Synchrotron
from ..compton import ExternalCompton
from ..compton import SynchrotronSelfCompton
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import SpectralModel


def set_spectral_pars_ranges_scales(parameters):
    """Properly set the ranges and scales for the particle energy distribution
    parameters. By default minimum and maximum of the energy distribution are
    frozen.

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        list of `SpectralModel` parameters
    """
    for parameter in parameters:
        # the normalisation of the electrons is the norm of the SpectralModel
        if parameter.name == "k_e":
            parameter.min = 1
            parameter.max = 1e9
            parameter.scale_method = "scale10"
            parameter.interp = "log"
        # Lorentz factors
        if parameter.name.startswith("gamma"):
            parameter.scale_method = "scale10"
            parameter.interp = "log"
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
            parameter.interp = "lin"


def set_emission_region_pars_ranges_scales(parameters):
    """Properly set the ranges and scales for the emission region parameters.

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        list of `SpectralModel` parameters
    """
    z_min = 0.001
    z_max = 10
    for parameter in parameters:
        if parameter.name == "z":
            parameter.min = z_min
            parameter.max = z_max
            parameter.frozen = True
        if parameter.name == "d_L":
            parameter.min = Distance(z=z_min).to_value("cm")
            parameter.max = Distance(z=z_max).to_value("cm")
            parameter.frozen = True
        if parameter.name == "delta_D":
            parameter.min = 1
            parameter.max = 100
            parameter.scale_method = "factor1"
            parameter.interp = "lin"
        if parameter.name == "B":
            parameter.min = 1e-4
            parameter.max = 1e3
            parameter.scale_method = "scale10"
            parameter.interp = "log"
        if parameter.name == "R_b":
            parameter.min = 1e12
            parameter.max = 1e18
            parameter.scale_method = "scale10"
            parameter.interp = "log"
        if parameter.name == "mu_s":
            parameter.min = 0
            parameter.max = 1
            parameter.frozen = True
        if parameter.name == "r":
            parameter.min = 1e16
            parameter.max = 1e20
            parameter.scale_method = "scale10"
            parameter.interp = "log"


def set_targets_pars_ranges_scales(parameters):
    """Properly set the ranges and scales for the targets for EC. All of them
    are fixed by default since typically the parameters of the targets are not
    fitted.

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        list of `SpectralModel` parameters
    """
    for parameter in parameters:
        if parameter.name == "L_disk":
            parameter.min = 1e42
            parameter.max = 1e48
            parameter.frozen = True
        if parameter.name == "xi_line":
            parameter.min = 1e-3
            parameter.max = 1
            parameter.frozen = True
        if parameter.name == "epsilon_line":
            parameter.min = 1e-7
            parameter.max = 1e-4
            parameter.frozen = True
        if parameter.name == "R_line":
            parameter.min = 1e15
            parameter.max = 1e18
            parameter.frozen = True
        if parameter.name == "xi_dt":
            parameter.min = 1e-3
            parameter.max = 1
            parameter.frozen = True
        if parameter.name == "epsilon_dt":
            parameter.min = 1e-6
            parameter.max = 1e-8
            parameter.frozen = True
        if parameter.name == "R_dt":
            parameter.min = 1e17
            parameter.max = 1e21
            parameter.frozen = True


def get_spectral_parameters_from_blob(blob):
    """Get the list of parameters of the particles energy distribution from a
    blob instance.

    Parameters
    ----------
    blob : `~agnpy.emission_regions.Blob`
        blob containing the particles energy distribution

    Returns
    -------
    spectral_pars_names : list of string
        list of parameters names
    spectral_pars : list of `~gammapy.modeling.Parameter`
        list of parameters of the particles energy distribution
    """
    spectral_pars = []
    spectral_pars_names = []
    pars = vars(blob.n_e).copy()
    # EED has in the attributes also the integrator
    pars.pop("integrator")

    for name in pars.keys():
        spectral_pars.append(Parameter(name, pars[name]))
        # create the list with the names of the spectral parameters
        spectral_pars_names.append(name)

    return dict(zip(spectral_pars_names, spectral_pars))


def get_emission_region_parameters_from_names(parameters_names, blob):
    """Return a list of `~gammapy.modeling.Parameter`s for the emission region,
    corresponding to the names specified in the list."""
    emission_region_pars = []

    for name in parameters_names:
        emission_region_pars.append(Parameter(name, getattr(blob, name)))

    return emission_region_pars


def get_blr_parameters(blr):
    """Return a dictionary of `~gammapy.modeling.Parameter`s (and a list of their names)
    from the BLR object."""
    target_pars = []
    target_pars_names = []

    blr_pars = vars(blr).copy()
    blr_pars.pop("name")
    blr_pars.pop("line")
    blr_pars.pop("lambda_line")
    for name in blr_pars:
        target_pars_names.append(name)
        target_pars.append(Parameter(name, blr_pars[name]))

    return dict(zip(target_pars_names, target_pars))


def get_dt_parameters(dt):
    """Return a dictionary of `~gammapy.modeling.Parameter`s (and a list of their names)
    from the DT object."""
    target_pars = []
    target_pars_names = []

    dt_pars = vars(dt).copy()
    dt_pars.pop("name")
    dt_pars.pop("T_dt")
    dt_pars.pop("Theta")
    for name in dt_pars:
        target_pars_names.append(name)
        target_pars.append(Parameter(name, dt_pars[name]))

    return dict(zip(target_pars_names, target_pars))


class SynchrotronSelfComptonSpectralModel(SpectralModel):

    tag = ["SynchrotronSelfComptonSpectralModel"]

    def __init__(self, blob, ssa=False):
        """Gammapy wrapper for a source emitting Synchrotron and SSC radiation.
        The parameters for the model are initialised from a Blob instance.

        Parameters
        ----------
        blob : `~agnpy.emission_regions.blob`
            emission region containing all the initial parameters of the model
        ssa : bool
            whether or not to calculate synchrotron self-absorption

        Returns
        -------
        `~gammapy.modeling.models.SpectralModel`
        """

        # the attributes representing the physical objects are internal, they
        # are used only for Parameters initialisation
        self._blob = copy.copy(blob)
        self.ssa = ssa

        # parameters of the particles energy distribution
        spectral_pars = get_spectral_parameters_from_blob(self._blob)
        self._spectral_parameters_names = list(spectral_pars.keys())

        # parameters of the emission region
        self._emission_region_parameters_names = ["z", "d_L", "delta_D", "B", "R_b"]
        emission_region_pars = get_emission_region_parameters_from_names(
            self._emission_region_parameters_names, self._blob
        )

        # group the model parameters, add the norm at the bottom of the list
        norm = Parameter("norm", 1, min=0.1, max=10, is_norm=True, frozen=True)
        self.default_parameters = Parameters(
            [*spectral_pars.values(), *emission_region_pars, norm]
        )

        # set min and maxes, scale them
        set_spectral_pars_ranges_scales(self.default_parameters)
        set_emission_region_pars_ranges_scales(self.default_parameters)

        super().__init__()

    @property
    def spectral_parameters(self):
        """Select all the parameters related to the particle distribution."""
        return self.parameters.select(self._spectral_parameters_names)

    @property
    def emission_region_parameters(self):
        """Select all the parameters related to the emission region."""
        return self.parameters.select(self._emission_region_parameters_names)

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

        # pop the norm and expand the remaining kwargs in the spectral parameters
        kwargs.pop("norm")
        args = kwargs.values()

        sed_synch = Synchrotron.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self._blob.n_e, *args, ssa=self.ssa
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self._blob.n_e, *args, ssa=self.ssa
        )

        sed = sed_synch + sed_ssc

        # to avoid the problem pointed out in
        # https://github.com/cosimoNigro/agnpy/issues/117
        # we can do here something like
        # sed = sed.reshape(energy.shape)
        # this is done in Gammapy's `NaimaSpectralModel`
        # https://github.com/gammapy/gammapy/blob/master/gammapy/modeling/models/spectral.py#L2119

        # gammapy requires a differential flux in input
        return (sed / energy ** 2).to("1 / (cm2 eV s)")


class ExternalComptonSpectralModel(SpectralModel):

    tag = ["ExternalComptonSpectralModel"]

    def __init__(self, blob, r, targets, ssa=False):
        """Gammapy wrapper for a source emitting Synchrotron, SSC, and EC on a
        list of targets. The parameters for the model are initialised from a Blob
        and targets instances.

        Parameters
        ----------
        blob : `~agnpy.emission_regions.blob`
            emission region containing all the initial parameters of the model
        r : `~astropy.units.Quantity`
            distance of the blob from the central BH
        targets : list of `~agnpy.targets`
            list of targets to be considered for external Compton
        Returns
        -------
        `~gammapy.modeling.models.SpectralModel`
        """

        # the attributes representing the physical objects are internal, they
        # are used only for Parameters initialisation
        self._blob = copy.copy(blob)
        for target in targets:
            if target.name == "SphericalShellBLR":
                self._blr = copy.copy(target)
            if target.name == "RingDustTorus":
                self._dt = copy.copy(target)
        self.ssa = ssa

        # parameters of the particles energy distribution
        spectral_pars = get_spectral_parameters_from_blob(self._blob)
        self._spectral_parameters_names = list(spectral_pars.keys())

        # parameters of the emission region
        self._emission_region_parameters_names = [
            "z",
            "d_L",
            "delta_D",
            "B",
            "R_b",
            "mu_s",
        ]
        emission_region_pars = get_emission_region_parameters_from_names(
            self._emission_region_parameters_names, self._blob
        )
        # add also the position of the blob to the emisison region parameters
        self._emission_region_parameters_names.append("r")
        emission_region_pars.append(Parameter("r", r))

        targets_pars = {}

        if hasattr(self, "_blr"):
            targets_pars.update(get_blr_parameters(self._blr))

        if hasattr(self, "_dt"):
            dt_pars = get_dt_parameters(self._dt)
            if hasattr(self, "_blr"):
                # if both DT and BLR are specified, check that they have the same L_disk
                # and remove the duplicate parameter
                if not u.isclose(self._blr.L_disk, self._dt.L_disk):
                    raise ValueError(
                        "The BLR and DT provided have the different disk luminosities."
                    )
                else:
                    dt_pars.pop("L_disk")

            targets_pars.update(dt_pars)

        # names of the target parameters, for the corresponding property
        self._targets_parameters_names = list(targets_pars.keys())
        # group the model parameters, add the norm at the bottom of the list
        norm = Parameter("norm", 1, min=0.1, max=10, is_norm=True, frozen=True)
        self.default_parameters = Parameters(
            [
                *spectral_pars.values(),
                *emission_region_pars,
                *targets_pars.values(),
                norm,
            ]
        )

        # set min and maxes, scale them
        set_spectral_pars_ranges_scales(self.default_parameters)
        set_emission_region_pars_ranges_scales(self.default_parameters)
        set_targets_pars_ranges_scales(self.default_parameters)

        super().__init__(norm=norm)

    @property
    def spectral_parameters(self):
        """Select all the parameters related to the particle distribution."""
        return self.parameters.select(self._spectral_parameters_names)

    @property
    def emission_region_parameters(self):
        """Select all the parameters related to the emission region."""
        return self.parameters.select(self._emission_region_parameters_names)

    @property
    def targets_parameters(self):
        """Select all the parameters related to the targets for EC."""
        return self.parameters.select(self._targets_parameters_names)

    def evaluate(self, energy, **kwargs):
        """evaluate"""
        nu = energy.to("Hz", equivalencies=u.spectral())

        # all the model parameters will be passed as kwargs, by SpectralModel.evaluate()
        # sort the ones related to the emission region out
        z = kwargs.pop("z")
        d_L = kwargs.pop("d_L")
        delta_D = kwargs.pop("delta_D")
        B = kwargs.pop("B")
        R_b = kwargs.pop("R_b")
        mu_s = kwargs.pop("mu_s")
        r = kwargs.pop("r")

        # now sort the target ones out
        L_disk = kwargs.pop("L_disk")

        if hasattr(self, "_blr"):
            # - BLR
            xi_line = kwargs.pop("xi_line")
            epsilon_line = kwargs.pop("epsilon_line")
            R_line = kwargs.pop("R_line")

        if hasattr(self, "_dt"):
            # - DT
            xi_dt = kwargs.pop("xi_dt")
            epsilon_dt = kwargs.pop("epsilon_dt")
            R_dt = kwargs.pop("R_dt")

        # pop the norm and expand the remaining kwargs in the spectral parameters
        kwargs.pop("norm")
        args = kwargs.values()

        sed_synch = Synchrotron.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self._blob.n_e, *args, ssa=self.ssa
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self._blob.n_e, *args, ssa=self.ssa
        )
        sed = sed_synch + sed_ssc

        if hasattr(self, "_blr"):
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
                self._blob.n_e,
                *args
            )
            sed += sed_ec_blr

        if hasattr(self, "_dt"):
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
                self._blob.n_e,
                *args
            )

            sed += sed_ec_dt

        # eventual reshaping
        # we can do here something like
        # sed = sed.reshape(energy.shape)

        return (sed / energy ** 2).to("1 / (cm2 eV s)")
