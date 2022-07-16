import copy
import numpy as np
import astropy.units as u
from astropy.constants import c, k_B
from astropy.coordinates import Distance
from ..synchrotron import Synchrotron
from ..compton import ExternalCompton
from ..compton import SynchrotronSelfCompton
from ..targets import lines_dictionary, SSDisk, RingDustTorus
from ..utils.conversion import mec2
from ..utils.math import gamma_to_integrate
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import SpectralModel


gamma_size = 300
gamma_to_integrate = np.logspace(1, 9, gamma_size)


def add_systematic_errors_flux_points(flux_points, syst):
    """Add the systematic error on the flux points in a given energy range.
    We symply sum the systematic error in quadrature with the statystical one.
    The systematic error is given as a percentage of the flux.

    Parameters
    ----------
    flux_points : `~gammapy.estimators.FluxPoints`
        Gammapy's flux points
    syst : float
        systematic error as a percentage of the dnde flux
    """
    dnde_err_syst = syst * flux_points.dnde.data
    # sum in quadrature with the stat error
    dnde_errn_tot = np.sqrt(flux_points.dnde_errn.data ** 2 + dnde_err_syst ** 2)
    dnde_errp_tot = np.sqrt(flux_points.dnde_errp.data ** 2 + dnde_err_syst ** 2)
    # the attributes we have to change is the norm_errn and norm_errp
    flux_points.norm_errn.data = dnde_errn_tot / flux_points.dnde_ref.data
    flux_points.norm_errp.data = dnde_errp_tot / flux_points.dnde_ref.data


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
            parameter._name = "log10_" + parameter.name
            parameter.value = np.log10(parameter.value)
            parameter.unit = ""
            parameter.min = -10
            parameter.max = 2
        # Lorentz factors
        if parameter.name == "gamma_min":
            parameter._name = "log10_" + parameter.name
            parameter.value = np.log10(parameter.value)
            parameter.min = 1
            parameter.max = 3
            parameter.frozen = True
        if parameter.name == "gamma_max":
            parameter._name = "log10_" + parameter.name
            parameter.value = np.log10(parameter.value)
            parameter.min = 5
            parameter.max = 8
            parameter.frozen = True
        if parameter.name in ["gamma_b", "gamma_0", "gamma_c"]:
            parameter._name = "log10_" + parameter.name
            parameter.value = np.log10(parameter.value)
            parameter.min = 2
            parameter.max = 6
        # indexes
        if parameter.name.startswith("p"):
            parameter.min = 1
            parameter.max = 5
        if parameter.name == "q":
            parameter.min = 0.01
            parameter.max = 1


def get_spectral_parameters_from_n_e(n_e):
    """Get the list of parameters of the particles energy distribution from an
    EED instance.

    Parameters
    ----------
    n_e : `~agnpy.spectra.ElectronDistribution`
        electron distribution

    Returns
    -------
    parameters : dict of `~gammapy.modeling.Parameter`
        dictionary of parameters of the particles energy distribution
    """
    spectral_pars_names = []
    spectral_pars = []
    pars = vars(n_e).copy()
    # EED has in the attributes also the integrator
    pars.pop("integrator")

    for name in pars.keys():
        if (name == "k_e") or name.startswith("gamma_"):
            spectral_pars_names.append("log10_" + name)
        else:
            spectral_pars_names.append(name)
        spectral_pars.append(Parameter(name, pars[name]))

    # add the limits and the scales to the spectral parameters
    set_spectral_pars_ranges_scales(spectral_pars)
    parameters = dict(zip(spectral_pars_names, spectral_pars))
    return parameters


def get_emission_region_parameters(model_type):
    """Return a dict of `~gammapy.modeling.Parameter`s for the emission region.
    The list of parameters is different whether we are considering a SSC or EC
    model.

    Parameters
    ----------
    model_type : ["ssc", "ec"]
        type of model for which this emission region is being considered
    """
    z_min = 0.001
    z_max = 10
    # parameters common to SSC and EC models
    z = Parameter("z", 0.1, min=z_min, max=z_max, frozen=True)
    delta_D = Parameter("delta_D", 10, min=1, max=100)
    log10_B = Parameter("log10_B", -2, min=-4, max=2)
    t_var = Parameter("t_var", "600 s", min=10, max=np.pi * 1e7)

    mu_s = Parameter("mu_s", 0, min=0, max=1, frozen=True)
    log10_r = Parameter("log10_r", 18, min=16, max=22, frozen=True)

    if model_type == "ssc":
        emission_region_pars_names = ["z", "delta_D", "log10_B", "t_var"]
        emission_region_pars = [z, delta_D, log10_B, t_var]
    elif model_type == "ec":
        emission_region_pars_names = ["z", "delta_D", "B", "t_var", "mu_s", "log10_r"]
        emission_region_pars = [z, delta_D, log10_B, t_var, mu_s, log10_r]

    parameters = dict(zip(emission_region_pars_names, emission_region_pars))
    return parameters


def get_targets_parameters(line):
    """Return a dict of `~gammapy.modeling.Parameter`s for the line and thermal
    emitters.

    Parameters
    ----------
    line : str
        a line from `~agnpy.targets.lines_dictionary`
    """

    # disk parameters
    L_disk = Parameter("L_disk", "1e45 erg s-1", min=1e42, max=1e48, frozen=True)
    M_BH = Parameter("M_BH", "1e42 g", min=1e32, max=1e45, frozen=True)
    m_dot = Parameter("m_dot", "1e26 g s-1", min=1e24, max=1e30, frozen=True)
    R_in = Parameter("R_in", "1e14 cm", min=1e12, max=1e16, frozen=True)
    R_out = Parameter("R_out", "1e17 cm", min=1e12, max=1e19, frozen=True)
    # BLR parameters
    xi_line = Parameter("xi_line", 0.6, min=0.0, max=1.0, frozen=True)
    lambda_line = Parameter(
        "lambda_line", lines_dictionary[line]["lambda"], min=900, max=7000, frozen=True
    )
    R_line = Parameter("R_line", "1e17 cm", min=1e16, max=1e18, frozen=True)
    # DT parameters
    xi_dt = Parameter("xi_dt", 0.6, min=0.0, max=1.0, frozen=True)
    T_dt = Parameter("T_dt", "1e3 K", min=1e2, max=1e4, frozen=True)
    R_dt = Parameter("R_dt", "2.5e18 cm", min=1.0e17, max=1.0e19, frozen=True)

    targets_pars_names = [
        "L_disk",
        "M_BH",
        "m_dot",
        "R_in",
        "R_out",
        "xi_line",
        "lambda_line",
        "R_line",
        "xi_dt",
        "T_dt",
        "R_dt",
    ]
    targets_pars = [
        L_disk,
        M_BH,
        m_dot,
        R_in,
        R_out,
        xi_line,
        lambda_line,
        R_line,
        xi_dt,
        T_dt,
        R_dt,
    ]
    parameters = dict(zip(targets_pars_names, targets_pars))
    return parameters


class SynchrotronSelfComptonSpectralModel(SpectralModel):

    tag = ["SynchrotronSelfComptonSpectralModel"]

    def __init__(self, n_e, ssa=False):
        """Gammapy wrapper for a source emitting Synchrotron and SSC radiation.

        Parameters
        ----------
        n_e : `~agnpy.spectra.ElectronDistribution`
            electron distribution to be used for this modelling
        ssa : bool
            whether or not to calculate synchrotron self-absorption

        Returns
        -------
        `~gammapy.modeling.models.SpectralModel`
        """
        self._n_e = n_e
        self.ssa = ssa

        # parameters of the particles energy distribution
        spectral_pars = get_spectral_parameters_from_n_e(self._n_e)
        self._spectral_pars_names = list(spectral_pars.keys())

        # parameters of the emission region
        emission_region_pars = get_emission_region_parameters("ssc")
        self._emission_region_pars_names = list(emission_region_pars.keys())

        # group the model parameters, add the norm at the bottom of the list
        norm = Parameter("norm", 1, min=0.1, max=10, is_norm=True, frozen=True)
        self.default_parameters = Parameters(
            [*spectral_pars.values(), *emission_region_pars.values(), norm]
        )

        super().__init__()

    @property
    def spectral_parameters(self):
        """Select all the parameters related to the particle distribution."""
        return self.parameters.select(self._spectral_pars_names)

    @property
    def emission_region_parameters(self):
        """Select all the parameters related to the emission region."""
        return self.parameters.select(self._emission_region_pars_names)

    def evaluate(self, energy, **kwargs):
        """Evaluate the SED model.
        NOTE: All the model parameters will be passed as kwargs by
        SpectralModel.evaluate()."""

        nu = energy.to("Hz", equivalencies=u.spectral())

        # sort the parameters related to the emission region
        z = kwargs.pop("z")
        delta_D = kwargs.pop("delta_D")
        B = 10 ** kwargs.pop("log10_B") * u.G
        t_var = kwargs.pop("t_var")
        # compute the luminosity distance and the size of the emission region
        d_L = Distance(z=z).to("cm")
        R_b = (c * t_var * delta_D / (1 + z)).to("cm")

        # pop the norm and raise to 10 the spectral parameters with log10_ names
        kwargs.pop("norm")
        args = [
            10 ** (kwargs[key].value) if key.startswith("log10_") else kwargs[key].value
            for key in kwargs
        ]
        # first comes the norm
        args[0] *= u.Unit("cm-3")

        sed_synch = Synchrotron.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self._n_e, *args, ssa=self.ssa
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self._n_e, *args, ssa=self.ssa
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

    def __init__(self, n_e, targets, blr_line="Lyalpha", ssa=False):
        """Gammapy wrapper for a source emitting Synchrotron, SSC, and EC on a
        list of targets.

        Parameters
        ----------
        n_e : `~agnpy.spectra.ElectronDistribution`
            electron distribution to be used for this modelling
        targets : list of strings ["blr", "dt"]
            targets to be considered for external Compton
            typically we consider the BLR or the DT or both, the EC on disk
            is not considered as it is subdominant at distances >> R_out
        blr_line : str
            line to be considered if EC on BLR is computed
        ssa : bool
            whether or not to calculate synchrotron self-absorption

        Returns
        -------
        `~gammapy.modeling.models.SpectralModel`
        """

        self._n_e = n_e
        self.targets = targets
        self.ssa = ssa

        # parameters of the particles energy distribution
        spectral_pars = get_spectral_parameters_from_n_e(self._n_e)
        self._spectral_pars_names = list(spectral_pars.keys())

        # parameters of the emission region
        emission_region_pars = get_emission_region_parameters("ec")
        self._emission_region_pars_names = list(emission_region_pars.keys())

        # parameters of the targets
        targets_pars = get_targets_parameters(blr_line)
        self._targets_pars_names = list(targets_pars.keys())

        # group the model parameters, add the norm at the bottom of the list
        norm = Parameter("norm", 1, min=0.1, max=10, is_norm=True, frozen=True)
        self.default_parameters = Parameters(
            [
                *spectral_pars.values(),
                *emission_region_pars.values(),
                *targets_pars.values(),
                norm,
            ]
        )

        super().__init__()

    @property
    def spectral_parameters(self):
        """Select all the parameters related to the particle distribution."""
        return self.parameters.select(self._spectral_pars_names)

    @property
    def emission_region_parameters(self):
        """Select all the parameters related to the emission region."""
        return self.parameters.select(self._emission_region_pars_names)

    @property
    def targets_parameters(self):
        """Select all the parameters related to the targets for EC."""
        return self.parameters.select(self._targets_pars_names)

    def evaluate(self, energy, **kwargs):
        """Evaluate the SED model.
        NOTE: All the model parameters will be passed as kwargs by
        SpectralModel.evaluate()."""

        nu = energy.to("Hz", equivalencies=u.spectral())

        # sort the parameters related to the emission region
        z = kwargs.pop("z")
        delta_D = kwargs.pop("delta_D")
        B = 10 ** kwargs.pop("log10_B") * u.G
        t_var = kwargs.pop("t_var")
        mu_s = kwargs.pop("mu_s")
        r = 10 ** kwargs.pop("log10_r") * u.cm
        # compute the luminosity distance and the size of the emission region
        d_L = Distance(z=z).to("cm")
        R_b = (c * t_var * delta_D / (1 + z)).to("cm")

        # now sort the parameters related to the targets
        # - Disk
        L_disk = kwargs.pop("L_disk")
        M_BH = kwargs.pop("M_BH")
        m_dot = kwargs.pop("m_dot")
        R_in = kwargs.pop("R_in")
        R_out = kwargs.pop("R_out")
        # - BLR
        xi_line = kwargs.pop("xi_line")
        lambda_line = kwargs.pop("lambda_line")
        R_line = kwargs.pop("R_line")
        epsilon_line = (
            lambda_line.to("erg", equivalencies=u.spectral()) / mec2
        ).to_value("")
        # - DT
        xi_dt = kwargs.pop("xi_dt")
        T_dt = kwargs.pop("T_dt")
        R_dt = kwargs.pop("R_dt")
        epsilon_dt = 2.7 * (k_B * T_dt / mec2).to_value("")

        # pop the norm and raise to 10 the spectral parameters with log10_ names
        kwargs.pop("norm")
        args = [
            10 ** (kwargs[key].value) if key.startswith("log10_") else kwargs[key].value
            for key in kwargs
        ]
        # first comes the norm
        args[0] *= u.Unit("cm-3")

        sed_synch = Synchrotron.evaluate_sed_flux(
            nu, z, d_L, delta_D, B, R_b, self._n_e, *args, ssa=self.ssa
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            nu,
            z,
            d_L,
            delta_D,
            B,
            R_b,
            self._n_e,
            *args,
            ssa=self.ssa,
            gamma=gamma_to_integrate
        )
        sed = sed_synch + sed_ssc

        if "blr" in self.targets:
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
                self._n_e,
                *args,
                gamma=gamma_to_integrate
            )
            sed += sed_ec_blr

        if "dt" in self.targets:
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
                self._n_e,
                *args,
                gamma=gamma_to_integrate
            )
            sed += sed_ec_dt

            # now add the thermal component as well
            sed_bb_dt = RingDustTorus.evaluate_bb_norm_sed(
                nu, z, xi_dt * L_disk, T_dt, R_dt, d_L
            )
            sed += sed_bb_dt

        # thermal disk component
        sed_bb_disk = SSDisk.evaluate_multi_T_bb_norm_sed(
            nu, z, L_disk, M_BH, m_dot, R_in, R_out, d_L
        )
        sed += sed_bb_disk

        # eventual reshaping
        # we can do here something like
        # sed = sed.reshape(energy.shape)
        # see the same comment in the SSC model

        return (sed / energy ** 2).to("1 / (cm2 eV s)")
