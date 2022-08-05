# wrap agnpy SED computation via sherpa's 1D model
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from astropy.constants import c, k_B
from sherpa.models import model
from ..utils.conversion import mec2
from ..spectra import PowerLaw
from ..targets import SSDisk, RingDustTorus
from ..synchrotron import Synchrotron
from ..compton import SynchrotronSelfCompton, ExternalCompton
from .core import (
    get_spectral_parameters_from_n_e,
    make_emission_region_parameters_dict,
    make_targets_parameters_dict,
)


gamma_size = 300
gamma_to_integrate = np.logspace(1, 9, gamma_size)


def _sort_spectral_parameters(args, n_e):
    """Sort and scale the parameters of the electron distribution."""
    # parameters of the spectrum are in log10 scale, first comes the norm
    args[0] = 10 ** args[0] * u.Unit("cm-3")
    # last two are the gamma min and gamma max
    args[-2] = 10 ** args[-2]
    args[-1] = 10 ** args[-1]
    # if this is not a power law, then there is a break or pivot Lorentz factor
    if not isinstance(n_e, PowerLaw):
        args[-3] = 10 ** args[-3]


def _evaluate_sed_ssc_scenario(x, pars, n_e, ssa):
    """At the model evaluation, sherpa passes the model parameters as a simple
    list, `pars`. This function sorts the parameters and evaluates the total SED
    for the SSC scenario.
    NOTE: sherpa parameters are NOT `~astropy.Quantities`, properly set them."""
    (*args, z, delta_D, log10_B, t_var) = pars

    _sort_spectral_parameters(args, n_e)

    # parameters of the emission region
    B = 10 ** log10_B * u.G
    # compute the luminosity distance and the size of the emission region
    d_L = Distance(z=z).to("cm")
    R_b = (c.to_value("cm s-1") * t_var * delta_D) / (1 + z) * u.cm

    # evaluate the SED
    x *= u.eV
    nu = x.to("Hz", equivalencies=u.spectral())
    sed_synch = Synchrotron.evaluate_sed_flux(
        nu, z, d_L, delta_D, B, R_b, n_e, *args, ssa=ssa
    )
    sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
        nu, z, d_L, delta_D, B, R_b, n_e, *args, ssa=ssa
    )
    return sed_synch + sed_ssc


def _evaluate_sed_ec_blr_scenario(x, pars, n_e, ssa):
    """At the model evaluation, sherpa passes the model parameters as a simple
    list, `pars`. This function sorts the parameters and evaluates the total SED
    for the EC on BLR scenario.
    NOTE: sherpa parameters are NOT `~astropy.Quantities`, properly set them."""
    (
        *args,
        z,
        delta_D,
        log10_B,
        t_var,
        mu_s,
        log10_r,
        log10_L_disk,
        M_BH,
        m_dot,
        R_in,
        R_out,
        xi_line,
        lambda_line,
        R_line,
    ) = pars

    _sort_spectral_parameters(args, n_e)

    # parameters of the emission region
    B = 10 ** log10_B * u.G
    # compute the luminosity distance and the size of the emission region
    d_L = Distance(z=z).to("cm")
    R_b = (c.to_value("cm s-1") * t_var * delta_D) / (1 + z) * u.cm
    r = 10 ** log10_r * u.cm

    # target parameters
    L_disk = 10 ** log10_L_disk * u.Unit("erg s-1")
    M_BH *= u.g
    m_dot *= u.Unit("g s-1")
    R_in *= u.cm
    R_out *= u.cm
    lambda_line *= u.Angstrom
    R_line *= u.cm
    epsilon_line = (lambda_line.to("erg", equivalencies=u.spectral()) / mec2).to_value(
        ""
    )

    # evaluate the SED
    x *= u.eV
    nu = x.to("Hz", equivalencies=u.spectral())
    sed_synch = Synchrotron.evaluate_sed_flux(
        nu, z, d_L, delta_D, B, R_b, n_e, *args, ssa=ssa
    )
    sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
        nu, z, d_L, delta_D, B, R_b, n_e, *args, ssa=ssa
    )
    sed_bb_disk = SSDisk.evaluate_multi_T_bb_norm_sed(
        nu, z, L_disk, M_BH, m_dot, R_in, R_out, d_L
    )
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
        n_e,
        *args,
        gamma=gamma_to_integrate
    )
    return sed_synch + sed_ssc + sed_bb_disk + sed_ec_blr


def _evaluate_sed_ec_dt_scenario(x, pars, n_e, ssa):
    """At the model evaluation, sherpa passes the model parameters as a simple
    list, `pars`. This function sorts the parameters and evaluates the total SED
    for the EC on DT scenario."""
    (
        *args,
        z,
        delta_D,
        log10_B,
        t_var,
        mu_s,
        log10_r,
        log10_L_disk,
        M_BH,
        m_dot,
        R_in,
        R_out,
        xi_dt,
        T_dt,
        R_dt,
    ) = pars

    _sort_spectral_parameters(args, n_e)

    # parameters of the emission region
    B = 10 ** log10_B * u.G
    # compute the luminosity distance and the size of the emission region
    d_L = Distance(z=z).to("cm")
    R_b = (c.to_value("cm s-1") * t_var * delta_D) / (1 + z) * u.cm
    r = 10 ** log10_r * u.cm

    # target parameters
    L_disk = 10 ** log10_L_disk * u.Unit("erg s-1")
    M_BH *= u.g
    m_dot *= u.Unit("g s-1")
    R_in *= u.cm
    R_out *= u.cm
    T_dt *= u.K
    R_dt *= u.cm
    epsilon_dt = 2.7 * (k_B * T_dt / mec2).to_value("")

    # evaluate the SED
    x *= u.eV
    nu = x.to("Hz", equivalencies=u.spectral())
    sed_synch = Synchrotron.evaluate_sed_flux(
        nu, z, d_L, delta_D, B, R_b, n_e, *args, ssa=ssa
    )
    sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
        nu, z, d_L, delta_D, B, R_b, n_e, *args, ssa=ssa
    )
    sed_bb_disk = SSDisk.evaluate_multi_T_bb_norm_sed(
        nu, z, L_disk, M_BH, m_dot, R_in, R_out, d_L
    )
    sed_bb_dt = RingDustTorus.evaluate_bb_norm_sed(
        nu, z, xi_dt * L_disk, T_dt, R_dt, d_L
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
        n_e,
        *args,
        gamma=gamma_to_integrate
    )
    return sed_synch + sed_ssc + sed_bb_disk + sed_bb_dt + sed_ec_dt


def _evaluate_sed_ec_blr_dt_scenario(x, pars, n_e, ssa):
    """At the model evaluation, sherpa passes the model parameters as a simple
    list, `pars`. This function sorts the parameters and evaluates the total SED
    for the EC on BLR and DT scenario.
    NOTE: sherpa parameters are NOT `~astropy.Quantities`, properly set them."""
    (
        *args,
        z,
        delta_D,
        log10_B,
        t_var,
        mu_s,
        log10_r,
        log10_L_disk,
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
    ) = pars

    _sort_spectral_parameters(args, n_e)

    # parameters of the emission region
    B = 10 ** log10_B * u.G
    # compute the luminosity distance and the size of the emission region
    d_L = Distance(z=z).to("cm")
    R_b = (c.to_value("cm s-1") * t_var * delta_D) / (1 + z) * u.cm
    r = 10 ** log10_r * u.cm

    # target parameters
    L_disk = 10 ** log10_L_disk * u.Unit("erg s-1")
    M_BH *= u.g
    m_dot *= u.Unit("g s-1")
    R_in *= u.cm
    R_out *= u.cm
    lambda_line *= u.Angstrom
    R_line *= u.cm
    epsilon_line = (lambda_line.to("erg", equivalencies=u.spectral()) / mec2).to_value(
        ""
    )
    T_dt *= u.K
    R_dt *= u.cm
    epsilon_dt = 2.7 * (k_B * T_dt / mec2).to_value("")

    # evaluate the SED
    x *= u.eV
    nu = x.to("Hz", equivalencies=u.spectral())
    sed_synch = Synchrotron.evaluate_sed_flux(
        nu, z, d_L, delta_D, B, R_b, n_e, *args, ssa=ssa
    )
    sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
        nu, z, d_L, delta_D, B, R_b, n_e, *args, ssa=ssa
    )
    sed_bb_disk = SSDisk.evaluate_multi_T_bb_norm_sed(
        nu, z, L_disk, M_BH, m_dot, R_in, R_out, d_L
    )
    sed_bb_dt = RingDustTorus.evaluate_bb_norm_sed(
        nu, z, xi_dt * L_disk, T_dt, R_dt, d_L
    )
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
        n_e,
        *args,
        gamma=gamma_to_integrate
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
        n_e,
        *args,
        gamma=gamma_to_integrate
    )
    return sed_synch + sed_ssc + sed_bb_disk + sed_bb_dt + sed_ec_blr + sed_ec_dt


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
        emission_region_pars = make_emission_region_parameters_dict(
            "ssc", backend="sherpa", modelname=self.name
        )

        pars_list = [*spectral_pars.values(), *emission_region_pars.values()]

        # each parameter should be declared as an attribute, see
        # https://sherpa.readthedocs.io/en/4.14.0/model_classes/usermodel.html
        pars_attr_list = []

        for par in pars_list:
            setattr(self, par.name, par)
            pars_attr_list.append(getattr(self, par.name))

        super().__init__(self.name, tuple(pars_attr_list))

    def set_emission_region_parameters_from_blob(self, blob):
        """Set the parameter of the emission region from a Blob instance"""
        self.z = blob.z
        self.delta_D = blob.delta_D
        self.log10_B = np.log10(blob.B.to_value("G"))
        self.t_var = blob.t_var.to_value("s")

    def calc(self, pars, x):
        """Evaluate the SED model."""
        return _evaluate_sed_ssc_scenario(x, pars, self._n_e, self.ssa)


class ExternalComptonRegriddableModel1D(model.RegriddableModel1D):
    def __init__(self, n_e, targets, ssa=False):
        """sherpa wrapper for a source emitting Synchrotron, SSC, and EC on a
        list of targets.

        Parameters
        ----------
        n_e : `~agnpy.spectra.ElectronDistribution`
            electron distribution to be used for this modelling
        targets : list of strings ["blr", "dt"]
            targets to be considered for external Compton
            typically we consider the BLR or the DT or both, the EC on disk
            is not considered as it is subdominant at distances >> R_out
        ssa : bool
            whether or not to calculate synchrotron self-absorption

        Returns
        -------
        `~sherpa.models.Regriddable1DModel`
        """
        self.name = "ec"
        self._n_e = n_e
        self.targets = targets
        self.ssa = ssa

        # parameters of the particles energy distribution
        spectral_pars = get_spectral_parameters_from_n_e(
            self._n_e, backend="sherpa", modelname=self.name
        )

        # parameters of the emission region
        emission_region_pars = make_emission_region_parameters_dict(
            "ec", backend="sherpa", modelname=self.name
        )

        # parameters of the targets
        targets_pars = make_targets_parameters_dict(
            self.targets, backend="sherpa", modelname=self.name
        )

        pars_list = [
            *spectral_pars.values(),
            *emission_region_pars.values(),
            *targets_pars.values(),
        ]

        # each parameter should be declared as an attribute, see
        # https://sherpa.readthedocs.io/en/4.14.0/model_classes/usermodel.html
        pars_attr_list = []

        for par in pars_list:
            setattr(self, par.name, par)
            pars_attr_list.append(getattr(self, par.name))

        super().__init__(self.name, tuple(pars_attr_list))

    def set_emission_region_parameters_from_blob(self, blob, r):
        """Set the parameter of the emission region from a Blob instance.
        Since this is EC, remember to specify also the distance"""
        self.z = blob.z
        self.delta_D = blob.delta_D
        self.log10_B = np.log10(blob.B.to_value("G"))
        self.t_var = blob.t_var.to_value("s")
        self.mu_s = blob.mu_s
        self.log10_r = np.log10(r.to_value("cm"))

    def set_targets_parameters_from_targets(self, disk, blr=None, dt=None):
        """Set the parameter of the targets for EC from instances of `~agnpy.targets`."""
        self.log10_L_disk = np.log10(disk.L_disk.to_value("erg s-1"))
        self.M_BH = disk.M_BH.to_value("g")
        self.m_dot = disk.m_dot.to_value("g s-1")
        self.R_in = disk.R_in.to_value("cm")
        self.R_out = disk.R_out.to_value("cm")

        if blr is not None:
            self.xi_line = blr.xi_line
            self.lambda_line = blr.lambda_line.to_value("Angstrom")
            self.R_line = blr.R_line.to_value("cm")

        if dt is not None:
            self.xi_dt = dt.xi_dt
            self.T_dt = dt.T_dt.to_value("K")
            self.R_dt = dt.R_dt.to_value("cm")

    def calc(self, pars, x):
        """Evaluate the SED model."""
        if self.targets == ["blr"]:
            return _evaluate_sed_ec_blr_scenario(x, pars, self._n_e, self.ssa)
        if self.targets == ["dt"]:
            return _evaluate_sed_ec_dt_scenario(x, pars, self._n_e, self.ssa)
        if self.targets == ["blr", "dt"]:
            return _evaluate_sed_ec_blr_dt_scenario(x, pars, self._n_e, self.ssa)
