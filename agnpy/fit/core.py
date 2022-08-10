# functions / classes shared by all wrapper types
import numpy as np
from sherpa.models import model
from gammapy import modeling


class Parameter:
    """Let us define a general parameter class. This parameter can then be
    casted as a parameter of the wrapping package, e.g. Gammapy or sherpa."""

    def __init__(self, name, value, unit, min, max, frozen=False):
        self.name = name
        self.value = value
        self.unit = unit
        self.min = min
        self.max = max
        self.frozen = frozen

    def to_sherpa_parameter(self, modelname):
        """Return a `~sherpa.models.parameter.Parameter`.

        Parameters
        ----------
        modelname : str
            The name of the model component containing the parameter.
        """
        # sherpa has a hard_max set to
        hard_max = 3.40282e38
        if self.max > hard_max:
            hard_max = 1e50
        return model.Parameter(
            modelname=modelname,
            name=self.name,
            val=self.value,
            units=self.unit,
            min=self.min,
            max=self.max,
            hard_max=hard_max,
            frozen=self.frozen,
        )

    def to_gammapy_parameter(self):
        """Return a `~gammapy.modeling.parameter."""
        return modeling.Parameter(
            name=self.name,
            value=self.value,
            unit=self.unit,
            min=self.min,
            max=self.max,
            frozen=self.frozen,
        )


def get_spectral_parameters_from_n_e(n_e, backend, modelname=None):
    """Get the list of parameters of the particles energy distribution.

    Parameters
    ----------
    n_e : `~agnpy.spectra.ElectronDistribution`
        electron distribution
    backend : str
        backend to be used (transform to gammapy or sherpa parameters)
    modelname : str
        if sherpa is selected as backend, set the modelname

    Returns
    -------
    parameters : dict of `~agnpy.fit.core.Parameter`
        dictionary of parameters of the particles energy distribution
    """
    _pars_names = []
    _pars = []

    pars = vars(n_e).copy()
    # particles distribution have the integrator among the attributes
    pars.pop("integrator")

    for name, value in zip(pars.keys(), pars.values()):
        if name == "k_e":
            par = Parameter(
                "log10_k_e", np.log10(value.to_value("cm-3")), "", min=-10, max=10
            )
        elif name == "gamma_min":
            par = Parameter(
                "log10_gamma_min", np.log10(value), "", min=0, max=4, frozen=True
            )
        elif name == "gamma_max":
            par = Parameter(
                "log10_gamma_max", np.log10(value), "", min=4, max=8, frozen=True
            )
        elif name in ["gamma_b", "gamma_0", "gamma_c"]:
            par = Parameter("log10_" + name, np.log10(value), "", min=2, max=6)
        elif name.startswith("p"):
            par = Parameter(name, value, "", min=1, max=5)
        elif name == "q":
            par = Parameter(name, value, "", min=0.001, max=1)

        _pars_names.append(par.name)
        _pars.append(par)

    # transform the parameters to sherpa o gammapy parameters
    if backend == "gammapy":
        _pars = [par.to_gammapy_parameter() for par in _pars]
    if backend == "sherpa":
        _pars = [par.to_sherpa_parameter(modelname) for par in _pars]

    parameters = dict(zip(_pars_names, _pars))
    return parameters


def make_emission_region_parameters_dict(scenario, backend, modelname=None):
    """Return a dict of `~agnpy.fit.core.Parameter`s for the emission region.
    The list of parameters is different whether we are considering a SSC or EC
    model.

    Parameters
    ----------
    scenario : "ssc" or "ec"
        the emission region parameters change depend on the scenario considered
    backend : str
        backend to be used (transform to gammapy or sherpa parameters)
    modelname : str
        if sherpa is selected as backend, set the modelname

    Returns
    -------
    parameters : dict of `~agnpy.fit.core.Parameter`
        dictionary of parameters of the emission region
    """
    _pars_names = []
    _pars = []

    # parameters common to SSC and EC models
    z = Parameter("z", 0.1, "", min=0.001, max=10, frozen=True)
    delta_D = Parameter("delta_D", 10, "", min=1, max=100)
    log10_B = Parameter("log10_B", -2, "", min=-4, max=2)
    t_var = Parameter("t_var", 600, "s", min=10, max=np.pi * 1e7)
    # parameters exclusive to EC
    mu_s = Parameter("mu_s", 0, "", min=0, max=1, frozen=True)
    log10_r = Parameter("log10_r", 18, "", min=16, max=22, frozen=True)

    if scenario == "ssc":
        _pars_names = ["z", "delta_D", "log10_B", "t_var"]
        _pars = [z, delta_D, log10_B, t_var]
    elif scenario == "ec":
        _pars_names = [
            "z",
            "delta_D",
            "log10_B",
            "t_var",
            "mu_s",
            "log10_r",
        ]
        _pars = [z, delta_D, log10_B, t_var, mu_s, log10_r]

    # transform the parameters to sherpa o gammapy parameters
    if backend == "gammapy":
        _pars = [par.to_gammapy_parameter() for par in _pars]
    if backend == "sherpa":
        _pars = [par.to_sherpa_parameter(modelname) for par in _pars]

    parameters = dict(zip(_pars_names, _pars))
    return parameters


def make_targets_parameters_dict(targets, backend, modelname=None):
    """Return a dict of `~agnpy.fit.core.Parameter`s for the line and thermal
    emitters.

    Parameters
    ----------
    targets : list of str
       targets for EC : `["blr"]`, or `["dt"]`, or `["blr", "dt"]`
    backend : str
        backend to be used (transform to gammapy or sherpa parameters)
    modelname : str
        if sherpa is selected as backend, set the modelname

    Returns
    -------
    parameters : dict of `~agnpy.fit.core.Parameter`
        dictionary of parameters of the emission region
    """
    _pars_names = []
    _pars = []

    # disk parameters
    log10_L_disk = Parameter("log10_L_disk", 45, "", min=42, max=48, frozen=True)
    M_BH = Parameter("M_BH", 1e42, "g", min=1e32, max=1e45, frozen=True)
    m_dot = Parameter("m_dot", 1e26, "g s-1", min=1e24, max=1e30, frozen=True)
    R_in = Parameter("R_in", 1e14, "cm", min=1e12, max=1e16, frozen=True)
    R_out = Parameter("R_out", 1e17, "cm", min=1e12, max=1e19, frozen=True)
    _pars_names.extend(["log10_L_disk", "M_BH", "m_dot", "R_in", "R_out"])
    _pars.extend([log10_L_disk, M_BH, m_dot, R_in, R_out])

    if "blr" in targets:
        xi_line = Parameter("xi_line", 0.6, "", min=0.0, max=1.0, frozen=True)
        lambda_line = Parameter(
            "lambda_line", 1215.67, "Angstrom", min=900, max=7000, frozen=True
        )
        R_line = Parameter("R_line", 1e17, "cm", min=1e16, max=1e18, frozen=True)
        _pars_names.extend(["xi_line", "lambda_line", "R_line"])
        _pars.extend([xi_line, lambda_line, R_line])

    if "dt" in targets:
        xi_dt = Parameter("xi_dt", 0.6, "", min=0.0, max=1.0, frozen=True)
        T_dt = Parameter("T_dt", 1e3, "K", min=1e2, max=1e4, frozen=True)
        R_dt = Parameter("R_dt", 3e18, "cm", min=1e17, max=1e20, frozen=True)
        _pars_names.extend(["xi_dt", "T_dt", "R_dt"])
        _pars.extend([xi_dt, T_dt, R_dt])

    # transform the parameters to sherpa o gammapy parameters
    if backend == "gammapy":
        _pars = [par.to_gammapy_parameter() for par in _pars]
    if backend == "sherpa":
        _pars = [par.to_sherpa_parameter(modelname) for par in _pars]

    parameters = dict(zip(_pars_names, _pars))
    return parameters
