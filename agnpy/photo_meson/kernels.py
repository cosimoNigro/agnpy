# integration kernels to be used for photomeson productions
from pathlib import Path

import astropy.units as u
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

from agnpy.utils.math import fmax, ftiny

data_dir = Path(__file__).parent.parent
secondaries = [
    "gamma",
    "electron",
    "positron",
    "electron_neutrino",
    "electron_antineutrino",
    "muon_neutrino",
    "muon_antineutrino",
]

# ratio between the pion and proton mass, to be used in several calculations
eta_0 = 0.313
r = 0.146


def log_interp(zz, xx, yy):
    logz = np.log10(zz)
    logx = np.log10(xx)

    nyy = np.clip(yy, ftiny, fmax)
    logy = np.log10(nyy)

    logf = interp1d(logx, logy, fill_value="extrapolate")

    return np.power(10, logf(logz))


def interpolate_phi_parameter(particle, parameter):
    """Interpolates the tables providing the parameters fo the phi functions
    as a function of eta. Table 1 and 2 in [KelnerAharonian2008]_.

    Parameters
    ----------
    particle : string
        secondary for which the spectrum has to be calculated.
    parameter: string
        the function to be interpolated
    """

    interp_file = (
        f"{data_dir}/data/photo_meson/kelner_aharonian_2008/phi_tables/{particle}.txt"
    )

    eta_eta0, s, delta, B = np.genfromtxt(
        interp_file, dtype="float", comments="#", usecols=(0, 1, 2, 3), unpack="True"
    )

    if parameter == "s":
        func = lambda x: log_interp(x, eta_eta0, s)
    elif parameter == "delta":
        func = lambda x: log_interp(x, eta_eta0, delta)
    elif parameter == "B":
        func = lambda x: log_interp(x, eta_eta0, B)
    else:
        raise ValueError(
            f"{parameter} not available among the parameters to be interpolated"
        )

    return func


def x_minus_gamma(eta):
    """Range of x values in which the phi expression is valid.
    Photon secondaries.
    """
    x_1 = eta + r**2
    x2_arg = (eta - r**2 - 2 * r) * (eta - r**2 + 2 * r)
    with np.errstate(invalid="ignore"):
        x_2 = np.where(x2_arg > 0, np.sqrt(x2_arg), 0)
    x_3 = 1 / (2 * (1 + eta))
    x_minus = x_3 * (x_1 - x_2)

    return np.where(x2_arg > 0, x_minus, 0)


def x_plus_gamma(eta):
    """Range of x values in which the phi expression is valid.
    Photon secondaries.
    """
    x_1 = eta + r**2
    x2_arg = (eta - r**2 - 2 * r) * (eta - r**2 + 2 * r)
    with np.errstate(invalid="ignore"):
        x_2 = np.where(x2_arg > 0, np.sqrt(x2_arg), 0)
    x_3 = 1 / (2 * (1 + eta))
    x_plus = x_3 * (x_1 + x_2)

    return np.where(x2_arg > 0, x_plus, 0)


def x_minus_leptons_1(eta):
    """Range of x values in which the phi expression is valid.
    Secondaries for which it is valid:
        * positrons,
        * muon antineutrinos,
        * electron neutrinos.
    """
    return x_minus_gamma(eta) / 4


def x_minus_leptons_2(eta):
    """Range of x values in which the phi expression is valid.
    Secondaries for which it is valid:
        * electrons,
        * electron antineutrinos.
    """
    x_1 = 2 * (1 + eta)
    x_2 = eta - (2 * r)
    x3_arg = eta * (eta - 4 * r * (1 + r))
    with np.errstate(invalid="ignore"):
        x_3 = np.where(x3_arg > 0, np.sqrt(x3_arg), 0)
    x_minus = (x_2 - x_3) / x_1

    return np.where(x3_arg > 0, x_minus / 2, 0)


def x_plus_leptons_2(eta):
    """Range of x values in which the phi expression is valid.
    Secondaries for which it is valid:
        * electrons,
        * electron antineutrinos.
    """
    x_1 = 2 * (1 + eta)
    x_2 = eta - (2 * r)
    x3_arg = eta * (eta - 4 * r * (1 + r))
    with np.errstate(invalid="ignore"):
        x_3 = np.where(x3_arg > 0, np.sqrt(x3_arg), 0)
    x_plus = (x_2 + x_3) / x_1

    return np.where(x3_arg > 0, x_plus, 0)


def x_minus_leptons_3(eta):
    """Range of x values in which the phi expression is valid.
    Secondaries for which it is valid:
        * muon neutrinos.
    """
    return 0.427 * x_minus_gamma(eta)


def x_plus_leptons_3(eta):
    """Range of x values in which the phi expression is valid.
    Secondaries for which it is valid:
        * muon neutrinos.
    """
    rho = eta / eta_0
    x_plus = x_plus_gamma(eta)

    x_plus = np.where(
        rho <= 2.14,
        0.427 * x_plus,
        np.where(
            (rho > 2.14) * (rho <= 10), (0.427 + 0.0729 * (rho - 2.14)) * x_plus, x_plus
        ),
    )

    return x_plus


def psi_gamma(eta):
    """Psi function for gamma rays"""
    return 2.5 + 0.4 * np.log(eta / eta_0)


def psi_1(eta):
    """psi function for electrons and electron antineutrinos"""
    return (
        6 * (1 - np.exp(1.5 * (4 - eta / eta_0))) * (np.sign(eta / eta_0 - 4) + 1) / 2.0
    )


def psi_2(eta):
    """Psi function for positrons, all neutrinos except electron antineutrinos"""
    return 2.5 + 1.4 * np.log(eta / eta_0)


class PhiKernel:
    """Phi function, Eq. (27) in [KelnerAharonian2008]_."""

    def __init__(self, particle):
        if particle not in secondaries:
            raise ValueError(f"{particle} not available among the secondaries")
        else:
            self.particle = particle
            # parameters of the phi function
            self.s = interpolate_phi_parameter(particle, "s")
            self.delta = interpolate_phi_parameter(particle, "delta")
            self.B = interpolate_phi_parameter(particle, "B")
            # maximum and minimum energies (these are functions of eta as well
            if self.particle == "gamma":
                self.x_minus = x_minus_gamma
                self.x_plus = x_plus_gamma
            elif self.particle in [
                "positron",
                "muon_antineutrino",
                "electron_neutrino",
            ]:
                self.x_minus = x_minus_leptons_1
                self.x_plus = x_plus_gamma
            elif self.particle in ["electron", "electron_antineutrino"]:
                self.x_minus = x_minus_leptons_2
                self.x_plus = x_plus_leptons_2
            elif self.particle == "muon_neutrino":
                self.x_minus = x_minus_leptons_3
                self.x_plus = x_plus_leptons_3
            # values of psi
            if self.particle == "gamma":
                self.psi = psi_gamma
            elif self.particle in ["electron", "electron_antineutrino"]:
                self.psi = psi_1
            else:
                self.psi = psi_2

    def __call__(self, eta, x):
        """Evaluate the phi function, Eq. (27) of [KelnerAharonian2008]_."""
        # evaluate the interpolated parameters
        s = self.s(eta / eta_0)
        delta = self.delta(eta / eta_0)
        B = self.B(eta / eta_0) * u.Unit("cm3 s-1")

        x_minus = self.x_minus(eta)
        x_plus = self.x_plus(eta)
        psi = self.psi(eta)

        with np.errstate(all="ignore"):
            # y = (x - x_minus) / (x_plus - x_minus)
            y = np.where(
                (x_plus - x_minus) != 0, (x - x_minus) / (x_plus - x_minus), -1
            )

            # x_x_min = x / x_minus
            x_x_min = np.where(x_minus != 0, x / x_minus, 0)

            # _exp = np.exp(-s * np.log(x / x_minus) ** delta)
            _exp = np.where(x_x_min >= 1, np.exp(-s * np.log(x_x_min) ** delta), 0)

            # _log = np.log(2 / (1 + y**2)) ** psi
            _logy = np.where(y != -1, np.log(2 / (1 + y**2)), 0)
            _log = np.where(_logy > 0, _logy**psi, 0)

        _phi = np.where(
            (x > x_minus) * (x < x_plus),
            B * _exp * _log,
            np.where(x < x_minus, B * np.log(2) ** psi, 0),
        )
        return _phi


def build_interp2d(x_grid, y_grid, values):

    interp = RegularGridInterpolator(
        (x_grid, y_grid),
        values,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    def f(x, y):
        x_arr, y_arr = np.broadcast_arrays(x, y)
        points = np.column_stack([x_arr.ravel(), y_arr.ravel()])
        result = interp(points)

        return result.reshape(x_arr.shape)

    return f


def interpolate_g_parameter(particle, parameter):
    interp_file = f"{data_dir}/data/photo_meson/dphi_dtheta_tables/{particle}.txt"

    eta_eta0_tab, theta_tab, x_cut, A0, A1, A2, A3, A4 = np.genfromtxt(
        interp_file,
        dtype=float,
        comments="#",
        usecols=(0, 1, 2, 3, 4, 5, 6, 7),
        unpack=True,
    )

    eta_unique = np.unique(eta_eta0_tab)
    theta_unique = np.unique(theta_tab)

    shape = (len(eta_unique), len(theta_unique))

    parameter_map = {
        "x_cut": x_cut,
        "A0": A0,
        "A1": A1,
        "A2": A2,
        "A3": A3,
        "A4": A4,
    }

    if parameter not in parameter_map:
        raise ValueError(
            f"{parameter} not available among the parameters to be interpolated"
        )

    values = parameter_map[parameter].reshape(shape)

    return build_interp2d(eta_unique, theta_unique, values)


class gKernel:
    def __init__(self, particle):
        if particle not in secondaries:
            raise ValueError(f"{particle} not available among the secondaries")
        else:
            self.particle = particle

            # parameters of the g function
            self.x_cut = interpolate_g_parameter(particle, "x_cut")
            self.A0 = interpolate_g_parameter(particle, "A0")
            self.A1 = interpolate_g_parameter(particle, "A1")
            self.A2 = interpolate_g_parameter(particle, "A2")
            self.A3 = interpolate_g_parameter(particle, "A3")
            self.A4 = interpolate_g_parameter(particle, "A4")

    def __call__(self, eta, theta, x):
        # evaluate the interpolated parameters
        eta_eta0 = eta / eta_0

        x_cut = self.x_cut(eta_eta0, theta)
        A0 = self.A0(eta_eta0, theta)
        A1 = self.A1(eta_eta0, theta)
        A2 = self.A2(eta_eta0, theta)
        A3 = self.A3(eta_eta0, theta)
        A4 = self.A4(eta_eta0, theta)

        X = -A3 * (np.log10(x) - A4)
        g = np.where(x <= x_cut, np.pow(10.0, A0 * X ** (A1 + np.log10(X)) + A2), 0.0)

        return g
