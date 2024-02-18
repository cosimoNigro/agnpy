# integration kernels to be used for photomeson productions
from scipy.interpolate import CubicSpline
import numpy as np
from pathlib import Path

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
        func = CubicSpline(eta_eta0, s)
    elif parameter == "delta":
        func = CubicSpline(eta_eta0, delta)
    elif parameter == "B":
        func = CubicSpline(eta_eta0, B)
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
    x_2 = np.sqrt((eta - r**2 - 2 * r) * (eta - r**2 + 2 * r))
    x_3 = 1 / (2 * (1 + eta))
    x_minus = x_3 * (x_1 - x_2)

    return x_minus


def x_plus_gamma(eta):
    """Range of x values in which the phi expression is valid.
    Photon secondaries.
    """
    x_1 = eta + r**2
    x_2 = np.sqrt((eta - r**2 - 2 * r) * (eta - r**2 + 2 * r))
    x_3 = 1 / (2 * (1 + eta))
    x_plus = x_3 * (x_1 + x_2)

    return x_plus


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
    x_3 = np.sqrt(eta * (eta - 4 * r * (1 + r)))
    x_minus = (x_2 - x_3) / x_1

    return x_minus / 2


def x_plus_leptons_2(eta):
    """Range of x values in which the phi expression is valid.
    Secondaries for which it is valid:
        * electrons,
        * electron antineutrinos.
    """
    x_1 = 2 * (1 + eta)
    x_2 = eta - (2 * r)
    x_3 = np.sqrt(eta * (eta - 4 * r * (1 + r)))
    x_plus = (x_2 + x_3) / x_1

    return x_plus


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
            elif self.particle in ["electrons", "electron_antineutrino"]:
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
        B = self.B(eta / eta_0)

        x_minus = self.x_minus(eta)
        x_plus = self.x_plus(eta)
        psi = self.psi(eta)

        y = (x - x_minus) / (x_plus - x_minus)
        _exp = np.exp(-s * np.log(x / x_minus) ** delta)
        _log = np.log(2 / (1 + y**2)) ** psi
        _phi = np.where(
            (x > x_minus) * (x < x_plus),
            B * _exp * _log,
            np.where(x < x_minus, B * np.log(2) ** psi, 0),
        )
        return _phi
