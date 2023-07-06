from scipy.interpolate import interp1d
import numpy as np
from pathlib import Path

data_dir = Path(__file__).parent.parent
secondaries = ["photon", "electron", "positron", "nu_electron", "nu_muon", "antinu_electron", "antinu_muon"]

# ratio between the pion and proton mass, to be used in several calculations
eta_0 = 0.313
r = 0.146


def interpolate_phi_parameters(eta, particle):
    """Interpolates the tables providing the parameters fo the phi functions
    as a function of eta. Table 1 and 2 in [KelnerAharonian2008]_

    Parameters
    ----------
    eta : float
        Eq. (10) [KelnerAharonian2008]_
    particle : string
        secondary for which the spectrum has to be calculated.
    """
    if particle not in secondaries:
        raise ValueError(f"{particle} not available among the secondaries")

    interp_file = f"{data_dir}/data/photo_mesons/kelner_aharonian_2008/phi_tables/{particle}.txt"

    eta_eta0, s, delta, B = np.genfromtxt(interp_file, dtype = 'float', comments = '#', usecols = (0,1,2,3), unpack = 'True')

    s_int = interp1d(eta_eta0, s, kind='linear', bounds_error=False, fill_value="extrapolate")
    delta_int = interp1d(eta_eta0, delta, kind='linear', bounds_error=False, fill_value="extrapolate")
    B_int = interp1d(eta_eta0, B, kind='linear', bounds_error=False, fill_value="extrapolate")

    return s_int(eta), delta_int(eta), B_int(eta)


def x_minus_plus_photon(eta):
    """Range of x values in which the phi expression is valid.
    Photon secondaries."""

    x_1 = eta + r ** 2
    x_2 = np.sqrt((eta - r ** 2 - 2 * r) * (eta - r ** 2 + 2 * r))
    x_3 = 1 / (2 * (1 + eta))

    x_plus = x_3 * (x_1 + x_2)
    x_minus = x_3 * (x_1 - x_2)

    return x_minus, x_plus


def x_minus_plus_leptons_1(eta):
    """Range of x values in which the phi expression is valid.
    Secondaries for which it is valid:
        * positrons,
        * muon antineutrinos,
        * electron neutrinos.
    """
    x_minus, x_plus = x_minus_plus_photon(eta)

    return x_minus / 4, x_plus


def x_minus_plus_leptons_2(eta):
    """Range of x values in which the phi expression is valid.
    Secondaries for which it is valid:
        * electrons,
        * electron antineutrinos.
    """

    x_1 = 2 * (1 + eta)
    x_2 = eta - (2 * r)
    x_3 = np.sqrt(eta * (eta - 4 * r * (1 + r)))

    x_plus = (x_2 + x_3) / x_1
    x_minus = (x_2 - x_3) / x_1

    return x_minus / 2, x_plus


def x_minus_plus_leptons_3(eta):
    """Range of x values in which the phi expression is valid.
    Secondaries for which it is valid:
        * muon neutrinos.
    """

    x_minus, x_plus = x_minus_plus_photon(eta)

    rho = eta / eta_0

    x_plus = np.where(
        rho < 2.14,
        0.427 * x_plus,
        np.where(
            (rho > 2.14) * (rho < 10),
            (0.427 + 0.0729 * (rho - 2.14)) * x_plus,
            x_plus
            )
    )

    return 0.427 * x_minus, x_plus


def phi(eta, x, x_minus, x_plus, psi, B, s, delta):
    """General phi function, Eq. (27) of [KelnerAharonian2008]_ """

    y = (x - x_minus) / (x_plus - x_minus)
    _exp = np.exp(- s * (np.log(x / x_minus))**delta)
    _log = np.log(2. / (1 + y**2))
    _phi = np.where(
        (x > x_minus) * (x < x_plus),
        B * _exp * _log ** psi,
        np.where(
            x < x_minus,
            B * (np.log(2) ** psi),
            0)
        )
    return _phi


def phi_gamma(eta, x):
    """phi function for photons """

    x_minus, x_plus = x_minus_plus_photon(eta)
    s, delta, B = interpolate_phi_parameters(eta/eta_0, "photon")
    psi = 2.5 + 0.4 * np.log(eta / eta_0)

    return phi(eta, x, x_minus, x_plus, psi, B, s, delta)


def phi_positron(eta, x):
    """phi function for positrons """

    x_minus, x_plus = x_minus_plus_leptons_1(eta)
    s, delta, B = interpolate_phi_parameters(eta/eta_0, "positron")
    psi = 2.5 + 1.4 * np.log(eta / eta_0)

    return phi(eta, x, x_minus, x_plus, psi, B, s, delta)


def phi_antinu_muon(eta, x):
    """phi function for muon antineutrinos """

    x_minus, x_plus = x_minus_plus_leptons_1(eta)
    s, delta, B = interpolate_phi_parameters(eta/eta_0, "antinu_muon")
    psi = 2.5 + 1.4 * np.log(eta / eta_0)

    return phi(eta, x, x_minus, x_plus, psi, B, s, delta)


def phi_nu_electron(eta, x):
    """phi function for electron neutrinos """

    x_minus, x_plus = x_minus_plus_leptons_1(eta)
    s, delta, B = interpolate_phi_parameters(eta/eta_0, "nu_electron")
    psi = 2.5 + 1.4 * np.log(eta / eta_0)

    return phi(eta, x, x_minus, x_plus, psi, B, s, delta)


def phi_nu_muon(eta, x):
    """phi function for muon neutrinos """

    x_minus, x_plus = x_minus_plus_leptons_3(eta)
    s, delta, B = interpolate_phi_parameters(eta/eta_0, 'nu_muon')
    psi = 2.5 + 1.4 * np.log(eta / eta_0)

    return phi(eta, x, x_minus, x_plus, psi, B, s, delta)


def phi_electron(eta, x):
    """phi function for electrons """

    x_minus, x_plus = x_minus_plus_leptons_2(eta)
    s, delta, B = interpolate_phi_parameters(eta/eta_0, 'electron')
    psi = 6 * (1 - np.exp(1.5 * (4 - eta/eta_0))) * (np.sign(eta / eta_0 - 4) + 1) / 2.

    return phi(eta, x, x_minus, x_plus, psi, B, s, delta)


def phi_antinu_electron(eta, x):
    """phi function for electron antineutrinos """

    x_minus, x_plus = x_minus_plus_leptons_2(eta)
    s, delta, B = interpolate_phi_parameters(eta/eta_0, 'antinu_electron')
    psi = 6 * (1 - np.exp(1.5 * (4 - eta / eta_0))) * (np.sign(eta / eta_0 - 4) + 1) / 2.

    return phi(eta, x, x_minus, x_plus, psi, B, s, delta)
