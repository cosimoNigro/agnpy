from scipy.interpolate import interp1d
import numpy as np


def lookup_tab1(eta, particle):

    data_dir = Path(__file__).parent.parent
    interp_file = f"{data_dir}/data/photo_meson/interpolation_tables_kelner_aharonian/{particle}.txt"

    eta_eta0, s, delta, B = np.genfromtxt(interp_file, dtype = 'float',  comments = '#', usecols = (0,1,2,3), unpack = 'True')

    s_int = interp1d(eta_eta0, s, kind='linear', bounds_error=False, fill_value="extrapolate")
    delta_int = interp1d(eta_eta0, delta, kind='linear', bounds_error=False, fill_value="extrapolate")
    B_int = interp1d(eta_eta0, B, kind='linear', bounds_error=False, fill_value="extrapolate")

    return s_int(eta), delta_int(eta), B_int(eta)


def x_plus_minus_photon(eta):
    # photon

    r = 0.146 # r = m_pi / M_p
    x_1 = eta + r ** 2
    x_2 = np.sqrt((eta - r ** 2 - 2 * r) * (eta - r ** 2 + 2 * r))
    x_3 = 1 / (2 * (1 + eta))

    x_plus = x_3 * (x_1 + x_2)
    x_minus = x_3 * (x_1 - x_2)

    return x_plus, x_minus


def x_plus_minus_positron(eta):
    # positron ,antinu_muon, nu_electron

    r = 0.146 # r = m_pi / M_p
    x_1 = eta + r ** 2
    x_2 = np.sqrt((eta - r ** 2 - 2 * r) * (eta - r ** 2 + 2 * r))
    x_3 = 1 / (2 * (1 + eta))

    x_plus = x_3 * (x_1 + x_2)
    x_minus = x_3 * (x_1 - x_2)

    return x_plus, x_minus / 4


def x_plus_minus_electron(eta):
    # electron, antinu_electron

    r = 0.146
    x_1 = 2 * (1 + eta)
    x_2 = eta - (2 * r)
    x_3 = np.sqrt(eta * (eta - 4 * r * (1 + r)))

    x_plus = (x_2 + x_3) / x_1
    x_minus = (x_2 - x_3) / x_1

    return x_plus, x_minus / 2


def x_plus_minus_nu_muon(eta):
    # nu_muon

    r = 0.146 # r = m_pi / M_p
    x_1 = eta + r ** 2
    x_2 = np.sqrt((eta - r ** 2 - 2 * r) * (eta - r ** 2 + 2 * r))
    x_3 = 1 / (2 * (1 + eta))

    x_plus = x_3 * (x_1 + x_2)
    x_minus = x_3 * (x_1 - x_2)

    rho = eta / 0.313

    if rho < 2.14:
        xp = 0.427 * x_plus
    elif rho > 2.14 and rho < 10:
        xp = (0.427 + 0.0729 * (rho - 2.14)) * x_plus
    elif rho > 10:
        xp = x_plus

    return xp, (x_minus * 0.427)


def phi_photon(eta, x):
    # photon

    x_p, x_n = x_plus_minus_photon(eta, particle)
    s, delta, B = lookup_tab1(eta / 0.313, 'photon') # eta_0 = 0.313
    psi = 2.5 + 0.4 * np.log(eta / 0.313)
    y = (x - x_n) / (x_p - x_n)
    ln1 = np.exp(- s * (np.log(x / x_n)) ** delta)
    ln2 = np.log(2. / (1 + y**2))

    return np.where((x > x_n)*(x < x_p), B * ln1 * ln2 ** psi,
            np.where(x < x_n, B * (np.log(2)) ** psi), 0
    )

def phi_positron(eta, x):
    # positron

    x_p, x_n = x_plus_minus_positron(eta)
    s, delta, B = lookup_tab1(eta / 0.313, 'positron') # eta_0 = 0.313
    psi = 2.5 + 1.4 * np.log(eta / 0.313)

    return np.where((x > x_n)*(x < x_p), B * ln1 * ln2 ** psi,
            np.where(x < x_n, B * (np.log(2)) ** psi), 0
    )

def phi_antinu_muon(eta, x):
    # antinu_muon

    x_p, x_n = x_plus_minus_positron(eta)
    s, delta, B = lookup_tab1(eta / 0.313, 'antinu_muon') # eta_0 = 0.313
    psi = 2.5 + 1.4 * np.log(eta / 0.313)

    return np.where((x > x_n)*(x < x_p), B * ln1 * ln2 ** psi,
            np.where(x < x_n, B * (np.log(2)) ** psi), 0
    )

def phi_nu_electron(eta, x):
    # nu_electron

    x_p, x_n = x_plus_minus_positron(eta)
    s, delta, B = lookup_tab1(eta / 0.313, 'nu_electron') # eta_0 = 0.313
    psi = 2.5 + 1.4 * np.log(eta / 0.313)

    return np.where((x > x_n)*(x < x_p), B * ln1 * ln2 ** psi,
            np.where(x < x_n, B * (np.log(2)) ** psi), 0
    )

def phi_nu_muon(eta, x):
    # nu_muon

    x_p, x_n = x_plus_minus_nu_muon(eta)
    s, delta, B = lookup_tab1(eta / 0.313, 'nu_muon') # eta_0 = 0.313
    psi = 2.5 + 1.4 * np.log(eta / 0.313)

    return np.where((x > x_n)*(x < x_p), B * ln1 * ln2 ** psi,
            np.where(x < x_n, B * (np.log(2)) ** psi), 0
    )

def phi_electron(eta, x):
    # electron

    x_p, x_n = x_plus_minus_electron(eta)
    s, delta, B = lookup_tab1(eta / 0.313, 'electron') # eta_0 = 0.313
    psi = 6 * (1 - np.exp(1.5 * (4 - eta/0.313))) * (np.sign(eta/0.313 - 4) + 1) / 2.

    return np.where((x > x_n)*(x < x_p), B * ln1 * ln2 ** psi,
            np.where(x < x_n, B * (np.log(2)) ** psi), 0
    )

def phi_antinu_electron(eta, x):
    # antinu_electron

    x_p, x_n = x_plus_minus_electron(eta)
    s, delta, B = lookup_tab1(eta / 0.313, 'antinu_electron') # eta_0 = 0.313
    psi = 6 * (1 - np.exp(1.5 * (4 - eta/0.313))) * (np.sign(eta/0.313 - 4) + 1) / 2.

    return np.where((x > x_n)*(x < x_p), B * ln1 * ln2 ** psi,
            np.where(x < x_n, B * (np.log(2)) ** psi), 0
    )
