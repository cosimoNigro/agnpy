# test different normalisations for the spectra
import astropy.units as u
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
import matplotlib.pyplot as plt

spectrum_norm = 1e-13 * u.Unit("cm-3")

spectrum_dict_pwl = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1, "gamma_max": 1e7},
}

spectrum_dict_bpl = {
    "type": "BrokenPowerLaw",
    "parameters": {
        "p1": 2.0,
        "p2": 3.0,
        "gamma_b": 1e2,
        "gamma_min": 1,
        "gamma_max": 1e7,
    },
}

spectrum_dict_bpl_2 = {
    "type": "BrokenPowerLaw2",
    "parameters": {
        "p1": 2.0,
        "p2": 3.0,
        "gamma_b": 1e2,
        "gamma_min": 1,
        "gamma_max": 1e7,
    },
}

R_b = 1e16 * u.cm
B = 1 * u.G
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10

for spectrum_dict in (spectrum_dict_pwl, spectrum_dict_bpl, spectrum_dict_bpl_2):
    for norm_type in ("integral", "differential", "gamma=1"):
        blob = Blob(
            R_b,
            z,
            delta_D,
            Gamma,
            B,
            spectrum_norm,
            spectrum_dict,
            spectrum_norm_type=norm_type,
        )
        blob.plot_n_e()

# let us trigger the error
blob = Blob(
    R_b,
    z,
    delta_D,
    Gamma,
    B,
    1e48 * u.erg,
    spectrum_dict_pwl,
    spectrum_norm_type="gamma=1",
)
