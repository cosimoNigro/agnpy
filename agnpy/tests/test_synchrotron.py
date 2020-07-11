# test on synchrotron module
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c, h
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron, nu_synch_peak, epsilon_B, synch_sed_param_bpl
from pathlib import Path
import pytest
import matplotlib.pyplot as plt


tests_dir = Path(__file__).parent

mec2 = m_e.to("erg", equivalencies=u.mass_energy())
epsilon_equivalency = [
    (u.Hz, u.Unit(""), lambda x: h.cgs * x / mec2, lambda x: x * mec2 / h.cgs)
]
# global blob, same parameters as the one defined to produce Figure 7.4 of
# Dermer Menon 2009
SPECTRUM_NORM = 1e48 * u.Unit("erg")
PWL_DICT = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e5},
}
R_B = 1e16 * u.cm
B = 1 * u.G
Z = Distance(1e27, unit=u.cm).z
DELTA_D = 10
GAMMA = 10
PWL_BLOB = Blob(R_B, Z, DELTA_D, GAMMA, B, SPECTRUM_NORM, PWL_DICT)

# blob with brokn power law of electrons, to test the parametrisation of the
# delta function approximation
# set the spectrum normalisation (total energy in electrons in this case)
BPL_DICT = {
    "type": "BrokenPowerLaw",
    "parameters": {
        "p1": 2.5,
        "p2": 3.5,
        "gamma_b": 1e4,
        "gamma_min": 1e2,
        "gamma_max": 1e7,
    },
}
BPL_BLOB = Blob(R_B, Z, DELTA_D, GAMMA, B, SPECTRUM_NORM, BPL_DICT)


class TestSynchrotron:
    """class grouping all tests related to the Synchrotron class"""

    def test_synch_reference_sed(self):
        """test agnpy synchrotron SED against the one sampled from Figure
        7.4 of Dermer Menon 2009"""
        sampled_synch_sed_table = np.loadtxt(
            f"{tests_dir}/sampled_seds/figure_7_4_dermer_menon_2009.txt",
            delimiter=",",
            comments="#",
        )
        sampled_synch_nu = sampled_synch_sed_table[:, 0] * u.Hz
        sampled_synch_sed = sampled_synch_sed_table[:, 1] * u.Unit("erg cm-2 s-1")
        synch = Synchrotron(PWL_BLOB)
        # recompute the SED at the same ordinates where the figure was sampled
        agnpy_synch_sed = synch.sed_flux(sampled_synch_nu)
        deviation = np.abs(1 - agnpy_synch_sed.value / sampled_synch_sed.value)
        # requires that the SED points deviate less than 20 % from the figure
        assert np.all(deviation < 0.15)

    def test_ssa_sed(self):
        """test this version SSA SED against the one generated with version 0.0.6"""
        sampled_ssa_sed_table = np.loadtxt(
            f"{tests_dir}/sampled_seds/ssa_sed_agnpy_v0_0_6.txt",
            delimiter=",",
            comments="#",
        )
        sampled_ssa_nu = sampled_ssa_sed_table[:, 0] * u.Hz
        sampled_ssa_sed = sampled_ssa_sed_table[:, 1] * u.Unit("erg cm-2 s-1")
        ssa = Synchrotron(PWL_BLOB, ssa=True)
        agnpy_ssa_sed = ssa.sed_flux(sampled_ssa_nu)
        assert np.allclose(agnpy_ssa_sed.value, sampled_ssa_sed.value, atol=0)

    def test_nu_synch_peak(self):
        gamma = 100
        nu_synch = nu_synch_peak(B, gamma).to_value("Hz")
        assert np.isclose(nu_synch, 27992489872.33304, atol=0)

    def test_epsilon_B(self):
        assert np.isclose(epsilon_B(B), 2.2655188038060715e-14, atol=0)

    def test_synch_sed_param_bpl(self):
        """check that the parametrised delta-function approximation for the
        synchrotron radiation gives exactly the same result of the full formula 
        from which it was derived"""
        nu = np.logspace(8, 23) * u.Hz
        synch = Synchrotron(BPL_BLOB)
        sed_delta_approx = synch.sed_flux_delta_approx(nu)
        # check that the synchrotron parameterisation work
        y = BPL_BLOB.B.value * BPL_BLOB.delta_D
        k_eq = (BPL_BLOB.u_e / BPL_BLOB.U_B).to_value("")
        sed_param = synch_sed_param_bpl(
            nu.value,
            y,
            k_eq,
            BPL_BLOB.n_e.p1,
            BPL_BLOB.n_e.p2,
            BPL_BLOB.n_e.gamma_b,
            BPL_BLOB.n_e.gamma_min,
            BPL_BLOB.n_e.gamma_max,
            BPL_BLOB.d_L.cgs.value,
            BPL_BLOB.R_b.cgs.value,
            BPL_BLOB.z,
        )
        assert np.allclose(sed_param, sed_delta_approx.value, atol=0, rtol=1e-2)
