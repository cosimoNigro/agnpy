# tests on spectra and emission region modules
import numpy as np
import astropy.units as u
from astropy.constants import e, c, m_e
from agnpy.emission_regions import Blob
import pytest


mec2 = m_e.to("erg", equivalencies=u.mass_energy())
# a global test blob with spectral index 2
SPECTRUM_NORM = 1e-13 * u.Unit("cm-3")
GAMMA_MIN = 1
GAMMA_B = 100
GAMMA_MAX = 1e6
PWL_IDX_2_DICT = {
    "type": "PowerLaw",
    "parameters": {"p": 2.0, "gamma_min": GAMMA_MIN, "gamma_max": GAMMA_MAX},
}
BPL_DICT = {
    "type": "BrokenPowerLaw",
    "parameters": {
        "p1": 2.4,
        "p2": 3.4,
        "gamma_b": GAMMA_B,
        "gamma_min": GAMMA_MIN,
        "gamma_max": GAMMA_MAX,
    },
}
# blob parameters
R_B = 1e16 * u.cm
Z = 0.1
DELTA_D = 10
GAMMA = 10
B = 0.1 * u.G
PWL_BLOB = Blob(R_B, Z, DELTA_D, GAMMA, B, SPECTRUM_NORM, PWL_IDX_2_DICT)
BPL_BLOB = Blob(R_B, Z, DELTA_D, GAMMA, B, SPECTRUM_NORM, BPL_DICT)
# useful for checks
BETA = 1 - 1 / np.power(GAMMA, 2)
V_B = 4 / 3 * np.pi * np.power(R_B, 3)


class TestBlob:
    """class grouping all tests related to the Blob emission region"""

    def test_default_norm_type(self):
        """the default norm type should be 'integral'"""
        assert PWL_BLOB.spectrum_norm_type == "integral"

    # tests for the power law normalisation
    # - tests for normalisations in cm3
    def test_pwl_integral_norm_cm3(self):
        """test if the integral norm in cm-3 is correctly set"""
        PWL_BLOB.set_n_e(SPECTRUM_NORM, PWL_IDX_2_DICT, "integral")
        assert u.allclose(
            PWL_BLOB.n_e_tot, SPECTRUM_NORM, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    def test_pwl_differential_norm_cm3(self):
        """test if the differential norm in cm-3 is correctly set"""
        PWL_BLOB.set_n_e(SPECTRUM_NORM, PWL_IDX_2_DICT, "differential")
        assert u.allclose(
            PWL_BLOB.n_e.k_e, SPECTRUM_NORM, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    def test_pwl_gamma_1_norm_cm3(self):
        """test if the norm at gamma = 1 in cm-3 is correctly set"""
        PWL_BLOB.set_n_e(SPECTRUM_NORM, PWL_IDX_2_DICT, "gamma=1")
        n_e_gamma_1 = PWL_BLOB.n_e(np.asarray([1]))
        assert u.allclose(
            n_e_gamma_1, SPECTRUM_NORM, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    # - tests for integral normalisations in erg cm-3 and erg
    def test_pwl_integral_norm_erg_cm3(self):
        """test if the integral norm in erg cm-3 is correctly set"""
        PWL_BLOB.set_n_e(3e-4 * u.Unit("erg cm-3"), PWL_IDX_2_DICT, "integral")
        assert u.allclose(
            PWL_BLOB.u_e,
            3e-4 * u.Unit("erg cm-3"),
            atol=0 * u.Unit("erg cm-3"),
            rtol=1e-2,
        )

    def test_pwl_integral_norm_erg(self):
        """test if the integral norm in erg is correctly set"""
        PWL_BLOB.set_n_e(1e48 * u.erg, PWL_IDX_2_DICT, "integral")
        assert u.allclose(PWL_BLOB.W_e, 1e48 * u.erg, atol=0 * u.erg, rtol=1e-2)

    # tests for the broken power law normalisation
    # - tests for normalisations in cm3
    def test_bpl_integral_norm_cm3(self):
        """test if the integral norm in cm-3 is correctly set"""
        BPL_BLOB.set_n_e(SPECTRUM_NORM, BPL_DICT, "integral")
        assert u.allclose(
            BPL_BLOB.n_e_tot, SPECTRUM_NORM, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    def test_bpl_differential_norm_cm3(self):
        """test if the differential norm in cm-3 is correctly set"""
        BPL_BLOB.set_n_e(SPECTRUM_NORM, BPL_DICT, "differential")
        assert u.allclose(
            BPL_BLOB.n_e.k_e, SPECTRUM_NORM, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    def test_bpl_gamma_1_norm_cm3(self):
        """test if the norm at gamma = 1 in cm-3 is correctly set"""
        BPL_BLOB.set_n_e(SPECTRUM_NORM, BPL_DICT, "gamma=1")
        n_e_gamma_1 = BPL_BLOB.n_e(np.asarray([1]))
        assert u.allclose(
            n_e_gamma_1, SPECTRUM_NORM, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    # - tests for integral normalisations in erg cm-3 and erg
    def test_bpl_integral_norm_erg_cm3(self):
        """test if the integral norm in erg cm-3 is correctly set"""
        BPL_BLOB.set_n_e(3e-4 * u.Unit("erg cm-3"), BPL_DICT, "integral")
        assert u.allclose(
            BPL_BLOB.u_e,
            3e-4 * u.Unit("erg cm-3"),
            atol=0 * u.Unit("erg cm-3"),
            rtol=1e-2,
        )

    def test_bpl_integral_norm_erg(self):
        """test if the integral norm in erg is correctly set"""
        BPL_BLOB.set_n_e(1e48 * u.erg, BPL_DICT, "integral")
        assert u.allclose(BPL_BLOB.W_e, 1e48 * u.erg, atol=0 * u.erg, rtol=1e-2)

    # test if mismatching unit and normalisation type raises an NameError
    @pytest.mark.parametrize(
        "spectrum_norm, spectrum_norm_type",
        [
            (1e48 * u.erg, "differential"),
            (1e48 * u.erg, "gamma=1"),
            (1e2 * u.Unit("erg cm-3"), "differential"),
            (1e2 * u.Unit("erg cm-3"), "gamma=1"),
        ],
    )
    def test_non_available_norm_type(self, spectrum_norm, spectrum_norm_type):
        """check that the spectrum_norm_type 'differential' and 'gamma=1' 
        raise a NameError for a spectrum_norm in erg or erg cm-3"""
        with pytest.raises(NameError):
            PWL_BLOB.set_n_e(spectrum_norm, PWL_IDX_2_DICT, spectrum_norm_type)

    # test on blob properties
    def test_set_delta_D(self):
        PWL_BLOB.set_delta_D(Gamma=10, theta_s=20 * u.deg)
        assert np.allclose(PWL_BLOB.delta_D, 1.53804, atol=0)

    def test_set_gamma_size(self):
        PWL_BLOB.set_gamma_size(1000)
        assert len(PWL_BLOB.gamma) == 1000

    def test_N_e(self):
        """check that N_e is n_e * V_b i.e. test their ratio to be V_b"""
        PWL_BLOB.set_n_e(SPECTRUM_NORM, PWL_IDX_2_DICT, "differential")
        n_e = PWL_BLOB.n_e(PWL_BLOB.gamma)
        N_e = PWL_BLOB.N_e(PWL_BLOB.gamma)
        assert u.allclose(N_e / n_e, V_B, atol=0 * u.Unit("cm3"), rtol=1e-3)

    def test_pwl_n_e_tot_analytical(self):
        n_e_expected = SPECTRUM_NORM * (1 / GAMMA_MIN - 1 / GAMMA_MAX)
        assert u.allclose(
            PWL_BLOB.n_e_tot, n_e_expected, atol=0 * u.Unit("cm-3"), rtol=1e-3
        )

    def test_pwl_N_e_tot_analytical(self):
        N_e_expected = V_B * SPECTRUM_NORM * (1 / GAMMA_MIN - 1 / GAMMA_MAX)
        assert np.allclose(PWL_BLOB.N_e_tot, N_e_expected.to_value(""), rtol=1e-3)

    def test_pwl_u_e_analyitcal(self):
        u_e_expected = mec2 * SPECTRUM_NORM * np.log(GAMMA_MAX / GAMMA_MIN)
        assert u.allclose(
            PWL_BLOB.u_e, u_e_expected, atol=0 * u.Unit("erg cm-3"), rtol=1e-4
        )

    def test_pwl_W_e_analytical(self):
        W_e_expected = mec2 * V_B * SPECTRUM_NORM * np.log(GAMMA_MAX / GAMMA_MIN)
        assert u.allclose(PWL_BLOB.W_e, W_e_expected, atol=0 * u.erg, rtol=1e-4)

    def test_U_B(self):
        # strip the units for convenience on this one
        U_B_expected = np.power(B.value, 2) / (8 * np.pi) * u.Unit("erg cm-3")
        assert np.allclose(PWL_BLOB.U_B, U_B_expected, atol=0 * u.Unit("erg cm-3"))

    def test_P_jet_e(self):
        u_e_expected = mec2 * SPECTRUM_NORM * np.log(GAMMA_MAX / GAMMA_MIN)
        P_jet_e_expected = (
            2 * np.pi * np.power(R_B, 2) * BETA * np.power(GAMMA, 2) * c * u_e_expected
        )
        assert u.allclose(
            PWL_BLOB.P_jet_e, P_jet_e_expected, atol=0 * u.Unit("erg s-1"), rtol=1e-2
        )

    def test_P_jet_B(self):
        U_B_expected = np.power(B.value, 2) / (8 * np.pi) * u.Unit("erg cm-3")
        P_jet_B_expected = (
            2 * np.pi * np.power(R_B, 2) * BETA * np.power(GAMMA, 2) * c * U_B_expected
        )
        assert u.allclose(
            PWL_BLOB.P_jet_B, P_jet_B_expected, atol=0 * u.Unit("erg s-1"), rtol=1e-2
        )
