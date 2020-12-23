# tests on spectra and emission region modules
import numpy as np
import astropy.units as u
from astropy.constants import e, c, m_e
from agnpy.emission_regions import Blob
import pytest


mec2 = m_e.to("erg", equivalencies=u.mass_energy())
# variables with _test are global and meant to be used in all tests
spectrum_norm_test = 1e-13 * u.Unit("cm-3")
gamma_min_test = 1
gamma_max_test = 1e6
# test ditctionaries for different spectra
pwl_dict_test = {
    "type": "PowerLaw",
    "parameters": {"p": 2, "gamma_min": gamma_min_test, "gamma_max": gamma_max_test},
}

bpwl_dict_test = {
    "type": "BrokenPowerLaw",
    "parameters": {
        "p1": 2.4,
        "p2": 3.4,
        "gamma_b": 1e2,
        "gamma_min": gamma_min_test,
        "gamma_max": gamma_max_test,
    },
}
lp_dict_test = {
    "type": "LogParabola",
    "parameters": {
        "p": 2.5,
        "q": 0.1,
        "gamma_0": 1e3,
        "gamma_min": gamma_min_test,
        "gamma_max": gamma_max_test,
    },
}
# blob parameters
R_b_test = 1e16 * u.cm
z_test = 0.1
delta_D_test = 10
Gamma_test = 10
B_test = 0.1 * u.G
pwl_blob_test = Blob(
    R_b_test,
    z_test,
    delta_D_test,
    Gamma_test,
    B_test,
    spectrum_norm_test,
    pwl_dict_test,
)
bpwl_blob_test = Blob(
    R_b_test,
    z_test,
    delta_D_test,
    Gamma_test,
    B_test,
    spectrum_norm_test,
    bpwl_dict_test,
)
lp_blob_test = Blob(
    R_b_test,
    z_test,
    delta_D_test,
    Gamma_test,
    B_test,
    spectrum_norm_test,
    lp_dict_test,
)
# useful for checks
Beta_test = 1 - 1 / np.power(Gamma_test, 2)
V_b_test = 4 / 3 * np.pi * np.power(R_b_test, 3)


class TestBlob:
    """class grouping all tests related to the Blob emission region"""

    def test_default_norm_type(self):
        """the default norm type should be 'integral'"""
        assert pwl_blob_test.spectrum_norm_type == "integral"

    # tests for the power-law spectrum
    # - printing test
    def test_print_pwl(self):
        print(pwl_blob_test)
        assert True

    # - tests for normalisations in cm3
    def test_pwl_integral_norm_cm3(self):
        """test if the integral norm in cm-3 is correctly set"""
        pwl_blob_test.set_spectrum(spectrum_norm_test, pwl_dict_test, "integral")
        assert u.allclose(
            pwl_blob_test.n_e_tot,
            spectrum_norm_test,
            atol=0 * u.Unit("cm-3"),
            rtol=1e-2,
        )

    def test_pwl_differential_norm_cm3(self):
        """test if the differential norm in cm-3 is correctly set"""
        pwl_blob_test.set_spectrum(spectrum_norm_test, pwl_dict_test, "differential")
        assert u.allclose(
            pwl_blob_test.n_e.k_e,
            spectrum_norm_test,
            atol=0 * u.Unit("cm-3"),
            rtol=1e-2,
        )

    def test_pwl_gamma_1_norm_cm3(self):
        """test if the norm at gamma = 1 in cm-3 is correctly set"""
        pwl_blob_test.set_spectrum(spectrum_norm_test, pwl_dict_test, "gamma=1")
        assert u.allclose(
            pwl_blob_test.n_e(1), spectrum_norm_test, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    # - tests for integral normalisations in erg cm-3 and erg
    def test_pwl_integral_norm_erg_cm3(self):
        """test if the integral norm in erg cm-3 is correctly set"""
        u_e = 3e-4 * u.Unit("erg cm-3")
        pwl_blob_test.set_spectrum(u_e, pwl_dict_test, "integral")
        assert u.allclose(
            pwl_blob_test.u_e, u_e, atol=0 * u.Unit("erg cm-3"), rtol=1e-2,
        )

    def test_pwl_integral_norm_erg(self):
        """test if the integral norm in erg is correctly set"""
        W_e = 1e48 * u.erg
        pwl_blob_test.set_spectrum(W_e, pwl_dict_test, "integral")
        assert u.allclose(pwl_blob_test.W_e, W_e, atol=0 * u.erg, rtol=1e-2)

    # tests for the broken power-law spectrum
    # - printing test
    def test_print_bpwl(self):
        print(bpwl_blob_test)
        assert True

    # - tests for normalisations in cm3
    def test_bpl_integral_norm_cm3(self):
        """test if the integral norm in cm-3 is correctly set"""
        bpwl_blob_test.set_spectrum(spectrum_norm_test, bpwl_dict_test, "integral")
        assert u.allclose(
            bpwl_blob_test.n_e_tot,
            spectrum_norm_test,
            atol=0 * u.Unit("cm-3"),
            rtol=1e-2,
        )

    def test_bpl_differential_norm_cm3(self):
        """test if the differential norm in cm-3 is correctly set"""
        bpwl_blob_test.set_spectrum(spectrum_norm_test, bpwl_dict_test, "differential")
        assert u.allclose(
            bpwl_blob_test.n_e.k_e,
            spectrum_norm_test,
            atol=0 * u.Unit("cm-3"),
            rtol=1e-2,
        )

    def test_bpl_gamma_1_norm_cm3(self):
        """test if the norm at gamma = 1 in cm-3 is correctly set"""
        bpwl_blob_test.set_spectrum(spectrum_norm_test, bpwl_dict_test, "gamma=1")
        assert u.allclose(
            bpwl_blob_test.n_e(1),
            spectrum_norm_test,
            atol=0 * u.Unit("cm-3"),
            rtol=1e-2,
        )

    # - tests for integral normalisations in erg cm-3 and erg
    def test_bpl_integral_norm_erg_cm3(self):
        """test if the integral norm in erg cm-3 is correctly set"""
        u_e = 3e-4 * u.Unit("erg cm-3")
        bpwl_blob_test.set_spectrum(u_e, bpwl_dict_test, "integral")
        assert u.allclose(
            bpwl_blob_test.u_e, u_e, atol=0 * u.Unit("erg cm-3"), rtol=1e-2,
        )

    def test_bpl_integral_norm_erg(self):
        """test if the integral norm in erg is correctly set"""
        W_e = 1e48 * u.erg
        bpwl_blob_test.set_spectrum(W_e, bpwl_dict_test, "integral")
        assert u.allclose(bpwl_blob_test.W_e, W_e, atol=0 * u.erg, rtol=1e-2)

    # tests for the log-parabola spectrum
    # - printing test
    def test_print_lp(self):
        print(lp_blob_test)
        assert True

    # - tests for normalisations in cm3
    def test_lp_integral_norm_cm3(self):
        """test if the integral norm in cm-3 is correctly set"""
        lp_blob_test.set_spectrum(spectrum_norm_test, lp_dict_test, "integral")
        assert u.allclose(
            lp_blob_test.n_e_tot,
            spectrum_norm_test,
            atol=0 * u.Unit("cm-3"),
            rtol=1e-2,
        )

    def test_lp_differential_norm_cm3(self):
        """test if the differential norm in cm-3 is correctly set"""
        lp_blob_test.set_spectrum(spectrum_norm_test, lp_dict_test, "differential")
        assert u.allclose(
            lp_blob_test.n_e.k_e,
            spectrum_norm_test,
            atol=0 * u.Unit("cm-3"),
            rtol=1e-2,
        )

    def test_lp_gamma_1_norm_cm3(self):
        """test if the norm at gamma = 1 in cm-3 is correctly set"""
        lp_blob_test.set_spectrum(spectrum_norm_test, lp_dict_test, "gamma=1")
        assert u.allclose(
            lp_blob_test.n_e(1), spectrum_norm_test, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    # - tests for integral normalisations in erg cm-3 and erg
    def test_lp_integral_norm_erg_cm3(self):
        """test if the integral norm in erg cm-3 is correctly set"""
        u_e = 3e-4 * u.Unit("erg cm-3")
        lp_blob_test.set_spectrum(u_e, lp_dict_test, "integral")
        assert u.allclose(
            lp_blob_test.u_e, u_e, atol=0 * u.Unit("erg cm-3"), rtol=1e-2,
        )

    def test_lp_integral_norm_erg(self):
        """test if the integral norm in erg is correctly set"""
        W_e = 1e48 * u.erg
        lp_blob_test.set_spectrum(W_e, lp_dict_test, "integral")
        assert u.allclose(pwl_blob_test.W_e, W_e, atol=0 * u.erg, rtol=1e-2)

    # test if mismatching unit and normalisation type raises a NameError
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
            pwl_blob_test.set_spectrum(spectrum_norm, pwl_dict_test, spectrum_norm_type)

    # test on blob properties
    def test_set_delta_D(self):
        pwl_blob_test.set_delta_D(Gamma=10, theta_s=20 * u.deg)
        assert np.allclose(pwl_blob_test.delta_D, 1.53804, atol=0)

    def test_set_gamma_size(self):
        pwl_blob_test.set_gamma_size(1000)
        assert len(pwl_blob_test.gamma) == 1000

    def test_N_e(self):
        """check that N_e is n_e * V_b i.e. test their ratio to be V_b"""
        pwl_blob_test.set_spectrum(spectrum_norm_test, pwl_dict_test, "differential")
        n_e = pwl_blob_test.n_e(pwl_blob_test.gamma)
        N_e = pwl_blob_test.N_e(pwl_blob_test.gamma)
        assert u.allclose(N_e / n_e, V_b_test, atol=0 * u.Unit("cm3"), rtol=1e-3)

    def test_U_B(self):
        # strip the units for convenience on this one
        U_B_expected = np.power(B_test.value, 2) / (8 * np.pi) * u.Unit("erg cm-3")
        assert np.allclose(pwl_blob_test.U_B, U_B_expected, atol=0 * u.Unit("erg cm-3"))

    def test_P_jet_e(self):
        u_e_expected = (
            mec2 * spectrum_norm_test * np.log(gamma_max_test / gamma_min_test)
        )
        P_jet_e_expected = (
            2
            * np.pi
            * np.power(R_b_test, 2)
            * Beta_test
            * np.power(Gamma_test, 2)
            * c
            * u_e_expected
        )
        assert u.allclose(
            pwl_blob_test.P_jet_e,
            P_jet_e_expected,
            atol=0 * u.Unit("erg s-1"),
            rtol=1e-2,
        )

    def test_P_jet_B(self):
        U_B_expected = np.power(B_test.value, 2) / (8 * np.pi) * u.Unit("erg cm-3")
        P_jet_B_expected = (
            2
            * np.pi
            * np.power(R_b_test, 2)
            * Beta_test
            * np.power(Gamma_test, 2)
            * c
            * U_B_expected
        )
        assert u.allclose(
            pwl_blob_test.P_jet_B,
            P_jet_B_expected,
            atol=0 * u.Unit("erg s-1"),
            rtol=1e-2,
        )
