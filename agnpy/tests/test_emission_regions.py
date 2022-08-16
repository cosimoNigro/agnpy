# tests the emission region modules
import numpy as np
import astropy.units as u
from astropy.constants import m_e, m_p
import pytest
from agnpy.spectra import PowerLaw, BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.utils.conversion import Gauss_cgs_unit


class TestBlob:
    """Class grouping all tests related to the Blob emission region."""

    def test_blob_properties(self):
        """Test that the blob properties are properly updated if the basic
        attributes are modified."""
        blob = Blob()
        blob.R_b = 1e17 * u.cm
        blob.z = 2.1
        blob.delta_D = 8
        blob.Gamma = 5
        blob.B = 2 * u.G
        assert u.isclose(
            blob.V_b, 4.1887902e51 * u.Unit("cm3"), atol=0 * u.Unit("cm3"), rtol=1e-3
        )
        assert u.isclose(blob.d_L, 5.21497473e28 * u.cm, atol=0 * u.cm, rtol=1e-3)
        assert np.isclose(blob.Beta, 0.9798, atol=0, rtol=1e-3)
        assert u.isclose(blob.theta_s, 5.67129265 * u.deg, atol=0 * u.deg, rtol=1e-3)
        assert u.isclose(
            blob.B_cgs,
            2 * u.Unit(Gauss_cgs_unit),
            atol=0 * u.Unit(Gauss_cgs_unit),
            rtol=1e-3,
        )

    def test_particle_spectra(self):
        """Test for the blob properties related to the particle spectra."""
        n_e = BrokenPowerLaw()
        n_p = PowerLaw(mass=m_p)
        # first we initialise the blob without the protons distribution
        blob = Blob(n_e=n_e)
        # assert that the proton distribution is not set
        with pytest.raises(AttributeError):
            assert blob.n_p
        assert blob.gamma_p is None
        # now let us set the proton distribution
        blob.n_p = n_p
        # change the grid of Lorentz factors
        gamma = np.logspace(2, 6, 50)
        blob.set_gamma_e(gamma[0], gamma[-1], len(gamma))
        assert np.array_equal(blob.gamma_e, gamma)
        blob.set_gamma_p(gamma[0], gamma[-1], len(gamma))
        assert np.array_equal(blob.gamma_p, gamma)

    def test_particles_densities(self):
        """Test different methods to initialise the protons and electrons
        distributions. Check the attributes related to the particles
        distributions computed by the blob.
        """
        blob = Blob()

        # intialise from total particle density
        n_tot = 1e-5 * u.Unit("cm-3")
        N_tot = blob.V_b * n_tot

        n_e = PowerLaw.from_total_density(
            n_tot=n_tot, mass=m_e, p=2.1, gamma_min=1e3, gamma_max=1e6
        )

        n_p = PowerLaw.from_total_density(
            n_tot=n_tot, mass=m_p, p=2.1, gamma_min=1e3, gamma_max=1e6
        )

        blob.n_e = n_e
        blob.n_p = n_p
        assert u.isclose(blob.n_e_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=0.05)
        assert u.isclose(blob.n_p_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=0.05)
        assert np.isclose(blob.N_e_tot, N_tot, atol=0, rtol=0.05)
        assert np.isclose(blob.N_p_tot, N_tot, atol=0, rtol=0.05)

        # intialise from total particle density
        u_tot = 3e-4 * u.Unit("erg cm-3")

        n_e = PowerLaw.from_total_energy_density(
            u_tot=u_tot, mass=m_e, p=2.5, gamma_min=1e2, gamma_max=1e6
        )

        n_p = PowerLaw.from_total_energy_density(
            u_tot=u_tot, mass=m_p, p=2.5, gamma_min=1e2, gamma_max=1e6
        )

        blob.n_e = n_e
        blob.n_p = n_p
        assert u.isclose(blob.u_e, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=0.05)
        assert u.isclose(blob.u_p, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=0.05)

        # intialise from total energy
        W = 1e48 * u.erg

        n_e = PowerLaw.from_total_energy(
            W=W, V=blob.V_b, mass=m_e, p=2.5, gamma_min=1e2, gamma_max=1e6
        )

        n_p = PowerLaw.from_total_energy(
            W=W, V=blob.V_b, mass=m_p, p=2.5, gamma_min=1e2, gamma_max=1e6
        )

        blob.n_e = n_e
        blob.n_p = n_p
        assert u.isclose(blob.W_e, W, atol=0 * u.Unit("erg"), rtol=0.05)
        assert u.isclose(blob.W_p, W, atol=0 * u.Unit("erg"), rtol=0.05)


class TestOld:
    def test_default_norm_type(self):
        """the default norm type should be 'integral'"""
        assert pwl_blob_test.spectrum_norm_type == "integral"

    def test_print_pwl(self):
        """tests for the power-law spectrum - printing test"""
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

    def test_print_bpwl(self):
        """tests for the broken power-law spectrum - printing test"""
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

    def test_print_lp(self):
        """tests for the log-parabola spectrum - printing test"""
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

    def test_set_delta_D(self):
        """test on blob properties"""
        pwl_blob_test.set_delta_D(Gamma=10, theta_s=20 * u.deg)
        assert np.allclose(pwl_blob_test.delta_D, 1.53804, atol=0)

    def test_set_gamma_size(self):
        """test set gamma size"""
        pwl_blob_test.set_gamma_size(1000)
        assert len(pwl_blob_test.gamma) == 1000

    def test_N_e(self):
        """check that N_e is n_e * V_b i.e. test their ratio to be V_b"""
        pwl_blob_test.set_spectrum(spectrum_norm_test, pwl_dict_test, "differential")
        n_e = pwl_blob_test.n_e(pwl_blob_test.gamma)
        N_e = pwl_blob_test.N_e(pwl_blob_test.gamma)
        assert u.allclose(N_e / n_e, V_b_test, atol=0 * u.Unit("cm3"), rtol=1e-3)

    def test_U_B(self):
        """test on blob properties"""
        # strip the units for convenience on this one
        U_B_expected = np.power(B_test.value, 2) / (8 * np.pi) * u.Unit("erg cm-3")
        assert np.allclose(pwl_blob_test.U_B, U_B_expected, atol=0 * u.Unit("erg cm-3"))

    def test_P_jet_e(self):
        """test on blob properties"""
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
        """test on blob properties"""
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
