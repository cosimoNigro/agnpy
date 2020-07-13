# tests on targets module
import numpy as np
import astropy.units as u
from astropy.constants import e, c, m_e, M_sun, G, sigma_sb
from agnpy.targets import CMB, SSDisk
import pytest

M_BH = 1.2 * 1e9 * M_sun
L_DISK = 1.512 * 1e46 * u.Unit("erg s-1")
ETA = 1 / 12
R_G = 1.77 * 1e14 * u.cm
R_IN_G_UNITS = 6
R_OUT_G_UNITS = 200
R_IN = R_IN_G_UNITS * R_G
R_OUT = R_OUT_G_UNITS * R_G

DISK = SSDisk(M_BH, L_DISK, ETA, R_IN, R_OUT)

# useful for checks
L_EDD = 15.12 * 1e46 * u.Unit("erg s-1")
M_DOT = 2.019 * 1e26 * u.Unit("g s-1")

RTOL = 1e-2

class TestDisk:
    """class grouping all the tests related to the SSDisk target"""
    
    # global quantities defining the test disk
    def test_L_Edd(self):
        assert u.allclose(DISK.L_Edd, L_EDD, rtol=RTOL)

    def test_l_Edd(self):
        assert u.allclose(DISK.l_Edd, 0.1, rtol=RTOL)

    def test_m_dot(self):
        assert u.allclose(DISK.m_dot, M_DOT, rtol=RTOL)

    @pytest.mark.parametrize(
        "R_in, R_out, R_g_units",
        [(R_IN_G_UNITS, R_OUT_G_UNITS, False), (R_IN, R_OUT, True),],
    )
    def test_R_in_R_out_units(self, R_in, R_out, R_g_units):
        """check if a TypeError is raised when passing R_in and R_out with 
        (without) units but specifiying R_g_units True (False)"""
        with pytest.raises(TypeError):
            disk = SSDisk(M_BH, L_DISK, ETA, R_in, R_out, R_g_units)

    def test_R_g(self):
        assert u.allclose(DISK.R_g, R_G, rtol=RTOL)

    def test_mu_from_r_tilde(self):
        mu = DISK.mu_from_r_tilde(10)
        mu_min_expected = 0.050
        mu_max_expected = 0.858
        assert np.allclose(mu[0], mu_min_expected, rtol=RTOL)
        assert np.allclose(mu[-1], mu_max_expected, rtol=RTOL)

    def test_phi_disk(self):
        R_tilde = 10
        phi_expected = 0.225
        assert np.allclose(DISK.phi_disk(R_tilde), phi_expected, rtol=RTOL)

    def test_phi_disk_mu(self):
        r_tilde = 10
        # assume R_tilde = 10 as before
        mu = 1 / np.sqrt(2)
        phi_expected = 0.225
        assert np.allclose(DISK.phi_disk_mu(mu, r_tilde), phi_expected, rtol=RTOL)

    def test_epsilon(self):
        R_tilde = 10
        epsilon_expected = 2.7e-5
        assert np.allclose(DISK.epsilon(R_tilde), epsilon_expected, rtol=RTOL)

    def test_epsilon_mu(self):
        r_tilde = 10
        # assume R_tilde = 10 as before
        mu = 1 / np.sqrt(2)
        epsilon_expected = 2.7e-5
        assert np.allclose(DISK.epsilon_mu(mu, r_tilde), epsilon_expected, rtol=1e-1)

    def test_T(self):
        R_tilde = 10
        R = 10 * R_G
        phi_expected = 0.225
        # Eq. 64 [Dermer2009]
        T_expected = np.power(
            3 * G * M_BH * M_DOT / (8 * np.pi * np.power(R, 3) * sigma_sb), 1 / 4
        ).to("K")
        assert u.allclose(DISK.T(R_tilde), T_expected, rtol=RTOL)

    def test_Theta(R_tilde):
        R_tilde = 10
        epsilon = DISK.epsilon(R_tilde)
        Theta = DISK.Theta(R_tilde)
        assert np.allclose(epsilon, 2.7 * Theta, rtol=RTOL)
