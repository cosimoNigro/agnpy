# tests on targets module
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from astropy.constants import e, c, m_e, M_sun, G, sigma_sb
from agnpy.emission_regions import Blob
from agnpy.targets import CMB, PointSourceBehindJet, SSDisk
import pytest

# global CMB at z = 1
CMB_Z_1 = CMB(z=1)

# global PointSourceBehindJet
PS = PointSourceBehindJet(1e46 * u.Unit("erg s-1"), 1e-5)
R_PS = np.asarray([1e18, 1e19, 1e20]) * u.cm

# global PWL blob, used to compute energy densities of targets
# in the reference frame comoving with the blob
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

# global disk
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


class TestCMB:
    """class grouping all the tests related to the CMB"""

    def test_u(self):
        """test u in the stationary reference frame"""
        assert u.isclose(
            6.67945605e-12 * u.Unit("erg / cm3"),
            CMB_Z_1.u(),
            atol=0 * u.Unit("erg / cm3"),
            rtol=1e-3,
        )

    def test_u_comoving(self):
        """test u in the reference frame comoving with the blob"""
        assert u.isclose(
            8.88373221e-10 * u.Unit("erg / cm3"),
            CMB_Z_1.u(PWL_BLOB),
            atol=0 * u.Unit("erg / cm3"),
            rtol=1e-3,
        )


class TestPointSourceBehindJet:
    """class grouping all the tests related to the PointSourceBehindJet"""

    def test_u(self):
        """test u in the stationary reference frame"""
        assert u.allclose(
            [2.65441873e-02, 2.65441873e-04, 2.65441873e-06] * u.Unit("erg / cm3"),
            PS.u(R_PS),
            atol=0 * u.Unit("erg / cm3"),
            rtol=1e-3,
        )

    def test_u_comoving(self):
        """test u in the reference frame comoving with the blob"""
        assert u.allclose(
            [6.6693519e-05, 6.6693519e-07, 6.6693519e-09] * u.Unit("erg / cm3"),
            PS.u(R_PS, PWL_BLOB),
            atol=0 * u.Unit("erg / cm3"),
            rtol=1e-3,
        )


class TestDisk:
    """class grouping all the tests related to the SSDisk target"""

    # global quantities defining the test disk
    def test_L_Edd(self):
        assert u.isclose(DISK.L_Edd, L_EDD, atol=0 * u.Unit("erg s-1"), rtol=1e-3)

    def test_l_Edd(self):
        assert u.isclose(DISK.l_Edd, 0.1, atol=0, rtol=1e-3)

    def test_m_dot(self):
        assert u.isclose(DISK.m_dot, M_DOT, atol=0 * u.Unit("g s-1"), rtol=1e-3)

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
        assert u.isclose(DISK.R_g, R_G, atol=0 * u.cm, rtol=1e-2)

    def test_mu_from_r_tilde(self):
        mu = DISK.mu_from_r_tilde(10)
        mu_min_expected = 0.050
        mu_max_expected = 0.858
        assert np.isclose(mu[0], mu_min_expected, atol=0, rtol=1e-3)
        assert np.isclose(mu[-1], mu_max_expected, atol=0, rtol=1e-3)

    def test_phi_disk(self):
        R_tilde = 10
        phi_expected = 0.225
        assert np.isclose(DISK.phi_disk(R_tilde), phi_expected, atol=0, rtol=1e-2)

    def test_phi_disk_mu(self):
        r_tilde = 10
        # assume R_tilde = 10 as before
        mu = 1 / np.sqrt(2)
        phi_expected = 0.225
        assert np.allclose(
            DISK.phi_disk_mu(mu, r_tilde), phi_expected, atol=0, rtol=1e-2
        )

    def test_epsilon(self):
        R_tilde = 10
        epsilon_expected = 2.7e-5
        assert np.allclose(DISK.epsilon(R_tilde), epsilon_expected, atol=0, rtol=1e-2)

    def test_epsilon_mu(self):
        r_tilde = 10
        # assume R_tilde = 10 as before
        mu = 1 / np.sqrt(2)
        epsilon_expected = 2.7e-5
        assert np.allclose(
            DISK.epsilon_mu(mu, r_tilde), epsilon_expected, atol=0, rtol=1e-2
        )

    def test_T(self):
        R_tilde = 10
        R = 10 * R_G
        phi_expected = 0.225
        # Eq. 64 [Dermer2009]
        T_expected = np.power(
            3 * G * M_BH * M_DOT / (8 * np.pi * np.power(R, 3) * sigma_sb), 1 / 4
        ).to("K")
        assert u.isclose(DISK.T(R_tilde), T_expected, atol=0 * u.K, rtol=1e-2)

    def test_Theta(R_tilde):
        R_tilde = 10
        epsilon = DISK.epsilon(R_tilde)
        assert np.isclose(epsilon, 2.7 * DISK.Theta(R_tilde), atol=0, rtol=1e-2)
