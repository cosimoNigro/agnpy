# tests on targets module
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from astropy.constants import e, c, m_e, M_sun, G, sigma_sb
from agnpy.emission_regions import Blob
from agnpy.targets import (
    CMB,
    PointSourceBehindJet,
    SSDisk,
    SphericalShellBLR,
    RingDustTorus,
)
import pytest

# global PWL blob, used to compute energy densities of targets
# in the reference frame comoving with it
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

# global CMB at z = 1
CMB_Z_1 = CMB(z=1)

# global PointSourceBehindJet
PS = PointSourceBehindJet(1e46 * u.Unit("erg s-1"), 1e-5)

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

# global SphericalShellBLR
BLR = SphericalShellBLR(L_DISK, 0.1, "Lyalpha", 1e17 * u.cm)

# dust torus definition
DT = RingDustTorus(L_DISK, 0.1, 1000 * u.K)


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
        r = np.asarray([1e18, 1e19, 1e20]) * u.cm
        assert u.allclose(
            [2.65441873e-02, 2.65441873e-04, 2.65441873e-06] * u.Unit("erg / cm3"),
            PS.u(r),
            atol=0 * u.Unit("erg / cm3"),
            rtol=1e-3,
        )

    def test_u_comoving(self):
        """test u in the reference frame comoving with the blob"""
        r = np.asarray([1e18, 1e19, 1e20]) * u.cm
        assert u.allclose(
            [6.6693519e-05, 6.6693519e-07, 6.6693519e-09] * u.Unit("erg / cm3"),
            PS.u(r, PWL_BLOB),
            atol=0 * u.Unit("erg / cm3"),
            rtol=1e-3,
        )


class TestSSDisk:
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


class TestSphericalShellBLR:
    """class grouping all the tests related to the SphericalShellBLR target"""

    @pytest.mark.parametrize(
        "line, lambda_line",
        [
            ("Lyalpha", 1215.67 * u.Angstrom),
            ("Lybeta", 1025.72 * u.Angstrom),
            ("Halpha", 6564.61 * u.Angstrom),
            ("Hbeta", 4862.68 * u.Angstrom),
        ],
    )
    def test_line_dict(self, line, lambda_line):
        """test correct loading of some of the emission line"""
        blr = SphericalShellBLR(1e46 * u.Unit("erg s-1"), 0.1, line, 1e17)
        assert u.isclose(blr.lambda_line, lambda_line, atol=0 * u.Angstrom)

    def test_u(self):
        """test u in the stationary reference frame"""
        r = np.logspace(16, 20, 10) * u.cm
        assert u.allclose(
            [
                4.02698710e-01,
                4.12267268e-01,
                5.49297935e-01,
                9.36951182e-02,
                1.12734943e-02,
                1.44410780e-03,
                1.86318218e-04,
                2.40606696e-05,
                3.10750072e-06,
                4.01348246e-07,
            ]
            * u.Unit("erg / cm3"),
            BLR.u(r),
            atol=0 * u.Unit("erg / cm3"),
        )

    def test_u_blr_vs_point_source(self):
        """test that for large enough distances the energy density of the 
        BLR tends to the one of a point like source approximating it"""
        # point source with the same luminosity as the BLR
        ps_blr = PointSourceBehindJet(BLR.xi_line * BLR.L_disk, BLR.epsilon_line)
        # r >> R_line
        r = np.logspace(19, 23, 10) * u.cm
        assert u.allclose(BLR.u(r), ps_blr.u(r), atol=0 * u.Unit("erg cm-3"), rtol=1e-2)

    def test_u_comoving(self):
        """test u in the reference frame comoving with the blob"""
        r = np.logspace(16, 20, 10) * u.cm
        assert u.allclose(
            [
                1.35145224e01,
                1.39362806e01,
                1.98574615e01,
                7.22733332e-02,
                2.50276530e-04,
                5.60160671e-06,
                4.97415343e-07,
                6.09348663e-08,
                7.81583912e-09,
                1.00855244e-09,
            ]
            * u.Unit("erg / cm3"),
            BLR.u(r, PWL_BLOB),
            atol=0 * u.Unit("erg / cm3"),
        )

    def test_u_blr_vs_point_source_comoving(self):
        """test that for large enough distances the energy density of the 
        BLR tends to the one of a point like source approximating it"""
        # point source with the same luminosity as the BLR
        ps_blr = PointSourceBehindJet(BLR.xi_line * BLR.L_disk, BLR.epsilon_line)
        # r >> R_line
        r = np.logspace(19, 23, 10) * u.cm
        assert u.allclose(
            BLR.u(r, PWL_BLOB),
            ps_blr.u(r, PWL_BLOB),
            atol=0 * u.Unit("erg cm-3"),
            rtol=1e-1,
        )


class TestRingDustTorus:
    """class grouping all the tests related to the RingDustTorus target"""

    def test_sublimation_radius(self):
        assert u.allclose(DT.R_dt, 1.361 * 1e19 * u.cm, atol=0 * u.cm, rtol=1e-3)

    def test_setting_radius(self):
        """check that, when passed manually, the radius is correctly set"""
        dt = RingDustTorus(L_DISK, 0.1, 1e3 * u.K, 1e19 * u.cm)
        assert u.allclose(dt.R_dt, 1e19 * u.cm, atol=0 * u.cm)

    def test_u(self):
        """test u in the stationary reference frame"""
        r = np.logspace(17, 23, 10) * u.cm
        assert u.allclose(
            [
                2.16675545e-05,
                2.16435491e-05,
                2.11389842e-05,
                1.40715277e-05,
                1.71541601e-06,
                8.61241559e-08,
                4.01273788e-09,
                1.86287690e-10,
                8.64677950e-12,
                4.01348104e-13,
            ]
            * u.Unit("erg / cm3"),
            DT.u(r),
            atol=0 * u.Unit("erg / cm3"),
        )

    def test_u_dt_vs_point_source(self):
        """test that in the stationary reference frame, for large enough 
        distances, the energy density of the DT tends to the one of a point like 
        source approximating it"""
        # point source with the same luminosity as the DT
        ps_dt = PointSourceBehindJet(DT.xi_dt * DT.L_disk, DT.epsilon_dt)
        # r >> R_dt
        r = np.logspace(21, 24, 10) * u.cm
        assert u.allclose(DT.u(r), ps_dt.u(r), atol=0 * u.Unit("erg cm-3"), rtol=1e-2)

    def test_u_comoving(self):
        """test u in the reference frame comoving with the blob"""
        r = np.logspace(17, 23, 10) * u.cm
        assert u.allclose(
            [
                2.13519004e-03,
                2.02003750e-03,
                1.50733234e-03,
                2.37521472e-04,
                3.50603715e-07,
                4.21027697e-10,
                1.04563603e-11,
                4.68861573e-13,
                2.17274347e-14,
                1.00842240e-15,
            ]
            * u.Unit("erg / cm3"),
            DT.u(r, PWL_BLOB),
            atol=0 * u.Unit("erg / cm3"),
        )

    def test_u_dt_vs_point_source_comoving(self):
        """test that in the reference frame comoving with the Blob, for large 
        enough distances, the energy density of the DT tends to the one of 
        a point like source approximating it"""
        # point source with the same luminosity as the DT
        ps_dt = PointSourceBehindJet(DT.xi_dt * DT.L_disk, DT.epsilon_dt)
        # r >> R_line
        r = np.logspace(21, 24, 10) * u.cm
        assert u.allclose(
            DT.u(r, PWL_BLOB),
            ps_dt.u(r, PWL_BLOB),
            atol=0 * u.Unit("erg cm-3"),
            rtol=1e-1,
        )
