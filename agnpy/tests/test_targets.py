# tests on targets module
import numpy as np
import astropy.units as u
import pytest
from astropy.coordinates import Distance
from astropy.constants import M_sun, G, sigma_sb
from agnpy.emission_regions import Blob
from agnpy.targets import (
    CMB,
    PointSourceBehindJet,
    SSDisk,
    SphericalShellBLR,
    RingDustTorus,
)


# variables with _test are global and meant to be used in all tests
blob_test = Blob()

# CMB at z = 1
cmb_test = CMB(z=1)

# PointSourceBehindJet
ps_test = PointSourceBehindJet(1e46 * u.Unit("erg s-1"), 1e-5)

# disk
M_BH_test = 1.2 * 1e9 * M_sun
L_disk_test = 1.512 * 1e46 * u.Unit("erg s-1")
eta_test = 1 / 12
R_g_test = 1.77 * 1e14 * u.cm
R_in_tilde_test = 6
R_out_tilde_test = 200
R_in_test = R_in_tilde_test * R_g_test
R_out_test = R_out_tilde_test * R_g_test
disk_test = SSDisk(M_BH_test, L_disk_test, eta_test, R_in_test, R_out_test)
# useful for checks
L_Edd_test = 15.12 * 1e46 * u.Unit("erg s-1")
m_dot_test = 2.019 * 1e26 * u.Unit("g s-1")

# SphericalShellBLR
blr_test = SphericalShellBLR(L_disk_test, 0.1, "Lyalpha", 1e17 * u.cm)

# dust torus definition
dt_test = RingDustTorus(L_disk_test, 0.1, 1000 * u.K)


class TestCMB:
    """class grouping all the tests related to the CMB"""

    def test_u(self):
        """test u in the stationary reference frame"""
        assert u.isclose(
            6.67945605e-12 * u.Unit("erg / cm3"),
            cmb_test.u(),
            atol=0 * u.Unit("erg / cm3"),
            rtol=1e-5,
        )

    def test_u_comoving(self):
        """test u in the reference frame comoving with the blob"""
        assert u.isclose(
            8.88373221e-10 * u.Unit("erg / cm3"),
            cmb_test.u(blob_test),
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
            ps_test.u(r),
            atol=0 * u.Unit("erg / cm3"),
            rtol=1e-3,
        )

    def test_u_comoving(self):
        """test u in the reference frame comoving with the blob"""
        r = np.asarray([1e18, 1e19, 1e20]) * u.cm
        assert u.allclose(
            [6.6693519e-05, 6.6693519e-07, 6.6693519e-09] * u.Unit("erg / cm3"),
            ps_test.u(r, blob_test),
            atol=0 * u.Unit("erg / cm3"),
            rtol=1e-3,
        )


class TestSSDisk:
    """class grouping all the tests related to the SSDisk target"""

    # global quantities defining the test disk
    def test_L_Edd(self):
        """test on disk properties"""
        assert u.isclose(
            disk_test.L_Edd, L_Edd_test, atol=0 * u.Unit("erg s-1"), rtol=1e-3
        )

    def test_l_Edd(self):
        """test on disk properties"""
        assert u.isclose(disk_test.l_Edd, 0.1, atol=0, rtol=1e-3)

    def test_m_dot(self):
        """test on disk properties"""
        assert u.isclose(
            disk_test.m_dot, m_dot_test, atol=0 * u.Unit("g s-1"), rtol=1e-3
        )

    @pytest.mark.parametrize(
        "R_in, R_out, R_g_units",
        [(R_in_tilde_test, R_out_tilde_test, False), (R_in_test, R_out_test, True),],
    )
    def test_R_in_R_out_units(self, R_in, R_out, R_g_units):
        """check if a TypeError is raised when passing R_in and R_out with
        (without) units but specifiying R_g_units True (False)"""
        with pytest.raises(TypeError):
            disk = SSDisk(M_BH_test, L_disk_test, eta_test, R_in, R_out, R_g_units)

    def test_R_g(self):
        """test on disk properties"""
        assert u.isclose(disk_test.R_g, R_g_test, atol=0 * u.cm, rtol=1e-2)

    def test_mu_from_r_tilde(self):
        """test on disk properties"""
        mu = SSDisk.evaluate_mu_from_r_tilde(
            R_in_tilde_test, R_out_tilde_test, r_tilde=10
        )
        mu_min_expected = 0.050
        mu_max_expected = 0.858
        assert np.isclose(mu[0], mu_min_expected, atol=0, rtol=1e-2)
        assert np.isclose(mu[-1], mu_max_expected, atol=0, rtol=1e-2)

    def test_phi_disk(self):
        """test on disk properties"""
        R_tilde = 10
        phi_expected = 0.225
        assert np.isclose(disk_test.phi_disk(R_tilde), phi_expected, atol=0, rtol=1e-2)

    def test_phi_disk_mu(self):
        """test on disk properties"""
        r_tilde = 10
        mu = 1 / np.sqrt(2)
        phi_disk_expected = 0.225
        phi_disk = SSDisk.evaluate_phi_disk_mu(mu, R_in_tilde_test, r_tilde)
        assert np.allclose(phi_disk, phi_disk_expected, atol=0, rtol=1e-2)

    def test_epsilon(self):
        """test on disk properties"""
        R_tilde = 10
        epsilon_expected = 2.7e-5
        assert np.allclose(
            disk_test.epsilon(R_tilde), epsilon_expected, atol=0, rtol=1e-2
        )

    def test_epsilon_mu(self):
        """test on disk properties"""
        r_tilde = 10
        # assume R_tilde = 10 as before
        mu = 1 / np.sqrt(2)
        epsilon_disk_expected = 2.7e-5
        epsilon_disk = SSDisk.evaluate_epsilon_mu(
            L_disk_test, M_BH_test, eta_test, mu, r_tilde
        )
        assert np.allclose(epsilon_disk, epsilon_disk_expected, atol=0, rtol=1e-2)

    def test_T(self):
        """test on disk properties"""
        R = 10 * R_g_test
        phi = 1 - np.sqrt((disk_test.R_in / R).to(""))
        T_expected = np.power(
            3
            * G
            * M_BH_test
            * m_dot_test
            * phi
            / (8 * np.pi * np.power(R, 3) * sigma_sb),
            1 / 4,
        ).to("K")
        assert u.isclose(disk_test.T(R), T_expected, atol=0 * u.K, rtol=1e-2)

    @pytest.mark.parametrize("R_out", [1e2, 1e3, 1e4])
    @pytest.mark.parametrize("mu_s", [0.5, 0.8, 1.0])
    def test_bb_sed_luminosity(self, R_out, mu_s):
        """test that the luminosity of the disk BB SED is the same as L_disk,
        create disks with different outer radii"""
        disk = SSDisk(M_BH_test, L_disk_test, 1 / 12, 6, R_out, R_g_units=True)
        # compute the SEDs, assume a random redshift
        z = 0.23
        nu = np.logspace(10, 20, 100) * u.Hz
        sed = disk.sed_flux(nu, z, mu_s)
        # compute back the luminosity
        d_L = Distance(z=z).to("cm")
        F_nu = sed / nu
        # renormalise, the factor 2 includes the two sides of the Disk
        L = 2 * (4 * np.pi * np.power(d_L, 2) * np.trapz(F_nu, nu, axis=0))
        assert u.isclose(L, L_disk_test, atol=0 * u.Unit("erg s-1"), rtol=0.1)


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
            blr_test.u(r),
            atol=0 * u.Unit("erg / cm3"),
        )

    def test_u_blr_vs_point_source(self):
        """test that for large enough distances the energy density of the
        BLR tends to the one of a point like source approximating it"""
        # point source with the same luminosity as the BLR
        ps_blr = PointSourceBehindJet(
            blr_test.xi_line * blr_test.L_disk, blr_test.epsilon_line
        )
        # r >> R_line
        r = np.logspace(19, 23, 10) * u.cm
        assert u.allclose(
            blr_test.u(r), ps_blr.u(r), atol=0 * u.Unit("erg cm-3"), rtol=1e-2
        )

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
            blr_test.u(r, blob_test),
            atol=0 * u.Unit("erg / cm3"),
        )

    def test_u_blr_vs_point_source_comoving(self):
        """test that for large enough distances the energy density of the
        BLR tends to the one of a point like source approximating it"""
        # point source with the same luminosity as the BLR
        ps_blr = PointSourceBehindJet(
            blr_test.xi_line * blr_test.L_disk, blr_test.epsilon_line
        )
        # r >> R_line
        r = np.logspace(19, 23, 10) * u.cm
        assert u.allclose(
            blr_test.u(r, blob_test),
            ps_blr.u(r, blob_test),
            atol=0 * u.Unit("erg cm-3"),
            rtol=1e-1,
        )


class TestRingDustTorus:
    """class grouping all the tests related to the RingDustTorus target"""

    def test_sublimation_radius(self):
        """test on RingDustTorus properties"""
        assert u.allclose(dt_test.R_dt, 1.361 * 1e19 * u.cm, atol=0 * u.cm, rtol=1e-3)

    def test_setting_radius(self):
        """check that, when passed manually, the radius is correctly set"""
        dt = RingDustTorus(L_disk_test, 0.1, 1e3 * u.K, 1e19 * u.cm)
        assert u.allclose(dt.R_dt, 1e19 * u.cm, atol=0 * u.cm)

    @pytest.mark.parametrize("T_dt", [1e3, 2e3, 5e3] * u.K)
    def test_bb_sed_luminosity(self, T_dt):
        """test that the luminosity of the DT BB SED is the same as xi_dt * L_disk,
        create DTs with different temperatrues (and radii)"""
        xi_dt = 0.5
        L_dt = xi_dt * L_disk_test
        dt = RingDustTorus(L_disk_test, xi_dt, T_dt)
        # compute the SEDs, assume a random redshift
        z = 0.23
        nu = np.logspace(10, 20, 100) * u.Hz
        sed = dt.sed_flux(nu, z)
        # compute back the luminosity
        d_L = Distance(z=z).to("cm")
        F_nu = sed / nu
        L = 4 * np.pi * np.power(d_L, 2) * np.trapz(F_nu, nu, axis=0)
        # this should be smaller than L_dt, which should be the luminosity
        # computed integrating the black body over the entire energy range
        assert L < L_dt

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
            dt_test.u(r),
            atol=0 * u.Unit("erg / cm3"),
        )

    def test_u_dt_vs_point_source(self):
        """test that in the stationary reference frame, for large enough
        distances, the energy density of the DT tends to the one of a point like
        source approximating it"""
        # point source with the same luminosity as the DT
        ps_dt = PointSourceBehindJet(dt_test.xi_dt * dt_test.L_disk, dt_test.epsilon_dt)
        # r >> R_dt
        r = np.logspace(21, 24, 10) * u.cm
        assert u.allclose(
            dt_test.u(r), ps_dt.u(r), atol=0 * u.Unit("erg cm-3"), rtol=1e-2
        )

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
            dt_test.u(r, blob_test),
            atol=0 * u.Unit("erg / cm3"),
        )

    def test_u_dt_vs_point_source_comoving(self):
        """test that in the reference frame comoving with the Blob, for large
        enough distances, the energy density of the DT tends to the one of
        a point like source approximating it"""
        # point source with the same luminosity as the DT
        ps_dt = PointSourceBehindJet(dt_test.xi_dt * dt_test.L_disk, dt_test.epsilon_dt)
        # r >> R_line
        r = np.logspace(21, 24, 10) * u.cm
        assert u.allclose(
            dt_test.u(r, blob_test),
            ps_dt.u(r, blob_test),
            atol=0 * u.Unit("erg cm-3"),
            rtol=1e-1,
        )
