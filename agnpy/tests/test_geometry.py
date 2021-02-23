# tests on the utils module
import pytest
import numpy as np
import agnpy.utils.geometry as geom
import astropy.units as u


class TestUtilsGeometry:
    """test utils.geometry"""

    @pytest.mark.parametrize("R_re", [1e16 * u.cm, 1e17 * u.cm])
    @pytest.mark.parametrize("r", [3.0e16 * u.cm, 2.0e17 * u.cm])
    @pytest.mark.parametrize("uu", [5.0e16 * u.cm, 4.0e17 * u.cm])
    @pytest.mark.parametrize("phi_re", np.linspace(0, 2 * np.pi, 5))
    def test_x_re_ring_vs_x_re_ring_mu_s(self, R_re, r, uu, phi_re):
        """test consistency of both methods if mu_s ~ 1"""
        x1 = geom.x_re_ring(R_re, r + uu)
        x2 = geom.x_re_ring_mu_s(R_re, r, phi_re, uu, 0.9999)
        assert np.isclose(x1, x2, atol=0, rtol=0.01)

    @pytest.mark.parametrize("R_re", [1e16 * u.cm, 1e17 * u.cm])
    @pytest.mark.parametrize("r", [3.0e16 * u.cm, 2.0e17 * u.cm])
    @pytest.mark.parametrize("uu", [5.0e16 * u.cm, 4.0e17 * u.cm])
    @pytest.mark.parametrize("phi_re", np.linspace(0, 2 * np.pi, 5))
    def test_x_re_ring_mu_s_perp(self, R_re, r, uu, phi_re):
        """test if the calculations are fine for mu_s ~=0"""
        mu_s = 0.01
        x1 = geom.x_re_ring_mu_s(R_re, r, phi_re, uu, mu_s)
        x2 = np.sqrt(
            r ** 2 + (uu - R_re * np.cos(phi_re)) ** 2 + (R_re * np.sin(phi_re)) ** 2
        )
        assert np.isclose(x1, x2, atol=0, rtol=0.01)

    @pytest.mark.parametrize("R_re", [1e17 * u.cm])
    @pytest.mark.parametrize("r", [3.0e16 * u.cm, 2.0e17 * u.cm])
    @pytest.mark.parametrize("uu", [5.0e16 * u.cm, 4.0e17 * u.cm])
    @pytest.mark.parametrize("phi_re", np.linspace(0 + 0.1, 2 * np.pi - 0.1, 5))
    def test_phi_mu_re_ring_1(self, R_re, r, uu, phi_re):
        """test for mu_s = 1"""
        mu_s = 0.99999
        twopi = 2 * np.pi
        x_re = geom.x_re_ring_mu_s(R_re, r, phi_re, uu, mu_s)
        phi, mu = geom.phi_mu_re_ring(R_re, r, phi_re, uu, mu_s)
        mu2 = (r + uu) / x_re
        assert np.isclose(mu, mu2, atol=0.01, rtol=0)

        # there is a pi of difference between phi and phi_re
        dphi = np.mod(phi.value - phi_re + twopi, twopi)
        # this one fails unless mu_s is very close to one, or uu << R_re
        assert np.isclose(dphi, np.pi, atol=0.1, rtol=0)

    @pytest.mark.parametrize("R_re", [1e16 * u.cm, 1e17 * u.cm])
    @pytest.mark.parametrize("r", [3.0e16 * u.cm, 2.0e17 * u.cm])
    @pytest.mark.parametrize("uu", [5.0e16 * u.cm, 4.0e17 * u.cm])
    @pytest.mark.parametrize("phi_re", np.linspace(0, 2 * np.pi, 5))
    def test_phi_mu_re_ring_0(self, R_re, r, uu, phi_re):
        """test for mu_s =~ 0"""
        mu_s = 0.001
        twopi = 2 * np.pi
        x_re = geom.x_re_ring_mu_s(R_re, r, phi_re, uu, mu_s)
        phi, mu = geom.phi_mu_re_ring(R_re, r, phi_re, uu, mu_s)
        mu2 = r / x_re
        assert np.isclose(mu, mu2, atol=0.01, rtol=0)

        phi2 = np.arctan2(-R_re * np.sin(phi_re), uu - R_re * np.cos(phi_re)).value

        # here we add on purpose pi to avoid slightly negative differences
        dphi = np.mod(phi.value - phi2 + 3 * np.pi, twopi)
        assert np.isclose(dphi, np.pi, atol=0.1, rtol=0)

    @pytest.mark.parametrize("R_re", [1e16 * u.cm, 1e17 * u.cm])
    @pytest.mark.parametrize("r", [3.0e16 * u.cm, 2.0e17 * u.cm])
    @pytest.mark.parametrize("uu", [1.0e19 * u.cm])
    @pytest.mark.parametrize("phi_re", np.linspace(0, 2 * np.pi, 5))
    @pytest.mark.parametrize("mu_s", np.linspace(0, 1, 5))
    def test_phi_mu_infty(self, R_re, r, uu, phi_re, mu_s):
        """test that for uu >> R_re for mu_s =~ 0"""
        phi, mu = geom.phi_mu_re_ring(R_re, r, phi_re, uu, mu_s)
        cospsi = geom.cos_psi(mu_s, mu, phi)
        assert np.isclose(cospsi, 1, atol=0.01, rtol=0)
