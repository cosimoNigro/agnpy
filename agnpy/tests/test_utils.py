# tests on the utils module
import pytest
import numpy as np
import astropy.units as u
from agnpy.spectra import PowerLaw
import agnpy.utils.math as math
import agnpy.utils.geometry as geom
from agnpy.utils.plot import load_mpl_rc, plot_eed, plot_sed


twopi = 2 * np.pi


def line_loglog(x, m, n):
    """a straight line in loglog-space"""
    return x ** m * np.exp(n)


def integral_line_loglog(x_min, x_max, m, n):
    """analytical integral of the line in log-log space"""
    if np.isclose(m, -1, atol=0, rtol=1e-5):
        return np.exp(n) * np.log(x_max / x_min)
    f_low = line_loglog(x_min, m + 1, n) / (m + 1)
    f_up = line_loglog(x_max, m + 1, n) / (m + 1)
    return f_up - f_low


class TestMathUtils:
    """test uitls.math"""

    @pytest.mark.parametrize("m", np.arange(-2, 2.5, 0.5))
    @pytest.mark.parametrize("n", np.arange(-2, 2.5, 0.5))
    def test_trapz_log_log(self, m, n):
        """test trapz loglog integral method"""
        x = np.logspace(2, 5)
        # generate syntethic power-law like data by defining a straight line
        y = line_loglog(x, m, n)
        trapz_loglog_integral = math.trapz_loglog(y, x, axis=0)
        analytical_integral = integral_line_loglog(x[0], x[-1], m, n)
        assert np.isclose(trapz_loglog_integral, analytical_integral, atol=0, rtol=0.01)


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

    @pytest.mark.parametrize("R_re", [1e16 * u.cm, 1e17 * u.cm])
    @pytest.mark.parametrize("r", [3.0e16 * u.cm, 2.0e17 * u.cm])
    @pytest.mark.parametrize("uu", [5.0e16 * u.cm, 4.0e17 * u.cm])
    @pytest.mark.parametrize("phi_re", np.linspace(0, 2 * np.pi, 5))
    @pytest.mark.parametrize("mu_s", np.linspace(0, 1, 5))
    def test_x_re_shell_mu_s_vs_ring(self, R_re, r, phi_re, uu, mu_s):
        """Test that for mu_re=0. x_re_shell_mu_s gives the same results as
        x_re_ring_mu_s"""
        x_ring = geom.x_re_ring_mu_s(R_re, r, phi_re, uu, mu_s)
        mu_re = 0.01
        x_shell = geom.x_re_shell_mu_s(R_re, r, phi_re, mu_re, uu, mu_s)
        assert np.isclose(x_shell, x_ring, atol=0, rtol=0.01)

    @pytest.mark.parametrize("R_re", [1e17 * u.cm])
    @pytest.mark.parametrize("r", [3.0e16 * u.cm, 2.0e17 * u.cm])
    @pytest.mark.parametrize("uu", [1.0e16 * u.cm, 4.0e17 * u.cm])
    @pytest.mark.parametrize("phi_re", np.linspace(0, 2 * np.pi, 5))
    @pytest.mark.parametrize("mu_re", np.linspace(-1, 1, 4))
    def test_x_re_shell_mu_s_vs_on_axis(self, R_re, r, phi_re, mu_re, uu):
        """Test that for mu_s=1. x_re_shell_mu_s gives the same results as
        x_re_shell"""
        mu_s = 0.9999
        x_on_axis = geom.x_re_shell(mu_re, R_re, r + uu)
        x_shell = geom.x_re_shell_mu_s(R_re, r, phi_re, mu_re, uu, mu_s)
        assert np.isclose(x_shell, x_on_axis, atol=0, rtol=0.01)

    @pytest.mark.parametrize("phi_re", np.linspace(0, 2 * np.pi, 5))
    def test_x_re_shell_mu_s(self, phi_re):
        """Test that for a simple case that  x_re_shell_mu_s gives correct results"""
        R_re = 1e17 * u.cm
        r = 3e17 * u.cm
        uu = 5e17 * u.cm
        mu_re = 0.999
        mu_s = 0.001
        x_true = np.sqrt((r - R_re) ** 2 + uu ** 2)
        x_shell = geom.x_re_shell_mu_s(R_re, r, phi_re, mu_re, uu, mu_s)
        assert np.isclose(x_shell, x_true, atol=0, rtol=0.01)

    @pytest.mark.parametrize("R_re", [1e16 * u.cm, 1e17 * u.cm])
    @pytest.mark.parametrize("r", [3.0e16 * u.cm, 2.0e17 * u.cm])
    @pytest.mark.parametrize("uu", [5.0e16 * u.cm, 4.0e17 * u.cm])
    @pytest.mark.parametrize("phi_re", np.linspace(0, 2 * np.pi, 5))
    @pytest.mark.parametrize("mu_s", np.linspace(0, 1, 5))
    def test_phi_mu_re_shell_vs_ring(self, R_re, r, phi_re, uu, mu_s):
        """Test that for mu_re=0 phi_mu_re_shell gives the same results as
        phi_mu_re_ring"""
        phi_ring, mu_ring = geom.phi_mu_re_ring(R_re, r, phi_re, uu, mu_s)
        mu_re = 0.001
        phi_shell, mu_shell = geom.phi_mu_re_shell(R_re, r, phi_re, mu_re, uu, mu_s)
        assert np.isclose(mu_shell, mu_ring, atol=0.01, rtol=0)

        # here we add on purpose pi to avoid slightly negative differences
        dphi = np.mod(phi_shell.value - phi_ring.value + 3 * np.pi, twopi)
        assert np.isclose(dphi, np.pi, atol=0.01, rtol=0)

    @pytest.mark.parametrize("R_re", [1e17 * u.cm])
    @pytest.mark.parametrize("r", [3.0e16 * u.cm, 2.0e17 * u.cm])
    @pytest.mark.parametrize("uu", [5.0e16 * u.cm, 4.0e17 * u.cm])
    @pytest.mark.parametrize("phi_re", np.linspace(0, 2 * np.pi, 5))
    @pytest.mark.parametrize("mu_s", np.linspace(0, 0.9, 5))
    def test_phi_mu_re_shell(self, R_re, r, phi_re, uu, mu_s):
        """Test that for mu_re=1 phi_mu_re_shell gives correct result"""
        mu_re = 0.99999
        phi_shell, mu_shell = geom.phi_mu_re_shell(R_re, r, phi_re, mu_re, uu, mu_s)
        phi_true = 0  # unless mu_s = 1 in which case phi_true can be whatever
        dx = uu * np.sqrt(1 - mu_s ** 2)
        dz = r - R_re + uu * mu_s
        mu_true = dz / np.sqrt(dx ** 2 + dz ** 2)
        assert np.isclose(mu_shell, mu_true, atol=0.01, rtol=0)
        print(phi_true, phi_shell, R_re, r, phi_re, uu, mu_s)

        # here we add on purpose pi to avoid slightly negative differences
        dphi = np.mod(phi_shell.value - phi_true + 3 * np.pi, twopi)
        assert np.isclose(dphi, np.pi, atol=0.03, rtol=0)


class TestPlotUtils:
    """test utils.plot"""

    def test_load_mpl_rc(self):
        """check that the matplotlibrc is properly loaded"""
        import matplotlib as mpl

        load_mpl_rc()

        assert mpl.rcParams["font.size"] == 12
        assert mpl.rcParams["lines.linewidth"] == 1.6
        assert mpl.rcParams["xtick.major.size"] == 7
        assert mpl.rcParams["xtick.minor.size"] == 4

    def test_plot_eed(self):
        """check that the functions for plotting EED can be called and that the
        **kwargs are correctly passed to matploltib"""
        kwargs = {"linewidth": 3, "color": "crimson"}
        gamma = np.logspace(2, 5)
        n_e = PowerLaw()

        # plot it without scaling by a power of gamma
        ax = plot_eed(gamma, n_e, **kwargs)
        line_2d = ax.get_lines()[0]

        assert line_2d.get_linewidth() == kwargs["linewidth"]
        assert line_2d.get_color() == kwargs["color"]

        # plot it by scaling by a power of gamma
        ax = plot_eed(gamma, n_e, gamma_power=2)
        assert ax.get_ylabel() == r"$\gamma^{2}$$\,n_e(\gamma)\,/\,{\rm cm}^{-3}$"

    def test_plot_sed(self):
        """check that the functions for plotting SED can be called and that the
        **kwargs are correctly passed to matploltib"""
        kwargs = {"linewidth": 3, "color": "crimson"}
        nu = np.logspace(10, 20) * u.Hz
        sed = np.logspace(-10, -20) * u.Unit("erg cm-2 s-1")

        ax = plot_sed(nu, sed, **kwargs)
        line_2d = ax.get_lines()[0]

        assert line_2d.get_linewidth() == kwargs["linewidth"]
        assert line_2d.get_color() == kwargs["color"]
