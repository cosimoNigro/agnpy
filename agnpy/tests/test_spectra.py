# tests on agnpy.spectra module
import numpy as np
import astropy.units as u
from astropy.constants import m_e, m_p
import pytest
from agnpy.spectra import PowerLaw, BrokenPowerLaw, LogParabola, ExpCutoffPowerLaw
from agnpy.utils.math import trapz_loglog
from agnpy.utils.conversion import mec2, mpc2


def power_law_integral(k_e, p, gamma_min, gamma_max):
    """Analytical integral of the power law."""
    if np.isclose(p, 1.0):
        integral = np.log(gamma_max / gamma_min)
    else:
        integral = (np.power(gamma_max, 1 - p) - np.power(gamma_min, 1 - p)) / (1 - p)
    return k_e * integral


def power_law_times_gamma_integral(k_e, p, gamma_min, gamma_max):
    """Analytical integral of the power law multiplied by gamma."""
    if np.isclose(p, 2.0):
        integral = np.log(gamma_max / gamma_min)
    else:
        integral = (np.power(gamma_max, 2 - p) - np.power(gamma_min, 2 - p)) / (2 - p)
    return k_e * integral


def broken_power_law_integral(k_e, p1, p2, gamma_b, gamma_min, gamma_max):
    """Analytical integral of the broken power law."""
    if np.allclose(p1, 1.0):
        term_1 = gamma_b * np.log(gamma_b / gamma_min)
    else:
        term_1 = gamma_b * (1 - np.power(gamma_min / gamma_b, 1 - p1)) / (1 - p1)
    if np.allclose(p2, 1.0):
        term_2 = gamma_b * np.log(gamma_max / gamma_b)
    else:
        term_2 = gamma_b * (np.power(gamma_max / gamma_b, 1 - p2) - 1) / (1 - p2)
    return k_e * (term_1 + term_2)


def broken_power_law_times_gamma_integral(k_e, p1, p2, gamma_b, gamma_min, gamma_max):
    """Analytical integral of the broken power law multiplied by gamma."""
    if np.allclose(p1, 2.0):
        term_1 = np.power(gamma_b, 2) * np.log(gamma_b / gamma_min)
    else:
        term_1 = (
            np.power(gamma_b, 2)
            * (1 - np.power(gamma_min / gamma_b, 2 - p1))
            / (2 - p1)
        )
    if np.allclose(p2, 2.0):
        term_2 = np.power(gamma_b, 2) * np.log(gamma_max / gamma_b)
    else:
        term_2 = (
            np.power(gamma_b, 2)
            * (np.power(gamma_max / gamma_b, 2 - p2) - 1)
            / (2 - p2)
        )
    return k_e * (term_1 + term_2)


class TestPowerLaw:
    """Class grouping all tests related to the PowerLaw spectrum."""

    def test_call(self):
        """assert that outside the bounding box the function returns 0"""
        gamma = np.logspace(0, 8)
        pwl = PowerLaw()
        values = pwl(gamma).value
        condition = (pwl.gamma_min <= gamma) * (gamma <= pwl.gamma_max)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    @pytest.mark.parametrize("p", np.arange(1, 5, 0.5))
    @pytest.mark.parametrize("gamma_min", np.logspace(0, 3, 4))
    @pytest.mark.parametrize("gamma_max", np.logspace(4, 7, 4))
    @pytest.mark.parametrize("gamma_power", [0, 1])
    @pytest.mark.parametrize("integrator", [np.trapz, trapz_loglog])
    def test_power_law_integral(self, p, gamma_min, gamma_max, gamma_power, integrator):
        """test the integration of the power law for different spectral indexes
        different integrating functions"""

        k = 1e-5 * u.Unit("cm-3")
        pwl = PowerLaw(k, p, gamma_min, gamma_max, integrator=integrator)
        numerical_integral = pwl.integrate(
            gamma_min, gamma_max, gamma_power=gamma_power
        )

        if gamma_power == 0:
            analytical_integral = power_law_integral(k, p, gamma_min, gamma_max)
        if gamma_power == 1:
            analytical_integral = power_law_times_gamma_integral(
                k, p, gamma_min, gamma_max
            )
        assert u.isclose(
            numerical_integral, analytical_integral, atol=0 * u.Unit("cm-3"), rtol=0.05
        )

    @pytest.mark.parametrize("p", np.arange(1, 5, 0.5))
    @pytest.mark.parametrize("gamma_min", np.logspace(0, 3, 4))
    @pytest.mark.parametrize("gamma_max", np.logspace(4, 7, 4))
    def test_from_normalised_density(self, p, gamma_min, gamma_max):
        """Test the intialisation of the power law from the total particle
        density. The same results should be obtained for proton and electrons."""
        n_tot = 1e-5 * u.Unit("cm-3")

        pwl_e = PowerLaw.from_normalised_density(
            n_tot=n_tot, mass=m_e, p=p, gamma_min=gamma_min, gamma_max=gamma_max,
        )
        assert pwl_e.particle == "electrons"

        pwl_p = PowerLaw.from_normalised_density(
            n_tot=n_tot, mass=m_p, p=p, gamma_min=gamma_min, gamma_max=gamma_max,
        )
        assert pwl_p.particle == "protons"

        n_e_tot = pwl_e.integrate(gamma_min, gamma_max)
        n_p_tot = pwl_p.integrate(gamma_min, gamma_max)
        # check that the numerical integral is close to the value we set initially
        assert u.isclose(n_e_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=1e-2)
        # total particle density should be the same for protons and electrons
        assert u.isclose(n_e_tot, n_p_tot, atol=0 * u.Unit("cm-3"), rtol=1e-2)

    @pytest.mark.parametrize("p", np.arange(1, 5, 0.5))
    @pytest.mark.parametrize("gamma_min", np.logspace(0, 3, 4))
    @pytest.mark.parametrize("gamma_max", np.logspace(4, 7, 4))
    def test_from_normalised_energy_density(self, p, gamma_min, gamma_max):
        """Test the intialisation of the power law from the total particle
        energy density. The same results should be obtained for proton and
        electrons."""
        u_tot = 3e-4 * u.Unit("erg cm-3")

        pwl_e = PowerLaw.from_normalised_energy_density(
            u_tot=u_tot, mass=m_e, p=p, gamma_min=gamma_min, gamma_max=gamma_max
        )
        assert pwl_e.particle == "electrons"

        pwl_p = PowerLaw.from_normalised_energy_density(
            u_tot=u_tot, mass=m_p, p=p, gamma_min=gamma_min, gamma_max=gamma_max,
        )
        assert pwl_p.particle == "protons"

        u_e_tot = mec2 * pwl_e.integrate(gamma_min, gamma_max, gamma_power=1)
        u_p_tot = mpc2 * pwl_p.integrate(gamma_min, gamma_max, gamma_power=1)
        # check that the numerical integral is close to the value we set initially
        assert u.isclose(u_e_tot, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)
        assert u.isclose(u_p_tot, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)
        # total particle energy density should be the same
        assert u.isclose(u_e_tot, u_p_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)

    @pytest.mark.parametrize("p", np.arange(1, 5, 0.5))
    @pytest.mark.parametrize("gamma_max", np.logspace(4, 7, 4))
    def test_from_norm_at_gamma_1(self, p, gamma_max):
        """Test the intialisation of the power law from the value at gamma=1.
        The same results should be obtained for proton and electrons."""
        norm = 1e-13 * u.Unit("cm-3")

        pwl_e = PowerLaw.from_norm_at_gamma_1(
            norm=norm, mass=m_e, p=p, gamma_min=1, gamma_max=gamma_max
        )
        assert pwl_e.particle == "electrons"

        pwl_p = PowerLaw.from_norm_at_gamma_1(
            norm=norm, mass=m_p, p=p, gamma_min=1, gamma_max=gamma_max
        )
        assert pwl_p.particle == "protons"

        # check that the value at gamma=1 is close to the value we set initially
        assert u.isclose(norm, pwl_e(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)
        assert u.isclose(norm, pwl_p(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)
        # particle density at gamma=1 should be the same for protons and electrons
        assert u.isclose(pwl_e(1), pwl_p(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)


class TestBrokenPowerLaw:
    """class grouping all tests related to the BrokenPowerLaw spectrum"""

    def test_call(self):
        """assert that outside the bounding box the function returns 0"""
        gamma = np.logspace(0, 8)
        values = bpwl_test(gamma).value
        condition = (gamma_min_test <= gamma) * (gamma <= gamma_max_test)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    @pytest.mark.parametrize("p1", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    @pytest.mark.parametrize("p2", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    @pytest.mark.parametrize("gamma_b", [1e2, 1e3, 1e4, 1e5, 1e6])
    @pytest.mark.parametrize("integrator", [np.trapz, trapz_loglog])
    def test_broken_power_law_integral(self, p1, p2, gamma_b, integrator):
        """test the integration of the log parabola for different spectral
        indexes and breaks and different integrating functions"""
        bpwl = BrokenPowerLaw(
            k_e_test,
            p1,
            p2,
            gamma_b,
            gamma_min_test,
            gamma_max_test,
            integrator=integrator,
        )
        numerical_integral = bpwl.integral(
            gamma_low=gamma_min_test, gamma_up=gamma_max_test, gamma_power=0
        )
        analytical_integral = broken_power_law_integral(
            k_e_test, p1, p2, gamma_b, gamma_min_test, gamma_max_test
        )
        assert u.isclose(
            numerical_integral, analytical_integral, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    @pytest.mark.parametrize("p1", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    @pytest.mark.parametrize("p2", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    @pytest.mark.parametrize("gamma_b", [1e2, 1e3, 1e4, 1e5, 1e6])
    @pytest.mark.parametrize("integrator", [np.trapz, trapz_loglog])
    def test_broken_power_law_times_gamma_integral(self, p1, p2, gamma_b, integrator):
        """test the integration of the broken power law times gamma for different
        spectral indexes and breaks and different integrating functions"""
        bpwl = BrokenPowerLaw(
            k_e_test,
            p1,
            p2,
            gamma_b,
            gamma_min_test,
            gamma_max_test,
            integrator=integrator,
        )
        numerical_integral = bpwl.integral(
            gamma_low=gamma_min_test, gamma_up=gamma_max_test, gamma_power=1
        )
        analytical_integral = broken_power_law_times_gamma_integral(
            k_e_test, p1, p2, gamma_b, gamma_min_test, gamma_max_test
        )
        assert u.isclose(
            numerical_integral, analytical_integral, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    def test_from_normalised_density(self):
        """test the intialisation of the broken power law from the total particle
        density"""
        n_e_tot = 1e-5 * u.Unit("cm-3")
        bpwl = BrokenPowerLaw.from_normalised_density(
            n_e_tot=n_e_tot,
            p1=p1_test,
            p2=p2_test,
            gamma_b=gamma_b_test,
            gamma_min=gamma_min_test,
            gamma_max=gamma_max_test,
        )
        # calculate n_e_tot
        n_e_tot_calc = bpwl.integral(
            gamma_low=gamma_min_test, gamma_up=gamma_max_test, gamma_power=0
        )
        assert u.isclose(n_e_tot, n_e_tot_calc, atol=0 * u.Unit("cm-3"), rtol=1e-2)

    def test_from_normalised_energy_density(self):
        """test the intialisation of the powerlaw from the total particle
        energy density"""
        u_e = 3e-4 * u.Unit("erg cm-3")
        bpwl = BrokenPowerLaw.from_normalised_energy_density(
            u_e=u_e,
            p1=p1_test,
            p2=p2_test,
            gamma_b=gamma_b_test,
            gamma_min=gamma_min_test,
            gamma_max=gamma_max_test,
        )
        # calculate u_e
        u_e_calc = mec2 * bpwl.integral(
            gamma_low=gamma_min_test, gamma_up=gamma_max_test, gamma_power=1
        )
        assert u.isclose(u_e, u_e_calc, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)

    def test_from_norm_at_gamma_1(self):
        """test the intialisation of the powerlaw from the normalisation at
        gamma = 1"""
        norm = 1e-13 * u.Unit("cm-3")
        bpwl = BrokenPowerLaw.from_norm_at_gamma_1(
            norm=norm,
            p1=p1_test,
            p2=p2_test,
            gamma_b=gamma_b_test,
            gamma_min=1,
            gamma_max=gamma_max_test,
        )
        assert u.isclose(norm, bpwl(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)


class TestLogParabola:
    """class grouping all tests related to the PowerLaw spectrum
    the analytical integral is non-trivial so we ignore the comparison with the
    analytical integral"""

    def test_call(self):
        """assert that outside the bounding box the function returns 0"""
        gamma = np.logspace(0, 8)
        values = lp_test(gamma).value
        condition = (gamma_min_test <= gamma) * (gamma <= gamma_max_test)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    def test_from_normalised_density(self):
        """test the intialisation of the log parabola from the total particle
        density"""
        n_e_tot = 1e-5 * u.Unit("cm-3")
        lp = LogParabola.from_normalised_density(
            n_e_tot=n_e_tot,
            p=p_test,
            q=q_test,
            gamma_0=gamma_0_test,
            gamma_min=gamma_min_test,
            gamma_max=gamma_max_test,
        )
        # calculate n_e_tot
        n_e_tot_calc = lp.integral(
            gamma_low=gamma_min_test, gamma_up=gamma_max_test, gamma_power=0
        )
        assert u.isclose(n_e_tot, n_e_tot_calc, atol=0 * u.Unit("cm-3"), rtol=1e-2)

    def test_from_normalised_energy_density(self):
        """test the intialisation of the powerlaw from the total particle
        energy density"""
        u_e = 3e-4 * u.Unit("erg cm-3")
        lp = LogParabola.from_normalised_energy_density(
            u_e=u_e,
            p=p_test,
            q=q_test,
            gamma_0=gamma_0_test,
            gamma_min=gamma_min_test,
            gamma_max=gamma_max_test,
        )
        # calculate u_e
        u_e_calc = mec2 * lp.integral(
            gamma_low=gamma_min_test, gamma_up=gamma_max_test, gamma_power=1
        )
        assert u.isclose(u_e, u_e_calc, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)

    def test_from_norm_at_gamma_1(self):
        """test the intialisation of the powerlaw from the normalisation at
        gamma = 1"""
        norm = 1e-13 * u.Unit("cm-3")
        lp = LogParabola.from_norm_at_gamma_1(
            norm=norm,
            p=p_test,
            q=q_test,
            gamma_0=gamma_0_test,
            gamma_min=1,
            gamma_max=gamma_max_test,
        )
        assert u.isclose(norm, lp(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)


class TestExpCutoffPowerLaw:
    """class grouping all tests related to the ExpCutoffPowerLaw spectrum"""

    def test_call(self):
        """assert that outside the bounding box the function returns 0"""
        gamma = np.logspace(0, 8)
        values = epwl_test(gamma).value
        condition = (gamma_min_test <= gamma) * (gamma <= gamma_max_test)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    def test_from_normalised_density(self):
        """test the intialisation of the power law from the total particle
        density"""
        n_e_tot = 1e-5 * u.Unit("cm-3")
        epwl = ExpCutoffPowerLaw.from_normalised_density(
            n_e_tot=n_e_tot,
            p=p_test,
            gamma_c=gamma_c_test,
            gamma_min=gamma_min_test,
            gamma_max=gamma_max_test,
        )
        # calculate n_e_tot
        n_e_tot_calc = epwl.integral(
            gamma_low=gamma_min_test, gamma_up=gamma_max_test, gamma_power=0
        )
        assert u.isclose(n_e_tot, n_e_tot_calc, atol=0 * u.Unit("cm-3"), rtol=1e-2)

    def test_from_normalised_energy_density(self):
        """test the intialisation of the power law from the total particle
        energy density"""
        u_e = 3e-4 * u.Unit("erg cm-3")
        epwl = ExpCutoffPowerLaw.from_normalised_energy_density(
            u_e=u_e,
            p=p_test,
            gamma_c=gamma_c_test,
            gamma_min=gamma_min_test,
            gamma_max=gamma_max_test,
        )
        # calculate u_e
        u_e_calc = mec2 * epwl.integral(
            gamma_low=gamma_min_test, gamma_up=gamma_max_test, gamma_power=1
        )
        print(u_e)
        print(u_e_calc)
        assert u.isclose(u_e, u_e_calc, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)

    def test_from_norm_at_gamma_1(self):
        """test the intialisation of the powerlaw from the normalisation at
        gamma = 1"""
        norm = 1e-13 * u.Unit("cm-3")
        epwl = ExpCutoffPowerLaw.from_norm_at_gamma_1(
            norm=norm,
            p=p_test,
            gamma_c=gamma_c_test,
            gamma_min=1,
            gamma_max=gamma_max_test,
        )
        assert u.isclose(norm, epwl(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)
