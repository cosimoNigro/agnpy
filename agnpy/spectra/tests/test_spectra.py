# tests on agnpy.spectra module
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.constants import m_e, m_p, c
import pytest

from agnpy import Blob, Synchrotron
from agnpy.radiative_process.time_evolution import TimeEvolution, synchrotron_loss
from agnpy.spectra import (
    PowerLaw,
    BrokenPowerLaw,
    LogParabola,
    ExpCutoffPowerLaw,
    ExpCutoffBrokenPowerLaw,
    InterpolatedDistribution,
)
from agnpy.utils.math import trapz_loglog, power_law_integral
from agnpy.utils.conversion import mec2, mpc2

agnpy_dir = Path(__file__).parent.parent.parent  # go to the agnpy root
# where to read sampled files
data_dir = agnpy_dir / "data"

gamma_init_interp = np.logspace(2, 5)
n_e_interp = 1e-3 * u.Unit("cm-3") * gamma_init_interp ** (-2.1)


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

class TestParticleDistribution:
    """Class grouping all the tests related to the general class
    ParticleDistribution, from which all the other classes inherit."""

    @pytest.mark.parametrize(
        "n_e",
        [
            PowerLaw(),
            BrokenPowerLaw(),
            LogParabola(),
            ExpCutoffPowerLaw(),
            ExpCutoffBrokenPowerLaw(),
            InterpolatedDistribution(gamma_init_interp, n_e_interp),
        ],
    )
    @pytest.mark.parametrize("gamma_power", [0, 1, 2])
    def test_plot(self, n_e, gamma_power):
        n_e.plot(gamma_power=gamma_power)
        assert True

    @pytest.mark.parametrize(
        "n_e, tag",
        [
            (PowerLaw(), "PowerLaw"),
            (BrokenPowerLaw(), "BrokenPowerLaw"),
            (LogParabola(), "LogParabola"),
            (ExpCutoffPowerLaw(), "ExpCutoffPowerLaw"),
            (ExpCutoffBrokenPowerLaw(), "ExpCutoffBrokenPowerLaw"),
            (
                InterpolatedDistribution(gamma_init_interp, n_e_interp),
                "InterpolatedDistribution",
            ),
        ],
    )
    def test_tags(self, n_e, tag):
        assert n_e.tag == tag


class TestPowerLaw:
    """Class grouping all tests related to the PowerLaw spectrum."""

    def test_call(self):
        """Assert that outside the bounding box the function returns 0."""
        gamma = np.logspace(0, 8)
        pwl = PowerLaw()
        values = pwl(gamma).value
        condition = (pwl.gamma_min <= gamma) * (gamma <= pwl.gamma_max)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    @pytest.mark.parametrize("p", np.arange(1, 4, 0.5))
    @pytest.mark.parametrize("gamma_min", np.logspace(0, 2, 3))
    @pytest.mark.parametrize("gamma_max", np.logspace(4, 6, 3))
    @pytest.mark.parametrize("gamma_power", [0, 1])
    @pytest.mark.parametrize("integrator", [np.trapz, trapz_loglog])
    def test_power_law_integral(self, p, gamma_min, gamma_max, gamma_power, integrator):
        """Test the integration of the power law for different parameters and
        integrating function."""
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

    @pytest.mark.parametrize("p", np.arange(1, 4, 0.5))
    @pytest.mark.parametrize("gamma_min", np.logspace(0, 2, 3))
    @pytest.mark.parametrize("gamma_max", np.logspace(4, 6, 3))
    def test_init(self, p, gamma_min, gamma_max):
        """Test the initialisation of the power law with the different methods."""
        # initialisation from total density
        n_tot = 1e-5 * u.Unit("cm-3")

        pwl_e = PowerLaw.from_total_density(
            n_tot=n_tot, mass=m_e, p=p, gamma_min=gamma_min, gamma_max=gamma_max
        )
        assert pwl_e.particle == "electrons"

        pwl_p = PowerLaw.from_total_density(
            n_tot=n_tot, mass=m_p, p=p, gamma_min=gamma_min, gamma_max=gamma_max
        )
        assert pwl_p.particle == "protons"

        # check the total density is the same value we set initially
        n_e_tot = pwl_e.integrate(gamma_min, gamma_max)
        n_p_tot = pwl_p.integrate(gamma_min, gamma_max)
        assert u.isclose(n_e_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=1e-2)
        assert u.isclose(n_p_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=1e-2)

        # initialisation from total energy density
        u_tot = 3e-4 * u.Unit("erg cm-3")

        pwl_e = PowerLaw.from_total_energy_density(
            u_tot=u_tot, mass=m_e, p=p, gamma_min=gamma_min, gamma_max=gamma_max
        )
        assert pwl_e.particle == "electrons"

        pwl_p = PowerLaw.from_total_energy_density(
            u_tot=u_tot, mass=m_p, p=p, gamma_min=gamma_min, gamma_max=gamma_max
        )
        assert pwl_p.particle == "protons"

        u_e_tot = mec2 * pwl_e.integrate(gamma_min, gamma_max, gamma_power=1)
        u_p_tot = mpc2 * pwl_p.integrate(gamma_min, gamma_max, gamma_power=1)

        # check the total energy density is the same value we set initially
        assert u.isclose(u_e_tot, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)
        assert u.isclose(u_p_tot, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)

        # initialisation from the density at gamma=1
        n_gamma_1 = 1e-13 * u.Unit("cm-3")

        pwl_e = PowerLaw.from_density_at_gamma_1(
            n_gamma_1=n_gamma_1, mass=m_e, p=p, gamma_min=1, gamma_max=gamma_max
        )
        assert pwl_e.particle == "electrons"

        pwl_p = PowerLaw.from_density_at_gamma_1(
            n_gamma_1=n_gamma_1, mass=m_p, p=p, gamma_min=1, gamma_max=gamma_max
        )
        assert pwl_p.particle == "protons"

        # check that the value at gamma=1 is the value we set initially
        assert u.isclose(n_gamma_1, pwl_e(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)
        assert u.isclose(n_gamma_1, pwl_p(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)


class TestBrokenPowerLaw:
    """Class grouping all tests related to the BrokenPowerLaw spectrum."""

    def test_call(self):
        """Assert that outside the bounding box the function returns 0."""
        gamma = np.logspace(0, 8)
        bpwl = BrokenPowerLaw()
        values = bpwl(gamma).value
        condition = (bpwl.gamma_min <= gamma) * (gamma <= bpwl.gamma_max)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    @pytest.mark.parametrize("p1", np.arange(1, 2, 0.5))
    @pytest.mark.parametrize("p2", np.arange(3, 4, 0.5))
    @pytest.mark.parametrize("gamma_b", np.logspace(3, 5, 3))
    @pytest.mark.parametrize("gamma_min", np.logspace(0, 2, 3))
    @pytest.mark.parametrize("gamma_max", np.logspace(6, 8, 3))
    @pytest.mark.parametrize("gamma_power", [0, 1])
    @pytest.mark.parametrize("integrator", [np.trapz, trapz_loglog])
    def test_broken_power_law_integral(
        self, p1, p2, gamma_b, gamma_min, gamma_max, gamma_power, integrator
    ):
        """Test the integration of the broken power law for different parameters
        and integrating function."""
        k = 1e-5 * u.Unit("cm-3")
        bpwl = BrokenPowerLaw(
            k, p1, p2, gamma_b, gamma_min, gamma_max, integrator=integrator
        )
        numerical_integral = bpwl.integrate(
            gamma_min, gamma_max, gamma_power=gamma_power
        )

        if gamma_power == 0:
            analytical_integral = broken_power_law_integral(
                k, p1, p2, gamma_b, gamma_min, gamma_max
            )
        if gamma_power == 1:
            analytical_integral = broken_power_law_times_gamma_integral(
                k, p1, p2, gamma_b, gamma_min, gamma_max
            )

        assert u.isclose(
            numerical_integral, analytical_integral, atol=0 * u.Unit("cm-3"), rtol=0.05
        )

    @pytest.mark.parametrize("p1", np.arange(1, 2, 0.5))
    @pytest.mark.parametrize("p2", np.arange(3, 4, 0.5))
    @pytest.mark.parametrize("gamma_b", np.logspace(3, 5, 3))
    @pytest.mark.parametrize("gamma_min", np.logspace(0, 2, 3))
    @pytest.mark.parametrize("gamma_max", np.logspace(6, 8, 3))
    def test_init(self, p1, p2, gamma_b, gamma_min, gamma_max):
        """Test the intialisation of the broken power law with the different methods."""
        # initialisation from total density
        n_tot = 1e-5 * u.Unit("cm-3")

        bpwl_e = BrokenPowerLaw.from_total_density(
            n_tot=n_tot,
            mass=m_e,
            p1=p1,
            p2=p2,
            gamma_b=gamma_b,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert bpwl_e.particle == "electrons"

        bpwl_p = BrokenPowerLaw.from_total_density(
            n_tot=n_tot,
            mass=m_p,
            p1=p1,
            p2=p2,
            gamma_b=gamma_b,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert bpwl_p.particle == "protons"

        # check the total density is the same value we set initially
        n_e_tot = bpwl_e.integrate(gamma_min, gamma_max)
        n_p_tot = bpwl_p.integrate(gamma_min, gamma_max)
        assert u.isclose(n_e_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=1e-2)
        assert u.isclose(n_p_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=1e-2)

        u_tot = 3e-4 * u.Unit("erg cm-3")

        # initialisation from total energy density
        bpwl_e = BrokenPowerLaw.from_total_energy_density(
            u_tot=u_tot,
            mass=m_e,
            p1=p1,
            p2=p2,
            gamma_b=gamma_b,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert bpwl_e.particle == "electrons"

        bpwl_p = BrokenPowerLaw.from_total_energy_density(
            u_tot=u_tot,
            mass=m_p,
            p1=p1,
            p2=p2,
            gamma_b=gamma_b,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert bpwl_p.particle == "protons"

        # check the total energy density is the same value we set initially
        u_e_tot = mec2 * bpwl_e.integrate(gamma_min, gamma_max, gamma_power=1)
        u_p_tot = mpc2 * bpwl_p.integrate(gamma_min, gamma_max, gamma_power=1)
        assert u.isclose(u_e_tot, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)
        assert u.isclose(u_p_tot, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)

        # initialisation from the density at gamma=1
        n_gamma_1 = 1e-13 * u.Unit("cm-3")

        bpwl_e = BrokenPowerLaw.from_density_at_gamma_1(
            n_gamma_1=n_gamma_1,
            mass=m_e,
            p1=p1,
            p2=p2,
            gamma_b=gamma_b,
            gamma_min=1,
            gamma_max=gamma_max,
        )
        assert bpwl_e.particle == "electrons"

        bpwl_p = BrokenPowerLaw.from_density_at_gamma_1(
            n_gamma_1=n_gamma_1,
            mass=m_p,
            p1=p1,
            p2=p2,
            gamma_b=gamma_b,
            gamma_min=1,
            gamma_max=gamma_max,
        )
        assert bpwl_p.particle == "protons"

        # check that the value at gamma=1 is the value we set initially
        assert u.isclose(n_gamma_1, bpwl_e(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)
        assert u.isclose(n_gamma_1, bpwl_p(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)


class TestLogParabola:
    """Class grouping all tests related to the LogParabola spectrum.
    The analytical integral is non-trivial so we ignore comparisons agains it."""

    def test_call(self):
        """Assert that outside the bounding box the function returns 0."""
        gamma = np.logspace(0, 8)
        lp = LogParabola()
        values = lp(gamma).value
        condition = (lp.gamma_min <= gamma) * (gamma <= lp.gamma_max)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    @pytest.mark.parametrize("p", np.arange(1, 4, 0.5))
    @pytest.mark.parametrize("q", [0.02, 0.05, 0.2, 0.5])
    @pytest.mark.parametrize("gamma_0", np.logspace(3, 5, 3))
    @pytest.mark.parametrize("gamma_min", np.logspace(0, 2, 3))
    @pytest.mark.parametrize("gamma_max", np.logspace(6, 8, 3))
    def test_init(self, p, q, gamma_0, gamma_min, gamma_max):
        """Test the intialisation of the log parabola with the different methods."""
        # initialisation from total density
        n_tot = 1e-5 * u.Unit("cm-3")

        lp_e = LogParabola.from_total_density(
            n_tot=n_tot,
            mass=m_e,
            p=p,
            q=q,
            gamma_0=gamma_0,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert lp_e.particle == "electrons"

        lp_p = LogParabola.from_total_density(
            n_tot=n_tot,
            mass=m_p,
            p=p,
            q=q,
            gamma_0=gamma_0,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert lp_p.particle == "protons"

        # check the total density is the same value we set initially
        n_e_tot = lp_e.integrate(gamma_min, gamma_max)
        n_p_tot = lp_p.integrate(gamma_min, gamma_max)
        assert u.isclose(n_e_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=1e-2)
        assert u.isclose(n_p_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=1e-2)

        # initialisation from total energy density
        u_tot = 3e-4 * u.Unit("erg cm-3")

        lp_e = LogParabola.from_total_energy_density(
            u_tot=u_tot,
            mass=m_e,
            p=p,
            q=q,
            gamma_0=gamma_0,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert lp_e.particle == "electrons"

        lp_p = LogParabola.from_total_energy_density(
            u_tot=u_tot,
            mass=m_p,
            p=p,
            q=q,
            gamma_0=gamma_0,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert lp_p.particle == "protons"

        # check the total energy density is the same value we set initially
        u_e_tot = mec2 * lp_e.integrate(gamma_min, gamma_max, gamma_power=1)
        u_p_tot = mpc2 * lp_p.integrate(gamma_min, gamma_max, gamma_power=1)
        assert u.isclose(u_e_tot, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)
        assert u.isclose(u_p_tot, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)

        # initialisation from the density at gamma=1
        n_gamma_1 = 1e-13 * u.Unit("cm-3")

        lp_e = LogParabola.from_density_at_gamma_1(
            n_gamma_1=n_gamma_1,
            mass=m_e,
            p=p,
            q=q,
            gamma_0=gamma_0,
            gamma_min=1,
            gamma_max=gamma_max,
        )
        assert lp_e.particle == "electrons"

        lp_p = LogParabola.from_density_at_gamma_1(
            n_gamma_1=n_gamma_1,
            mass=m_p,
            p=p,
            q=q,
            gamma_0=gamma_0,
            gamma_min=1,
            gamma_max=gamma_max,
        )
        assert lp_p.particle == "protons"

        # check that the value at gamma=1 is the value we set initially
        assert u.isclose(n_gamma_1, lp_e(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)
        assert u.isclose(n_gamma_1, lp_p(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)


class TestExpCutoffPowerLaw:
    """Class grouping all tests related to the ExpCutoffPowerLaw spectrum."""

    def test_call(self):
        """Assert that outside the bounding box the function returns 0."""
        gamma = np.logspace(0, 8)
        epwl = ExpCutoffPowerLaw()
        values = epwl(gamma).value
        condition = (epwl.gamma_min <= gamma) * (gamma <= epwl.gamma_max)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    @pytest.mark.parametrize("p", np.arange(1, 4, 0.5))
    @pytest.mark.parametrize("gamma_c", np.logspace(3, 5, 3))
    @pytest.mark.parametrize("gamma_min", np.logspace(0, 2, 3))
    @pytest.mark.parametrize("gamma_max", np.logspace(6, 8, 3))
    def test_init(self, p, gamma_c, gamma_min, gamma_max):
        """Test the intialisation of the exp. cutoff power law with the
        different methods."""
        # initialise from total density
        n_tot = 1e-5 * u.Unit("cm-3")

        epwl_e = ExpCutoffPowerLaw.from_total_density(
            n_tot=n_tot,
            mass=m_e,
            p=p,
            gamma_c=gamma_c,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert epwl_e.particle == "electrons"

        epwl_p = ExpCutoffPowerLaw.from_total_density(
            n_tot=n_tot,
            mass=m_p,
            p=p,
            gamma_c=gamma_c,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert epwl_p.particle == "protons"

        # check the total density is the same value we set initially
        n_e_tot = epwl_e.integrate(gamma_min, gamma_max)
        n_p_tot = epwl_p.integrate(gamma_min, gamma_max)
        assert u.isclose(n_e_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=1e-2)
        assert u.isclose(n_p_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=1e-2)

        # initialise from total energy density
        u_tot = 3e-4 * u.Unit("erg cm-3")

        epwl_e = ExpCutoffPowerLaw.from_total_energy_density(
            u_tot=u_tot,
            mass=m_e,
            p=p,
            gamma_c=gamma_c,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert epwl_e.particle == "electrons"

        epwl_p = ExpCutoffPowerLaw.from_total_energy_density(
            u_tot=u_tot,
            mass=m_p,
            p=p,
            gamma_c=gamma_c,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert epwl_p.particle == "protons"

        # check the total density is the same value we set initially
        u_e_tot = mec2 * epwl_e.integrate(gamma_min, gamma_max, gamma_power=1)
        u_p_tot = mpc2 * epwl_p.integrate(gamma_min, gamma_max, gamma_power=1)
        assert u.isclose(u_e_tot, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)
        assert u.isclose(u_p_tot, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)

        # initialisation from the density at gamma=1
        n_gamma_1 = 1e-13 * u.Unit("cm-3")

        epwl_e = ExpCutoffPowerLaw.from_density_at_gamma_1(
            n_gamma_1=n_gamma_1,
            mass=m_e,
            p=p,
            gamma_c=gamma_c,
            gamma_min=1,
            gamma_max=gamma_max,
        )
        assert epwl_e.particle == "electrons"

        epwl_p = ExpCutoffPowerLaw.from_density_at_gamma_1(
            n_gamma_1=n_gamma_1,
            mass=m_p,
            p=p,
            gamma_c=gamma_c,
            gamma_min=1,
            gamma_max=gamma_max,
        )
        assert epwl_p.particle == "protons"

        # check that the value at gamma=1 is close to the value we set initially
        assert u.isclose(n_gamma_1, epwl_e(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)
        assert u.isclose(n_gamma_1, epwl_p(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)


class TestExpCutoffBrokenPowerLaw:
    """Class grouping all tests related to the ExpCutoffBrokenPowerLaw spectrum."""

    def test_call(self):
        """Assert that outside the bounding box the function returns 0."""
        gamma = np.logspace(0, 8)
        ebpwl = ExpCutoffBrokenPowerLaw()
        values = ebpwl(gamma).value
        condition = (ebpwl.gamma_min <= gamma) * (gamma <= ebpwl.gamma_max)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    @pytest.mark.parametrize("p1", np.arange(1, 2, 0.5))
    @pytest.mark.parametrize("p2", np.arange(3, 4, 0.5))
    @pytest.mark.parametrize("gamma_b", np.logspace(3, 5, 3))
    @pytest.mark.parametrize("gamma_min", np.logspace(0, 2, 3))
    @pytest.mark.parametrize("gamma_max", np.logspace(6, 8, 3))
    @pytest.mark.parametrize("gamma_c", np.logspace(3, 5, 3))
    def test_init(self, p1, p2, gamma_c, gamma_b, gamma_min, gamma_max):
        """Test the intialisation of the broken power law with the different methods."""
        # initialisation from total density
        n_tot = 1e-5 * u.Unit("cm-3")

        ebpwl_e = ExpCutoffBrokenPowerLaw.from_total_density(
            n_tot=n_tot,
            mass=m_e,
            p1=p1,
            p2=p2,
            gamma_c=gamma_c,
            gamma_b=gamma_b,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert ebpwl_e.particle == "electrons"

        ebpwl_p = ExpCutoffBrokenPowerLaw.from_total_density(
            n_tot=n_tot,
            mass=m_p,
            p1=p1,
            p2=p2,
            gamma_c=gamma_c,
            gamma_b=gamma_b,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert ebpwl_p.particle == "protons"

        # check the total density is the same value we set initially
        n_e_tot = ebpwl_e.integrate(gamma_min, gamma_max)
        n_p_tot = ebpwl_p.integrate(gamma_min, gamma_max)
        assert u.isclose(n_e_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=1e-2)
        assert u.isclose(n_p_tot, n_tot, atol=0 * u.Unit("cm-3"), rtol=1e-2)

        u_tot = 3e-4 * u.Unit("erg cm-3")

        # initialisation from total energy density
        ebpwl_e = ExpCutoffBrokenPowerLaw.from_total_energy_density(
            u_tot=u_tot,
            mass=m_e,
            p1=p1,
            p2=p2,
            gamma_c=gamma_c,
            gamma_b=gamma_b,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert ebpwl_e.particle == "electrons"

        ebpwl_p = ExpCutoffBrokenPowerLaw.from_total_energy_density(
            u_tot=u_tot,
            mass=m_p,
            p1=p1,
            p2=p2,
            gamma_c=gamma_c,
            gamma_b=gamma_b,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )
        assert ebpwl_p.particle == "protons"

        # check the total energy density is the same value we set initially
        u_e_tot = mec2 * ebpwl_e.integrate(gamma_min, gamma_max, gamma_power=1)
        u_p_tot = mpc2 * ebpwl_p.integrate(gamma_min, gamma_max, gamma_power=1)
        assert u.isclose(u_e_tot, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)
        assert u.isclose(u_p_tot, u_tot, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)

        # initialisation from the density at gamma=1
        n_gamma_1 = 1e-13 * u.Unit("cm-3")

        ebpwl_e = ExpCutoffBrokenPowerLaw.from_density_at_gamma_1(
            n_gamma_1=n_gamma_1,
            mass=m_e,
            p1=p1,
            p2=p2,
            gamma_c=gamma_c,
            gamma_b=gamma_b,
            gamma_min=1,
            gamma_max=gamma_max,
        )
        assert ebpwl_e.particle == "electrons"

        ebpwl_p = ExpCutoffBrokenPowerLaw.from_density_at_gamma_1(
            n_gamma_1=n_gamma_1,
            mass=m_p,
            p1=p1,
            p2=p2,
            gamma_c=gamma_c,
            gamma_b=gamma_b,
            gamma_min=1,
            gamma_max=gamma_max,
        )
        assert ebpwl_p.particle == "protons"

        # check that the value at gamma=1 is the value we set initially
        assert u.isclose(n_gamma_1, ebpwl_e(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)
        assert u.isclose(n_gamma_1, ebpwl_p(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)


class TestInterpolatedDistribution:
    @pytest.mark.parametrize(
        "n",
        [
            PowerLaw(),
            BrokenPowerLaw(),
            LogParabola(),
            ExpCutoffPowerLaw(),
            ExpCutoffBrokenPowerLaw(),
        ],
    )
    def test_interpolation_analytical(self, n):
        """Assert that the interpolated distribution does not have large
        deviations from the original one."""
        gamma = np.logspace(np.log10(n.gamma_min), np.log10(n.gamma_max))
        n_interp = InterpolatedDistribution(gamma, n(gamma))

        assert u.allclose(n(gamma), n_interp(gamma), atol=0 * u.Unit("cm-3"), rtol=1e-6)

        # also assert that outside the bounding box the values are 0
        gamma_broad = np.logspace(0, 8)
        values = n_interp(gamma_broad).to_value("cm-3")
        condition = (n_interp.gamma_min <= gamma_broad) * (
            gamma_broad <= n_interp.gamma_max
        )

        assert not np.all(values[~condition])

        # test the part of the SSA integrand involving the electron distribution
        ssa_integrand = n.SSA_integrand(gamma)
        ssa_integrand_interp = n_interp.SSA_integrand(gamma)

        assert u.allclose(
            ssa_integrand, ssa_integrand_interp, atol=0 * u.Unit("cm-3"), rtol=1e-1
        )

    def test_interpolation_physical(self):
        """Test the interpolation of a physical distribution, test also the norm factor."""
        data = np.loadtxt(f"{data_dir}/particles_distributions/eed_lepton_cooling.txt")
        gamma_init = data[:, 0]
        n_init = data[:, 1] * u.Unit("cm-3")

        # interpolate its values, change the scale factor
        n_e = InterpolatedDistribution(gamma_init, n_init, norm=2)

        assert u.allclose(
            n_e(gamma_init), 2 * n_init, atol=0 * u.Unit("cm-3"), rtol=1e-3
        )

    def test_integration_is_consistent(self):
        blob = Blob(n_e=PowerLaw(1 * u.cm ** -3, 2.3, gamma_min=1e2, gamma_max=1e7))
        synch = Synchrotron(blob)
        initial_integrate = blob.n_e.integrate(blob.n_e.gamma_min, blob.n_e.gamma_max)
        # make sure the evaluation and integral are consistent no matter how the distribution changes
        for i in range(100):
            TimeEvolution(blob, 1 * u.s, synchrotron_loss(synch)).eval_with_automatic_intervals()
            integrate = blob.n_e.integrate(blob.n_e.gamma_min, blob.n_e.gamma_max)
            assert u.isclose(initial_integrate, integrate, rtol=0.001)
