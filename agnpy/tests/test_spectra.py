# tests on spectra module
import numpy as np
import astropy.units as u
import pytest
from agnpy.spectra import PowerLaw, BrokenPowerLaw, LogParabola, ExpCutoffPowerLaw, Interpolation
from agnpy.utils.math import trapz_loglog
from agnpy.utils.conversion import mec2

# variables with _test are global and meant to be used in all tests
# global PowerLaw
k_e_test = 1e-13 * u.Unit("cm-3")
p_test = 2.1
gamma_min_test = 10
gamma_max_test = 1e7
pwl_test = PowerLaw(
    k_e_test, p_test, gamma_min_test, gamma_max_test
)


# global BrokenPowerLaw
p1_test = 2.1
p2_test = 3.1
gamma_b_test = 1e3
bpwl_test = BrokenPowerLaw(
    k_e_test, p1_test, p2_test, gamma_b_test, gamma_min_test, gamma_max_test
)


# global LogParabola
q_test = 0.2
gamma_0_test = 1e4
lp_test = LogParabola(
    k_e_test, p_test, q_test, gamma_0_test, gamma_min_test, gamma_max_test
)

# global PowerLaw exp Cutoff
gamma_c_test = 1e3
epwl_test = ExpCutoffPowerLaw(
    k_e_test, p_test, gamma_c_test, gamma_min_test, gamma_max_test
)


def power_law_integral(k_e, p, gamma_min, gamma_max):
    """analytical integral of the power law"""
    if np.isclose(p, 1.0):
        integral = np.log(gamma_max / gamma_min)
    else:
        integral = (np.power(gamma_max, 1 - p) - np.power(gamma_min, 1 - p)) / (1 - p)
    return k_e * integral


def power_law_times_gamma_integral(k_e, p, gamma_min, gamma_max):
    """analytical integral of the power law multiplied by gamma"""
    if np.isclose(p, 2.0):
        integral = np.log(gamma_max / gamma_min)
    else:
        integral = (np.power(gamma_max, 2 - p) - np.power(gamma_min, 2 - p)) / (2 - p)
    return k_e * integral


def broken_power_law_integral(k_e, p1, p2, gamma_b, gamma_min, gamma_max):
    """analytical integral of the power law with two spectral indexes"""
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
    """analytical integral of the power law with two spectral indexes multiplied
    by gamma"""
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


# Data are generated, following the four already implemented distributions of agnpy.
# The interpolation function is derived for each data set and then is interpolated and
# data points are being compared:

def pwl_data(k_e_test,p_test,gamma_min_test,gamma_max_test):

    pwl_data = np.zeros((2,100),float)
    gamma1 = np.logspace(np.log10(gamma_min_test),np.log10(gamma_max_test),100)

    for i in range(len(gamma1)):
        pwl_data[0,i] = gamma1[i]
        pwl_data[1,i] = pwl_test(gamma1[i]).value

    return pwl_data


def bpwl_data(k_e_test, p1_test, p2_test, gamma_b_test, gamma_min_test, gamma_max_test):

    bpwl_data = np.zeros((2,100),float)
    gamma1 = np.logspace(np.log10(gamma_min_test),np.log10(gamma_max_test),100)

    for i in range(len(gamma1)):
        bpwl_data[0,i] = gamma1[i]
        bpwl_data[1,i] = bpwl_test(gamma1[i]).value

    return bpwl_data


def logparabola_data(k_e_test, p1_test, p2_test, gamma_b_test, gamma_min_test, gamma_max_test):

    lp_data = np.zeros((2,100),float)
    gamma1 = np.logspace(np.log10(gamma_min_test),np.log10(gamma_max_test),100)

    for i in range(len(gamma1)):
        lp_data[0,i] = gamma1[i]
        lp_data[1,i] = lp_test(gamma1[i]).value

    return lp_data

def epwl_data(k_e_test, p1_test, p2_test, gamma_b_test, gamma_min_test, gamma_max_test):

    ep_data = np.zeros((2,100),float)
    gamma1 = np.logspace(np.log10(gamma_min_test),np.log10(gamma_max_test),100)

    for i in range(len(gamma1)):
        ep_data[0,i] = gamma1[i]
        ep_data[1,i] = epwl_test(gamma1[i]).value

    return ep_data

#power law data and interpolation
pwl_data = pwl_data(
    k_e_test,p_test,gamma_min_test,gamma_max_test
)
pwl_inter = Interpolation(
    pwl_data[0,:], pwl_data[1,:]*u.Unit('cm-3')
)
# broken power law data and interpolation
bpwl_data = bpwl_data(
    k_e_test, p1_test, p2_test, gamma_b_test, gamma_min_test, gamma_max_test
)
bpwl_inter = Interpolation(
    bpwl_data[0,:], bpwl_data[1,:]*u.Unit('cm-3')
)
# log parabola data and interpolation
lp_data = logparabola_data(
    k_e_test, p1_test, p2_test, gamma_b_test, gamma_min_test, gamma_max_test
)
lp_inter = Interpolation(
    lp_data[0,:], lp_data[1,:]*u.Unit('cm-3')
)
# exp cut off power law data and interpolation
epwl_data = epwl_data(
    k_e_test, p1_test, p2_test, gamma_b_test, gamma_min_test, 1e5
)
epwl_inter = Interpolation(
    epwl_data[0,:], epwl_data[1,:]*u.Unit('cm-3')
)
class TestPowerLaw:
    """class grouping all tests related to the BrokenPowerLaw spectrum"""

    def test_call(self):
        """assert that outside the bounding box the function returns 0"""
        gamma = np.logspace(0, 8)
        values = pwl_test(gamma).value
        condition = (gamma_min_test <= gamma) * (gamma <= gamma_max_test)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    @pytest.mark.parametrize("integrator", [np.trapz, trapz_loglog])
    def test_power_law_integral(self, p, integrator):
        """test the integration of the power law for different spectral indexes
        different integrating functions"""
        pwl = PowerLaw(
            k_e_test, p, gamma_min_test, gamma_max_test, integrator=integrator
        )
        numerical_integral = pwl.integral(gamma_min_test, gamma_max_test)
        analytical_integral = power_law_integral(
            k_e_test, p, gamma_min_test, gamma_max_test
        )
        assert u.isclose(
            numerical_integral, analytical_integral, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    @pytest.mark.parametrize("p", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    @pytest.mark.parametrize("integrator", [np.trapz, trapz_loglog])
    def test_power_law_times_gamma_integral(self, p, integrator):
        """test the integration of the power law times gamma for different
        spectral indexes and different integrating functions"""
        pwl = PowerLaw(
            k_e_test, p, gamma_min_test, gamma_max_test, integrator=integrator
        )
        numerical_integral = pwl.integral(
            gamma_low=gamma_min_test, gamma_up=gamma_max_test, gamma_power=1
        )
        analytical_integral = power_law_times_gamma_integral(
            k_e_test, p, gamma_min_test, gamma_max_test,
        )
        assert u.isclose(
            numerical_integral, analytical_integral, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    def test_from_normalised_density(self):
        """test the intialisation of the power law from the total particle
        density"""
        n_e_tot = 1e-5 * u.Unit("cm-3")
        pwl = PowerLaw.from_normalised_density(
            n_e_tot=n_e_tot,
            p=p_test,
            gamma_min=gamma_min_test,
            gamma_max=gamma_max_test,
        )
        # calculate n_e_tot
        n_e_tot_calc = pwl.integral(
            gamma_low=gamma_min_test, gamma_up=gamma_max_test, gamma_power=0
        )
        assert u.isclose(n_e_tot, n_e_tot_calc, atol=0 * u.Unit("cm-3"), rtol=1e-2)

    def test_from_normalised_energy_density(self):
        """test the intialisation of the power law from the total particle
        energy density"""
        u_e = 3e-4 * u.Unit("erg cm-3")
        pwl = PowerLaw.from_normalised_energy_density(
            u_e=u_e, p=p_test, gamma_min=gamma_min_test, gamma_max=gamma_max_test
        )
        # calculate u_e
        u_e_calc = mec2 * pwl.integral(
            gamma_low=gamma_min_test, gamma_up=gamma_max_test, gamma_power=1
        )
        assert u.isclose(u_e, u_e_calc, atol=0 * u.Unit("erg cm-3"), rtol=1e-2)

    def test_from_norm_at_gamma_1(self):
        """test the intialisation of the powerlaw from the normalisation at
        gamma = 1"""
        norm = 1e-13 * u.Unit("cm-3")
        pwl = PowerLaw.from_norm_at_gamma_1(
            norm=norm, p=p_test, gamma_min=1, gamma_max=gamma_max_test
        )
        assert u.isclose(norm, pwl(1), atol=0 * u.Unit("cm-3"), rtol=1e-2)


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

class TestInterpolation:

    """ 1. assert that outside the bounding box the function returns 0"""

    def test_call_pow(self):
        gamma = np.logspace(0, 8)
        values = pwl_inter(gamma).value
        condition = (gamma_min_test <= gamma) * (gamma <= gamma_max_test)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    def test_call_bpwl(self):
        gamma = np.logspace(0, 8)
        values = bpwl_inter(gamma).value
        condition = (gamma_min_test <= gamma) * (gamma <= gamma_max_test)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    def test_call_lp(self):
        gamma = np.logspace(0, 8)
        values = lp_inter(gamma).value
        condition = (gamma_min_test <= gamma) * (gamma <= gamma_max_test)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    def test_call_ep(self):
        gamma = np.logspace(0, 8)
        values = epwl_inter(gamma).value
        condition = (gamma_min_test <= gamma) * (gamma <= gamma_max_test)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    """ 2. assert that the interpolated function does not have large deviations from the original one """

    def test_power_vs_inter(self):
        a = abs(pwl_inter(pwl_data[0,:]).value - pwl_data[1,:])/pwl_data[1,:]
        assert max(a) < 1e-5

    def test_bpwl_vs_inter(self):
        a = abs(bpwl_inter(bpwl_data[0,:]).value - bpwl_data[1,:])/bpwl_data[1,:]
        assert max(a) < 1e-5

    def test_lp_vs_inter(self):
        a = abs(lp_inter(pwl_data[0,:]).value - lp_data[1,:])/lp_data[1,:]
        assert max(a) < 1e-5

    def test_ep_vs_inter(self):
        a = abs(epwl_inter(epwl_data[0,:]).value - epwl_data[1,:])/epwl_data[1,:]
        assert max(a) < 1e-5


    """ 3. assert that the SSA integrand of the interpolated function does not have large deviations from the original one """

    def test_SSA_pow(self):
        SSA_inter = pwl_inter.SSA_integrand(pwl_data[0,:]).value
        SSA_power = pwl_test.SSA_integrand(pwl_data[0,:]).value
        a = abs((SSA_inter - SSA_power)/SSA_power)
        assert max(a) < 1e-2

    #only test that does not pass. The problem is where the distribution breaks. For the 100 points, only 1 does not pass the test, the one that
    #corresponds to the brake.
    def test_SSA_bpwl(self):
        SSA_inter = bpwl_inter.SSA_integrand(bpwl_data[0,:]).value
        SSA_power = bpwl_test.SSA_integrand(bpwl_data[0,:]).value
        a = abs((SSA_inter - SSA_power)/SSA_power)
        assert max(a) < 1e-2

    def test_SSA_lp(self):
        SSA_inter = lp_inter.SSA_integrand(lp_data[0,:]).value
        SSA_power = lp_test.SSA_integrand(lp_data[0,:]).value
        a = abs((SSA_inter - SSA_power)/SSA_power)
        assert max(a) < 1e-2

    def test_SSA_ep(self):
        SSA_inter = epwl_inter.SSA_integrand(epwl_data[0,:]).value
        SSA_power = epwl_test.SSA_integrand(epwl_data[0,:]).value
        a = abs((SSA_inter - SSA_power)/SSA_power)
        assert max(a) < 1e-2
