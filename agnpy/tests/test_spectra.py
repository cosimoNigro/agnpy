# tests on spectra module
import numpy as np
import astropy.units as u
from astropy.constants import m_e
from agnpy.spectra import PowerLaw, BrokenPowerLaw, LogParabola
import pytest


mec2 = m_e.to("erg", equivalencies=u.mass_energy())
gamma_min = 10
gamma_max = 1e7
pwl = PowerLaw(k_e=1e-13 * u.Unit("cm-3"), p=2)
pwl.set_bounding(gamma_min, gamma_max)


def power_law_integral(gamma_min, gamma_max, k_e, p):
    """analytical integral of the power law"""
    if np.isclose(p, 1.0):
        integral = np.log(gamma_max / gamma_min)
    else:
        integral = (np.power(gamma_max, 1 - p) - np.power(gamma_min, 1 - p)) / (1 - p)
    return k_e * integral


def power_law_times_gamma_integral(gamma_min, gamma_max, k_e, p):
    """analytical integral of the power law multiplied by gamma"""
    if np.isclose(p, 2.0):
        integral = np.log(gamma_max / gamma_min)
    else:
        integral = (np.power(gamma_max, 2 - p) - np.power(gamma_min, 2 - p)) / (2 - p)
    return k_e * integral


class TestPowerLaw:
    """class grouping all tests related to the PowerLaw spectrum"""

    def test_set_bounding_box(self):
        """test changing of the boundaries"""
        new_gamma_min = 1e2
        new_gamma_max = 1e6
        pwl.set_bounding(new_gamma_min, new_gamma_max)
        assert pwl.bounding_box == (new_gamma_min, new_gamma_max)

    def test_call(self):
        """assert that outside the bounding box the function returns 0"""
        gamma = np.logspace(0, 8)
        values = pwl(gamma).value
        condition = (gamma_min <= gamma) * (gamma <= gamma_max)
        # check that outside the boundaries values are all 0
        assert not np.all(values[~condition])

    @pytest.mark.parametrize("p", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    def test_power_law_integral(self, p):
        """test the integration with different power law indexes"""
        k_e = 1e-13 * u.Unit("cm-3")
        gamma_min = 1e2
        gamma_max = 1e6
        numerical_integral = PowerLaw().integral(
            gamma_min=gamma_min, gamma_max=gamma_max, gamma_power=0, k_e=k_e, p=p
        )
        analytical_integral = power_law_integral(
            gamma_min=gamma_min, gamma_max=gamma_max, k_e=k_e, p=p,
        )
        assert u.isclose(
            numerical_integral, analytical_integral, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )

    @pytest.mark.parametrize("p", [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    def test_power_law_times_gamma_integral(self, p):
        """test the integration with different power law indexes"""
        k_e = 1e-13 * u.Unit("cm-3")
        gamma_min = 1e2
        gamma_max = 1e6
        numerical_integral = PowerLaw().integral(
            gamma_min=gamma_min, gamma_max=gamma_max, gamma_power=1, k_e=k_e, p=p
        )
        analytical_integral = power_law_times_gamma_integral(
            gamma_min=gamma_min, gamma_max=gamma_max, k_e=k_e, p=p,
        )
        assert u.isclose(
            numerical_integral, analytical_integral, atol=0 * u.Unit("cm-3"), rtol=1e-2
        )
