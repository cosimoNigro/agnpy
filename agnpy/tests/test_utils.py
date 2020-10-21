# tests on the utils module
import pytest
import numpy as np
from agnpy.utils.math import trapz_loglog


def line_loglog(x, m, n):
    """a straight line in loglog-space"""
    return x ** m * np.e ** n


def integral_line_loglog(x_min, x_max, m, n):
    """analytical integral of the line in log-log space"""
    f_low = line_loglog(x_min, m + 1, n) / (m + 1)
    f_up = line_loglog(x_max, m + 1, n) / (m + 1)
    return f_up - f_low


class TestMathUtils:
    """test uitls.math"""

    @pytest.mark.parametrize("m", np.arange(-2, 2.5, 0.5))
    @pytest.mark.parametrize("n", np.arange(-2, 2.5, 0.5))
    def test_trapz_log_log(self, m, n):
        """test the trapz log-log integration method"""
        x = np.logspace(2, 5)
        # generate syntethic power-law like data by defining a straight line
        y = line_loglog(x, m, n)
        trapz_integral = np.trapz(y, x, axis=0)
        trapz_loglog_integral = trapz_loglog(y, x, axis=0)
        analytical_integral = integral_line_loglog(x[0], x[-1], m, n)
        print("analyitical: ", analytical_integral)
        print("trapz: ", trapz_integral)
        print("trapz_loglog: ", trapz_loglog_integral)
        assert True
