
from copy import deepcopy
import numpy as np
import astropy.units as u
from astropy.constants import m_e, m_p, c
import pytest

from agnpy import Blob, Synchrotron, SynchrotronSelfCompton
from agnpy.radiative_process.time_evolution import TimeEvolution, synchrotron_loss, ssc_loss, ssc_thomson_limit_loss
from agnpy.spectra import (
    PowerLaw,
    BrokenPowerLaw,
    InterpolatedDistribution,
)
from agnpy.utils.conversion import mec2
from agnpy.utils.math import power_law_integral


def gamma_before_time(prefactor, gamma_after_time, time):
    """
    Reverse-time calculation of the gamma value before the energy loss from formula -dE/dt ~ (E ** 2)
    Applicable to synchrotron and Thomson losses
    """
    return 1 / ((1 / gamma_after_time) - time * prefactor() / (m_e * c ** 2))

class TestSpectraTimeEvolution:

    @pytest.mark.parametrize("p", [0.5, 1.2, 2.4])
    @pytest.mark.parametrize("time_and_steps", [(1, 10), (60, 60), (200, 100)])
    def test_compare_numerical_results_with_analytical_calculation_synch(self, p, time_and_steps):
        """
        Test time evolution of spectral electron density for synchrotron energy losses.
        Use a simple power law spectrum for easy calculation of analytical results.
        """
        time = time_and_steps[0] * u.s
        steps = time_and_steps[1]
        gamma_min = 1e2
        gamma_max = 1e7
        k = 0.1 * u.Unit("cm-3")
        initial_n_e = PowerLaw(k, p, gamma_min=gamma_min, gamma_max=gamma_max, mass=m_e)
        blob = Blob(n_e=initial_n_e, delta_D=1)
        synch = Synchrotron(blob)
        TimeEvolution(blob, time, synchrotron_loss(synch)).eval_with_fixed_intervals(
                                                      intervals_count=steps, max_change_per_interval=0.1)
        evaluated_n_e = blob.n_e

        def gamma_before_synch(gamma_after_time, time):
            return gamma_before_time(synch._electron_energy_loss_formula_prefactor, gamma_after_time, time)

        assert u.isclose(
            gamma_before_synch(evaluated_n_e.gamma_min, time),
            gamma_min,
            rtol=0.05
        )
        assert u.isclose(
            gamma_before_synch(evaluated_n_e.gamma_max, time),
            gamma_max,
            rtol=0.05
        )
        assert u.isclose(
            evaluated_n_e.integrate(evaluated_n_e.gamma_min, evaluated_n_e.gamma_max),
            power_law_integral(k, p, gamma_min, gamma_max),
            rtol=0.05
        )
        assert u.isclose(
            # synchrotron losses are highest at the highest energy, so test the highest energy range, as the most affected
            evaluated_n_e.integrate(evaluated_n_e.gamma_max/10, evaluated_n_e.gamma_max),
            power_law_integral(k, p, gamma_before_synch(evaluated_n_e.gamma_max/10, time), gamma_max),
            rtol=0.05
        )

    def test_compare_calculation_with_calculation_split_into_two(self):
        initial_n_e = BrokenPowerLaw(
            k=1e-8 * u.Unit("cm-3"),
            p1=1.9,
            p2=2.6,
            gamma_b=1e4,
            gamma_min=10,
            gamma_max=1e6,
            mass=m_e,
        )
        blob = Blob(n_e=initial_n_e)
        synch = Synchrotron(blob)

        # iterate over 60 s in 20 steps
        TimeEvolution(blob, 60 * u.s, synchrotron_loss(synch)).eval_with_fixed_intervals(intervals_count=20)
        eval_1 = deepcopy(blob.n_e)
        # iterate first over 30 s, and then, starting with interpolated distribution, over the remaining 30 s,
        # with slightly different number of subintervals
        blob.n_e = initial_n_e
        TimeEvolution(blob, 30 * u.s, synchrotron_loss(synch)).eval_with_fixed_intervals(intervals_count=10)
        TimeEvolution(blob, 30 * u.s, synchrotron_loss(synch)).eval_with_fixed_intervals(intervals_count=8)
        eval_2 = blob.n_e

        gamma_min = eval_1.gamma_min
        gamma_max = eval_1.gamma_max
        gammas = np.logspace(np.log10(gamma_min), np.log10(gamma_max))
        assert u.allclose(
            eval_1.evaluate(gammas, 1, gamma_min, gamma_max),
            eval_2.evaluate(gammas, 1, gamma_min, gamma_max),
            0.001)

    def test_total_number_of_electrons_should_stay_after_time_evolution(self):
        r_b = 1e16 * u.cm
        n_e = PowerLaw.from_total_energy(
            1e48 * u.erg,
            4 / 3 * np.pi * r_b ** 3,
            p=2.0,
            gamma_min=1e2,
            gamma_max=1e7,
            mass=m_e,
        )

        blob = Blob(r_b, n_e=n_e)
        synch = Synchrotron(blob)
        ssc = SynchrotronSelfCompton(blob)

        time = 6 * u.s
        intervals = 300

        electrons_before = n_e.integrate()

        TimeEvolution(blob, time, synchrotron_loss(synch)).eval_with_fixed_intervals(
                                                      intervals_count=intervals, max_change_per_interval=0.5)
        electrons_after_sync = blob.n_e.integrate()

        blob.n_e = n_e
        TimeEvolution(blob, time, ssc_thomson_limit_loss(ssc)).eval_with_fixed_intervals(
                                                      intervals_count=intervals, max_change_per_interval=0.5)
        electrons_after_ssc_th = blob.n_e.integrate()

        blob.n_e = n_e
        TimeEvolution(blob, time, ssc_loss(ssc)).eval_with_fixed_intervals(
                                                      intervals_count=intervals, max_change_per_interval=0.5)
        electrons_after_ssc = blob.n_e.integrate()

        assert u.isclose(electrons_before, electrons_after_sync, rtol=0.005)
        assert u.isclose(electrons_before, electrons_after_ssc_th, rtol=0.005)
        assert u.isclose(electrons_before, electrons_after_ssc, rtol=0.005)

    def test_ssc_losses_in_thompson_limit(self):
        r_b = 1e16 * u.cm
        n_e = PowerLaw.from_total_energy(
            1e30 * u.erg,
            4 / 3 * np.pi * r_b ** 3,
            p=2.8,
            gamma_min=1e1,
            gamma_max=1e3,
            mass=m_e,
        )

        blob = Blob(r_b, n_e=n_e)
        ssc = SynchrotronSelfCompton(blob)

        thomson = ssc.electron_energy_loss_rate_thomson(np.logspace(1, 3, 200)).to("erg/s")
        ssc = ssc.electron_energy_loss_rate(np.logspace(1, 3, 200)).to("erg/s")

        assert u.allclose(thomson, ssc, 0.01)

    def test_compare_numerical_results_with_analytical_calculation_ssc_thomson(self):
        """Test time evolution of spectral electron density for SSC energy losses in Thomson range.
         Similar to the corresponding test for Synchrotron losses."""
        time = 1e26 * u.s # energy losses are very low, of the order of 1e-33 erg/s,
                          # so we need to multiply them by a long time to observe any change in the electron distribution
        steps = 1
        p = 0.5
        r_b = 1e16 * u.cm
        gamma_min = 1e3
        gamma_max = 1e4
        n_e = PowerLaw.from_total_energy(
            1e20 * u.erg,
            4 / 3 * np.pi * r_b ** 3,
            p=p,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            mass=m_e,
        )
        k = n_e.k
        blob = Blob(r_b, n_e=n_e, B=10*u.G)
        initial_gammas = blob.gamma_e

        ssc = SynchrotronSelfCompton(blob)
        TimeEvolution(blob, time, ssc_loss(ssc)).eval_with_fixed_intervals(intervals_count=steps)
        evaluated_n_e: InterpolatedDistribution = blob.n_e

        def gamma_before(prefactor, gamma_after_time, before_time):
            return 1 / ((1 / gamma_after_time) - before_time * prefactor() / (m_e * c ** 2))

        def gamma_before_ssc_th(gamma_after_time, before_time):
            return gamma_before(ssc._electron_energy_loss_thomson_formula_prefactor, gamma_after_time, before_time)

        # map result gammas to original gammas and assert they are equal
        for evaluated_gamma, initial_gamma in zip(evaluated_n_e.gamma_input, initial_gammas):
            assert u.isclose(
                gamma_before_ssc_th(evaluated_gamma, time),
                initial_gamma,
                rtol=0.001
            )

        # split the result into log-equal intervals, integrate them, and compare with the analytical result
        logspace = np.logspace(np.log10(n_e.gamma_min), np.log10(evaluated_n_e.gamma_max), 20)
        for gamma_min, gamma_max in zip(logspace, logspace[1:]):
            integral = evaluated_n_e.integrate(gamma_min, gamma_max)
            assert u.isclose(
                integral,
                power_law_integral(k, p, gamma_before_ssc_th(gamma_min, time), gamma_before_ssc_th(gamma_max, time)),
                rtol=0.001
            )

    def test_combined_energy_loss_from_two_processes(self):
        """Test time evolution of combined two fake processes - one decreases energy by 20%, another increases by 5%.
           Hence, the combined process in case of the Euler method should lose exactly 15% in each step"""
        blob = Blob(n_e= PowerLaw())
        initial_gammas = blob.gamma_e
        decrease_by_20_perc = lambda g: -1 * (g * mec2).to("erg") * 0.2 / u.s
        increase_by_5_perc = lambda g: -1 * (g * mec2) * -0.05 / u.s
        number_of_steps = 10
        TimeEvolution(blob, number_of_steps * u.s, [decrease_by_20_perc, increase_by_5_perc]).eval_with_fixed_intervals(
                                                      intervals_count=number_of_steps, method="Euler", max_change_per_interval=0.2)
        euler_expected = initial_gammas * (0.85 ** number_of_steps)
        assert u.allclose(blob.gamma_e, euler_expected)

    def test_heun_method_compared_to_euler(self):
        """ Heun method should give better results than Euler method (note: Heun method makes calculations twice,
            so it only makes sense to compare them with 2x more Euler steps)
        """
        time = 60 * u.s
        steps_euler = 600
        steps_heun = steps_euler // 2

        blob1 = Blob(n_e=PowerLaw())
        initial_gamma = blob1.gamma_e
        synch1 = Synchrotron(blob1)
        TimeEvolution(blob1, time, synchrotron_loss(synch1)).eval_with_fixed_intervals(intervals_count=steps_euler, method="Euler")
        euler_reversed_analytically = gamma_before_time(synch1._electron_energy_loss_formula_prefactor, blob1.gamma_e, time)

        blob2 = Blob(n_e=PowerLaw())
        synch2 = Synchrotron(blob2)
        TimeEvolution(blob2, time, synchrotron_loss(synch2)).eval_with_fixed_intervals(intervals_count=steps_heun, method="Heun")
        heun_reversed_analytically = gamma_before_time(synch2._electron_energy_loss_formula_prefactor, blob2.gamma_e, time)

        errors_heun = np.abs((heun_reversed_analytically - initial_gamma) / initial_gamma)
        errors_euler = np.abs((euler_reversed_analytically - initial_gamma) / initial_gamma)
        average_error_euler = np.average(errors_euler)
        average_error_heun = np.average(errors_heun)
        print("Average error Euler method", average_error_euler)
        print("Average error Heun  method", average_error_heun)
        assert average_error_heun < average_error_euler
        assert np.alltrue(errors_heun <= errors_euler)

    def test_heun_method_compared_to_euler_for_automatic_intervals_method(self):
        """ Heun method should give better results than Euler method.
            Interestingly, using Heun instead of Euler method, gives lower errors of the final result
            even when the step threshold is 2 orders of magnitude less restrictive
        """
        time = 60 * u.s
        precision_euler = 0.00001
        precision_heun  = 0.001

        blob1 = Blob(n_e=PowerLaw())
        initial_gamma = blob1.gamma_e
        synch1 = Synchrotron(blob1)
        TimeEvolution(blob1, time, synchrotron_loss(synch1)).eval_with_automatic_intervals(max_change_per_interval=precision_euler,
                                                           method="Euler")
        euler_reversed_analytically = gamma_before_time(synch1._electron_energy_loss_formula_prefactor,
                                                        blob1.gamma_e, time)

        blob2 = Blob(n_e=PowerLaw())
        synch2 = Synchrotron(blob2)
        TimeEvolution(blob2, time, synchrotron_loss(synch2)).eval_with_automatic_intervals(max_change_per_interval=precision_heun,
                                                           method="Heun")
        heun_reversed_analytically = gamma_before_time(synch2._electron_energy_loss_formula_prefactor,
                                                       blob2.gamma_e, time)

        errors_euler = np.abs((euler_reversed_analytically - initial_gamma) / initial_gamma)
        errors_heun = np.abs((heun_reversed_analytically - initial_gamma) / initial_gamma)
        average_error_euler = np.average(errors_euler)
        average_error_heun = np.average(errors_heun)
        print("Average error Euler method", average_error_euler)
        print("Average error Heun  method", average_error_heun)
        assert average_error_heun < average_error_euler
        assert np.alltrue(errors_heun <= errors_euler)

    def test_automatic_and_fixed_intervals_approach_with_single_loop(self):
        """ Automatic intervals approach with threshold low enough should use just one run of calculations, so should give the same result
            as fixed intervals method with one subinterval
        """
        time = 60 * u.s

        blob1 = Blob(n_e=PowerLaw())
        synch1 = Synchrotron(blob1)
        TimeEvolution(blob1, time, synchrotron_loss(synch1)).eval_with_fixed_intervals(intervals_count=1,
                                                       method="Euler")

        blob2 = Blob(n_e=PowerLaw())
        synch2 = Synchrotron(blob2)
        TimeEvolution(blob2, time, synchrotron_loss(synch2)).eval_with_automatic_intervals(max_change_per_interval=0.01,
                                                           method="Euler")

        assert np.all(np.isclose(blob1.n_e.gamma_input, blob2.n_e.gamma_input, rtol=0.000001))

