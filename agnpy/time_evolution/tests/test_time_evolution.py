
import numpy as np
import astropy.units as u
import pandas as pd
import pytest
from astropy.constants import m_e, m_p, c, e
from astropy.coordinates import Distance
from copy import deepcopy
from math import pi
from agnpy import Blob, Synchrotron, SynchrotronSelfCompton, SpectralConstraints, EmptyDistribution
from agnpy.time_evolution._time_evolution_utils import to_total_energy_gev_sqr, DistributionToSinglePointCollapseError
from agnpy.time_evolution.time_evolution import TimeEvolution, synchrotron_loss, ssc_loss, ssc_thomson_limit_loss, \
    fermi_acceleration
from agnpy.spectra import (
    PowerLaw,
    BrokenPowerLaw,
    InterpolatedDistribution,
)
from agnpy.utils.conversion import mec2
from agnpy.utils.math import power_law_integral

def assert_series_equal_ignore_pos(s1, s2, ignore_pos, **kwargs):
    s1_cmp = s1.drop(s1.index[ignore_pos])
    s2_cmp = s2.drop(s2.index[ignore_pos])
    pd.testing.assert_series_equal(s1_cmp, s2_cmp, **kwargs)

def gamma_before_time(prefactor, gamma_after_time, time):
    """
    Reverse-time calculation of the gamma value before the energy loss from formula -dE/dt ~ (E ** 2)
    Applicable to synchrotron and Thomson losses
    """
    return 1 / (1 / gamma_after_time - time * prefactor() / mec2)

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
        TimeEvolution(blob, time, synchrotron_loss(synch), step_duration=time/steps, max_energy_change_per_interval=0.1).evaluate()
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
        blob = Blob(n_e=initial_n_e, B=1 * u.G)
        synch = Synchrotron(blob)

        # iterate over 60 s in 20 steps
        TimeEvolution(blob, 60 * u.s, synchrotron_loss(synch), step_duration=3 * u.s).evaluate()
        eval_1 = deepcopy(blob.n_e)
        # iterate first over 30 s, and then, starting with interpolated distribution, over the remaining 30 s,
        # with slightly different number of subintervals
        blob.n_e = initial_n_e
        TimeEvolution(blob, 30 * u.s, synchrotron_loss(synch), step_duration=3 * u.s).evaluate()
        TimeEvolution(blob, 30 * u.s, synchrotron_loss(synch), step_duration=3.5 * u.s).evaluate()
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
        step_time = time/300

        electrons_before = n_e.integrate()

        TimeEvolution(blob, time, synchrotron_loss(synch), step_duration=step_time, max_energy_change_per_interval=0.5,
                      max_density_change_per_interval=10.0).evaluate()
        electrons_after_sync = blob.n_e.integrate()

        blob.n_e = n_e
        TimeEvolution(blob, time, ssc_thomson_limit_loss(ssc), step_duration=step_time, max_energy_change_per_interval=0.5,
                      max_density_change_per_interval=10.0).evaluate()
        electrons_after_ssc_th = blob.n_e.integrate()

        blob.n_e = n_e
        TimeEvolution(blob, time, ssc_loss(ssc), step_duration=step_time, max_energy_change_per_interval=0.5,
                      max_density_change_per_interval=10.0).evaluate()
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
        TimeEvolution(blob, time, ssc_loss(ssc), step_duration=time).evaluate()
        evaluated_n_e: InterpolatedDistribution = blob.n_e

        def gamma_before_ssc_th(gamma_after_time, before_time):
            return gamma_before_time(ssc._electron_energy_loss_thomson_formula_prefactor, gamma_after_time, before_time)

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
        decrease_by_20_perc = lambda args: -1 * (args.gamma * mec2).to("erg") * 0.2 / u.s
        increase_by_5_perc = lambda args: -1 * (args.gamma * mec2) * -0.05 / u.s
        number_of_steps = 10
        TimeEvolution(blob, number_of_steps * u.s, [decrease_by_20_perc, increase_by_5_perc], method="euler",
                      max_energy_change_per_interval=0.2, max_density_change_per_interval = 0.5, step_duration=1*u.s).evaluate()
        euler_expected = initial_gammas * (0.85 ** number_of_steps)
        assert u.allclose(blob.gamma_e, euler_expected)

    def test_heun_method_compared_to_euler(self):
        """ Heun method should give better results than Euler method (note: Heun method makes calculations twice,
            so it only makes sense to compare them with 2x shorter Euler steps)
        """
        time = 60 * u.s

        blob1 = Blob(n_e=PowerLaw())
        initial_gamma = blob1.gamma_e
        synch1 = Synchrotron(blob1)
        TimeEvolution(blob1, time, synchrotron_loss(synch1), method="euler", step_duration=0.1*u.s).evaluate()
        euler_reversed_analytically = gamma_before_time(synch1._electron_energy_loss_formula_prefactor, blob1.gamma_e, time)

        blob2 = Blob(n_e=PowerLaw())
        synch2 = Synchrotron(blob2)
        TimeEvolution(blob2, time, synchrotron_loss(synch2), method="heun", step_duration=0.2*u.s).evaluate()
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
            Interestingly, in this example, using Heun instead of Euler method, gives lower errors of the final result
            even when the step threshold is 2 orders of magnitude less restrictive,
            and, as a result, needs 2 orders of magnitude less iterations (~20 vs ~2000) - so is faster than Euler!
        """
        time = 60 * u.s
        precision_euler = 0.00001
        precision_heun  = 0.001

        blob1 = Blob(n_e=PowerLaw())
        initial_gamma = blob1.gamma_e
        synch1 = Synchrotron(blob1)
        TimeEvolution(blob1, time, synchrotron_loss(synch1), max_energy_change_per_interval=precision_euler,
            method="euler", optimize_recalculating_slow_rates=False).evaluate()
        euler_reversed_analytically = gamma_before_time(synch1._electron_energy_loss_formula_prefactor,
                                                        blob1.gamma_e, time)

        blob2 = Blob(n_e=PowerLaw())
        synch2 = Synchrotron(blob2)
        TimeEvolution(blob2, time, synchrotron_loss(synch2), max_energy_change_per_interval=precision_heun,
            method="heun", optimize_recalculating_slow_rates=False).evaluate()
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
        """ Automatic intervals approach with an "undemanding" threshold should use just one run of calculations, so should give the same result
            as fixed intervals method with one subinterval
        """
        time = 60 * u.s

        blob1 = Blob(n_e=PowerLaw())
        synch1 = Synchrotron(blob1)
        TimeEvolution(blob1, time, synchrotron_loss(synch1), step_duration=time).evaluate()

        blob2 = Blob(n_e=PowerLaw())
        synch2 = Synchrotron(blob2)
        TimeEvolution(blob2, time, synchrotron_loss(synch2), max_energy_change_per_interval=0.5).evaluate()

        assert np.all(np.isclose(blob1.n_e.gamma_input, blob2.n_e.gamma_input, rtol=0.000001))

    def test_simulation_trending_to_delta_function(self):
        """ Simulate the process with acceleration dominating at lower energies, and losses dominating at higher energies,
            causing the distribution converging to the delta function
        """
        time = 280 * u.s

        blob = Blob(n_e=PowerLaw(), xi=0.1)
        delta_function_energy = SpectralConstraints(blob).gamma_max_synch
        blob.n_e.gamma_min = delta_function_energy / 2
        blob.n_e.gamma_max = delta_function_energy * 2

        # acceleration formula from SpectralConstraints.gamma_max_synch
        fermi_acc = lambda args: (blob.xi * blob.B_cgs * c * e.gauss) * np.ones_like(args.gamma)

        synch = Synchrotron(blob)
        with pytest.raises(DistributionToSinglePointCollapseError) as ex:
            TimeEvolution(blob, time, [synchrotron_loss(synch), fermi_acc],
                           max_energy_change_per_interval=0.1, max_density_change_per_interval=1.0).evaluate()
        assert np.isclose(ex.value.gamma_point, delta_function_energy, rtol=0.001)

    def test_synch_losses_with_continues_injection_and_acceleration(self):
        """ Simulate the process with Fermi acceleration, synch losses, and contiunous injection, and compare
        with results obtained using different simulation methods
        """

        # load reference data
        ref_data=[]
        offset=2
        nrows = 45
        for t in range(5):
            ref = pd.read_csv(
                "agnpy/time_evolution/tests/out_inject_continuous_B=10.10_tacc=5.000000e+00.txt",
                sep=r"\s+", header=None, skiprows=offset, nrows=nrows, usecols=[1, 2, 3, 4]
            )
            ref.columns = ["eed_x", "eed_y", "sed_x", "sed_y"]
            ref_data.append(ref)
            offset += (nrows+1)

        # prepare objects
        R_b = (100 * c * u.s).to(u.cm)
        V_b = 4 / 3 * np.pi * R_b ** 3
        distance = Distance(z=0.01)
        delta_D = 1.001
        Gamma = 1.001
        B = 10.1 * u.G
        tacc = 5 * u.s
        blob = Blob(R_b, distance.z, delta_D, Gamma, B, n_e=EmptyDistribution(gamma_min=10, gamma_max=1e3))
        synch = Synchrotron(blob)
        injection_dist = PowerLaw.from_total_energy(100000 * u.GeV, V_b, gamma_min=10, gamma_max=1000, p=2, mass=m_e)
        def injection_function(args):
            return injection_dist(args.gamma) / u.s

        # run simulations for 5 different time scales
        times = [0, 4, 11, 31, 101, 301]
        gamma_array = np.logspace(1, 3, 200)
        for i in range(len(times)-1):
            single_iteration_time = (times[i + 1] - times[i]) * u.s
            _, gamma_array, densities, _, energy_change_rates, _, _ = (TimeEvolution(
                blob,
                single_iteration_time,
                energy_change_functions={"Synch": synchrotron_loss(synch), "Acc": fermi_acceleration(tacc)},
                injection_functions_abs={"Injection": injection_function},
                initial_gamma_array=gamma_array,
                gamma_bounds=(1e1, 1e7),
                method="euler",
                max_density_change_per_interval=1.0,
                max_energy_change_per_interval=0.1,
                max_injection_per_interval=50,
            ).evaluate())

            distribution = blob.n_e
            ev_per_gamma = mec2.to("eV")
            ref = ref_data[i]
            eed_x = ref.eed_x.values
            density = distribution(eed_x)
            eed_y = to_total_energy_gev_sqr(eed_x, density, V_b)
            sed_x = ref.sed_x.values

            if i < 4:
                sed_nu_fnu = synch.sed_flux(sed_x * ev_per_gamma.to("Hz", equivalencies=u.spectral()))
            else:
                # for the last iteration, the default gamma points used in the SED calculation tend to cause significant
                # interpolation error coming from a density peak gathering at the last bin; to avoid this error, we must use
                # exactly the same interpolation points as the reference data
                blob.n_e = InterpolatedDistribution(eed_x, density)
                sed_nu_fnu = synch.sed_flux(sed_x * ev_per_gamma.to("Hz", equivalencies=u.spectral()))

            sed_y = (sed_nu_fnu * 4 * pi * distance**2).to("GeV/s")
            df = pd.DataFrame({
                "eed_y": eed_y.value,
                "sed_y": sed_y.value
            })
            df.columns = ["eed_y", "sed_y"]

            ignore_pos = [27,30] # positions where sharp density changes occur
            assert_series_equal_ignore_pos(ref.eed_y, df.eed_y, ignore_pos, rtol=0.4)
            ignore_pos = [31, 40]  # positions where agnpy returns already -inf but reference data still haves some very small value
            assert_series_equal_ignore_pos(ref.sed_y.apply(np.log10), df.sed_y.apply(np.log10), ignore_pos, rtol=0.5, atol=0.6)


    def test_synch_losses_with_continues_injection_and_acceleration_with_two_zones(self):
        """ Simulate the process with Fermi acceleration and synch losses in one zone,
        and synch losses only with second zone, with continues constant injection of electrons to the first zone,
        and escape from first to second zone;  then compares with results obtained using different simulation methods
        """

        # load reference data
        ref_data=[]
        my_data=[]
        offset=2
        nrows = 45
        for t in range(4):
            ref = pd.read_csv(
                "agnpy/time_evolution/tests/out_escape_to_blob_B=10.30_tacc=1.000000e+01_tesc=1.000000e+01_no_merging.txt",
                sep=r"\s+", header=None, skiprows=offset, nrows=nrows, usecols=[1, 2, 3], dtype=float,
            )
            ref.columns = ["eed_x", "eed_y_zone1", "eed_y_zone2"]
            ref_data.append(ref)
            offset += (nrows+1)

        # prepare objects
        R_b = (100 * c * u.s).to(u.cm)
        V_b = 4 / 3 * np.pi * R_b ** 3
        distance = Distance(z=0.01)
        delta_D = 1.001
        Gamma = 1.001
        B = 10.3 * u.G
        tacc = 10 * u.s
        tesc = 10 * u.s
        blob = Blob(R_b, distance.z, delta_D, Gamma, B, n_e=EmptyDistribution(gamma_min=10, gamma_max=1e3))
        synch = Synchrotron(blob)
        injection_dist = PowerLaw.from_total_energy(100000 * u.GeV, V_b, gamma_min=10, gamma_max=20, p=2, mass=m_e)
        def injection_function_zone1(args):
            return injection_dist(args.gamma) / u.s

        def escape_fn(args):
            return -1 * np.ones(args.densities.shape) / tesc

        def injection_function_zone2(args):
            return -1 * escape_fn(args) * args.densities * args.density_subgroups[0]

        # run simulations over different time scales
        times = [0, 11, 31, 101, 301]
        number_of_bins = 11
        density_groups = np.array([np.ones(number_of_bins, dtype=float), np.zeros(number_of_bins, dtype=float)])
        gamma_array = np.logspace(np.log10(10), np.log10(20), number_of_bins)
        for i in range(len(times)-1):
            single_iteration_time = (times[i + 1] - times[i]) * u.s
            _, gamma_array, densities, density_groups, energy_change_rates, _, _ = (TimeEvolution(
                blob,
                single_iteration_time,
                energy_change_functions={"Synch": synchrotron_loss(synch), "Acc": fermi_acceleration(tacc)},
                injection_functions_abs={"Zone1-inj": injection_function_zone1, "Zone2-inj": injection_function_zone2},
                injection_functions_rel={"Zone1-esc": escape_fn},
                subgroups=[["Synch", "Acc", "Zone1-inj", "Zone1-esc"],
                           ["Synch", "Zone2-inj"]],
                subgroups_initial_density=density_groups,
                initial_gamma_array=gamma_array,
                gamma_bounds=(1e1, 1e9),
                method="euler",
                step_duration=0.2 * u.s,
                max_bin_creep_from_bounds=0.005,
            ).evaluate())

            distribution_zone1 = InterpolatedDistribution(gamma_array, densities * density_groups[0])
            distribution_zone2 = InterpolatedDistribution(gamma_array, densities * density_groups[1])
            ref = ref_data[i]
            eed_x = ref.eed_x.values
            eed_y_zone1 = to_total_energy_gev_sqr(eed_x, distribution_zone1(eed_x), V_b)
            eed_y_zone2 = to_total_energy_gev_sqr(eed_x, distribution_zone2(eed_x), V_b)


            df = pd.DataFrame({
                "eed_x": ref.eed_x.values,
                "eed_y_zone1": eed_y_zone1.value,
                "eed_y_zone2": eed_y_zone2.value
            })
            my_data.append(df)

            ignore_pos = [8,13,29] # positions where sharp density changes occur at different stages; for simplicity, just filter them all out
            assert_series_equal_ignore_pos(ref.eed_y_zone1, df.eed_y_zone1, ignore_pos, rtol=0.3)
            assert_series_equal_ignore_pos(ref.eed_y_zone2, df.eed_y_zone2, ignore_pos, rtol=0.3)
