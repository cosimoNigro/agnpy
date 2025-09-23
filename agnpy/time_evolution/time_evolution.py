import logging
import astropy.units as u
from agnpy import Blob, Synchrotron, SynchrotronSelfCompton
from agnpy.time_evolution._time_evolution_utils import *
from agnpy.time_evolution.types import *
from agnpy.utils.conversion import mec2
from astropy.constants import c, e
from astropy.units import Quantity
from numpy._typing import NDArray
from typing import Iterable, Tuple

log = logging.getLogger(__name__)

erg_per_s = u.Unit("erg s-1")
per_s_cm3 = u.Unit("s-1 cm-3")
per_s = u.Unit("s-1")

def synchrotron_loss(sync: Synchrotron) -> EnergyChangeFn:
    return lambda gamma: sync.electron_energy_loss_rate(gamma) * -1

def ssc_loss(sync: SynchrotronSelfCompton) -> EnergyChangeFn:
    return lambda gamma: sync.electron_energy_loss_rate(gamma) * -1

def ssc_thomson_limit_loss(sync: SynchrotronSelfCompton) -> EnergyChangeFn:
    return lambda gamma: sync.electron_energy_loss_rate_thomson(gamma) * -1

def fermi_acceleration(t_acc: Quantity) -> EnergyChangeFn:
    return lambda gamma: to_erg(gamma)/t_acc

def bohm_diffusion_losses(blob) -> InjectionRelFn:
    def bohm_diffusion_loss_formula(gamma):
        r_l = (gamma * mec2 / e.gauss / blob.B_cgs).to("cm") # Larmor radius
        d_b = r_l * c / 3
        t = (blob.R_b**2 / (2*d_b)).to("s")
        return -1/t
    return bohm_diffusion_loss_formula

class TimeEvolution:
    """
    Evaluates the change of electron spectral density inside the blob.

    Parameters
    ----------
    blob :
        The blob for which the time evolution will be performed. As a result of the time evolution,
        the blob.n_e will be replaced with the InterpolatedDistribution.
    total_time :
         Total time for the calculation
    energy_change_functions :
         A function, or an array of functions, to be used for calculation of energy change rate for gamma values.
         For energy gain processes, function should return positive values; for loss, negative values.
    injection_functions_rel :
         A function, or an array of functions, to be used for calculation of electron density change caused by
         direct particle injection (or escape). This function is used for relative density change.
         For particle injection, the return coefficient should be >0, and for escape, <0
    injection_functions_abs :
         A function, or an array of functions, to be used for calculation of electron density change caused by
         direct particle injection (or escape). This function is used for absolute density change.
         For particle injection, the return density change rate should be >0, and for escape, <0
    initial_gamma_array :
        Optional array of gamma values (electron energies); if empty, blob.gamma_e() will be used.
        This is primarily intended for cases when simulation is done in multiple steps, and you want to use
        the final bins from one step as a starting point for the next step
    gamma_bounds :
        Optional tuple of gamma lower and upper bounds for calculations
    max_bin_creep_from_bounds :
        Maximum distance (in log10) the first or last bin can creep away from the gamma bounds towards the center of distribution,
        before the new bin is injected at the boundary. Applicable only if gamma_bounds was set.
    min_bin_distance :
        Minimum distance between the bins, below which the bins will be merged.
    max_energy_change_per_interval :
        maximum relative change of the electron energy allowed in one time interval for each bin
    max_density_change_per_interval :
        maximum relative change of the electron density allowed in one time interval for each bin
    max_injection_per_interval :
        maximum absolute injection or escape of particles in one time interval for each bin (in cm-3)
    method :
        numerical method for calculating energy evolution; accepted values: "euler" (faster) or "heun" (more precise)
    distribution_change_callback :
        This optional function will be called each time the blob's electron distribution has been updated.
        You can use it, for example, for updating the distribution plot while the simulation is running.
    """

    def __init__(self,
                 blob: Blob,
                 total_time: Quantity,
                 energy_change_functions: EnergyChangeFns,
                 injection_functions_rel: InjectionRelFns = None,
                 injection_functions_abs: InjectionAbsFns = None,
                 initial_gamma_array: NDArray[np.floating] = None,
                 gamma_bounds: Tuple[float, float] = None,
                 max_bin_creep_from_bounds: float = 0.1,
                 min_bin_distance : float = 1.5e-3,
                 max_energy_change_per_interval: float = 0.01,
                 max_density_change_per_interval: float = 0.1,
                 max_injection_per_interval: float = 1.0,
                 method: NumericalMethod = "euler",
                 distribution_change_callback: CallbackFnType = None):
        self._blob = blob
        self._total_time_sec = total_time.to("s")
        self._energy_change_functions = energy_change_functions if isinstance(energy_change_functions, dict) \
            else {str(v): v for v in energy_change_functions} if isinstance(energy_change_functions, Iterable) \
            else {str(energy_change_functions): energy_change_functions}
        self._injection_functions_rel = injection_functions_rel if isinstance(injection_functions_rel, dict) \
            else {str(v): v for v in injection_functions_rel} if isinstance(injection_functions_rel, Iterable) \
            else {str(injection_functions_rel): injection_functions_rel} if injection_functions_rel is not None \
            else {}
        self._injection_functions_abs = injection_functions_abs if isinstance(injection_functions_abs, dict) \
            else {str(v): v for v in injection_functions_abs} if isinstance(injection_functions_abs, Iterable) \
            else {str(injection_functions_abs): injection_functions_abs} if injection_functions_abs is not None \
            else {}
        duplicated_keys = (set(self._energy_change_functions) & set(self._injection_functions_rel) |
                       set(self._energy_change_functions) & set(self._injection_functions_abs) |
                       set(self._injection_functions_rel) & set(self._injection_functions_abs))
        if duplicated_keys:
            raise ValueError("Found duplicate keys of energy change or injection functions: " + str(duplicated_keys))
        self._initial_gamma_array = initial_gamma_array
        self._gamma_bounds = gamma_bounds
        if gamma_bounds is not None and initial_gamma_array is not None and np.any(initial_gamma_array < gamma_bounds[0]):
            raise ValueError("initial_gamma_array might not contain elements smaller than gamma_min")
        if gamma_bounds is not None and initial_gamma_array is not None and np.any(initial_gamma_array > gamma_bounds[1]):
            raise ValueError("initial_gamma_array might not contain elements greater than gamma_max")
        if gamma_bounds is not None and initial_gamma_array is None:
            self._blob.n_e.gamma_min = gamma_bounds[0]
            self._blob.n_e.gamma_max = gamma_bounds[1]
        self._max_bin_creep_from_bounds = max_bin_creep_from_bounds
        self._min_bin_distance = min_bin_distance
        self._max_energy_change_per_interval = max_energy_change_per_interval
        self._max_density_change_per_interval = max_density_change_per_interval
        self._max_injection_per_interval = max_injection_per_interval * u.Unit("cm-3")
        valid_methods = {"heun", "euler"}
        if method.lower() not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Expected one of: {', '.join(valid_methods)}")
        self._method = method
        self._distribution_change_callback = distribution_change_callback

    def _calculate_initial_values(self):
        gm_bins_lb = self._initial_gamma_array if self._initial_gamma_array is not None else self._blob.gamma_e
        gm_bins_up = np.zeros_like(gm_bins_lb)
        gm_bins = np.array([gm_bins_lb, gm_bins_up])
        update_bin_ub(gm_bins)
        density = self._blob.n_e(gm_bins_lb)
        en_chg_rates = recalc_new_rates(gm_bins, self._energy_change_functions, erg_per_s)
        abs_injection_rates = recalc_new_rates(gm_bins_lb, self._injection_functions_abs, per_s_cm3)
        rel_injection_rates = recalc_new_rates(gm_bins_lb, self._injection_functions_rel, per_s)
        return gm_bins, density, en_chg_rates, abs_injection_rates, rel_injection_rates

    def eval_with_fixed_intervals(self, intervals_count: int) -> tuple[NDArray,Quantity,dict[str,Quantity],dict[str,Quantity],dict[str,Quantity]]:
        """
        Performs the time evolution of the electron distribution inside the blob, repeating the calculation in the
        equal-length time intervals. If the max_energy_change_per_interval or max_density_change_per_interval is exceeded
        in such a step, the error will be raised.

        Parameters
        ----------
        intervals_count : int
            the calculation will be performed in the intervals_count steps of equal duration (time/intervals_count)

        Side Effects
        ------------
        Replaces the blob.n_e with the new InterpolatedDistribution

        Returns
        ------------
        A tuple consisting of: new gamma array, new densities, new change rates, new absolute injection rates, new relative injection rates
        """

        if intervals_count <= 0:
            raise ValueError("intervals_count must be > 0")
        unit_time_interval = (self._total_time_sec / intervals_count).to("s")
        gm_bins, density, en_chg_rates, abs_injection_rates, rel_injection_rates = self._calculate_initial_values()
        for counter in range(1, intervals_count + 1):
            gm_bins, density, en_chg_rates, abs_injection_rates, rel_injection_rates = (
                self._do_iterative(gm_bins, density, en_chg_rates, abs_injection_rates, rel_injection_rates,
                unit_time_interval, counter, intervals_count))
        en_chg_rates_lb = energy_changes_lb(en_chg_rates)
        mask = density > 0
        return gm_bins[0][mask], density[mask], remap(mask,en_chg_rates_lb), remap(mask,abs_injection_rates), remap(mask,rel_injection_rates)

    def _do_iterative(self, gm_bins, density, en_chg_rates, abs_injection_rates, rel_injection_rates, unit_time_interval_sec, iteration, intervals_count):
        """
        Iterative algorithm for evaluating the electron energy and density. For each gamma point it creates a narrow bin,
        calculates the energy change for a start and an end of the bin, and scales up density by the bin narrowing factor.
        """

        # ======== stage 1: check change rates ========
        en_bins, abs_en_changes, total_injection, new_low_change_rates_mask = self._check_change_rates(
            abs_injection_rates, density, en_chg_rates, gm_bins, rel_injection_rates, unit_time_interval_sec)
        if not np.all(new_low_change_rates_mask):
            raise ValueError("Energy change or injection formula returned too big value. Use shorter time ranges.")
        log.info("%*d / %d", len(str(intervals_count)), iteration, intervals_count)

        # ======== stage 2: apply time evaluation ========
        new_gm_bins, new_dens = recalc_gamma_bins_and_density(en_bins, abs_en_changes, density)
        new_dens = new_dens + total_injection
        update_distribution(new_gm_bins[0], new_dens, self._blob)

        # ======== stage 3: for Heun method, recalculate bins ========
        if self._method.lower() == "heun":
            en_chg_rates_recalced = recalc_new_rates(new_gm_bins, self._energy_change_functions, erg_per_s)
            averaged_en_chg_rates = {key: (en_chg_rates_recalced[key] + en_chg_rates[key]) / 2
                 for key in en_chg_rates_recalced}
            abs_en_chg_recalc = sum_change_rates(averaged_en_chg_rates, new_gm_bins.shape, erg_per_s) * unit_time_interval_sec
            new_gm_bins, new_dens = recalc_gamma_bins_and_density(en_bins, abs_en_chg_recalc, density)
            new_dens = new_dens + total_injection
            update_distribution(new_gm_bins[0], new_dens, self._blob)
            en_chg_rates = averaged_en_chg_rates

        # ======== stage 4: sort and merge bins if needed ========
        new_gm_bins, new_dens, sort_and_merge_mapping = self._sort_and_merge_bins(new_gm_bins, new_dens)
        en_chg_rates = remap(sort_and_merge_mapping, en_chg_rates)

        # ======== stage 5: remove bins falling behind the edges ========
        if self._gamma_bounds is not None:
            new_gm_bins, mapping_wo_bins_beyond_bounds = remove_gamma_beyond_bounds(new_gm_bins, self._gamma_bounds)
            new_dens = remap(mapping_wo_bins_beyond_bounds, new_dens)
            en_chg_rates = remap(mapping_wo_bins_beyond_bounds, en_chg_rates, np.nan)

        # ======== stage 6: add bins at edges if they move towards the center ========
        if self._gamma_bounds is not None:
            new_gm_bins, mapping_with_new_bins = add_boundary_bins(
                new_gm_bins, en_chg_rates, self._gamma_bounds, self._max_bin_creep_from_bounds)
            new_dens = remap(mapping_with_new_bins, new_dens, 0 * u.Unit("cm-3"))

        # ======== stage 7: recalculate change rates ========
        update_bin_ub(new_gm_bins)
        en_chg_rates = recalc_new_rates(new_gm_bins, self._energy_change_functions, erg_per_s)
        abs_injection_rates = recalc_new_rates(new_gm_bins[0], self._injection_functions_abs, per_s_cm3)
        rel_injection_rates = recalc_new_rates(new_gm_bins[0], self._injection_functions_rel, per_s)

        # ======== stage 8: call the callback function =========
        if self._distribution_change_callback is not None:
            total_time = unit_time_interval_sec * iteration
            en_chg_rates_lb = energy_changes_lb(en_chg_rates)
            mask = new_dens > 0
            self._distribution_change_callback(CallbackParams(total_time, new_gm_bins[0][mask], new_dens[mask],
                   remap(mask,en_chg_rates_lb), remap(mask,abs_injection_rates), remap(mask,rel_injection_rates)))

        return new_gm_bins, new_dens, en_chg_rates, abs_injection_rates, rel_injection_rates

    def eval_with_automatic_intervals(self) -> tuple[NDArray,Quantity,dict[str,Quantity],dict[str,Quantity],dict[str,Quantity]]:
        """
        Performs the time evolution of the electron distribution inside the blob, repeating the calculation by the
        automatically selected time intervals such that the energy change in each interval does not exceed the
        max_change_per_interval rate. The automatically selected time interval duration may differ per energy bins
        - shorter intervals will be used for bins where energy change is faster.
        It assumes the energy-change-functions are costly to call so it tries to recalculate them only for bins
        with high energy changes, and reuse earlier calculated values for bins with low energy changes.
        It also assumes the particle-injection functions are cheap to call, so they can be called often and for every bin.

        Side Effects
        ------------
        Replaces the blob.n_e with the new InterpolatedDistribution

        Returns
        ------------
        A tuple consisting of: new gamma array, new densities, new change rates dict, new absolute injection rates, new relative injection rates
        """

        gm_bins, density, en_chg_rates, abs_injection_rates, rel_injection_rates = self._calculate_initial_values()
        empty_mask = np.repeat(False, len(density))
        (gm_bins, density, en_chg_rates, abs_injection_rates, rel_injection_rates, _, _) = self._do_recursive(
            gm_bins, density, en_chg_rates, abs_injection_rates, rel_injection_rates, empty_mask, False, self._total_time_sec, [], 1,)
        en_chg_rates_lb = energy_changes_lb(en_chg_rates)
        mask = density > 0
        return gm_bins[0][mask], density[mask], remap(mask, en_chg_rates_lb), remap(mask, abs_injection_rates), remap(mask, rel_injection_rates)

    def _do_recursive(self, gm_bins, density, en_chg_rates, abs_injection_rates, rel_injection_rates, low_chg_rates_mask, recalc_high_chg, time_sec, prev_times, depth):
        """
        Recursive algorithm for evaluating the electron energy and density.
        Parameters:
            gm_bins - array of shape (2N,) containing interlaced lower and upper bounds of gamma bins
            en_chg_rates - dict with values of array of shape (2N,) consisting of most recently calculated energy change rates corresponding to interlaced gm_bins values
            low_chg_rates_mask - array of shape (N,), a mask of bins that do not need recalculation
            recalc_high_chg - boolean; if false, recalculation of en_chg_rates array is skipped, irrespective of the mask
            time_sec - next time to calculate
            prev_times - list of previous times already calculated (but potentially not fully corrected yet with Heun method)
            depth - recursion depth
        Side Effects:
            Replaces the blob.n_e with the new InterpolatedDistribution
        Returns:
            mapping, gm_bins_lb, gm_bins_ub, new_times, dens, en_chg_rates, abs_injection_rate
        """

        # ======== stage 1: check change rates ========
        en_bins, abs_en_changes, total_injection, new_low_change_rates_mask = self._check_change_rates(
            abs_injection_rates, density, en_chg_rates, gm_bins, rel_injection_rates, time_sec)
        low_chg_rates_flip_mask = ~low_chg_rates_mask & new_low_change_rates_mask
        prev_times_sum = sum(prev_times) if len(prev_times) > 0 else 0 * u.s
        self._log_mask_info(new_low_change_rates_mask, depth, prev_times_sum)

        # ======== stage 2: apply time evaluation, directly or in two recursive substeps ========
        times = prev_times.copy()
        if np.all(new_low_change_rates_mask):
            new_gm_bins, dens_after_en_eval = recalc_gamma_bins_and_density(en_bins, abs_en_changes, density)
            new_dens = dens_after_en_eval + total_injection
            mapping_timeeval = np.arange(len(new_dens))
            times.append(time_sec)
            update_distribution(new_gm_bins[0], new_dens, self._blob)
        else:
            half_time = time_sec / 2
            (gm_bins_halftime, density_halftime, en_chg_rates_halftime, abs_injection_rates_halftime, rel_injection_rates_halftime, mapping_halftime, times)\
                = self._do_recursive(gm_bins,
                                     density,
                                     en_chg_rates,
                                     abs_injection_rates,
                                     rel_injection_rates,
                                     new_low_change_rates_mask,
                                     True,
                                     half_time, times, depth + 1)
            low_chg_rates_mask_halftime = remap(mapping_halftime, new_low_change_rates_mask, False)
            (new_gm_bins, new_dens, en_chg_rates, abs_injection_rates, rel_injection_rates, mapping_2nd_halftime, times) \
                = self._do_recursive(gm_bins_halftime,
                                     density_halftime,
                                     en_chg_rates_halftime,
                                     abs_injection_rates_halftime,
                                     rel_injection_rates_halftime,
                                     low_chg_rates_mask_halftime,
                                     False,
                                     half_time, times, depth + 1)
            mapping_timeeval = combine_mapping(mapping_halftime, mapping_2nd_halftime)
            low_chg_rates_flip_mask = remap(mapping_timeeval, low_chg_rates_flip_mask, False)
            low_chg_rates_mask = remap(mapping_timeeval, low_chg_rates_mask, False)

        # ======== stage 3: for Heun method, recalculate bins covered by the flip mask ========
        if self._method.lower() == "heun" and np.any(low_chg_rates_flip_mask) and not np.all(density == 0):
            en_chg_rates_recalced_twice = recalc_new_rates(new_gm_bins, self._energy_change_functions, erg_per_s, en_chg_rates, low_chg_rates_flip_mask)
            averaged_en_chg_rates = {key: (en_chg_rates[key][..., low_chg_rates_flip_mask] +
                                           en_chg_rates_recalced_twice[key][..., low_chg_rates_flip_mask]) / 2
                                     for key in en_chg_rates_recalced_twice}
            recalc_time_start_idx = -1 * (len(times) - len(prev_times))
            times_to_recalc = times[recalc_time_start_idx:]
            gm_bins_recalc = remap(mapping_timeeval, gm_bins, np.nan)[..., low_chg_rates_flip_mask]
            dens_recalc = remap(mapping_timeeval, density)[low_chg_rates_flip_mask]
            for t in times_to_recalc:
                rel_injection_rates_recalc = recalc_new_rates(gm_bins_recalc[0], self._injection_functions_rel, per_s)
                abs_scaling = sum_change_rates(rel_injection_rates_recalc, gm_bins_recalc.shape[1], per_s) * t
                abs_injection_rates_recalc = recalc_new_rates(gm_bins_recalc[0], self._injection_functions_abs, per_s_cm3)
                abs_injection = sum_change_rates(abs_injection_rates_recalc, gm_bins_recalc.shape[1], per_s_cm3) * t
                total_injection = dens_recalc * abs_scaling + abs_injection
                abs_en_changes = sum_change_rates(averaged_en_chg_rates, gm_bins_recalc.shape, erg_per_s) * t
                gm_bins_recalc, dens_recalc = recalc_gamma_bins_and_density(to_erg(gm_bins_recalc), abs_en_changes, dens_recalc)
                dens_recalc = dens_recalc + total_injection
                if any(np.isnan(dens_recalc)):
                    # in this case, we should roll back the sub-calculation progress and retry with the new mask from the beginning of the current method
                    raise ValueError(
                        "Illegal negative density obtained - the resolution algorithm not yet implemented, please report it to the agnpy maintainers")
            new_gm_bins[..., low_chg_rates_flip_mask] = gm_bins_recalc
            new_dens[low_chg_rates_flip_mask] = dens_recalc
            for k in en_chg_rates.keys():
                en_chg_rates[k][..., low_chg_rates_flip_mask] = averaged_en_chg_rates[k]
            update_distribution(new_gm_bins[0], new_dens, self._blob)

        # ======== stage 4: sort and merge bins if needed ========
        if recalc_high_chg:
            new_gm_bins, new_dens, additional_mapping = self._sort_and_merge_bins(new_gm_bins, new_dens, ~low_chg_rates_mask)
            en_chg_rates = remap(additional_mapping, en_chg_rates)
        else:
            additional_mapping = np.arange(new_gm_bins.shape[1])

        # ======== stage 5: remove bins falling behind the edges ========
        if self._gamma_bounds is not None:
            new_gm_bins, mapping_wo_bins_beyond_bounds = remove_gamma_beyond_bounds(new_gm_bins, self._gamma_bounds)
            new_dens = remap(mapping_wo_bins_beyond_bounds, new_dens)
            en_chg_rates = remap(mapping_wo_bins_beyond_bounds, en_chg_rates, np.nan)
            additional_mapping = combine_mapping(additional_mapping, mapping_wo_bins_beyond_bounds)

        # ======== stage 6: add bins at edges if they move towards the center ========
        if self._gamma_bounds is not None and recalc_high_chg:
            new_gm_bins, mapping_with_new_bins = add_boundary_bins(
                new_gm_bins, en_chg_rates, self._gamma_bounds, self._max_bin_creep_from_bounds)
            new_dens = remap(mapping_with_new_bins, new_dens, 0 * u.Unit("cm-3"))
            en_chg_rates = remap(mapping_with_new_bins, en_chg_rates, np.nan)
            additional_mapping = combine_mapping(additional_mapping, mapping_with_new_bins)

        # ======== stage 7: recalculate change rates if needed, but only for unmasked bins ========
        abs_injection_rates = remap(additional_mapping, abs_injection_rates, np.nan)
        rel_injection_rates = remap(additional_mapping, rel_injection_rates, np.nan)
        if recalc_high_chg:
            update_bin_ub(new_gm_bins)
            recalc_mask = ~ remap(additional_mapping, low_chg_rates_mask, False)
            en_chg_rates = recalc_new_rates(new_gm_bins, self._energy_change_functions, erg_per_s, en_chg_rates, recalc_mask)
            abs_injection_rates = recalc_new_rates(new_gm_bins[0], self._injection_functions_abs, per_s_cm3, abs_injection_rates, recalc_mask)
            rel_injection_rates = recalc_new_rates(new_gm_bins[0], self._injection_functions_rel, per_s, rel_injection_rates, recalc_mask)

        # ======== stage 8: call the callback function =========
        if recalc_high_chg and self._distribution_change_callback is not None:
            total_time = prev_times_sum + time_sec
            en_chg_rates_lb = energy_changes_lb(en_chg_rates)
            mask = new_dens > 0
            self._distribution_change_callback(CallbackParams(total_time, new_gm_bins[0][mask], new_dens[mask],
                   remap(mask,en_chg_rates_lb), remap(mask,abs_injection_rates), remap(mask,rel_injection_rates)))

        final_mapping = combine_mapping(mapping_timeeval, additional_mapping)
        return new_gm_bins, new_dens, en_chg_rates, abs_injection_rates, rel_injection_rates, final_mapping, times

    def _log_mask_info(self, low_change_rates_mask, depth, elapsed_time_sec):
        if log.isEnabledFor(logging.INFO):
            count = sum(low_change_rates_mask)
            total = len(low_change_rates_mask)
            width = min(3, len(str(total)))
            progress_percent = int(100 * elapsed_time_sec / self._total_time_sec)
            msg = f"{str(progress_percent).rjust(3)}% {str(count).rjust(width)}/{str(total).rjust(width)} " + depth * "-"
            log.info(msg)

    def _check_change_rates(self, abs_injection_rates, density, en_chg_rates, gm_bins, rel_injection_rates,
                            unit_time_interval_sec):
        abs_en_changes = sum_change_rates(en_chg_rates, gm_bins.shape, erg_per_s) * unit_time_interval_sec
        abs_scaling = sum_change_rates(rel_injection_rates, density.shape, per_s) * unit_time_interval_sec
        abs_injection = sum_change_rates(abs_injection_rates, density.shape, per_s_cm3) * unit_time_interval_sec
        total_injection = density * abs_scaling + abs_injection
        en_bins = to_erg(gm_bins)
        new_low_change_rates_mask = calc_new_low_change_rates_mask(en_bins, density, abs_en_changes, abs_scaling,
                                                                   abs_injection,
                                                                   self._max_energy_change_per_interval,
                                                                   self._max_density_change_per_interval,
                                                                   self._max_injection_per_interval)
        return en_bins, abs_en_changes, total_injection, new_low_change_rates_mask

    def _sort_and_merge_bins(self, gm_bins, densities, mask=None):
        if mask is None:
            mask = np.repeat(True, gm_bins.shape[1])
        gm_bins, densities, mapping_deduplication = sort_and_merge_duplicates(gm_bins, densities, mask)
        gm_bins, densities, mapping_too_close = remove_too_close_bins(gm_bins, densities, self._min_bin_distance,
                                                              mask[mapping_deduplication])
        additional_mapping = combine_mapping(mapping_deduplication, mapping_too_close)
        return gm_bins, densities, additional_mapping