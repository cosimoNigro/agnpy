import logging
from collections import defaultdict
from agnpy import Blob, Synchrotron, SynchrotronSelfCompton
from agnpy.time_evolution._time_evolution_utils import *
from agnpy.time_evolution.types import *
from agnpy.utils.conversion import mec2
from astropy.constants import c, e
from astropy.units import Quantity
from numpy._typing import NDArray
from typing import Iterable

log = logging.getLogger(__name__)

def synchrotron_loss(sync: Synchrotron) -> EnergyChangeFnType:
    return LabeledFunction(lambda gamma: sync.electron_energy_loss_rate(gamma) * -1, "Synch")

def ssc_loss(sync: SynchrotronSelfCompton) -> EnergyChangeFnType:
    return LabeledFunction(lambda gamma: sync.electron_energy_loss_rate(gamma) * -1, "SSC")

def ssc_thomson_limit_loss(sync: SynchrotronSelfCompton) -> EnergyChangeFnType:
    return LabeledFunction(lambda gamma: sync.electron_energy_loss_rate_thomson(gamma) * -1, "SSC-Thomson")

def fermi_acceleration(t_acc: Quantity) -> EnergyChangeFnType:
    return LabeledFunction(lambda gamma: to_erg(gamma)/t_acc, "FermiAcc")

def bohm_diffusion_losses(blob) -> InjectionRelFnType:
    def larmor_radius(gamma, B):
        return (gamma * mec2 / e.gauss / B.cgs).to("cm")
    def bohm_diffusion_loss_formula(gamma, _, time):
        r_l = larmor_radius(gamma, blob.B_cgs)
        d_b = r_l * c / 3
        t = (blob.R_b**2 / (2*d_b)).to("s")
        t_fract = (time.to("s") / t).value
        return np.exp(-t_fract)
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
         For particle injection, the return coefficient should be >1.0, and for escape, <1.0
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
        Maximum distance the first or last bin can creep away from the gamma bounds towards the center of distribution,
        before the new bin is injected at the boundary. Measured as a ratio of a distance from a boundary
        and the total distribution length, in a log scale. Applicable only if gamma_bounds was set.
    min_bin_distance :
        Minimum distance between the bins, below which the bins will be merged.
    max_energy_change_per_interval :
        maximum relative change of the electron energy allowed in one time interval
    max_density_change_per_interval :
        maximum relative change of the electron density allowed in one time interval
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
                 max_bin_creep_from_bounds: float = 1e-2,
                 min_bin_distance : float = 1.5e-3,
                 max_energy_change_per_interval: float = 0.01,
                 max_density_change_per_interval: float = 0.1,
                 method: NumericalMethod = "euler",
                 distribution_change_callback: CallbackFnType = None):
        self._blob = blob
        self._total_time_sec = total_time.to("s")
        self._energy_change_functions = defaultdict(list)
        for lf in energy_change_functions if isinstance(energy_change_functions, Iterable) else [energy_change_functions]:
            self._energy_change_functions[str(lf)].append(lf)
        self._injection_functions_rel = [] if injection_functions_rel is None \
            else injection_functions_rel if isinstance(injection_functions_rel, Iterable) \
            else [injection_functions_rel]
        self._injection_functions_abs = [] if injection_functions_abs is None \
            else injection_functions_abs if isinstance(injection_functions_abs, Iterable) \
            else [injection_functions_abs]
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
        valid_methods = {"heun", "euler"}
        if method.lower() not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Expected one of: {', '.join(valid_methods)}")
        self._method = method
        self._distribution_change_callback = distribution_change_callback

    def eval_with_fixed_intervals(self, intervals_count: int) -> tuple[Quantity,Quantity,dict,Quantity]:
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
        A tuple consisting of: new gamma array, new densities, new change rates dict, new absolute injection rates
        """

        if intervals_count <= 0:
            raise ValueError("intervals_count must be > 0")
        unit_time_interval = (self._total_time_sec / intervals_count).to("s")
        gm_bins_lb = self._initial_gamma_array if self._initial_gamma_array is not None else self._blob.gamma_e
        dens = []
        en_chg_rates_lb = {}
        abs_injection_rate = []
        for counter in range(1, intervals_count + 1):
            gm_bins_lb, dens, en_chg_rates_lb, abs_injection_rate = self._do_iterative(gm_bins_lb, unit_time_interval,
                                                      counter, intervals_count, en_chg_rates_lb)
        return gm_bins_lb, dens, en_chg_rates_lb, abs_injection_rate

    def _do_iterative(self, gm_bins_lb, unit_time_interval_sec, iteration, intervals_count, en_chg_rates_lb):
        """
        Iterative algorithm for evaluating the electron energy and density. For each gamma point it creates a narrow bin,
        calculates the energy change for a start and an end of the bin, and scales up density by the bin narrowing factor.
        """
        gm_bins_ub = gm_bins_lb.copy()

        # ======== stage 1: add bins at edges ========
        if self._gamma_bounds is not None:
            if iteration == 1:
                en_chg_rates_lb = recalc_chg_rates_boundary_bins(gm_bins_lb, self._energy_change_functions)
            first_bin_en_chg = sum_change_rates(en_chg_rates_lb)[0]
            last_bin_en_chg = sum_change_rates(en_chg_rates_lb)[-1]
            gm_bins_lb, gm_bins_ub, _ = add_boundary_bins(gm_bins_lb, gm_bins_ub, first_bin_en_chg, last_bin_en_chg,
                                                          self._gamma_bounds, self._max_bin_creep_from_bounds)

        # ======== stage 2: recalculate change rates ========
        update_bin_ub(gm_bins_lb, gm_bins_ub)
        gm_bins = interlace(gm_bins_lb, gm_bins_ub)
        en_bins = to_erg(gm_bins)
        n_e = self._blob.n_e
        density = n_e(gm_bins_lb)
        new_en_chg_rates = recalc_change_rates(gm_bins, self._energy_change_functions)
        abs_changes = sum_change_rates(new_en_chg_rates) * unit_time_interval_sec
        abs_injection = eval_abs_particle_injection(gm_bins_lb, density,
                                                    self._injection_functions_rel,
                                                    self._injection_functions_abs, unit_time_interval_sec)
        abs_injection_rate = abs_injection / unit_time_interval_sec
        new_low_change_rates_mask = calc_new_low_change_rates_mask(en_bins, density, abs_changes, abs_injection,
                                                                   self._max_energy_change_per_interval,
                                                                   self._max_density_change_per_interval)
        if not np.all(new_low_change_rates_mask):
            raise ValueError(
                "Energy change formula returned too big value. Use shorter time ranges.")
        log.info("%*d / %d", len(str(intervals_count)), iteration, intervals_count)

        # ======== stage 3: apply time evaluation ========
        new_gm_bins, new_dens = recalc_gamma_bins_and_density(en_bins, abs_changes, n_e)
        new_gm_bins_lb, _ = deinterlace(new_gm_bins)
        new_dens = new_dens + abs_injection
        update_distribution(new_gm_bins_lb, new_dens, self._blob)
        en_chg_rates_lb = deinterlace_energy_changes_dict(new_en_chg_rates)

        # ======== stage 4: for Heun method, recalculate bins ========
        if self._method.lower() == "heun":
            en_chg_rates_recalced = recalc_change_rates(new_gm_bins, self._energy_change_functions)
            averaged_en_chg_rates = \
                {key: (en_chg_rates_recalced[key] + new_en_chg_rates[key]) / 2
                 for key in en_chg_rates_recalced}
            abs_en_chg_recalced = sum_change_rates(averaged_en_chg_rates) * unit_time_interval_sec
            new_gm_bins, new_dens = \
                recalc_gamma_bins_and_density(en_bins, abs_en_chg_recalced, n_e)
            new_gm_bins_lb, _ = deinterlace(new_gm_bins)
            new_dens = new_dens + abs_injection
            update_distribution(new_gm_bins_lb, new_dens, self._blob)
            en_chg_rates_lb = deinterlace_energy_changes_dict(averaged_en_chg_rates)

        # ======== stage 5: sort and merge bins if needed ========
        mapping_deduplication = sorting_and_deduplicating_mapping(new_gm_bins_lb)
        new_gm_bins_lb = new_gm_bins_lb[mapping_deduplication]
        new_dens = new_dens[mapping_deduplication]
        abs_injection_rate = abs_injection_rate[mapping_deduplication]
        en_chg_rates_lb = remap_energy_change_rates_lb(mapping_deduplication, en_chg_rates_lb)

        # ======== stage 6: remove bins which are beyond edges ========
        new_gm_bins_lb, _, mapping_wo_bins_beyond_bounds = remove_gamma_beyond_bounds(new_gm_bins_lb, None, self._gamma_bounds)
        new_dens = new_dens[mapping_wo_bins_beyond_bounds]
        abs_injection_rate = abs_injection_rate[mapping_wo_bins_beyond_bounds]
        en_chg_rates_lb = remap_energy_change_rates_lb(mapping_wo_bins_beyond_bounds, en_chg_rates_lb)

        # ======== stage 7: remove too close bins ========
        new_gm_bins_lb, _, mapping_too_close = remove_too_close_bins(new_gm_bins_lb, None, self._min_bin_distance)
        new_dens = new_dens[mapping_too_close]
        abs_injection_rate = abs_injection_rate[mapping_too_close]
        en_chg_rates_lb = remap_energy_change_rates_lb(mapping_too_close, en_chg_rates_lb)

        # ======== stage 8: call the callback function =========
        if self._distribution_change_callback is not None:
            total_time = unit_time_interval_sec * iteration
            self._distribution_change_callback(total_time, new_gm_bins_lb, new_dens, en_chg_rates_lb,
                                               abs_injection_rate)

        return new_gm_bins_lb, new_dens, en_chg_rates_lb, abs_injection_rate

    def eval_with_automatic_intervals(self) -> tuple[Quantity,Quantity,dict,Quantity]:
        """
        Performs the time evolution of the electron distribution inside the blob, repeating the calculation by the
        automatically selected time intervals such that the energy change in each interval does not exceed the
        max_change_per_interval rate. The automatically selected time interval duration may differ per energy bins
        - shorter intervals will be used for bins where energy change is faster.
        It assumes the energy-change-functions are costly to call so it tries to recalculate them only for bins
        with how energy changes, and reuse earlier calculated values for bins with low energy changes.
        It also assumes the particle-injection functions are cheap to call, so they can be called often and for every bin.

        Side Effects
        ------------
        Replaces the blob.n_e with the new InterpolatedDistribution

        Returns
        ------------
        A tuple consisting of: new gamma array, new densities, new change rates dict, new absolute injection rates
        """

        initial_gamma_bins_from = self._initial_gamma_array if self._initial_gamma_array is not None else self._blob.gamma_e
        dummy_bins_ends = np.zeros_like(initial_gamma_bins_from)
        empty_mask = np.repeat(False, len(initial_gamma_bins_from))
        (_, gamma_bins, _, _, density, energy_change_rates, injection) = self._do_recursive(
            initial_gamma_bins_from, dummy_bins_ends, {}, empty_mask, True, self._total_time_sec, [], 1)
        return gamma_bins, density, deinterlace_energy_changes_dict(energy_change_rates), injection

    def _do_recursive(self, gm_bins_lb, gm_bins_ub, en_chg_rates, low_chg_rates_mask, recalc_high_chg, time_sec, prev_times, depth):
        """
        Recursive algorithm for evaluating the electron energy and density.
        Parameters:
            gm_bins_lb - array of shape (N,) containing lower bounds of gamma bins
            gm_bins_ub - array of shape (N,) containing upper bounds of gamma bins
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

        # ======== stage 1: add bins at edges ========
        if recalc_high_chg and self._gamma_bounds is not None:
            if depth == 1:
                chg_rates_lub = recalc_chg_rates_boundary_bins(gm_bins_lb, self._energy_change_functions)
                first_bin_en_chg = sum_change_rates(chg_rates_lub)[0]
                last_bin_en_chg = sum_change_rates(chg_rates_lub)[-1]
            else:
                first_bin_en_chg = sum_change_rates(en_chg_rates)[0]
                last_bin_en_chg = sum_change_rates(en_chg_rates)[-2]
            gm_bins_lb, gm_bins_ub, mapping_with_edge_bins = add_boundary_bins(gm_bins_lb, gm_bins_ub,
                                                                               first_bin_en_chg, last_bin_en_chg,
                                                                               self._gamma_bounds, self._max_bin_creep_from_bounds)
            en_chg_rates = remap_energy_change_rates(mapping_with_edge_bins, en_chg_rates)
            low_chg_rates_mask = remap_low_changes_mask(mapping_with_edge_bins, low_chg_rates_mask)
        else:
            mapping_with_edge_bins = np.arange(len(gm_bins_lb))

        # ======== stage 2: recalculate change rates if needed, and only for unmasked bins ========
        if recalc_high_chg:
            update_bin_ub(gm_bins_lb, gm_bins_ub, low_chg_rates_mask)
            gm_bins = interlace(gm_bins_lb, gm_bins_ub)
            en_chg_rates = recalc_change_rates(gm_bins, self._energy_change_functions, en_chg_rates, low_chg_rates_mask)
        else:
            gm_bins = interlace(gm_bins_lb, gm_bins_ub)
        abs_en_chg = (sum_change_rates(en_chg_rates) * time_sec).to("erg")
        en_bins = to_erg(gm_bins)
        gm_bins_lb_begin, gm_bins_ub_begin = deinterlace(gm_bins)
        n_e_begin = self._blob.n_e
        dens_begin = n_e_begin(gm_bins_lb_begin)
        abs_injection = eval_abs_particle_injection(gm_bins_lb_begin, dens_begin,
                                                    self._injection_functions_rel,
                                                    self._injection_functions_abs, time_sec)
        low_chg_rates_mask_recalced = calc_new_low_change_rates_mask(en_bins,
                                                                     dens_begin,
                                                                     abs_en_chg,
                                                                     abs_injection,
                                                                     self._max_energy_change_per_interval,
                                                                     self._max_density_change_per_interval)
        low_chg_rates_flip_mask = ~low_chg_rates_mask & low_chg_rates_mask_recalced
        prev_times_sum = sum(prev_times) if len(prev_times) > 0 else 0 * u.s
        self._log_mask_info(low_chg_rates_mask_recalced, depth, prev_times_sum)

        # ======== stage 3: apply time evaluation, directly or in two recursive substeps ========
        times = prev_times.copy()
        if np.all(low_chg_rates_mask_recalced):
            if np.all(dens_begin == 0):
                dens_after_en_eval = dens_begin.copy()
            else:
                gm_bins, dens_after_en_eval = recalc_gamma_bins_and_density(en_bins, abs_en_chg, n_e_begin)
                gm_bins_lb, gm_bins_ub = deinterlace(gm_bins)
            dens = dens_after_en_eval + abs_injection
            abs_injection_rate = abs_injection / time_sec
            new_en_chg_rates = en_chg_rates
            mapping_timeeval = np.arange(len(gm_bins_lb_begin))
            final_mapping = mapping_with_edge_bins
            times.append(time_sec)
            update_distribution(gm_bins_lb, dens, self._blob)
        else:
            half_time = time_sec / 2
            (mapping_halftime, gm_bins_lb_halftime, gm_bins_ub_halftime, times, dens, en_chg_rates_halftime,
             abs_injection_rate) \
                = self._do_recursive(gm_bins_lb, gm_bins_ub,
                               en_chg_rates,
                               low_chg_rates_mask_recalced,
                               False,
                               half_time, times, depth + 1)

            low_chg_rates_mask_halftime = remap_low_changes_mask(mapping_halftime, low_chg_rates_mask_recalced)

            if self._distribution_change_callback is not None:
                self._distribution_change_callback(prev_times_sum + half_time, gm_bins_lb_halftime, dens,
                                                   deinterlace_energy_changes_dict(en_chg_rates_halftime),
                                                   abs_injection_rate)

            (mapping_2nd_halftime, gm_bins_lb, gm_bins_ub, times, dens, new_en_chg_rates, abs_injection_rate) \
                = self._do_recursive(gm_bins_lb_halftime,
                               gm_bins_ub_halftime,
                               en_chg_rates_halftime,
                               low_chg_rates_mask_halftime,
                               True,
                               half_time, times, depth + 1)
            gm_bins = interlace(gm_bins_lb, gm_bins_ub)
            mapping_timeeval = combine_mapping(mapping_halftime, mapping_2nd_halftime)
            low_chg_rates_flip_mask = remap_low_changes_mask(mapping_timeeval, low_chg_rates_flip_mask)
            final_mapping = combine_mapping(mapping_with_edge_bins, mapping_timeeval)

        # ======== stage 4: for Heun method, recalculate bins covered by the flip mask ========
        if self._method.lower() == "heun" and np.any(low_chg_rates_flip_mask) and not np.all(dens_begin == 0):
            en_chg_rates_recalced_twice = recalc_change_rates(
                gm_bins, self._energy_change_functions, new_en_chg_rates, ~low_chg_rates_flip_mask)
            low_chg_rates_flip_mask_dbl = np.repeat(low_chg_rates_flip_mask, 2)
            averaged_en_chg_rates = {key: (new_en_chg_rates[key][low_chg_rates_flip_mask_dbl] +
                                           en_chg_rates_recalced_twice[key][low_chg_rates_flip_mask_dbl]) / 2
                                     for key in en_chg_rates_recalced_twice}
            recalc_time_start_idx = -1 * (len(times) - len(prev_times))
            times_to_recalc = times[recalc_time_start_idx:]
            gm_bins_lb_recalc = remap_simple(mapping_timeeval, gm_bins_lb_begin)[low_chg_rates_flip_mask]
            gm_bins_ub_recalc = remap_simple(mapping_timeeval, gm_bins_ub_begin)[low_chg_rates_flip_mask]
            gm_bins_recalc = interlace(gm_bins_lb_recalc, gm_bins_ub_recalc)
            dens_recalc = n_e_begin(gm_bins_lb_recalc)
            for t in times_to_recalc:
                abs_injection_recalc = eval_abs_particle_injection(gm_bins_lb_recalc, dens_recalc,
                                                                   self._injection_functions_rel,
                                                                   self._injection_functions_abs, t)
                gm_bins_recalc, dens_recalc = recalc_gamma_bins_and_density(
                    to_erg(gm_bins_recalc),
                    sum_change_rates(averaged_en_chg_rates) * t,
                    dens_recalc)
                gm_bins_lb_recalc = deinterlace(gm_bins_recalc)[0]
                dens_recalc = dens_recalc + abs_injection_recalc
                if any(np.isnan(dens_recalc)):
                    # in this case, we should roll back the sub-calculation progress and retry with the new mask from the beginning of the current method
                    raise ValueError(
                        "Illegal negative density obtained - the resolution algorithm not yet implemented, please report it to the agnpy maintainers")
                abs_injection_rate[low_chg_rates_flip_mask] = abs_injection_recalc / t
            gm_bins[low_chg_rates_flip_mask_dbl] = gm_bins_recalc
            gm_bins_lb, gm_bins_ub = deinterlace(gm_bins)
            dens[low_chg_rates_flip_mask] = dens_recalc
            update_distribution(gm_bins_lb, dens, self._blob)

        # ======== stage 5: sort and merge bins if needed ========
        mapping_deduplication = sorting_and_deduplicating_mapping(gm_bins_lb, low_chg_rates_flip_mask)
        gm_bins_lb = gm_bins_lb[mapping_deduplication]
        gm_bins_ub = gm_bins_ub[mapping_deduplication]
        dens = dens[mapping_deduplication]
        abs_injection_rate = abs_injection_rate[mapping_deduplication]
        new_en_chg_rates = remap_energy_change_rates(mapping_deduplication, new_en_chg_rates)
        final_mapping = combine_mapping(final_mapping, mapping_deduplication)

        # ======== stage 6: remove bins which are beyond edges ========
        gm_bins_lb, gm_bins_ub, mapping_wo_bins_beyond_bounds = remove_gamma_beyond_bounds(gm_bins_lb, gm_bins_ub,
                                                                                           self._gamma_bounds)
        abs_injection_rate = abs_injection_rate[mapping_wo_bins_beyond_bounds]
        dens = dens[mapping_wo_bins_beyond_bounds]
        new_en_chg_rates = remap_energy_change_rates(mapping_wo_bins_beyond_bounds, new_en_chg_rates)
        final_mapping = final_mapping[mapping_wo_bins_beyond_bounds]

        # ======== stage 7: remove too close bins ========
        gm_bins_lb, gm_bins_ub, mapping_too_close = remove_too_close_bins(gm_bins_lb, gm_bins_ub, self._min_bin_distance)
        dens = dens[mapping_too_close]
        abs_injection_rate = abs_injection_rate[mapping_too_close]
        new_en_chg_rates = remap_energy_change_rates(mapping_too_close, new_en_chg_rates)
        final_mapping = final_mapping[mapping_too_close]

        # ======== stage 8: if this is end of the simulation, publish the final callback =========
        if depth == 1 and self._distribution_change_callback is not None:
            self._distribution_change_callback(prev_times_sum + time_sec, gm_bins_lb, dens,
                                               deinterlace_energy_changes_dict(new_en_chg_rates),
                                               abs_injection_rate)

        return final_mapping, gm_bins_lb, gm_bins_ub, times, dens, new_en_chg_rates, abs_injection_rate

    def _log_mask_info(self, low_change_rates_mask, depth, elapsed_time_sec):
        if log.isEnabledFor(logging.INFO):
            count = sum(low_change_rates_mask)
            total = len(low_change_rates_mask)
            width = min(3, len(str(total)))
            progress_percent = int(100 * elapsed_time_sec / self._total_time_sec)
            msg = f"{str(progress_percent).rjust(3)}% {str(count).rjust(width)}/{str(total).rjust(width)} " + depth * "-"
            log.info(msg)
