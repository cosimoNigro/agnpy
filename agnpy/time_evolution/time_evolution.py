import logging
import astropy.units as u
from agnpy import Blob, Synchrotron, SynchrotronSelfCompton
from agnpy.time_evolution._time_evolution_utils import *
from agnpy.time_evolution.types import *
from astropy.constants import c, e
from astropy.units import Quantity
from numpy._typing import NDArray
from typing import Iterable, Tuple

log = logging.getLogger(__name__)

erg_per_s = u.Unit("erg s-1")
per_s_cm3 = u.Unit("s-1 cm-3")
per_s = u.Unit("s-1")

def synchrotron_loss(sync: Synchrotron) -> EnergyChangeFn:
    """
    Returns the energy change function for synchrotron losses
    """
    return lambda args: sync.electron_energy_loss_rate(args.gamma) * -1

def ssc_loss(sync: SynchrotronSelfCompton) -> EnergyChangeFn:
    """
    Returns the energy change function for SSC losses
    """
    return lambda args: sync.electron_energy_loss_rate(args.gamma) * -1

def ssc_thomson_limit_loss(sync: SynchrotronSelfCompton) -> EnergyChangeFn:
    """
    Returns the simplified energy change function implementation for SynchrotronSelfCompton (valid only in the Thomson limit)
    """
    return lambda args: sync.electron_energy_loss_rate_thomson(args.gamma) * -1

def fermi_acceleration(t_acc: Quantity) -> EnergyChangeFn:
    """
    Returns the acceleration function, parametrized by the acceleration timescale `t_acc`
    """
    return lambda args: to_erg(args.gamma)/t_acc


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
    rel_injection_functions :
         A function, or an array of functions, to be used for calculation of electron density change caused by
         direct particle injection (or escape). This function is used for relative density change.
         For particle injection, the return coefficient should be >0, and for escape, <0
    abs_injection_functions :
         A function, or an array of functions, to be used for calculation of electron density change caused by
         direct particle injection (or escape). This function is used for absolute density change.
         For particle injection, the return density change rate should be >0, and for escape, <0
    step_duration:
        specific time for equal-length steps, or "auto"
    initial_gamma_array :
        Optional array of gamma values (electron energies); if empty, blob.gamma_e() will be used.
        This is primarily intended for cases when simulation is done in multiple steps, and you want to use
        the final bins from one step as a starting point for the next step
    gamma_bounds :
        Optional tuple of gamma lower and upper bounds for calculations
    max_bin_creep_from_bounds :
        Maximum distance (in log10) the first or last bin can creep away from the gamma bounds towards the center of distribution,
        before the new bin is injected at the boundary. Applicable only if gamma_bounds was set.
    merge_bins_closer_than :
        Minimum distance( in log10) between the bins, below which the bins will be merged.
    max_energy_change_per_interval :
        Maximum relative change of the electron energy allowed in one time interval for each bin
    max_density_change_per_interval :
        Maximum relative change of the electron density allowed in one time interval for each bin
    max_injection_per_interval :
        Maximum absolute injection or escape of particles in one time interval for each bin (in cm-3)
    optimize_recalculating_slow_rates :
        If True, the automatically selected time interval duration may differ per energy bins - longer intervals
        will be used for bins where energy change is slower, and the rate of change will not be recalculated until the
        end of such interval
    method :
        Numerical method for calculating energy evolution; accepted values: "euler" (faster) or "heun" (more precise)
    subgroups :
        A list of lists, each sublist contains the names of the functions (keys from energy_change_functions
        and/or rel_injection_functions and/or abs_injection_functions) which contribute to each subgroup;
        if empty, the single group (i.e. a list of all keys from energy and injection functions) will be used.
    subgroups_initial_density :
        A (M,N) matrix of split ratios of total density per group; M must be equal to number of subgroups, N equal to initial_gamma_array length;
        and the sum of ratios across all groups must be equal to [1.0, ..., 1.0].
        If not provided, it is assumed [1.0, ..., 1.0] for the first subgroup, and [0.0, ..., 0.0] for any other subgroups;
        in other words, all particles are assigned to the first subgroup.
    distribution_change_callback :
        This optional function will be called each time the blob's electron distribution has been updated.
        You can use it, for example, for updating the distribution plot while the simulation is running.
    """

    def __init__(self,
                 blob: Blob,
                 total_time: Quantity,
                 energy_change_functions: EnergyChangeFns,
                 rel_injection_functions: InjectionRelFns = None,
                 abs_injection_functions: InjectionAbsFns = None,
                 step_duration: Union[str, Quantity] = "auto",
                 initial_gamma_array: NDArray[np.floating] = None,
                 gamma_bounds: Tuple[float, float] = None,
                 max_bin_creep_from_bounds: float = 0.1,
                 merge_bins_closer_than : float = 1.5e-3,
                 max_energy_change_per_interval: float = 0.01,
                 max_density_change_per_interval: float = 0.1,
                 max_injection_per_interval: float = 1.0,
                 optimize_recalculating_slow_rates: bool = None,
                 method: NumericalMethod = "euler",
                 subgroups: SubgroupsList = None,
                 subgroups_initial_density: NDArray[np.floating] =None,
                 distribution_change_callback: CallbackFnType = None):
        self._blob = blob
        self._total_time_sec = total_time.to("s")
        if isinstance(step_duration, Quantity):
            self._step_duration = step_duration.to("s")
        elif isinstance(step_duration, str) and step_duration == "auto":
            self._step_duration = "auto"
        else:
            raise ValueError("step_duration must be a time Quantity, or a string \"auto\"")
        self._energy_change_functions = energy_change_functions if isinstance(energy_change_functions, dict) \
            else {str(v): v for v in energy_change_functions} if isinstance(energy_change_functions, Iterable) \
            else {str(energy_change_functions): energy_change_functions}
        self._rel_injection_functions = rel_injection_functions if isinstance(rel_injection_functions, dict) \
            else {str(v): v for v in rel_injection_functions} if isinstance(rel_injection_functions, Iterable) \
            else {str(rel_injection_functions): rel_injection_functions} if rel_injection_functions is not None \
            else {}
        self._abs_injection_functions = abs_injection_functions if isinstance(abs_injection_functions, dict) \
            else {str(v): v for v in abs_injection_functions} if isinstance(abs_injection_functions, Iterable) \
            else {str(abs_injection_functions): abs_injection_functions} if abs_injection_functions is not None \
            else {}
        duplicated_keys = (set(self._energy_change_functions) & set(self._rel_injection_functions) |
                           set(self._energy_change_functions) & set(self._abs_injection_functions) |
                           set(self._rel_injection_functions) & set(self._abs_injection_functions))
        if duplicated_keys:
            raise ValueError("Found duplicate keys of energy change or injection functions: " + str(duplicated_keys))
        self._gamma_bounds = gamma_bounds
        if gamma_bounds is not None and initial_gamma_array is not None and np.any(initial_gamma_array < gamma_bounds[0]):
            raise ValueError("initial_gamma_array might not contain elements smaller than gamma_min")
        if gamma_bounds is not None and initial_gamma_array is not None and np.any(initial_gamma_array > gamma_bounds[1]):
            raise ValueError("initial_gamma_array might not contain elements greater than gamma_max")
        if gamma_bounds is not None and initial_gamma_array is None:
            self._blob.n_e.gamma_min = gamma_bounds[0]
            self._blob.n_e.gamma_max = gamma_bounds[1]
        if initial_gamma_array is not None and any(initial_gamma_array[1:] <= initial_gamma_array[:-1]):
            raise ValueError("initial_gamma_array must be strictly increasing")
        self._initial_gamma_array = initial_gamma_array if initial_gamma_array is not None else blob.gamma_e
        self._max_bin_creep_from_bounds = max_bin_creep_from_bounds
        self._merge_bins_closer_than = merge_bins_closer_than
        self._max_energy_change_per_interval = max_energy_change_per_interval
        self._max_density_change_per_interval = max_density_change_per_interval
        self._max_injection_per_interval = max_injection_per_interval * u.Unit("cm-3")
        if optimize_recalculating_slow_rates and step_duration != 'auto':
            raise ValueError("optimize_recalculating_slow_rates is only supported with automatic step duration")
        if optimize_recalculating_slow_rates is None:
            optimize_recalculating_slow_rates = step_duration == 'auto'
        self._optimize_recalculating_slow_rates = optimize_recalculating_slow_rates
        valid_methods = {"heun", "euler"}
        if method.lower() not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Expected one of: {', '.join(valid_methods)}")
        if optimize_recalculating_slow_rates and method.lower() == 'heun':
            raise ValueError("Heun method is not supported when optimize_recalculating_slow_rates is set")
        self._method = method.lower()
        self._subgroups = subgroups
        for subgroup in subgroups or []:
            for s in subgroup:
                if s not in self._energy_change_functions and s not in self._rel_injection_functions and s not in self._abs_injection_functions:
                    raise ValueError(f"Function {s} not defined")
        if subgroups_initial_density is not None:
            if not np.allclose(subgroups_initial_density.sum(axis=0), 1.0, atol=1e-6):
                raise ValueError("Initial densities should sum to 1.0")
        self._subgroups_initial_density = subgroups_initial_density
        if subgroups_initial_density is not None and subgroups is None:
            raise ValueError("subgroups_initial_density provided without subgroups")
        if subgroups_initial_density is not None and initial_gamma_array is None:
            raise ValueError("subgroups_initial_density provided without initial_gamma_array")
        if subgroups_initial_density is not None and initial_gamma_array is not None and subgroups_initial_density.shape[1] != len(initial_gamma_array):
            raise ValueError("Incorrect shape for subgroups_initial_density")
        self._distribution_change_callback = distribution_change_callback

    def _calculate_initial_values(self):
        gm_bins_lb = self._initial_gamma_array
        gm_bins_up = np.zeros_like(gm_bins_lb)
        gm_bins = np.array([gm_bins_lb, gm_bins_up])
        update_bin_ub(gm_bins)
        density = self._blob.n_e(gm_bins_lb)
        if self._subgroups is not None:
            if self._subgroups_initial_density is not None:
                subgroups_density = self._subgroups_initial_density
            else:
                # first group all-ones, other groups all-zeros
                subgroups_density = np.zeros((len(self._subgroups), len(self._initial_gamma_array)), dtype=float)
                subgroups_density[0] = np.ones(len(self._initial_gamma_array), dtype=float)
        else:
            subgroups_density = np.ones((1, len(self._initial_gamma_array)), dtype=float)
        en_chg_rates = recalc_new_rates(gm_bins, self._energy_change_functions, density, subgroups_density)
        rel_injection_rates = recalc_new_rates(gm_bins_lb, self._rel_injection_functions, density, subgroups_density)
        abs_injection_rates = recalc_new_rates(gm_bins_lb, self._abs_injection_functions, density, subgroups_density)
        return gm_bins, density, subgroups_density, en_chg_rates, rel_injection_rates, abs_injection_rates

    def evaluate(self) -> TimeEvaluationResult:
        """
        Performs the time evolution of the electron distribution inside the blob. In the loop, for each gamma point it creates a narrow bin,
        calculates the energy change for a start and an end of the bin, and scales up density by the bin narrowing factor and injection/escape rate.

        Side Effects
        ------------
        Replaces the blob.n_e with the new InterpolatedDistribution

        Returns
        ------------
        A tuple consisting of final values of: gamma, densities, density groups, change rates, relative injection rates, absolute injection rates
        """

        gm_bins, density, subgroups_density, en_chg_rates, rel_injection_rates, abs_injection_rates = self._calculate_initial_values()
        total_time_elapsed = 0 * u.s
        log.info("Progress: %3d%% %s (%i bins)", 0, total_time_elapsed, gm_bins.shape[-1])
        time_left_per_bin = np.zeros_like(gm_bins[0]) * u.s

        while total_time_elapsed < self._total_time_sec:
            en_bins = to_erg(gm_bins)
            en_chg_rates_grouped, rel_injection_rates_grouped, abs_injection_rates_grouped = self._group_change_rates(
                gm_bins, en_chg_rates, rel_injection_rates, abs_injection_rates)
            densities_grouped = to_densities_grouped(density, subgroups_density)
            total_injection_rates_grouped = densities_grouped * rel_injection_rates_grouped + abs_injection_rates_grouped
            if self._step_duration != "auto":
                step_time = self._step_duration
            else:
                max_times = self._calc_max_time_per_bin(
                    en_bins, density, subgroups_density, en_chg_rates_grouped, rel_injection_rates_grouped, abs_injection_rates_grouped)
                if self._optimize_recalculating_slow_rates:
                    bins_with_no_time_left = time_left_per_bin == 0
                    time_left_per_bin[bins_with_no_time_left] = max_times[bins_with_no_time_left]
                    step_time = np.min(time_left_per_bin)
                else:
                    step_time = np.min(max_times)
            if step_time > self._total_time_sec - total_time_elapsed:
                step_time = self._total_time_sec - total_time_elapsed

            gm_bins, density, subgroups_density, mapping = self._apply_time_changes(
                en_bins, density, subgroups_density, en_chg_rates_grouped, total_injection_rates_grouped, step_time)

            if self._optimize_recalculating_slow_rates:
                time_left_per_bin -= step_time
                time_left_per_bin = remap(mapping, time_left_per_bin, 0 * u.s)
                time_left_per_bin[gm_bins[1] == 0] = 0  # gm_bins[1]==0 is a signal the bin needs recalculation
                recalc_mask = time_left_per_bin < step_time
                time_left_per_bin[recalc_mask] = 0
            else:
                recalc_mask = np.ones_like(density, dtype=bool)
            update_bin_ub(gm_bins, recalc_mask)
            en_chg_rates = recalc_new_rates(gm_bins, self._energy_change_functions, density, subgroups_density,
                                            remap(mapping,en_chg_rates,np.nan), recalc_mask)
            abs_injection_rates = recalc_new_rates(gm_bins[0], self._abs_injection_functions, density, subgroups_density,
                                                   remap(mapping, abs_injection_rates, np.nan), recalc_mask)
            rel_injection_rates = recalc_new_rates(gm_bins[0], self._rel_injection_functions, density, subgroups_density,
                                                   remap(mapping, rel_injection_rates, np.nan), recalc_mask)
            total_time_elapsed += step_time
            if self._distribution_change_callback is not None:
                self._distribution_change_callback(TimeEvaluationResult(total_time_elapsed,
                    gm_bins[0], density, subgroups_density, energy_changes_lb(en_chg_rates), abs_injection_rates,
                    rel_injection_rates))
            progress_percent = int(100 * total_time_elapsed / self._total_time_sec)
            log.info("Progress: %3d%% %s (%i bins)", progress_percent, total_time_elapsed, gm_bins.shape[-1])

        en_chg_rates_lb = energy_changes_lb(en_chg_rates)
        return TimeEvaluationResult(total_time_elapsed, gm_bins[0], density, subgroups_density, en_chg_rates_lb, rel_injection_rates, abs_injection_rates)

    def _calc_max_time_per_bin(self, en_bins, density, subgroups_density, en_chg_rates_grouped,
                               rel_injection_rates_grouped, abs_injection_rates_grouped):
        density_grouped = to_densities_grouped(density, subgroups_density)
        max_times = np.repeat(np.inf, len(density))
        for i in (range(subgroups_density.shape[0])):
            if np.any((density_grouped[i] == 0) & (abs_injection_rates_grouped[i] < 0)):
                raise ValueError("Particles cannot escape when density is zero")

            en_chg_rates = en_chg_rates_grouped[i]
            density = density_grouped[i]
            rel_injection_rates = rel_injection_rates_grouped[i]
            abs_injection_rates = abs_injection_rates_grouped[i]

            # 1. relative energy change per bin must be lower than the threshold: (abs(en_changes) * time) / energy_bins <= max_energy_change_per_interval
            time_1 = np.min(self._max_energy_change_per_interval * en_bins / abs(en_chg_rates), axis=0)

            # 2. density increase/increase (from energy changes) = Δx/(Δx+tΔf(x)) where x is bin energy and Δx is bin width;
            # if Δf(x)>0 then density decreases, and it must not decrease more than a factor 1+threshold, so: [density decrease >= 1/(1+thr)] => [t <= (Δx/Δf)*thr]
            # if Δf(x)>0 then density increases, and it must stay below 1+thr factor, hence: [density increase <= 1+thr] => [t <= -(Δx/Δf)(thr/(1+thr))]
            # note: if density increases, the bin width decreases, and condition for not making it negative is: [t < -Δx/Δf(x)]
            time_2 = np.repeat(np.inf, len(density)) * u.Unit("s")
            x2_x1 = en_bins[1] - en_bins[0]  # Δx, always positive
            f2_f1 = en_chg_rates[1] - en_chg_rates[0]  # Δf(x), can be positive/0/negative
            mask = f2_f1 != 0
            x_f = x2_x1[mask] / f2_f1[mask]  # Δx/Δf
            thr = self._max_density_change_per_interval
            time_2[mask] = np.where(x_f > 0, x_f * thr, -x_f * thr / (1 + thr))

            # 3. particle scaling must stay in a reasonable range, say not more than half can escape
            # rel_injection_rates * time > -0.5
            time_3 = np.repeat(np.inf, len(density)) * u.Unit("s")
            time_3[rel_injection_rates < 0] = -0.5 / rel_injection_rates[rel_injection_rates < 0]

            # 4. abs particle injection must be below the threshold
            # abs_injection_rates * time < max_abs_injection_per_interval
            time_4 = np.repeat(np.inf, len(density)) * u.Unit("s")
            time_4[abs_injection_rates != 0] = self._max_injection_per_interval / np.abs(
                abs_injection_rates[abs_injection_rates != 0])

            max_times_per_group = Quantity([time_1, time_2, time_3, time_4]).min(axis=0)
            max_times = np.minimum(max_times, max_times_per_group)
        return max_times

    def _apply_time_changes(self, en_bins, density, subgroups_density, en_changes_grouped,
                            total_injection_grouped, unit_time_interval_sec):

        # ======== stage 1: apply time evaluation ========
        new_gm_bins, new_dens, subgroups_density = self._apply_time_eval(en_bins, density, subgroups_density,
                en_changes_grouped * unit_time_interval_sec, total_injection_grouped * unit_time_interval_sec)
        dist = update_distribution(new_gm_bins, new_dens, self._blob)

        # ======== stage 2: for Heun method, recalculate bins ========
        if self._method.lower() == "heun":
            en_chg_rates_recalced = recalc_new_rates(new_gm_bins, self._energy_change_functions, new_dens, subgroups_density)
            en_changes_grouped_recalced = sum_change_rates(en_chg_rates_recalced, new_gm_bins.shape, erg_per_s, self._subgroups)
            averaged_en_chg_rates_grouped = (en_changes_grouped + en_changes_grouped_recalced) / 2
            abs_en_chg_recalc = averaged_en_chg_rates_grouped * unit_time_interval_sec
            new_gm_bins, new_dens, subgroups_density = (
                self._apply_time_eval(en_bins, density, subgroups_density, abs_en_chg_recalc,
                                total_injection_grouped * unit_time_interval_sec))
            dist = update_distribution(new_gm_bins, new_dens, self._blob)
            en_changes_grouped = averaged_en_chg_rates_grouped

        # ======== stage 3: sort and merge bins if needed ========
        new_gm_bins, new_dens, sort_and_merge_mapping = self._sort_and_merge_bins(new_gm_bins, new_dens, dist)
        subgroups_density = remap_subgroups_density(sort_and_merge_mapping, subgroups_density)
        final_mapping = sort_and_merge_mapping

        if self._gamma_bounds is not None:
            # ======== stage 4: remove bins falling behind the edges ========
            new_gm_bins, mapping_wo_bins_beyond_bounds = self._remove_gamma_beyond_bounds(new_gm_bins)
            new_dens = remap(mapping_wo_bins_beyond_bounds, new_dens)
            subgroups_density = remap_subgroups_density(mapping_wo_bins_beyond_bounds, subgroups_density)
            final_mapping = combine_mapping(final_mapping, mapping_wo_bins_beyond_bounds)

            # ======== stage 5: add bins at edges if they move towards the center ========
            new_gm_bins, mapping_with_new_bins = self._add_boundary_bins(new_gm_bins, en_changes_grouped[0])
            new_dens = remap(mapping_with_new_bins, new_dens, 0 * u.Unit("cm-3"))
            subgroups_density = remap_subgroups_density(mapping_with_new_bins, subgroups_density)
            final_mapping = combine_mapping(final_mapping, mapping_with_new_bins)
        return new_gm_bins, new_dens, subgroups_density, final_mapping

    def _apply_time_eval(self, en_bins, density, subgroups_density, abs_en_changes_grouped, total_injection_grouped):
        new_bins_and_densities = [
            recalc_gamma_bins_and_density(en_bins, abs_en_changes_grouped[i], density * subgroups_density[i])
            for i in range(len(subgroups_density))]
        for i in range(len(new_bins_and_densities)):
            new_bin_density = new_bins_and_densities[i].densities + total_injection_grouped[i]
            if np.any(new_bin_density < 0):
                raise ValueError("Negative density obtained")
            new_bins_and_densities[i] = BinsWithDensities(new_bins_and_densities[i].gamma_bins, new_bin_density)

        new_gm_bins = new_bins_and_densities[0].gamma_bins
        new_densities = np.zeros_like(subgroups_density) * u.Unit("cm-3")
        new_densities[0] = new_bins_and_densities[0].densities

        for i in range(1, len(new_bins_and_densities)):
            new_gm_bins_subgroup = new_bins_and_densities[i].gamma_bins
            new_dens_subgroup = new_bins_and_densities[i].densities
            if np.all(new_dens_subgroup == 0) or np.all(new_bins_and_densities[i].gamma_bins == new_gm_bins):
                new_densities[i] = new_dens_subgroup
            else:
                distribution = InterpolatedDistribution(new_gm_bins_subgroup[0], new_dens_subgroup, interpolator=PchipInterpolator)
                densities_interpolated = distribution(new_gm_bins[0])
                new_densities[i] = densities_interpolated

        new_dens = Quantity(new_densities.sum(axis=0))
        subgroups_density_new = subgroups_density.copy()
        mask = new_dens != 0
        subgroups_density_new[:, mask] = (new_densities[:, mask] / new_dens[mask]).value
        return new_gm_bins, new_dens, subgroups_density_new

    def _add_boundary_bins(self, gm_bins, en_chg_rates):
        gm_bins_log = np.log10(gm_bins[0, [0, -1]])
        gamma_bounds_log = np.log10(self._gamma_bounds)
        mapping_with_edge_bins = np.arange(gm_bins.shape[1])
        is_first_bin_far_from_start = gm_bins_log[0] - gamma_bounds_log[0] >= self._max_bin_creep_from_bounds
        if is_first_bin_far_from_start:
            first_bin_en_chg = en_chg_rates[0, 0]
            if first_bin_en_chg > 0:
                log.info("Adding a bin at the lower gamma bound")
                new_col = np.array([[self._gamma_bounds[0]], [0]])
                gm_bins = np.concatenate([new_col, gm_bins], axis=1)
                mapping_with_edge_bins = np.concatenate([np.array([-1]), mapping_with_edge_bins])
        is_last_bin_far_from_end = gamma_bounds_log[-1] - gm_bins_log[-1] >= self._max_bin_creep_from_bounds
        if is_last_bin_far_from_end:
            last_bin_en_chg = en_chg_rates[0, -1]
            if last_bin_en_chg < 0:
                log.info("Adding a bin at the upper gamma bound")
                new_col = np.array([[self._gamma_bounds[1] / (1 + bin_size_factor)], [0]])
                gm_bins = np.concatenate([gm_bins, new_col], axis=1)
                mapping_with_edge_bins = np.concatenate([mapping_with_edge_bins, np.array([-1])])
        return gm_bins, mapping_with_edge_bins

    def _group_change_rates(self, gm_bins, en_chg_rates, rel_injection_rates, abs_injection_rates):
        en_chg_rates_grouped = sum_change_rates(en_chg_rates, gm_bins.shape, erg_per_s, self._subgroups)
        rel_injection_rates_grouped = sum_change_rates(rel_injection_rates, gm_bins.shape[-1:], per_s, self._subgroups)
        abs_injection_rates_grouped = sum_change_rates(abs_injection_rates, gm_bins.shape[-1:], per_s_cm3, self._subgroups)
        return en_chg_rates_grouped, rel_injection_rates_grouped, abs_injection_rates_grouped

    def _sort_and_merge_bins(self, gm_bins, densities, interpolated_distribution):
        gm_bins, densities, mapping_deduplication = sort_and_merge_duplicates(gm_bins, densities, interpolated_distribution)
        if self._merge_bins_closer_than is None:
            return gm_bins, densities, mapping_deduplication
        gm_bins, densities, mapping_too_close = self._remove_too_close_bins(gm_bins, densities, interpolated_distribution)
        additional_mapping = combine_mapping(mapping_deduplication, mapping_too_close)
        return gm_bins, densities, additional_mapping

    def _remove_gamma_beyond_bounds(self, gm_bins):
        gm_bins_lb = gm_bins[0]
        initial = np.arange(len(gm_bins_lb))
        mapping = np.where((self._gamma_bounds[0] <= gm_bins_lb) & (gm_bins_lb <= self._gamma_bounds[1] / (1 + bin_size_factor)))[0]
        removed = np.setdiff1d(initial, mapping)
        if len(removed) > 0:
            log.info("Removed bins %s with gamma values %s", removed, gm_bins_lb[removed])
            return gm_bins[..., mapping], mapping
        return gm_bins, initial

    def _remove_too_close_bins(self, gm_bins, densities, interpolated_distribution):
        gm_bins_lb_log = np.log10(gm_bins[0])
        gaps = np.diff(gm_bins_lb_log)
        too_close_bins_mask = gaps < self._merge_bins_closer_than
        # don't merge first and last bin
        too_close_bins_mask[0] = too_close_bins_mask[-1] = False

        if np.any(too_close_bins_mask):
            close_bins_idx = too_close_bins_mask.nonzero()[0] + 1
            keep_idx = np.setdiff1d(np.arange(len(densities)), close_bins_idx, assume_unique=True)
            close_bins_groups = group_duplicates(close_bins_idx)
            log.info("Merging groups of bins %s", close_bins_groups)
            gm_bins_lb_merged, densities_merged = merge_points(gm_bins[0], densities, close_bins_groups,
                                                               interpolated_distribution)
            groups_start_idx = [lst[0] for lst in close_bins_groups]
            groups_start_mask = np.isin(np.arange(gm_bins.shape[1]), groups_start_idx)
            gm_bins_ub_merged = gm_bins[1][keep_idx]
            gm_bins_ub_merged[groups_start_mask[keep_idx]] = 0
            return np.array([gm_bins_lb_merged, gm_bins_ub_merged]), densities_merged, keep_idx
        else:
            return gm_bins, densities, np.arange(gm_bins.shape[-1])