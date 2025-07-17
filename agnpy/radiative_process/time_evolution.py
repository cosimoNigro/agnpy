import astropy.units as u
import logging
import numpy as np
from agnpy import Blob, InterpolatedDistribution, Synchrotron, SynchrotronSelfCompton
from agnpy.utils.conversion import mec2
from astropy.constants import c, e
from astropy.units import Quantity
from numpy._typing import NDArray
from typing import Iterable, Callable, Union, Sequence, Literal

log = logging.getLogger(__name__)

# A function type that takes a Quantity-array (unitless Lorentz gamma factors)
# and returns a new Quantity-array (with units of energy/time, same length).
EnergyChangeFnType = Callable[[Quantity], Quantity]
NumericalMethod = Literal["euler", "heun"]
EnergyChangeFns = Union[EnergyChangeFnType, Sequence[EnergyChangeFnType]]

def synchrotron_loss(sync: Synchrotron) -> EnergyChangeFnType:
    return lambda gamma: sync.electron_energy_loss_rate(gamma) * -1

def ssc_loss(sync: SynchrotronSelfCompton) -> EnergyChangeFnType:
    return lambda gamma: sync.electron_energy_loss_rate(gamma) * -1

def ssc_thomson_limit_loss(sync: SynchrotronSelfCompton) -> EnergyChangeFnType:
    return lambda gamma: sync.electron_energy_loss_rate_thomson(gamma) * -1

def fermi_acceleration(blob) -> EnergyChangeFnType:
    return lambda gamma: blob.xi * blob.B_cgs * c * e.gauss

class TimeEvolution:
    """
    Evaluates the change of electron spectral density inside the blob.

    Parameters
    ----------
    blob : ~agnpy.Blob
        The blob for which the time evolution will be performed. As a result of the time evolution,
        the blob.n_e will be replaced with the InterpolatedDistribution.
    time : ~astropy.units.Quantity
         Total time for the calculation
    energy_change_functions : a function, or an array of functions, of EnergyChangeFnType type
         The function(s) to be used for calculation of energy change rate per gamma values
         (for energy gain processes, function should return positive values; for loss, negative values)
    """

    def __init__(self, blob: Blob, time: Quantity, energy_change_functions: EnergyChangeFns):
        self._blob = blob
        self._total_time_sec = time.to("s")
        self._energy_change_functions = energy_change_functions if isinstance(energy_change_functions, Iterable) else [
            energy_change_functions]

    def eval_with_fixed_intervals(self,
                                  intervals_count: int,
                                  method: NumericalMethod = "heun",
                                  max_energy_change_per_interval: float = 0.01,
                                  max_density_change_per_interval: float = 0.5) -> None:
        """
        Performs the time evolution of the electron distribution inside the blob, repeating the calculation in the
        equal-length time intervals.

        Parameters
        ----------
        intervals_count : int
            the calculation will be performed in the N steps of equal duration (time/N),
            calculating energy change for every gamma value in each step
        method
            numerical method for calculating energy evolution; accepted values:
             "heun" (default, more precise) or "euler" (2x faster, but more than 2x less precise)
        max_energy_change_per_interval
            maximum relative change of the electron energy allowed in one subinterval (if exceeded, will raise an error)
        max_density_change_per_interval
            maximum relative change of the electron density allowed in one time interval (if exceeded, will raise an error)

        Side Effects
        ------------
        Replaces the blob.n_e with the new InterpolatedDistribution
        """

        if intervals_count <= 0:
            raise ValueError("intervals_count must be > 0")
        self._validate_method(method)

        unit_time_interval_sec = (self._total_time_sec / intervals_count).to("s")
        # for logging only:
        fmt_width = len(str(intervals_count))

        def do_iterative(gamma_bins_from, iteration_count):
            """
            Iterative algorithm for evaluating the electron energy and density. For each gamma point it creates a narrow bin,
            calculates the energy change for a start and an end of the bin, and scales up density by the bin narrowing factor.
            Parameters:
                gamma_bins_from - array of gamma values (will be used as a lower bound for each bin)
                iteration_count - only for logging
            Side Effects:
                Replaces the blob.n_e with the new InterpolatedDistribution
            Returns:
                new_gamma_bins_from
            """
            bin_size_factor = 0.0001
            gamma_bins = self._interlace(gamma_bins_from, gamma_bins_from * (1 + bin_size_factor))
            energy_bins = (gamma_bins * mec2).to("erg")
            new_energy_change_rates = self._recalc_change_rates(gamma_bins)
            abs_changes = new_energy_change_rates * unit_time_interval_sec
            new_low_change_rates_mask = self._calc_new_low_change_rates_mask(
                abs_changes, energy_bins, max_energy_change_per_interval, max_density_change_per_interval)
            log.info("%*d / %d", fmt_width, iteration_count, intervals_count)
            n_e = self._blob.n_e
            if np.all(new_low_change_rates_mask):
                new_gamma_bins, new_n_array = self._recalc_gamma_bins_and_density(energy_bins, abs_changes, n_e)
                new_gamma_bins_from = self._deinterlace(new_gamma_bins)[0]
                self._blob.n_e = self._make_interpolated_distribution(new_gamma_bins_from, new_n_array)
            else:
                raise ValueError(
                    "Energy change formula returned too big value. Use shorter time ranges.")
            if method.lower() == "heun":
                energy_change_rates_recalc = self._recalc_change_rates(new_gamma_bins)
                averaged_energy_change_rates = (new_energy_change_rates + energy_change_rates_recalc) / 2
                abs_changes_recalc = averaged_energy_change_rates * unit_time_interval_sec
                new_gamma_bins, new_n_array =\
                    self._recalc_gamma_bins_and_density(energy_bins, abs_changes_recalc, n_e)
                new_gamma_bins_from = self._deinterlace(new_gamma_bins)[0]
                self._blob.n_e = self._make_interpolated_distribution(new_gamma_bins_from, new_n_array)

            new_gamma_bins_from = np.sort(new_gamma_bins_from)
            duplicates = TimeEvolution._get_duplicates(new_gamma_bins_from)
            new_gamma_bins_from = np.delete(new_gamma_bins_from, duplicates)
            return new_gamma_bins_from

        last_gamma_bins_from = self._blob.gamma_e
        for i in range(intervals_count):
            last_gamma_bins_from = do_iterative(last_gamma_bins_from, i + 1)

    def eval_with_automatic_intervals(self,
                                      method: NumericalMethod = "heun",
                                      max_energy_change_per_interval: float = 0.01,
                                      max_density_change_per_interval: float = 0.1) -> None:
        """
        Performs the time evolution of the electron distribution inside the blob, repeating the calculation by the
        automatically selected time intervals such that the energy change in each interval does not exceed the
        max_change_per_interval rate. The automatically selected time interval duration may differ per energy bins
        - shorter intervals will be used for bins where energy change is faster.

        Parameters
        ----------
        method
            numerical method for calculating energy evolution; accepted values:
            "heun" (default, more precise) or "euler" (usually 2x-3x faster, but less precise - but in some scenarios might be actually slower)
        max_energy_change_per_interval
            maximum relative change of the electron energy allowed in one time interval
        max_density_change_per_interval
            maximum relative change of the electron density allowed in one time interval

        Side Effects
        ------------
        Replaces the blob.n_e with the new InterpolatedDistribution
        """
        self._validate_method(method)

        def do_recursive(gamma_bins, energy_change_rates, low_change_rates_mask, recalculate_high_changes, time_sec,
                         depth, elapsed_time_sec):
            """
            Recursive algorithm for evaluating the electron energy and density.
            Parameters:
                gamma_bins - array of shape (2N,) consisting of inlined pairs of gamma values (lower and upper bounds of gamma bins)
                energy_change_rates - array of shape (2N,) consisting of most recently calculated energy change rates corresponding to gamma_bins values
                low_change_rates_mask - array of shape (N,), a mask of bins that do not need recalculation
                recalculate_high_changes - boolean; if false, recalculation of energy_change_rate array is skipped, irrespective of the mask
                depth - recursion depth, only for logging
                elapsed_time_sec - total elapsed time in seconds, only for logging
            Side Effects:
                Replaces the blob.n_e with the new InterpolatedDistribution
            Returns:
                new_gamma_bins, mapping
            """

            if recalculate_high_changes:
                # update bin ends, but only for unmasked bins
                start_bin_indices = np.arange(0, gamma_bins.size, 2)
                end_bin_indices = np.arange(1, gamma_bins.size, 2)
                gamma_bins[end_bin_indices[~low_change_rates_mask]] = gamma_bins[start_bin_indices[~low_change_rates_mask]] * (1 + bin_size_factor)
                new_energy_change_rates = self._recalc_change_rates(gamma_bins, energy_change_rates, low_change_rates_mask)
            else:
                new_energy_change_rates = np.copy(energy_change_rates)
            abs_changes = (new_energy_change_rates * time_sec).to("erg")
            energy_bins = (gamma_bins * mec2).to("erg")
            new_low_change_rates_mask = self._calc_new_low_change_rates_mask(
                abs_changes, energy_bins, max_energy_change_per_interval, max_density_change_per_interval)

            log_mask_info(new_low_change_rates_mask, depth, elapsed_time_sec)
            n_e = self._blob.n_e
            if np.all(new_low_change_rates_mask):
                new_gamma_bins, new_n_array = self._recalc_gamma_bins_and_density(energy_bins, abs_changes, n_e)
                new_gamma_bins_from, new_gamma_bins_to = self._deinterlace(new_gamma_bins)
                self._blob.n_e = self._make_interpolated_distribution(new_gamma_bins_from, new_n_array)
                bin_idx_mapping = np.arange(len(new_gamma_bins_from))
            else:
                half_time = time_sec / 2
                new_gamma_bins, first_mapping = do_recursive(gamma_bins,
                                                             new_energy_change_rates,
                                                             new_low_change_rates_mask,
                                                             False,
                                                             half_time, depth + 1, elapsed_time_sec)
                new_energy_change_rates = remap_energy_change_rates(first_mapping, new_energy_change_rates)
                new_gamma_bins, second_mapping = do_recursive(new_gamma_bins,
                                                              new_energy_change_rates,
                                                              new_low_change_rates_mask[first_mapping],
                                                              True,
                                                              half_time, depth + 1, elapsed_time_sec + half_time)
                new_energy_change_rates = remap_energy_change_rates(second_mapping, new_energy_change_rates)
                bin_idx_mapping = first_mapping[second_mapping] # combined first and second mapping
                new_gamma_bins_from, new_gamma_bins_to = self._deinterlace(new_gamma_bins)
            # only recalculate the bins not masked by low_change_rates_mask but masked by new_low_change_rates_mask
            recalc_mask = (new_low_change_rates_mask & ~low_change_rates_mask)[bin_idx_mapping]
            if method.lower() == "heun" and np.any(recalc_mask):
                recalc_mask_dbl = np.repeat(recalc_mask, 2)
                energy_change_rates_recalc = self._recalc_change_rates(new_gamma_bins, new_energy_change_rates, ~recalc_mask)
                averaged_energy_change_rates = (new_energy_change_rates[recalc_mask_dbl] + energy_change_rates_recalc[recalc_mask_dbl]) / 2
                abs_changes_recalc = averaged_energy_change_rates * time_sec
                new_gamma_bins_recalc, new_n_array_recalc = self._recalc_gamma_bins_and_density(
                    energy_bins[recalc_mask_dbl],
                    abs_changes_recalc,
                    n_e)
                if any(np.isnan(new_n_array_recalc)):
                    # in this case, we should roll back the sub-calculation progress and retry with the new mask from the beginning of the current method
                    raise ValueError("Illegal negative density obtained - the resolution algorithm not yet implemented, please report it to the agnpy maintainers")
                new_gamma_bins[recalc_mask_dbl] = new_gamma_bins_recalc
                new_gamma_bins_from, new_gamma_bins_to = self._deinterlace(new_gamma_bins)
                new_n_e = self._blob.n_e
                new_n_array = np.zeros_like(new_gamma_bins_from) * u.Unit("cm-3")
                new_n_array[~recalc_mask] = new_n_e(new_gamma_bins_from[~recalc_mask])
                new_n_array[recalc_mask] = new_n_array_recalc
                self._blob.n_e = self._make_interpolated_distribution(new_gamma_bins_from, new_n_array)

            second_bin_idx_mapping = TimeEvolution._sort_and_deduplicate(new_gamma_bins_from, recalc_mask)
            new_gamma_bins_from = new_gamma_bins_from[second_bin_idx_mapping]
            new_gamma_bins_to = new_gamma_bins_to[second_bin_idx_mapping]
            return self._interlace(new_gamma_bins_from, new_gamma_bins_to), bin_idx_mapping[second_bin_idx_mapping]

        def remap_energy_change_rates(first_mapping, new_energy_change_rates):
            new_energy_change_rates_from, new_energy_change_rates_to = self._deinterlace(new_energy_change_rates)
            return self._interlace(new_energy_change_rates_from[first_mapping], new_energy_change_rates_to[first_mapping])

        def log_mask_info(low_change_rates_mask: NDArray[np.bool_], depth: int, elapsed_time_sec):
            if log.isEnabledFor(logging.INFO):
                count = sum(low_change_rates_mask)
                total = len(low_change_rates_mask)
                width = min(3, len(str(total)))
                progress_percent = int(100 * elapsed_time_sec / self._total_time_sec)
                msg = f"{str(progress_percent).rjust(3)}% {str(count).rjust(width)}/{str(total).rjust(width)} " + depth * "-"
                log.info(msg)

        bin_size_factor = 0.0001
        initial_gamma_bins_from = self._blob.gamma_e
        dummy_bins_ends = np.zeros_like(initial_gamma_bins_from)
        initial_gamma_bins = self._interlace(initial_gamma_bins_from, dummy_bins_ends)
        dummy_change_rates = np.zeros_like(initial_gamma_bins) * u.Unit("erg s-1")
        empty_mask = np.repeat(False, len(initial_gamma_bins_from))
        do_recursive(initial_gamma_bins, dummy_change_rates, empty_mask, True, self._total_time_sec, 1, 0*u.s)

    def _recalc_change_rates(self, gamma_bins, previous_energy_change_rates=None, low_change_rates_mask=None):
        """
        Calculates (or recalculates) the energy changes array.
        If the mask is provided, only unmasked elements will be recalculated, and previous_energy_change_rates will be used for masked elements.
        gamma_bins - array of shape (2N,), consisting of inlined pairs of gamma values (lower and upper bounds of gamma bins)
        previous_energy_change_rates - optional array of shape (2N,), consisting of most recently calculated energy change rates corresponding to gamma_bins values
        low_change_rates_mask - optional array of shape (N,) a mask of bins that do not need recalculation
        """
        mask = np.ones_like(gamma_bins, dtype=bool) if low_change_rates_mask is None else np.repeat(~low_change_rates_mask, 2)
        new_energy_change_rates = np.zeros_like(gamma_bins) * u.Unit(
            "erg s-1") if previous_energy_change_rates is None else previous_energy_change_rates.copy()
        new_energy_change_rates[mask] = np.zeros_like(gamma_bins[mask]) * u.Unit("erg s-1")
        for en_change_fn in self._energy_change_functions:
            new_energy_change_rates[mask] += en_change_fn(gamma_bins[mask]).to("erg s-1")
        return new_energy_change_rates

    @staticmethod
    def _interlace(gamma_bins_start: NDArray, gamma_bins_end: NDArray) -> NDArray:
        """ Combines two gamma arrays into one array, to make the calculations on them in one steps instead of two;
            the first array goes into odd indices, the second into even indices """
        return np.column_stack((gamma_bins_start, gamma_bins_end)).ravel()

    @staticmethod
    def _deinterlace(interlaced: NDArray) -> (NDArray, NDArray):
        """ reverts the interlace method """
        gamma_bins_start = interlaced[0::2]
        gamma_bins_end = interlaced[1::2]
        return gamma_bins_start, gamma_bins_end

    @staticmethod
    def _validate_method(method):
        valid_methods = {"heun", "euler"}
        if method.lower() not in valid_methods:
            raise ValueError(f"Invalid method '{method}'. Expected one of: {', '.join(valid_methods)}")

    @staticmethod
    def _recalc_gamma_bins_and_density(energy_bins, abs_energy_changes, n_e):
        gamma_bins_start = (TimeEvolution._deinterlace(energy_bins)[0] / mec2).to_value('')
        n_array = n_e(gamma_bins_start)
        new_energy_bins, density_increase = TimeEvolution._recalc_energy_bins_and_density_increase(abs_energy_changes, energy_bins)
        new_n_array = n_array * density_increase
        new_gamma_bins = (new_energy_bins / mec2).to_value('')
        return new_gamma_bins, new_n_array

    @staticmethod
    def _recalc_energy_bins_and_density_increase(abs_energy_changes, energy_bins):
        """
        abs_energy_changes - array of shape (2N,)
        energy_bins - array of shape (2N,)
        return ((2N,), (N,)) - note: some of the elements of density_increase may be nan
        """
        energy_bins_from, energy_bins_to = TimeEvolution._deinterlace(energy_bins)
        energy_bins_width = energy_bins_to - energy_bins_from
        new_energy_bins = energy_bins + abs_energy_changes
        new_energy_bins_from, new_energy_bins_to = TimeEvolution._deinterlace(new_energy_bins)
        new_energy_bins_width = new_energy_bins_to - new_energy_bins_from
        density_increase = energy_bins_width / new_energy_bins_width
        invalid_width = new_energy_bins_to - new_energy_bins_from < 0
        density_increase[invalid_width] = np.nan # mark with nan the bins where bin's start and end get swapped
        return new_energy_bins, density_increase

    @staticmethod
    def _calc_new_low_change_rates_mask(abs_changes, energy_bins, max_energy_change_per_interval, max_density_change_per_interval):
        relative_changes_bin_start, relative_changes_bin_end = TimeEvolution._deinterlace(abs_changes / energy_bins)
        density_increase = TimeEvolution._recalc_energy_bins_and_density_increase(abs_changes, energy_bins)[1]
        new_low_change_rates_mask = ((abs(relative_changes_bin_start) < max_energy_change_per_interval) &
                                     (abs(relative_changes_bin_end) < max_energy_change_per_interval) &
                                     (~np.isnan(density_increase)) &
                                     (density_increase < 1 + max_density_change_per_interval) &
                                     (1 / (1 + max_density_change_per_interval) < density_increase))
        return new_low_change_rates_mask

    @staticmethod
    def _make_interpolated_distribution(gamma_array, n_array):
        sort_indices = np.argsort(gamma_array)
        gamma_array_sorted = gamma_array[sort_indices]
        n_array_sorted = n_array[sort_indices]
        # If any two different gamma points map into the same point, we will not be able to make the interpolated distribution,
        # hence we need to merge them into a single point. Also consider the fact the InterpolatedDistribution
        # uses the log10 values of the gamma points, so two close-enough gamma points can collapse into a single point after log10.
        duplicated_indices = TimeEvolution._get_duplicates(gamma_array_sorted)
        gamma_array_sorted = np.delete(gamma_array_sorted, duplicated_indices)
        for index in duplicated_indices[::-1]:
            n_array_sorted[index - 1] = n_array_sorted[index - 1] + n_array_sorted[index]
            n_array_sorted = np.delete(n_array_sorted, index)
        if len(gamma_array_sorted) == 1:
            raise ValueError("Unsupported state, cannot create InterpolatedDistribution - distribution collapsed to a single gamma point " + str(gamma_array_sorted[0]))
        return InterpolatedDistribution(gamma_array_sorted, n_array_sorted)

    @staticmethod
    def _sort_and_deduplicate(array, mask=None, element_transform = np.log10):
        sort_idx = np.argsort(array)
        sorted_array = array[sort_idx]
        sorted_mask = mask[sort_idx]
        duplicate_idx = TimeEvolution._get_duplicates(sorted_array, mask=sorted_mask, element_transform=element_transform)
        keep_idx = np.setdiff1d(np.arange(len(sort_idx)), duplicate_idx, assume_unique=True)
        return sort_idx[keep_idx]

    @staticmethod
    def _get_duplicates(sorted_array, mask=None, element_transform = np.log10):
        if mask is None:
            mask = np.repeat(True, len(sorted_array))
        duplicates = element_transform(sorted_array)[:-1] == element_transform(sorted_array)[1:]
        both_masked = mask[:-1] & mask[1:]
        match = duplicates & both_masked
        if np.any(match):
            removed_mask_indices = np.where(match)[0] + 1
            return removed_mask_indices
        else:
            return []