import logging
from typing import Iterable, Callable, Union, Sequence

import numpy as np
import astropy.units as u
from astropy.units import Quantity
from astropy.constants import c, e
from numpy._typing import NDArray

from agnpy import Blob, InterpolatedDistribution, Synchrotron, SynchrotronSelfCompton
from agnpy.utils.conversion import mec2

log = logging.getLogger(__name__)

# A function type that takes a Quantity-array (unitless Lorentz gamma factors)
# and returns a new Quantity-array (with units of energy/time, same length).
EnergyChangeFnType = Callable[[Quantity], Quantity]

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

    def __init__(self, blob: Blob, time: Quantity, energy_change_functions: Union[EnergyChangeFnType, Sequence[EnergyChangeFnType]]):
        self._blob = blob
        self._total_time_sec = time.to("s")
        self._energy_change_functions = energy_change_functions if isinstance(energy_change_functions, Iterable) else [
            energy_change_functions]

    def eval_with_fixed_intervals(self, intervals_count=1, method="heun", max_energy_change_per_interval=0.01, max_density_change_per_interval=0.5):
        """
        Performs the time evolution of the electron distribution inside the blob, repeating the calculation in the
        equal-length time intervals.

        Parameters
        ----------
        intervals_count : int
            the calculation will be performed in the N steps of equal duration (time/N),
            calculating energy change for every gamma value in each step
        method
            numerical method for calculating energy evolution; accepted values (case-insensitive):
             "heun" (default, more precise) or "euler" (2x faster, but more than 2x less precise)
        max_energy_change_per_interval
            maximum relative change of the electron energy allowed in one subinterval (if exceeded, will raise an error)
        max_density_change_per_interval
            maximum relative change of the electron density allowed in one time interval (if exceeded, will raise an error)
        """

        if intervals_count <= 0:
            raise ValueError("intervals_count must be > 0")
        self._validate_method(method)

        unit_time_interval_sec = (self._total_time_sec / intervals_count).to("s")
        # for logging only:
        fmt_width = len(str(intervals_count))

        def do_iterative(gamma_bins_from, n_array, iteration_count):
            """
            Iterative algorithm for evaluating the electron energy and density. For each gamma point it creates a narrow bin,
            calculates the energy change for a start and an end of the bin, and scales up density by the bin narrowing factor.
            Parameters:
            gamma_bins_from - 1D array of gamma values (will be used as a lower bound for each bin)
            n_array - 1D array of differential electron densities corresponding to gamma values
            iteration_count - only for logging
            """
            bin_size_factor = 0.0001
            gamma_bins = self._interlace(gamma_bins_from, gamma_bins_from * (1 + bin_size_factor))
            energy_bins = (gamma_bins * mec2).to("erg")
            new_energy_change_rates = self._recalc_change_rates(gamma_bins)
            abs_changes = new_energy_change_rates * unit_time_interval_sec
            new_low_change_rates_mask = self._calc_new_low_change_rates_mask(
                abs_changes, energy_bins, max_energy_change_per_interval, max_density_change_per_interval)
            log.info("%*d / %d", fmt_width, iteration_count, intervals_count)
            if np.all(new_low_change_rates_mask):
                new_gamma_bins, new_n_array = self._recalc_gamma_bins_and_density(energy_bins, abs_changes, n_array)
                new_gamma_bins_from = self._deinterlace(new_gamma_bins)[0]
                self._blob.n_e = InterpolatedDistribution(new_gamma_bins_from, new_n_array)
            else:
                raise ValueError(
                    "Energy change formula returned too big value. Use shorter time ranges.")
            if method.lower() == "heun":
                energy_change_rates_recalc = self._recalc_change_rates(new_gamma_bins)
                averaged_energy_change_rates = (new_energy_change_rates + energy_change_rates_recalc) / 2
                abs_changes_recalc = averaged_energy_change_rates * unit_time_interval_sec
                new_gamma_bins_recalc, new_n_array_recalc = self._recalc_gamma_bins_and_density(energy_bins,
                                                                                                abs_changes_recalc,
                                                                                                n_array)
                new_gamma_bins = new_gamma_bins_recalc
                new_n_array = new_n_array_recalc
                self._blob.n_e = InterpolatedDistribution(self._deinterlace(new_gamma_bins)[0], new_n_array)

            return self._deinterlace(new_gamma_bins)[0], new_n_array

        last_gamma_bins_from = self._blob.gamma_e
        last_n_array = self._blob.n_e(last_gamma_bins_from)
        for i in range(intervals_count):
            last_gamma_bins_from, last_n_array = do_iterative(last_gamma_bins_from, last_n_array, i + 1)

    def eval_with_automatic_intervals(self, method="heun", max_energy_change_per_interval=0.01, max_density_change_per_interval=0.1):
        """
        Performs the time evolution of the electron distribution inside the blob, repeating the calculation by the
        automatically selected time intervals such that the energy change in each interval does not exceed the
        max_change_per_interval rate. The automatically selected time interval duration may differ per energy bins
        - shorter intervals will be used for bins where energy change is faster.

        Parameters
        ----------
        method
            numerical method for calculating energy evolution; accepted values (case-insensitive):
            "heun" (default, more precise) or "euler" (2x-3x faster, but less precise)
        max_energy_change_per_interval
            maximum relative change of the electron energy allowed in one time interval
        max_density_change_per_interval
            maximum relative change of the electron density allowed in one time interval
        """
        self._validate_method(method)

        def do_recursive(gamma_bins, energy_change_rates, n_array, low_change_rates_mask, recalculate_high_changes, time_sec,
                         depth, elapsed_time_sec):
            """
            Recursive algorithm for evaluating the electron energy and density.
            Parameters:
            gamma_bins - array of shape (2N,) consisting of inlined pairs of gamma values (lower and upper bounds of gamma bins)
            energy_change_rates - array of shape (2N,) consisting of most recently calculated energy change rates corresponding to gamma_bins values
            n_array - array of shape (N,), contains differential electron densities corresponding to lower ends of gamma bins
            low_change_rates_mask - array of shape (N,), a mask of bins that do not need recalculation
            recalculate_high_changes - boolean; if false, recalculation of energy_change_rate array is skipped, irrespective of the mask
            depth - recursion depth, only for logging
            elapsed_time_sec - total elapsed time in seconds, only for logging
            """

            if recalculate_high_changes:
                # update bin ends, but only for unmasked bins
                start_bin_indices = np.arange(0, gamma_bins.size, 2)
                end_bin_indices = np.arange(1, gamma_bins.size, 2)
                gamma_bins[end_bin_indices[~low_change_rates_mask]] = gamma_bins[start_bin_indices[~low_change_rates_mask]] * (1 + bin_size_factor)

            energy_bins = (gamma_bins * mec2).to("erg")
            new_energy_change_rates = self._recalc_change_rates(
                gamma_bins, energy_change_rates, low_change_rates_mask) if recalculate_high_changes else np.copy(
                energy_change_rates)
            abs_changes = new_energy_change_rates * time_sec
            new_low_change_rates_mask = self._calc_new_low_change_rates_mask(
                abs_changes, energy_bins, max_energy_change_per_interval, max_density_change_per_interval)
            if np.all(new_low_change_rates_mask):
                new_gamma_bins, new_n_array = self._recalc_gamma_bins_and_density(energy_bins, abs_changes, n_array)
                invalid_density = np.isnan(new_n_array)
                if any(invalid_density):
                    new_low_change_rates_mask[invalid_density] = False

            self._log_mask_info(new_low_change_rates_mask, depth, elapsed_time_sec)
            removed_mask_indices = np.empty((0,), dtype=int)
            if np.all(new_low_change_rates_mask):
                new_gamma_bins_from, new_gamma_bins_to = self._deinterlace(new_gamma_bins)
                if not np.all(new_gamma_bins_from[:-1] <= new_gamma_bins_from[1:]):
                    sort_indices = np.argsort(new_gamma_bins_from)
                    new_gamma_bins_from = new_gamma_bins_from[sort_indices]
                    new_gamma_bins_to = new_gamma_bins_to[sort_indices]
                    new_gamma_bins = self._interlace(new_gamma_bins_from, new_gamma_bins_to)
                    new_n_array = new_n_array[sort_indices]

                # if any two different gamma points map into the same point, we will not be able to make the interpolated distribution
                # So we need to merge them into a single point
                # (other option is to mark them as invalid in the mask and try recalculating in half-time, but in practice it tends to end in endless loops)
                duplicates = (np.log10(new_gamma_bins_from)[:-1] == np.log10(new_gamma_bins_from)[1:])
                if np.any(duplicates):
                    removed_mask_indices = np.where(duplicates)[0] + 1
                    for index in removed_mask_indices[::-1]:
                        new_n_array[index-1] = new_n_array[index-1] + new_n_array[index]
                        new_n_array = np.delete(new_n_array, index)
                        new_gamma_bins_from = np.delete(new_gamma_bins_from, index)
                        new_gamma_bins_to = np.delete(new_gamma_bins_to, index)
                        new_gamma_bins = self._interlace(new_gamma_bins_from, new_gamma_bins_to)

                self._blob.n_e = InterpolatedDistribution(new_gamma_bins_from, new_n_array)
            else:
                half_time = time_sec / 2
                new_gamma_bins, new_n_array, removed_mask_indices = do_recursive(gamma_bins, new_energy_change_rates, n_array,
                                                           new_low_change_rates_mask, False, half_time,
                                                           depth + 1, elapsed_time_sec)
                new_low_change_rates_mask = np.delete(new_low_change_rates_mask, removed_mask_indices)
                removed_mask_indices_doubled = [x * 2 for x in removed_mask_indices] + [x * 2 + 1 for x in removed_mask_indices]
                new_energy_change_rates = np.delete(new_energy_change_rates, removed_mask_indices_doubled)
                new_gamma_bins, new_n_array, new_removed_mask_indices = do_recursive(new_gamma_bins, new_energy_change_rates, new_n_array,
                                                           new_low_change_rates_mask, True, half_time,
                                                           depth + 1, elapsed_time_sec + half_time)
                removed_mask_indices = self._merge_removed_mask_indices(removed_mask_indices, new_removed_mask_indices)
            if method.lower() == "heun":
                # only recalculate the bins unmasked by low_change_rates_mask but masked by new_low_change_rates_mask
                heun_recalc_mask = new_low_change_rates_mask & ~low_change_rates_mask
                heun_recalc_mask_2 = np.repeat(heun_recalc_mask, 2)
                energy_change_rates_recalc = \
                    self._recalc_change_rates(new_gamma_bins, new_energy_change_rates, ~heun_recalc_mask)[heun_recalc_mask_2]
                averaged_energy_change_rates = (new_energy_change_rates[heun_recalc_mask_2] + energy_change_rates_recalc) / 2
                abs_changes_recalc = averaged_energy_change_rates * time_sec
                new_gamma_bins_recalc, new_n_array_recalc = self._recalc_gamma_bins_and_density(
                    energy_bins[heun_recalc_mask_2],
                    abs_changes_recalc,
                    n_array[heun_recalc_mask])
                if any(np.isnan(new_n_array_recalc)):
                    raise ValueError("Illegal negative density obtained") #TODO how to handle it correctly?
                new_gamma_bins[heun_recalc_mask_2] = new_gamma_bins_recalc
                new_n_array[heun_recalc_mask] = new_n_array_recalc
                new_gamma_bins_from, new_gamma_bins_to = self._deinterlace(new_gamma_bins)
                if not np.all(new_gamma_bins_from[:-1] <= new_gamma_bins_from[1:]):
                    sort_indices = np.argsort(new_gamma_bins_from)
                    new_gamma_bins_from = new_gamma_bins_from[sort_indices]
                    new_gamma_bins_to = new_gamma_bins_to[sort_indices]
                    new_gamma_bins = self._interlace(new_gamma_bins_from, new_gamma_bins_to)
                    new_n_array = new_n_array[sort_indices]
                self._blob.n_e = InterpolatedDistribution(new_gamma_bins_from, new_n_array)

            return new_gamma_bins, new_n_array, removed_mask_indices

        bin_size_factor = 0.0001
        initial_gamma_bins_from = self._blob.gamma_e
        dummy_bins_ends = np.zeros_like(initial_gamma_bins_from)
        initial_gamma_bins = self._interlace(initial_gamma_bins_from, dummy_bins_ends)
        dummy_change_rates = np.zeros_like(initial_gamma_bins) * u.Unit("erg s-1")
        initial_n_array = self._blob.n_e(initial_gamma_bins_from)
        empty_mask = np.repeat(False, len(initial_n_array))
        do_recursive(initial_gamma_bins, dummy_change_rates, initial_n_array, empty_mask, True, self._total_time_sec, 1, 0*u.s)

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
    def _recalc_gamma_bins_and_density(energy_bins, abs_energy_changes, n_array):
        new_energy_bins, density_increase = TimeEvolution._recalc_energy_bins_and_density_increase(abs_energy_changes, energy_bins)
        new_n_array = n_array * density_increase
        new_gamma_bins = (new_energy_bins / mec2).to_value('')
        return new_gamma_bins, new_n_array

    @staticmethod
    def _recalc_energy_bins_and_density_increase(abs_energy_changes, energy_bins):
        energy_bins_from, energy_bins_to = TimeEvolution._deinterlace(energy_bins)
        energy_bins_width = energy_bins_to - energy_bins_from
        new_energy_bins = energy_bins + abs_energy_changes
        new_energy_bins_from, new_energy_bins_to = TimeEvolution._deinterlace(new_energy_bins)
        new_energy_bins_width = new_energy_bins_to - new_energy_bins_from
        invalid_width = new_energy_bins_to - new_energy_bins_from < 0
        density_increase = energy_bins_width / new_energy_bins_width
        density_increase[invalid_width] = np.nan
        return new_energy_bins, density_increase

    def _log_mask_info(self, low_change_rates_mask: NDArray[np.bool_], depth: int, elapsed_time_sec):
        if log.isEnabledFor(logging.INFO):
            count = sum(low_change_rates_mask)
            total = len(low_change_rates_mask)
            width = min(3, len(str(total)))
            progress_percent = int(100 * elapsed_time_sec / self._total_time_sec)
            msg = f"{str(progress_percent).rjust(3)}% {str(count).rjust(width)}/{str(total).rjust(width)} " + depth * "-"
            log.info(msg)

    @staticmethod
    def _calc_new_low_change_rates_mask(abs_changes, energy_bins, max_energy_change_per_interval, max_density_change_per_interval):
        relative_changes_bin_start, relative_changes_bin_end = TimeEvolution._deinterlace(abs_changes / energy_bins)
        density_increase = TimeEvolution._recalc_energy_bins_and_density_increase(abs_changes, energy_bins)[1]
        new_low_change_rates_mask = ((abs(relative_changes_bin_start) < max_energy_change_per_interval) &
                                     (abs(relative_changes_bin_end) < max_energy_change_per_interval) &
                                     (density_increase < 1 + max_density_change_per_interval) &
                                     (1 / (1 + max_density_change_per_interval) < density_increase))
        return new_low_change_rates_mask

    @staticmethod
    def _merge_removed_mask_indices(removed_indices_step1, removed_indices_step2):
        """
        Merge two sets of removed indices from two consecutive removal steps.

        Parameters
        ----------
        removed_indices_step1 : numpy array
            Indices removed in step 1 (relative to the original array)
        removed_indices_step2 : numpy array
            Indices removed in step 2 (relative to the array after step 1)

        Returns
        -------
        result: np.ndarray
            Sorted array of all removed indices relative to the original array
        """
        removed_indices_step1 = np.sort(removed_indices_step1)

        remapped_indices_step2 = np.empty((0,), dtype=int)
        for val2 in removed_indices_step2:
            found = False
            for val1_idx in range(len(removed_indices_step1) - 1, -1, -1):
                if val2 >= removed_indices_step1[val1_idx] - val1_idx:
                    remapped_indices_step2 = np.append(remapped_indices_step2, val2 + val1_idx + 1)
                    found = True
                    break
            if not found:
                remapped_indices_step2 = np.append(remapped_indices_step2, val2)

        combined = np.concatenate((removed_indices_step1, remapped_indices_step2))
        combined = np.sort(combined)
        return combined
