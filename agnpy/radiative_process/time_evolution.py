import logging
from typing import Iterable, Callable, Union, Sequence

import numpy as np
import astropy.units as u
from astropy.units import Quantity
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
        self._time = time
        self._energy_change_functions = energy_change_functions

    def eval_with_fixed_intervals(self, intervals_count=1, method="heun", max_change_per_interval=0.01):
        """
        Performs the time evolution of the electron distribution inside the blob, repeating the calculation in the
        equal-length time intervals.

        Parameters
        ----------
        intervals_count : int
            the calculation will be performed in the N steps of equal duration (time/N),
            calculating energy change for every gamma value in each step
            if set to 0 (default), then calculation will use the automatic time splitting, and recalculating energy change rate
            only for gamma values for which the relative energy change during that time exceeds max_change_per_interval
        method
            numerical method for calculating energy evolution; accepted values (case-insensitive):
             "heun" (default, more precise) or "euler" (2x faster, but more than 2x less precise)
        max_change_per_interval
            maximum relative change of the electron energy allowed in one subinterval (if exceeded, will raise an error)
        """

        if intervals_count <= 0:
            raise ValueError("intervals_count must be > 0")
        if method.lower() not in ("heun", "euler"):
            raise ValueError("Invalid method, expected HEUN or EULER")

        unit_time_interval_sec = (self._time / intervals_count).to("s")
        # for logging only:
        fmt_width = len(str(intervals_count))

        def do_iterative(gamma_bins_from, n_array, iteration):
            """
            Iterative algorithm for evaluating the electron energy and density. For each gamma point create a narrow bin,
            calculate the energy change for start and end of the bin, and scale up density by the bin narrowing factor.
            gamma_bins_from - size N
            n_array - size N
            iteration - only for debugging
            """
            bin_size_factor = 0.0001
            gamma_bins = self._interlace(gamma_bins_from, gamma_bins_from * (1 + bin_size_factor))
            energy_bins = (gamma_bins * mec2).to("erg")
            energy_bins_from, energy_bins_to = self._deinterlace(energy_bins)
            new_energy_change_rates = self.recalc_change_rates(gamma_bins)
            abs_changes = new_energy_change_rates * unit_time_interval_sec
            relative_changes = self._deinterlace(abs_changes)[0] / energy_bins_from
            new_low_change_rates_mask = abs(relative_changes) < max_change_per_interval
            log.info("%*d / %d", fmt_width, iteration, intervals_count)
            if np.all(new_low_change_rates_mask):
                new_gamma_bins, new_n_array = self._recalc_gamma_bins_and_density(energy_bins, abs_changes, n_array)
                new_gamma_bins_from = self._deinterlace(new_gamma_bins)[0]
                self._blob.n_e = InterpolatedDistribution(new_gamma_bins_from, new_n_array)
            else:
                raise ValueError(
                    "Energy change formula returned too big value. Use shorter time ranges.")
            if method.lower() == "heun":
                energy_change_rates_recalc = self.recalc_change_rates(new_gamma_bins)
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

    def eval_with_automatic_intervals(self, method="heun", max_change_per_interval=0.01):
        """
        Performs the time evolution of the electron distribution inside the blob, repeating the calculation by the
        automatically selected time intervals such that the energy change in each interval does not exceed the
        max_change_per_interval rate. The automatically selected time interval duration may differ per energy bins
        - shorter intvervals will be used for bins where energy change is faster.

        Parameters
        ----------
        method
            numerical method for calculating energy evolution; accepted values (case-insensitive):
            "heun" (default, more precise) or "euler" (2x-3x faster, but less precise)
        max_change_per_interval
            maximum relative change of the electron energy allowed in one time interval
        """
        if method.lower() not in ("heun", "euler"):
            raise ValueError("Invalid method, expected HEUN or EULER")

        def do_recursive(gamma_bins, energy_change_rates, n_array, low_change_rates_mask, recalculate_high_changes, time_sec,
                         depth):
            """
            Recursive algorithm for evaluating the electron energy and density
            gamma_bins - size 2N
            energy_change_rates - size 2N
            n_array - size N
            low_change_rates_mask - size N
            recalculate_high_changes - boolean
            depth - only for debugging
            """

            if recalculate_high_changes:
                # update bin ends, but only for unmasked bins
                start_bin_indices = np.arange(0, gamma_bins.size, 2)
                end_bin_indices = np.arange(1, gamma_bins.size, 2)
                gamma_bins[end_bin_indices[~low_change_rates_mask]] = start_bin_indices[~low_change_rates_mask] * (1 + bin_size_factor)

            energy_bins = (gamma_bins * mec2).to("erg")
            energy_bins_from = self._deinterlace(energy_bins)[0]
            new_energy_change_rates = self.recalc_change_rates(
                gamma_bins, energy_change_rates, low_change_rates_mask) if recalculate_high_changes else np.copy(
                energy_change_rates)
            abs_changes = new_energy_change_rates * time_sec
            relative_changes = self._deinterlace(abs_changes)[0] / energy_bins_from
            new_low_change_rates_mask = abs(relative_changes) < max_change_per_interval
            self._log_mask_info(new_low_change_rates_mask, depth)
            if np.all(new_low_change_rates_mask):
                new_gamma_bins, new_n_array = self._recalc_gamma_bins_and_density(energy_bins, abs_changes, n_array)
                if any(new_n_array < 0):
                    raise ValueError("Illegal negative density obtained")
                new_gamma_bins_from = self._deinterlace(new_gamma_bins)[0]
                self._blob.n_e = InterpolatedDistribution(new_gamma_bins_from, new_n_array)
            else:
                half_time = time_sec / 2
                new_gamma_bins, new_n_array = do_recursive(gamma_bins, new_energy_change_rates, n_array,
                                                           new_low_change_rates_mask, False, half_time, depth + 1)
                new_gamma_bins, new_n_array = do_recursive(new_gamma_bins, new_energy_change_rates, new_n_array,
                                                           new_low_change_rates_mask, True, half_time, depth + 1)
            if method.lower() == "heun":
                # only recalculate the bins unmasked by low_change_rates_mask but masked by new_low_change_rates_mask
                heun_recalc_mask = new_low_change_rates_mask & ~low_change_rates_mask
                heun_recalc_mask_2 = np.repeat(heun_recalc_mask, 2)
                energy_change_rates_recalc = \
                    self.recalc_change_rates(new_gamma_bins, new_energy_change_rates, ~heun_recalc_mask)[heun_recalc_mask_2]
                averaged_energy_change_rates = (new_energy_change_rates[heun_recalc_mask_2] + energy_change_rates_recalc) / 2
                abs_changes_recalc = averaged_energy_change_rates * time_sec
                new_gamma_bins_recalc, new_n_array_recalc = self._recalc_gamma_bins_and_density(
                    energy_bins[heun_recalc_mask_2],
                    abs_changes_recalc,
                    n_array[heun_recalc_mask])
                if any(new_n_array_recalc < 0):
                    raise ValueError("Illegal negative density obtained")
                new_gamma_bins[heun_recalc_mask_2] = new_gamma_bins_recalc
                new_n_array[heun_recalc_mask] = new_n_array_recalc
                self._blob.n_e = InterpolatedDistribution(self._deinterlace(new_gamma_bins)[0], new_n_array)

            return new_gamma_bins, new_n_array

        bin_size_factor = 0.0001
        initial_gamma_bins_from = self._blob.gamma_e
        dummy_bins_ends = np.zeros_like(initial_gamma_bins_from)
        initial_gamma_bins = self._interlace(initial_gamma_bins_from, dummy_bins_ends)
        dummy_change_rates = np.zeros_like(initial_gamma_bins) * u.Unit("erg s-1")
        initial_n_array = self._blob.n_e(initial_gamma_bins_from)
        empty_mask = np.repeat(False, len(initial_n_array))
        do_recursive(initial_gamma_bins, dummy_change_rates, initial_n_array, empty_mask, True, self._time.to("s"), 1)

    def recalc_change_rates(self, gamma_bins, previous_energy_change_rates=None, low_change_rates_mask=None):
        """
        Calculates (or recalculates) the energy changes array.
        If the mask is provided, only unmasked elements will be recalculated, and previous_energy_change_rates will be used for masked elements.
        gamma_bins - size 2N
        previous_energy_change_rates - size 2N
        low_change_rates_mask - size N
        """
        mask = np.ones_like(gamma_bins, dtype=bool) if low_change_rates_mask is None else np.repeat(~low_change_rates_mask, 2)
        new_energy_change_rates = np.zeros_like(gamma_bins) * u.Unit(
            "erg s-1") if previous_energy_change_rates is None else previous_energy_change_rates.copy()
        new_energy_change_rates[mask] = np.zeros_like(gamma_bins[mask]) * u.Unit("erg s-1")
        for en_change_fn in self._energy_change_functions if isinstance(self._energy_change_functions, Iterable) else [
            self._energy_change_functions]:
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
    def _recalc_gamma_bins_and_density(energy_bins, abs_energy_changes, n_array):
        energy_bins_from, energy_bins_to = TimeEvolution._deinterlace(energy_bins)
        energy_bins_width = energy_bins_to - energy_bins_from
        new_energy_bins = energy_bins + abs_energy_changes
        new_energy_bins_from, new_energy_bins_to = TimeEvolution._deinterlace(new_energy_bins)
        new_energy_bins_width = new_energy_bins_to - new_energy_bins_from
        density_increase = energy_bins_width / new_energy_bins_width
        div_by_zero = np.flatnonzero(np.isnan(density_increase))
        if div_by_zero.size > 0:
            raise ValueError("Illegal value obtained: the bin widths changed to 0 for bins " + str(div_by_zero))
        new_n_array = n_array * density_increase
        new_gamma_bins = (new_energy_bins / mec2).value
        return new_gamma_bins, new_n_array

    @staticmethod
    def _log_mask_info(low_change_rates_mask: NDArray[np.bool_], depth: int):
        if log.isEnabledFor(logging.INFO):
            count = sum(low_change_rates_mask)
            total = len(low_change_rates_mask)
            width = len(str(total))
            msg = f"{str(count).rjust(width)} / {str(total).rjust(width)} " + depth * "-"
            log.info(msg)

