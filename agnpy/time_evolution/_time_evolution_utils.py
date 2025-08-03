import random

import numpy as np
import astropy.units as u
from agnpy import InterpolatedDistribution, ParticleDistribution
from agnpy.utils.conversion import mec2
from numpy._typing import NDArray
from scipy.interpolate import PchipInterpolator

bin_size_factor = 0.0001
zero_density_approximation = 1e-1 * u.Unit("cm-3")


def update_bin_ub(gamma_bins_from, gamma_bins_to, low_change_rates_mask = None):
    """ Update bin upper bounds, but only for unmasked bins
    """
    low_change_rates_mask = low_change_rates_mask if low_change_rates_mask is not None else np.zeros_like(gamma_bins_from, bool)
    gamma_bins_to[~low_change_rates_mask] = gamma_bins_from[~low_change_rates_mask] * (1 + bin_size_factor)


def recalc_change_rates(gamma_bins, energy_change_functions, previous_energy_change_rates={},
                        low_change_rates_mask=None):
    """
    Calculates (or recalculates) the energy changes array.
    If the mask is provided, only unmasked elements will be recalculated, and previous_energy_change_rates will be used for masked elements.
    gamma_bins - array of shape (2N,), consisting of inlined pairs of gamma values (lower and upper bounds of gamma bins)
    previous_energy_change_rates - optional dict of arrays of shape (2N,) each, consisting of most recently calculated energy change rates corresponding to gamma_bins values
    low_change_rates_mask - optional array of shape (N,) a mask of bins that do not need recalculation
    """
    mask = np.ones_like(gamma_bins, dtype=bool) if low_change_rates_mask is None else np.repeat(~low_change_rates_mask,
                                                                                                2)
    new_energy_change_rates = {}
    for label, fns in energy_change_functions.items():
        new_energy_change_rates[label] = np.zeros_like(gamma_bins) * u.Unit("erg s-1") \
            if previous_energy_change_rates.get(label) is None \
            else previous_energy_change_rates[label].copy()
        new_energy_change_rates[label][mask] = np.zeros_like(gamma_bins[mask]) * u.Unit("erg s-1")
        for en_change_fn in fns:
            new_energy_change_rates[label][mask] += en_change_fn(gamma_bins[mask]).to("erg s-1")
    return new_energy_change_rates


def recalc_chg_rates_boundary_bins(gm_bins_lb, energy_change_functions):
    return recalc_change_rates(np.array([gm_bins_lb[0], gm_bins_lb[-1]]), energy_change_functions)


def sum_change_rates(last_energy_change_rates):
    return sum(last_energy_change_rates.values())


def remap_simple(mapping, values):
    injection_remapped = values[mapping]
    injection_remapped[mapping == -1] = np.nan
    return injection_remapped


def remap_energy_change_rates(mapping, energy_change_rates):
    result = {}
    for key, val in energy_change_rates.items():
        en_chg_rates_lb, en_ch_rates_ub = deinterlace(val)
        en_chg_rates_lb = en_chg_rates_lb[mapping]
        en_ch_rates_ub = en_ch_rates_ub[mapping]
        en_chg_rates_lb[mapping == -1] = np.nan
        en_ch_rates_ub[mapping == -1] = np.nan
        result[key] = interlace(en_chg_rates_lb, en_ch_rates_ub)
    return result


def remap_energy_change_rates_lb(mapping, energy_change_rates):
    result = {}
    for key, val in energy_change_rates.items():
        en_chg_rates_lb = val[mapping]
        en_chg_rates_lb[mapping == -1] = np.nan
        result[key] = en_chg_rates_lb
    return result


def remap_low_changes_mask(mapping, low_change_rates_mask):
    low_change_rates_mask_remapped = low_change_rates_mask[mapping]
    low_change_rates_mask_remapped[mapping == -1] = False
    return low_change_rates_mask_remapped


def combine_mapping(first_mapping, second_mapping):
    """ Combines two mappings, but with special treatment of "-1" value in the mapping:
        -1 does not mean the last element; it means a "new element", at must be retained across the mappings
    """
    combined = first_mapping[second_mapping]
    combined[second_mapping == -1] = -1
    return combined


def recalc_gamma_bins_and_density(energy_bins, abs_energy_changes, densities_or_distribution):
    gamma_bins_start = (deinterlace(energy_bins)[0] / mec2).to_value('')
    n_array = densities_or_distribution(gamma_bins_start) if isinstance(densities_or_distribution,
                                                                        ParticleDistribution) else densities_or_distribution
    new_energy_bins, density_increase = recalc_energy_bins_and_density_increase(abs_energy_changes, energy_bins)
    new_n_array = n_array * density_increase
    new_gamma_bins = (new_energy_bins / mec2).to_value('')
    return new_gamma_bins, new_n_array


def to_erg(gamma_bins):
    return (gamma_bins * mec2).to("erg")


def recalc_energy_bins_and_density_increase(abs_energy_changes, energy_bins):
    """
    abs_energy_changes - array of shape (2N,)
    energy_bins - array of shape (2N,)
    return ((2N,), (N,)) - note: some of the elements of density_increase may be set to the "nan" value
    """
    energy_bins_from, energy_bins_to = deinterlace(energy_bins)
    energy_bins_width = energy_bins_to - energy_bins_from
    new_energy_bins = energy_bins + abs_energy_changes
    new_energy_bins_from, new_energy_bins_to = deinterlace(new_energy_bins)
    new_energy_bins_width = new_energy_bins_to - new_energy_bins_from
    density_increase = energy_bins_width / new_energy_bins_width
    invalid_width = new_energy_bins_to - new_energy_bins_from < 0
    density_increase[invalid_width] = np.nan  # mark with nan the bins where bin's start and end get swapped
    return new_energy_bins, density_increase


def calc_new_low_change_rates_mask(energy_bins, density, abs_energy_change, abs_particle_injection,
                                   max_energy_change_per_interval, max_density_change_per_interval):
    if np.any((density == 0) & (abs_particle_injection < 0)):
        raise ValueError("Particles cannot escape when density is zero")
    relative_changes_bin_lb, relative_changes_bin_ub = deinterlace(abs_energy_change / energy_bins)
    # If we start with a bin of zero density, and inject absolute amount of particles, the injection rate will be infinite,
    # no matter how short time we take. So instead, we convert zero density to some very small value number, and then
    # we can always find time short enough to make the relative density change also low enough.
    # TODO find a better solution
    density_corrected = np.where(density > 0, density, zero_density_approximation)
    rel_density_change_from_injection = (density_corrected + abs_particle_injection) / density_corrected
    rel_density_increase_from_energy_change = recalc_energy_bins_and_density_increase(abs_energy_change, energy_bins)[1]
    rel_density_increase_combined = rel_density_change_from_injection * rel_density_increase_from_energy_change
    new_low_change_rates_mask = ((abs(relative_changes_bin_lb) < max_energy_change_per_interval) &
                                 (abs(relative_changes_bin_ub) < max_energy_change_per_interval) &
                                 (rel_density_change_from_injection >= 0) &
                                 (~np.isnan(rel_density_increase_from_energy_change)) &
                                 (rel_density_increase_combined < 1 + max_density_change_per_interval) &
                                 (1 / (1 + max_density_change_per_interval) < rel_density_increase_combined))
    return new_low_change_rates_mask


def eval_abs_particle_injection(gamma_bins, densities,
                                rel_injection_functions, abs_injection_functions, time):
    scaling = np.repeat(1.0, len(densities))
    for injection_function in rel_injection_functions:
        scaling = scaling * injection_function(gamma_bins, densities, time)
    densities_scaled = densities * scaling
    for injection_function in abs_injection_functions:
        densities_scaled += injection_function(gamma_bins) * time
    return densities_scaled - densities


def sorting_and_deduplicating_mapping(array, mask=None, element_transform=np.log10):
    sort_idx = np.argsort(array)
    sorted_array = array[sort_idx]
    sorted_mask = mask[sort_idx] if mask is not None else np.repeat(True, len(sort_idx))
    duplicate_idx = get_duplicates(sorted_array, mask=sorted_mask, element_transform=element_transform)
    keep_idx = np.setdiff1d(np.arange(len(sort_idx)), duplicate_idx, assume_unique=True)
    return sort_idx[keep_idx]


def get_duplicates(sorted_array, mask=None, element_transform=np.log10):
    """ Return indices of the elements in the array which can be removed, i.e. these fulfilling the criteria:
        they are "masked" (i.e. mask is True for them) and equal (after transformation) to any other masked element
        which is not going to be removed (note: if no mask is provided, all elements are considered masked)
    """
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


def add_boundary_bins(gm_bins_lb, gm_bins_ub, first_bin_en_chg, last_bin_en_chg, gamma_bounds, max_bin_creep):
    mapping_with_edge_bins = np.arange(len(gm_bins_lb))
    is_first_bin_far_from_start = is_far_from_start(gamma_bounds, gm_bins_lb[0], max_bin_creep)
    if is_first_bin_far_from_start and first_bin_en_chg > 0:
        gm_bins_lb = np.concatenate([np.array([gamma_bounds[0]]), gm_bins_lb])
        gm_bins_ub = np.concatenate([np.array([gamma_bounds[0] * (1 + bin_size_factor)]), gm_bins_ub])
        mapping_with_edge_bins = np.concatenate([np.array([-1]), mapping_with_edge_bins])
    is_last_bin_far_from_end = is_far_from_end(gamma_bounds, gm_bins_lb[-1], max_bin_creep)
    if is_last_bin_far_from_end and last_bin_en_chg < 0:
        gm_bins_lb = np.concatenate([gm_bins_lb, np.array([gamma_bounds[1] / (1 + bin_size_factor)])])
        gm_bins_ub = np.concatenate([gm_bins_ub, np.array([gamma_bounds[1]])])
        mapping_with_edge_bins = np.concatenate([mapping_with_edge_bins, np.array([-1])])
    return gm_bins_lb, gm_bins_ub, mapping_with_edge_bins


def remove_gamma_beyond_bounds(gm_bins_lb, gm_bins_ub, gamma_bounds):
    if gamma_bounds is not None:
        mapping = np.where((gamma_bounds[0] <= gm_bins_lb) & (gm_bins_lb <= gamma_bounds[1] / (1 + bin_size_factor)))[0]
        if len(mapping) != len(gm_bins_lb):
            return gm_bins_lb[mapping], None if gm_bins_ub is None else gm_bins_ub[mapping], mapping
    return gm_bins_lb, gm_bins_ub, np.arange(len(gm_bins_lb))


def remove_too_close_bins(gm_bins_lb, gm_bins_ub, min_bin_creep):
    gm_bins_lb_log = np.log10(gm_bins_lb)
    gaps = np.diff(gm_bins_lb_log)
    too_close_bins = gaps < min_bin_creep
    if np.any(too_close_bins):
        indices = np.arange(len(gm_bins_lb))
        idx = int(np.argmin(np.where(too_close_bins, gaps, np.inf)))
        # for better uniformity of the bins, randomly remove lower of upper bin, unless they are extreme bins
        idx_to_remove = 1 if idx == 0 else idx if idx + 1 == len(gaps) else idx if bool(random.randint(0, 1)) else idx + 1
        mapping = np.delete(indices, idx_to_remove)
        return gm_bins_lb[mapping], None if gm_bins_ub is None else gm_bins_ub[mapping], mapping
    else:
        return gm_bins_lb, gm_bins_ub, np.arange(len(gm_bins_lb))


def update_distribution(gamma_array, n_array, blob):
    sort_indices = np.argsort(gamma_array)
    gamma_array_sorted = gamma_array[sort_indices]
    n_array_sorted = n_array[sort_indices]
    # If any two different gamma points map into the same point, we will not be able to make the interpolated distribution,
    # hence we need to merge them into a single point. Also consider the fact the InterpolatedDistribution
    # uses the log10 values of the gamma points, so two close-enough gamma points can collapse into a single point after log10.
    duplicated_indices = get_duplicates(gamma_array_sorted)
    gamma_array_sorted = np.delete(gamma_array_sorted, duplicated_indices)
    for index in duplicated_indices[::-1]:
        n_array_sorted[index - 1] = n_array_sorted[index - 1] + n_array_sorted[index]
        n_array_sorted = np.delete(n_array_sorted, index)
    if len(gamma_array_sorted) == 1:
        raise ValueError(
            "Unsupported state, cannot create InterpolatedDistribution - distribution collapsed to a single gamma point " + str(
                gamma_array_sorted[0]))
    blob.n_e = InterpolatedDistribution(gamma_array_sorted, n_array_sorted, interpolator=PchipInterpolator)


def is_far_from_start(bounds, point, max_bin_creep):
    return _is_far_from_bounds(bounds, point, max_bin_creep, True)

def is_far_from_end(bounds, point, max_bin_creep):
    return _is_far_from_bounds(bounds, point, max_bin_creep, False)

def _is_far_from_bounds(bounds, point, max_bin_creep, from_start):
    if bounds is None:
        return False
    log_start = np.log10(bounds[0])
    log_end = np.log10(bounds[1])
    log_point = np.log10(point)
    nom = log_point - log_start if from_start else log_end - log_point
    return nom / (log_end - log_start) > max_bin_creep


def is_farther_than_coeff_from_end(start, end, point, coeff):
    if start >= end:
        raise ValueError("Start must be lower than end")
    if not (start <= point <= end):
        raise ValueError("Point is not between start an end")
    log_start = np.log10(start)
    log_end = np.log10(end)
    log_point = np.log10(point)
    return (log_end - log_point) / (log_end - log_start) > coeff


def interlace(gamma_bins_start: NDArray, gamma_bins_end: NDArray) -> NDArray:
    """ Combines two gamma arrays into one array, to make the calculations on them in one steps instead of two;
        the first array goes into odd indices, the second into even indices """
    return np.column_stack((gamma_bins_start, gamma_bins_end)).ravel()


def deinterlace(interlaced: NDArray) -> tuple[NDArray, NDArray]:
    """ reverts the interlace method """
    gamma_bins_start = interlaced[0::2]
    gamma_bins_end = interlaced[1::2]
    return gamma_bins_start, gamma_bins_end


def deinterlace_energy_changes_dict(energy_changes_dict) -> dict:
    return {key: deinterlace(value)[0] for key, value in energy_changes_dict.items()}

