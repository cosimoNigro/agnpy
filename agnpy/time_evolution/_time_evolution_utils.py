import numpy as np
from astropy.units import Quantity
from pygments.styles import vs

from agnpy import InterpolatedDistribution, ParticleDistribution
from agnpy.utils.conversion import mec2
from numpy._typing import NDArray
from scipy.interpolate import PchipInterpolator

bin_size_factor = 1e-4

def update_bin_ub(gamma_bins_from, gamma_bins_to, low_change_rates_mask = None):
    """ Update bin upper bounds, but only for unmasked bins
    """
    low_change_rates_mask = low_change_rates_mask if low_change_rates_mask is not None else np.zeros_like(gamma_bins_from, bool)
    gamma_bins_to[~low_change_rates_mask] = gamma_bins_from[~low_change_rates_mask] * (1 + bin_size_factor)


def recalc_new_rates(gm_bins, functions, unit, previous_rates=None, recalc_mask=None):
    """
    Calculates (or recalculates) the energy change rates or injection rates.
    If the mask is provided, only unmasked elements will be recalculated, and previous_rates will be used for masked elements.
    gamma_bins - array of shape (N,) consisting of gamma values (lower bounds of gamma bins), or (2N,) consisting of inlined pairs of gamma values (lower and upper bounds of gamma bins)
    previous_rates - optional dict of arrays of shape (N,) or (2N,), consisting of most recently calculated rates corresponding to gamma_bins values
    low_change_rates_mask - optional array of shape (N,) a mask of bins that do not need recalculation
    """
    if previous_rates is None:
        previous_rates = {}
    mask = np.ones_like(gm_bins, dtype=bool) if recalc_mask is None else\
        np.repeat(recalc_mask, 2) if len(recalc_mask) * 2 == len(gm_bins) else recalc_mask
    new_injection_rates = {}
    for label, fn in functions.items():
        new_injection_rates[label] = np.zeros_like(gm_bins) * unit \
            if previous_rates.get(label) is None \
            else previous_rates[label].copy()
        new_injection_rates[label][mask] = fn(gm_bins[mask]).to(unit)
    return new_injection_rates


def sum_change_rates_at_index(change_rates, index):
    return np.sum([arr[index].value for arr in change_rates.values()])


def sum_change_rates(change_rates, shape, unit):
    result = np.zeros(shape) * unit
    for ch_rates in change_rates.values():
        result += ch_rates.to(unit)
    return result


def remap(mapping, values, default_value=None):
    if isinstance(values, dict):
        return {key: remap(mapping, val, default_value) for key, val in values.items()}
    else:
        remapped = values[mapping]
        new_elements_mask = mapping == -1
        if np.any(new_elements_mask):
            if default_value is None:
                raise ValueError("No default value provided for new elements")
            else:
                remapped[new_elements_mask] = default_value
        return remapped


def remap_interlaced(mapping, values, default_value = None):
    if isinstance(values, dict):
        return {key: remap_interlaced(mapping, val, default_value) for key, val in values.items()}
    else:
        v1, v2 = deinterlace(values)
        return interlace(remap(mapping, v1, default_value), remap(mapping, v2, default_value))


def combine_mapping(first_mapping, second_mapping):
    """ Combines two mappings, but with special treatment of "-1" value in the mapping:
        -1 does not mean the last element; it means a "new element", at must be retained across the mappings
    """
    combined = first_mapping[second_mapping]
    combined[second_mapping == -1] = -1
    return combined


def to_erg(gamma_bins):
    return (gamma_bins * mec2).to("erg")


def recalc_gamma_bins_and_density(energy_bins, abs_energy_changes, density):
    new_energy_bins, density_increase = recalc_energy_bins_and_density_increase(abs_energy_changes, energy_bins)
    new_density = density * density_increase
    new_gamma_bins = (new_energy_bins / mec2).to_value('')
    return new_gamma_bins, new_density


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


def calc_new_low_change_rates_mask(energy_bins, density, abs_energy_change, abs_particle_scaling, abs_particle_injection,
                                   max_energy_change_per_interval, max_density_change_per_interval, max_abs_injection_per_interval):
    if np.any((density == 0) & (abs_particle_injection < 0)):
        raise ValueError("Particles cannot escape when density is zero")
    # The obvious hard limit for escape is -1.0, which means "all particles escaped"; more negative value would clearly
    # mean we used too long time. The -0.5 is somewhat arbitrary, perhaps it should be configurable like other limits?
    # But it is tricky how to configure it, because e.g. -1.1 value makes no physical sense, on the other hand +1.1 is completely fine.
    max_escape = -0.5
    relative_changes_bin_lb, relative_en_changes_bin_ub = deinterlace(abs_energy_change / energy_bins)
    rel_density_increase_from_energy_change = recalc_energy_bins_and_density_increase(abs_energy_change, energy_bins)[1]
    rel_density_change_from_injection = np.divide(density * abs_particle_scaling + abs_particle_injection, density,
                                                  out=Quantity(np.zeros_like(density.value, dtype=float)),
                                                  where=density!=0).value
    rel_density_increase_combined = rel_density_change_from_injection * rel_density_increase_from_energy_change
    new_low_change_rates_mask = ((abs(relative_en_changes_bin_ub) <= max_energy_change_per_interval) &
                                 (~np.isnan(rel_density_increase_from_energy_change)) &
                                 (abs(abs_particle_scaling) <= max_density_change_per_interval) &
                                 (abs(abs_particle_injection) <= max_abs_injection_per_interval) &
                                 (abs_particle_scaling >= max_escape) &
                                 (abs(rel_density_increase_combined) <= max_density_change_per_interval))
    return new_low_change_rates_mask


def sort_and_merge_duplicates(gma_bins_lb, densities, mask=None, element_transform=np.log10):
    sort_idx = np.argsort(gma_bins_lb)
    gm_bins_lb_sorted = gma_bins_lb[sort_idx]
    mask_sorted = mask[sort_idx] if mask is not None else None
    densities_sorted = densities[sort_idx]
    duplicate_idx = get_duplicates(gm_bins_lb_sorted, mask=mask_sorted, element_transform=element_transform)
    if any(duplicate_idx):
        keep_idx = np.setdiff1d(np.arange(len(sort_idx)), duplicate_idx, assume_unique=True)
        gm_bins_merged, densities_merged = merge_points_preserve_integral(gm_bins_lb_sorted, densities_sorted, group_duplicates(duplicate_idx))
        return gm_bins_merged, densities_merged, sort_idx[keep_idx]
    else:
        return gm_bins_lb_sorted, densities_sorted, sort_idx



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
    return np.where(match)[0] + 1


def group_duplicates(dup_indices):
    dup_indices = np.array(dup_indices)
    if len(dup_indices) == 0:
        return []
    # Include the starting element (one before each duplicate)
    starts = dup_indices - 1
    # Combine starts and duplicates
    combined = np.concatenate([starts[:, None], dup_indices[:, None]], axis=1).flatten()
    # Find where consecutive numbers break
    breaks = np.where(np.diff(combined) != 1)[0]
    # Split at breaks
    groups = np.split(combined, breaks + 1)
    # Convert each to list
    return [list(g) for g in groups]


def merge_points_preserve_integral(x, y, merge_groups):
    """
    Merge groups of points in (x, y) while preserving trapezoidal integral.

    Parameters:
    - x: array-like, x-coordinates (must be sorted)
    - y: array-like, y-coordinates
    - merge_groups: list of lists, each inner list contains indices of x to merge

    Returns:
    - x_new, y_new: arrays with merged points
    """

    remove_mask = np.zeros_like(x, dtype=bool)
    x_new_points = []
    y_new_points = []
    for group in merge_groups:
        if len(group) < 2:
            continue  # nothing to merge
        i_first = group[0]
        i_last = group[-1]
        # Determine left and right neighbors outside the group if available
        n_first = i_first if i_first == 0 else i_first - 1
        n_last = i_last if i_last == len(x)-1 else i_last + 1
        integral = np.trapz(y=y[n_first:n_last+1], x=x[n_first:n_last+1])
        x_left = x[n_first]
        y_left = y[n_first]
        x_right = x[n_last]
        y_right = y[n_last]
        if i_first == n_first or i_last == n_last:
            # for boundary groups, use the boundary x value as the replacement x
            x_new = x_left if i_first == n_first else x_right
        else:
            # for other groups, use weighted geometric mean for x
            weights = y[group]
            x_new = np.exp(np.sum(weights * np.log(x[group])) / np.sum(weights))
        interval_width = x_right - x_left
        y_new = (2 * integral - y_left * (x_new - x_left) - y_right * (x_right - x_new)) / interval_width
        # Save new point and mark original points in the group (except first) for removal
        x_new_points.append((i_first, x_new))
        y_new_points.append((i_first, y_new))
        remove_mask[group[1:]] = True

    # Remove old points and replace first point of each group with new point
    x_merged = x[~remove_mask]
    y_merged = y[~remove_mask]
    for idx, x_val in x_new_points:
        y_val = dict(y_new_points)[idx]
        i_new = np.where(x_merged == x[idx])[0][0]  # find index in merged array
        x_merged[i_new] = x_val
        y_merged[i_new] = y_val
    return x_merged, y_merged

def add_boundary_bins(gm_bins_lb, en_chg_rates, gamma_bounds, max_bin_creep):
    mapping_with_edge_bins = np.arange(len(gm_bins_lb))
    point1 = gm_bins_lb[0]
    is_first_bin_far_from_start = is_far_from_bounds(gamma_bounds, point1, max_bin_creep, True)
    if is_first_bin_far_from_start:
        first_bin_en_chg = sum_change_rates_at_index(en_chg_rates, 0)
        if first_bin_en_chg > 0:
            gm_bins_lb = np.concatenate([np.array([gamma_bounds[0]]), gm_bins_lb])
            mapping_with_edge_bins = np.concatenate([np.array([-1]), mapping_with_edge_bins])
    point = gm_bins_lb[-1]
    is_last_bin_far_from_end = is_far_from_bounds(gamma_bounds, point, max_bin_creep, False)
    if is_last_bin_far_from_end:
        last_bin_en_chg = sum_change_rates_at_index(en_chg_rates, -2) # -2 is the lower bound of the last bin
        if last_bin_en_chg < 0:
            gm_bins_lb = np.concatenate([gm_bins_lb, np.array([gamma_bounds[1] / (1 + bin_size_factor)])])
            mapping_with_edge_bins = np.concatenate([mapping_with_edge_bins, np.array([-1])])
    return gm_bins_lb, mapping_with_edge_bins


def is_far_from_bounds(bounds, point, max_bin_creep, from_start):
    if bounds is None:
        return False
    log_start = np.log10(bounds[0])
    log_end = np.log10(bounds[1])
    log_point = np.log10(point)
    nom = log_point - log_start if from_start else log_end - log_point
    return nom > max_bin_creep


def remove_gamma_beyond_bounds(gm_bins_lb, gamma_bounds):
    mapping = np.where((gamma_bounds[0] <= gm_bins_lb) & (gm_bins_lb <= gamma_bounds[1] / (1 + bin_size_factor)))[0]
    if len(mapping) != len(gm_bins_lb):
        return gm_bins_lb[mapping], mapping
    return gm_bins_lb, np.arange(len(gm_bins_lb))


def remove_too_close_bins(gm_bins_lb, densities, min_bin_creep, mask=None):
    gm_bins_lb_log = np.log10(gm_bins_lb)
    gaps = np.diff(gm_bins_lb_log)
    too_close_bins_mask = gaps < min_bin_creep
    if np.any(too_close_bins_mask):
        groups = find_consecutive_groups(np.arange(len(too_close_bins_mask))[too_close_bins_mask])
        for group in groups:
            group.append(max(group) + 1) # add the final bin to each group
        if len(groups) == 1 and len(groups[0]) == len(gm_bins_lb):
            # All bins are so close they could be merged into one bin, but we can't let it happen
            if len(gm_bins_lb):
                # Only 2 bins left, no merging possible
                return gm_bins_lb, densities, np.arange(len(gm_bins_lb))
            else:
                # Leave 1st bin unmerged, and merge others (in practice, still rather erroneous state...)
                groups[0].pop(0)
        removed = []
        for group in groups:
            removed.extend(group[1:])
        gm_bins_merged, densities_merged = merge_points_preserve_integral(gm_bins_lb, densities, groups)
        remaining = np.setdiff1d(np.arange(len(gm_bins_lb)), removed, assume_unique=True)
        return gm_bins_merged, densities_merged, remaining
    else:
        return gm_bins_lb, densities, np.arange(len(gm_bins_lb))

def find_consecutive_groups(arr):
    breaks = np.where(np.diff(arr) != 1)[0] + 1
    groups = np.split(arr, breaks)
    return [group.tolist() for group in groups]


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


def interlace(gamma_bins_start: NDArray, gamma_bins_end: NDArray) -> NDArray:
    """ Combines two gamma arrays into one array, to make the calculations on them in one steps instead of two;
        the first array goes into odd indices, the second into even indices """
    return np.column_stack((gamma_bins_start, gamma_bins_end)).ravel()


def deinterlace(interlaced: NDArray) -> tuple[NDArray, NDArray]:
    """ reverts the interlace method """
    gamma_bins_start = interlaced[0::2]
    gamma_bins_end = interlaced[1::2]
    return gamma_bins_start, gamma_bins_end


def energy_changes_lb(energy_changes_dict) -> dict:
    return {key: deinterlace(value)[0] for key, value in energy_changes_dict.items()}

