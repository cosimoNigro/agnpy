import numpy as np
from astropy.units import Quantity, Unit
from pygments.styles import vs
from astropy.constants import m_e, c, h, e
from agnpy import InterpolatedDistribution
from agnpy.time_evolution.types import BinsWithDensities, FnParams
from agnpy.utils.conversion import mec2
from scipy.interpolate import PchipInterpolator

bin_size_factor = 1e-4

class DistributionToSinglePointCollapseError(Exception):
    def __init__(self, gamma_point):
        self.gamma_point = gamma_point
        super().__init__(f"Unsupported state, cannot create InterpolatedDistribution - distribution collapsed to a single gamma point {gamma_point}")

def update_bin_ub(gm_bins, mask = None):
    """ Update bin upper bounds, but only for bins indicated by the mask
    """
    mask = mask if mask is not None else np.ones_like(gm_bins[0], bool)
    gm_bins[1][mask] = gm_bins[0][mask] * (1 + bin_size_factor)


def recalc_new_rates(gm_bins, functions, unit, dens, dens_groups, previous_rates=None, recalc_mask=None):
    """
    Calculates (or recalculates) the energy change rates or injection rates.
    If the mask is provided, only unmasked elements will be recalculated, and previous_rates will be used for masked elements.
    gm_bins - array of shape (N,) consisting of gamma values (lower bounds of gamma bins), or (2,N) consisting of lower and upper bounds of gamma bins
    previous_rates - optional dict of arrays of shape (N,) or (2,N), consisting of most recently calculated rates corresponding to gm_bins values
    recalc_mask - optional array of shape (N,) a mask of bins that need recalculation
    """
    if previous_rates is None:
        previous_rates = {}
    mask = np.ones((gm_bins.shape[-1]), dtype=bool) if recalc_mask is None else recalc_mask
    gm_bins_masked = gm_bins[..., mask]
    dens_masked = dens[mask]
    dens_groups_masked = remap_subgroups_density(mask_to_mapping(mask), dens_groups)
    new_rates = {}
    for label, fn in functions.items():
        new_rates[label] = np.zeros_like(gm_bins) * unit \
            if previous_rates.get(label) is None \
            else previous_rates[label].copy()
        gm_bins_interlaced = gm_bins_masked.reshape(-1)
        params = FnParams(gm_bins_interlaced, dens_masked, dens_groups_masked)
        new_rates[label][..., mask] = fn(params).reshape(gm_bins_masked.shape).to(unit)
    return new_rates


def sum_change_rates_at_index(change_rates, index):
    return np.sum([arr[index].value for arr in change_rates.values()])


def sum_change_rates(change_rates, shape, unit, subgroups=None):
    if subgroups is None:
        result = np.zeros(shape) * unit
        for ch_rates in change_rates.values():
            result += ch_rates.to(unit)
        return Quantity([result])
    else:
        return Quantity([
            np.sum(Quantity([change_rates[x] for x in row if x in change_rates]), axis=0)
            if any(x in change_rates for x in row) else np.zeros(shape) * unit
            for row in subgroups
        ])


def remap(mapping, values, default_value=None):
    if isinstance(values, dict):
        return {key: remap(mapping, val, default_value) for key, val in values.items()}
    else:
        remapped = values[..., mapping]
        new_elements_mask = mapping == -1
        if np.any(new_elements_mask):
            if default_value is None:
                raise ValueError("No default value provided for new elements")
            else:
                remapped[..., new_elements_mask] = default_value
        return remapped


def remap_subgroups_density(mapping, subgroups_density):
    new_subgroups_density = subgroups_density[:,mapping]
    new_subgroups_density[0, mapping == -1] = 1.0
    new_subgroups_density[1:, mapping == -1] = 0.0
    return new_subgroups_density


def mask_to_mapping(mask):
    return np.arange(len(mask))[mask]


def combine_mapping(first_mapping, second_mapping):
    """ Combines two mappings, but with special treatment of "-1" value in the mapping:
        -1 does not mean the last element; it means a "new element", at must be retained across the mappings
    """
    combined = first_mapping[second_mapping]
    combined[second_mapping == -1] = -1
    return combined


def to_erg(gamma_bins):
    return (gamma_bins * mec2).to("erg")


def to_densities_grouped(density, subgroups_density):
    density_grouped = density * subgroups_density if subgroups_density is not None else density.reshape(
        (1, len(density)))
    return density_grouped

def second_axis_size(a: np.ndarray) -> int:
    return a.shape[1] if a.ndim > 1 else 1


def recalc_gamma_bins_and_density(energy_bins, abs_energy_changes, density) -> BinsWithDensities:
    new_energy_bins, density_increase = recalc_energy_bins_and_density_increase(abs_energy_changes, energy_bins)
    new_density = density * density_increase
    new_gamma_bins = (new_energy_bins / mec2).to_value('')
    return BinsWithDensities(new_gamma_bins, new_density)


def recalc_energy_bins_and_density_increase(abs_energy_changes, energy_bins):
    """
    abs_energy_changes - array of shape (2,N)
    energy_bins - array of shape (2,N)
    return ((2,N), (N,)) - note: some of the elements of density_increase may be set to the "nan" value
    """
    energy_bins_lb, energy_bins_ub = energy_bins
    energy_bins_width = energy_bins_ub - energy_bins_lb
    new_energy_bins = energy_bins + abs_energy_changes
    new_energy_bins_from, new_energy_bins_to = new_energy_bins
    new_energy_bins_width = new_energy_bins_to - new_energy_bins_from
    density_increase = energy_bins_width / new_energy_bins_width
    invalid_width = new_energy_bins_to - new_energy_bins_from < 0
    density_increase[invalid_width] = np.nan  # mark with nan the bins where bin's start and end get swapped
    return new_energy_bins, density_increase


def calc_new_low_change_rates_mask(energy_bins, density, subgroups_density, abs_energy_change_grouped, abs_particle_scaling_grouped, abs_particle_injection_grouped,
                                   max_energy_change_per_interval, max_density_change_per_interval, max_abs_injection_per_interval):
    # The obvious hard limit for escape is -1.0, which means "all particles escaped"; more negative value would clearly
    # mean we used too long time. The -0.5 is somewhat arbitrary, perhaps it should be configurable like other limits?
    # But it is tricky how to configure it, because e.g. -1.1 value makes no physical sense, on the other hand +1.1 is completely fine.
    max_escape = -0.5
    new_low_change_rates_mask = np.ones(len(density), dtype=bool)
    density_grouped = to_densities_grouped(density, subgroups_density)
    for i in (range(subgroups_density.shape[0])):
        if np.any((density_grouped[i] == 0) & (abs_particle_injection_grouped[i] < 0)):
            raise ValueError("Particles cannot escape when density is zero")

        abs_energy_change = abs_energy_change_grouped[i]
        density = density_grouped[i]
        abs_particle_scaling = abs_particle_scaling_grouped[i]
        abs_particle_injection = abs_particle_injection_grouped[i]
        relative_en_change = abs_energy_change / energy_bins
        rel_density_increase_from_energy_change = recalc_energy_bins_and_density_increase(abs_energy_change, energy_bins)[1]
        rel_density_change_from_injection = np.divide(density * abs_particle_scaling + abs_particle_injection, density,
                                                      out=Quantity(np.zeros_like(density.value, dtype=float)),
                                                      where=density!=0).value
        rel_density_increase_combined = rel_density_change_from_injection * rel_density_increase_from_energy_change

        rel_en_chg_mask = (abs(relative_en_change) <= max_energy_change_per_interval).all(axis=0)
        rel_density_increase_mask = (~np.isnan(rel_density_increase_from_energy_change))
        particle_scaling_mask = (abs(abs_particle_scaling) <= max_density_change_per_interval)
        particle_injection_mask = (abs(abs_particle_injection) <= max_abs_injection_per_interval)
        escape_mask = (abs_particle_scaling >= max_escape)
        rel_density_comb_mask = (abs(rel_density_increase_combined) <= max_density_change_per_interval)
        group_mask = (rel_en_chg_mask & rel_density_increase_mask & particle_scaling_mask & particle_injection_mask & escape_mask & rel_density_comb_mask)
        new_low_change_rates_mask &= group_mask
    return new_low_change_rates_mask


def sort_and_merge_duplicates(gm_bins, densities, mask=None, element_transform=np.log10):
    sort_idx = np.argsort(gm_bins[0])
    gm_bins_sorted = gm_bins[..., sort_idx]
    mask_sorted = mask[..., sort_idx] if mask is not None else None
    densities_sorted = densities[sort_idx]
    duplicate_idx = get_duplicates(gm_bins_sorted[0], mask=mask_sorted, element_transform=element_transform)
    if any(duplicate_idx):
        keep_idx = np.setdiff1d(np.arange(len(sort_idx)), duplicate_idx, assume_unique=True)
        gm_bins_lb_merged, densities_merged = merge_points_preserve_integral(gm_bins_sorted[0], densities_sorted, group_duplicates(duplicate_idx))
        return np.array([gm_bins_lb_merged,np.zeros_like(gm_bins_lb_merged)]), densities_merged, sort_idx[keep_idx]
    else:
        return gm_bins_sorted, densities_sorted, sort_idx



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
    # Find where consecutive numbers break
    breaks = np.where(np.diff(dup_indices) != 1)[0]
    # Split at breaks
    groups = np.split(dup_indices, breaks + 1)
    # Convert each to list, adding first element
    return [[g[0]-1, *list(g)] for g in groups]


def merge_points_preserve_integral(x, y, merge_groups):
    """
    Merge groups of points in (x, y) while preserving trapezoidal integral.
    TODO: this requires some rethinking, because there are cases when preserving integral is not possible,
    unless using negative densities - what should happen in such a case?

    Parameters:
    - x: 1D numpy array, x-coordinates (must be sorted)
    - y: 1D numpy array, y-coordinates
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
            # for other groups, use simple arithmetic mean for x (geometric and/or weighted sum causes trouble if some or all densities are 0)
            x_new = np.sum(x[group]) / len(group)
            if y_left != y_right:
                thr = (2 * integral - (y_right * x_right) + (y_left * x_left)) / (y_left - y_right)
                if y_left > y_right:
                    if not x_new < thr:
                        x_new = max(x[x < thr])
                else:
                    if not x_new > thr:
                        x_new = min(x[x > thr])
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
    if len(merge_groups) == 1 and set(range(len(x))).issubset(merge_groups[0]):
        # all points merged to a single point, the integral is meaningless
        y_merged[0] = np.nan
    return x_merged, y_merged

def add_boundary_bins(gm_bins, en_chg_rates, gamma_bounds, max_bin_creep):
    gm_bins_lb = gm_bins[0]
    mapping_with_edge_bins = np.arange(gm_bins.shape[1])
    point1 = gm_bins_lb[0]
    is_first_bin_far_from_start = is_far_from_bounds(gamma_bounds, point1, max_bin_creep, True)
    if is_first_bin_far_from_start:
        first_bin_en_chg = sum_change_rates_at_index(en_chg_rates, 0)
        if first_bin_en_chg > 0:
            new_col = np.array([[gamma_bounds[0]], [gamma_bounds[0] * (1 + bin_size_factor)]])
            gm_bins = np.concatenate([new_col, gm_bins], axis=1)
            mapping_with_edge_bins = np.concatenate([np.array([-1]), mapping_with_edge_bins])
    point = gm_bins_lb[-1]
    is_last_bin_far_from_end = is_far_from_bounds(gamma_bounds, point, max_bin_creep, False)
    if is_last_bin_far_from_end:
        last_bin_en_chg = sum_change_rates_at_index(en_chg_rates, -2) # -2 is the lower bound of the last bin
        if last_bin_en_chg < 0:
            new_col = np.array([[gamma_bounds[1] / (1 + bin_size_factor)], [gamma_bounds[1]]])
            gm_bins = np.concatenate([gm_bins,new_col], axis=1)
            mapping_with_edge_bins = np.concatenate([mapping_with_edge_bins, np.array([-1])])
    return gm_bins, mapping_with_edge_bins


def is_far_from_bounds(bounds, point, max_bin_creep, from_start):
    if bounds is None:
        return False
    log_start = np.log10(bounds[0])
    log_end = np.log10(bounds[1])
    log_point = np.log10(point)
    nom = log_point - log_start if from_start else log_end - log_point
    return nom > max_bin_creep


def remove_gamma_beyond_bounds(gm_bins, gamma_bounds):
    gm_bins_lb = gm_bins[0]
    mapping = np.where((gamma_bounds[0] <= gm_bins_lb) & (gm_bins_lb <= gamma_bounds[1] / (1 + bin_size_factor)))[0]
    if len(mapping) != len(gm_bins_lb):
        return gm_bins[..., mapping], mapping
    return gm_bins, np.arange(len(gm_bins_lb))


def remove_too_close_bins(gm_bins, densities, min_bin_creep, mask=None):
    unmasked_bins = np.arange(gm_bins.shape[1])[mask]
    gm_bins_lb_log = np.log10(gm_bins[0])
    gaps = np.diff(gm_bins_lb_log)
    too_close_bins_mask = gaps < min_bin_creep

    if not np.any(too_close_bins_mask):
        return gm_bins, densities, np.arange(gm_bins.shape[-1])

    groups = find_consecutive_groups(np.arange(len(too_close_bins_mask))[too_close_bins_mask])
    for group in groups:
        group.append(max(group) + 1) # add the final bin to each group
    # remove the masked bins from the group, and then remove the groups that were left empty after the filtering
    for i in range(len(groups)):
        groups[i] = [x for x in groups[i] if x in unmasked_bins]
    groups = [sub for sub in groups if sub]
    if not groups:
        return gm_bins, densities, np.arange(gm_bins.shape[-1]) # all bins have been filter out, skip further processing

    if len(groups) == 1 and len(groups[0]) == gm_bins.shape[-1]:
        # All bins are so close they could be merged into one bin, but we can't let it happen
        if gm_bins.shape[-1] == 2:
            # Only 2 bins left, no merging possible
            return gm_bins, densities, np.arange(gm_bins.shape[-1])
        else:
            # Leave 1st bin unmerged, and merge others (in practice, still rather erroneous state...)
            groups[0].pop(0)
    removed = []
    for group in groups:
        removed.extend(group[1:])
    gm_bins_lb_merged, densities_merged = merge_points_preserve_integral(gm_bins[0], densities, groups)
    remaining = np.setdiff1d(np.arange(gm_bins.shape[-1]), removed, assume_unique=True)
    return np.array([gm_bins_lb_merged,np.zeros_like(gm_bins_lb_merged)]), densities_merged, remaining


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
    if any(duplicated_indices):
        gm_bins_lb_merged, densities_merged = merge_points_preserve_integral(gamma_array_sorted, n_array_sorted,
                                                                         group_duplicates(duplicated_indices))
    else:
        gm_bins_lb_merged, densities_merged = gamma_array_sorted, n_array_sorted
    if len(gm_bins_lb_merged) == 1:
        raise DistributionToSinglePointCollapseError(gamma_array_sorted[0])
    blob.n_e = InterpolatedDistribution(gm_bins_lb_merged, densities_merged, interpolator=PchipInterpolator)


def apply_time_eval(en_bins, density, subgroups_density, abs_en_changes_grouped, total_injection_grouped):
    new_bins_and_densities = [recalc_gamma_bins_and_density(en_bins, abs_en_changes_grouped[i], density*subgroups_density[i])
                              for i in range(len(subgroups_density))]
    for i in range(len(new_bins_and_densities)):
        new_bins_and_densities[i] = BinsWithDensities(new_bins_and_densities[i].gamma_bins,
                                                      new_bins_and_densities[i].densities + total_injection_grouped[i])

    new_gm_bins = new_bins_and_densities[0].gamma_bins
    new_densities = np.zeros_like(subgroups_density) * Unit("cm-3")
    new_densities[0] = new_bins_and_densities[0].densities

    for i in range(1, len(new_bins_and_densities)):
        new_gm_bins_subgroup = new_bins_and_densities[i].gamma_bins
        new_dens_subgroup = new_bins_and_densities[i].densities
        if np.all(new_dens_subgroup == 0) or np.all(new_bins_and_densities[i].gamma_bins == new_gm_bins):
            new_densities[i] = new_dens_subgroup
        else:
            distribution = InterpolatedDistribution(new_gm_bins_subgroup[0], new_dens_subgroup)
            densities_interpolated = distribution(new_gm_bins[0])
            new_densities[i] = densities_interpolated

    new_dens = Quantity(new_densities.sum(axis=0))
    subgroups_density_new = subgroups_density.copy()
    mask = new_dens != 0
    subgroups_density_new[:,mask] = (new_densities[:,mask] / new_dens[mask]).value
    return new_gm_bins, new_dens, subgroups_density_new


def energy_changes_lb(energy_changes_dict) -> dict:
    return {key: value[0] for key, value in energy_changes_dict.items()}

