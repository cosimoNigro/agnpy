import numpy as np
from astropy.units import Quantity, Unit
from numpy._typing import NDArray
from pygments.styles import vs
from astropy.constants import m_e, c, h, e
from agnpy import InterpolatedDistribution, EmptyDistribution
from agnpy.time_evolution.types import BinsWithDensities, FnParams, TimeEvaluationResult, GammaFn, SubgroupsList
from agnpy.utils.conversion import mec2
from scipy.interpolate import PchipInterpolator
from typing import Tuple

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


def recalc_new_rates(gm_bins:NDArray, functions:dict[str,GammaFn], unit:Unit, dens:Quantity, subgroups_density:NDArray, previous_rates=None, recalc_mask=None) -> dict[str,Quantity]:
    """
    Calculates (or recalculates) the energy change rates or injection rates.
    If the mask is provided, only unmasked elements will be recalculated, and previous_rates will be used for masked elements.
    gm_bins - array of shape (N,) or (2,N) of gamma values (lower bounds of gamma bins, or lower and upper bounds of gamma bins)
    functions - a dict of functions to use in the calculation of energy change rates or injection rates
    dens - current total density array
    subgroups_density - array of density split per subgroup, if subgroups have been used
    previous_rates - optional dict of arrays of shape (N,) or (2,N), consisting of most recently calculated rates corresponding to gm_bins values
    recalc_mask - optional array of shape (N,) a mask of bins that need recalculation
    Returns a dictionary of recalculated rates, with keys from input functions dict
    """
    if previous_rates is None:
        previous_rates = {}
    mask = np.ones((gm_bins.shape[-1]), dtype=bool) if recalc_mask is None else recalc_mask
    gm_bins_masked = gm_bins[..., mask]
    dens_masked = dens[mask]
    dens_groups_masked = remap_subgroups_density(mask_to_mapping(mask), subgroups_density)
    new_rates = {}
    for label, fn in functions.items():
        new_rates[label] = np.zeros_like(gm_bins) * unit \
            if previous_rates.get(label) is None \
            else previous_rates[label].copy()
        gm_bins_interlaced = gm_bins_masked.reshape(-1)
        params = FnParams(gm_bins_interlaced, dens_masked, dens_groups_masked)
        new_rates[label][..., mask] = fn(params).to(unit).reshape(gm_bins_masked.shape)
    return new_rates


def sum_change_rates(change_rates: dict[str,Quantity], shape:Tuple, unit:Unit, subgroups:SubgroupsList=None) -> Quantity:
    """
    Returns a Quantity(N,M), the sum of all values from the change_rates dictionary, optionally grouped by subgroups.
    If subgroups are not provided, a single group is returned (i.e. N=1)

    Parameters
    ----------
    change_rates - a dict of Quantities to sum over
    shape - a (M,) shape of each value in the change_rates dict, and of each group in the result list
    unit - a unit for each group
    subgroups - optional subgroups list
    """
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
        -1 does not mean the last element; it means a "new element", and must be retained across the mappings
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
    return ((2,N), (N,))
    """
    energy_bins_lb, energy_bins_ub = energy_bins
    energy_bins_width = energy_bins_ub - energy_bins_lb
    new_energy_bins = energy_bins + abs_energy_changes
    new_energy_bins_from, new_energy_bins_to = new_energy_bins
    invalid_bins = np.logical_or(new_energy_bins_from <= 0, new_energy_bins_to <= 0)
    if np.any(invalid_bins):
        raise ValueError("Negative gamma obtained at bins " + str(np.arange(len(energy_bins_lb))[invalid_bins]))
    new_energy_bins_width = new_energy_bins_to - new_energy_bins_from
    density_increase = energy_bins_width / new_energy_bins_width
    invalid_width = new_energy_bins_to - new_energy_bins_from < 0
    if np.any(invalid_width):
        raise ValueError("Negative density obtained at bins " + str(np.arange(len(energy_bins_lb))[invalid_width]))
    return new_energy_bins, density_increase


def sort_and_merge_duplicates(gm_bins, densities, interpolated_distribution):
    # If any two different gamma points map into the same point, we will not be able to make the interpolated distribution,
    # hence we need to merge them into a single point. Also consider the fact the InterpolatedDistribution
    # uses the log10 values of the gamma points, so two close-enough gamma points can collapse into a single point after log10.
    sort_idx = np.argsort(gm_bins[0])
    gm_bins_sorted = gm_bins[..., sort_idx]
    densities_sorted = densities[sort_idx]
    sorted_array = gm_bins_sorted[0]
    gm_bins_log10 = np.log10(sorted_array)
    duplicate_idx = np.asarray(gm_bins_log10[:-1] == gm_bins_log10[1:]).nonzero()[0] + 1

    if any(duplicate_idx):
        keep_idx = np.setdiff1d(np.arange(len(sort_idx)), duplicate_idx, assume_unique=True)
        duplicate_groups = group_duplicates(duplicate_idx)
        gm_bins_lb_merged, densities_merged = merge_points(gm_bins_sorted[0], densities_sorted, duplicate_groups, interpolated_distribution)
        return np.array([gm_bins_lb_merged,np.zeros_like(gm_bins_lb_merged)]), densities_merged, sort_idx[keep_idx]
    else:
        return gm_bins_sorted, densities_sorted, sort_idx


def group_duplicates(dup_indices):
    dup_indices = np.array(dup_indices)
    if len(dup_indices) == 0:
        return []
    # Find where consecutive numbers break
    breaks = np.where(np.diff(dup_indices) != 1)[0]
    # Split at breaks
    groups = np.split(dup_indices, breaks + 1)
    # Convert each to list, adding first element
    return [[g[0]-1, *list(g)] for g in groups]


def merge_points(x, y, merge_groups, interpolated_distribution):
    """
    Merge groups of points in (x, y).
    Ideally we should try to make it such that integral is preserved, but in special cases this could lead to
    negative densities. So instead we just take the average.

    Parameters:
    - x: 1D numpy array, x-coordinates (must be sorted)
    - y: 1D numpy array, y-coordinates
    - merge_groups: list of lists, each inner list contains indices of x to merge
    - interpolated_distribution: if present, it will be used for finding the y value of merged point; if absent, simple average is used

    Returns:
    - x_new, y_new: arrays with merged points
    """

    labels = np.full(len(x), -1)
    for g_idx, group in enumerate(merge_groups):
        labels[group] = g_idx
    x_means = np.array([np.exp(np.mean(np.log(x[labels == g]))) for g in range(len(merge_groups))])
    if interpolated_distribution is None:
        y_means = Quantity([np.mean(y[labels == g]) for g in range(len(merge_groups))])
    else:
        y_means = Quantity([interpolated_distribution(x_means[g]) for g in range(len(merge_groups))])
    group_tails = [item for group in merge_groups for item in group[1:]]
    retain_mask = ~np.isin(np.arange(len(x)), group_tails)
    labels_destination = labels[retain_mask]
    merged_x = np.where(labels_destination == -1, x[retain_mask], x_means[labels_destination])
    merged_y = np.where(labels_destination == -1, y[retain_mask], y_means[labels_destination])
    return merged_x, merged_y


def remove_empty_densities(result: TimeEvaluationResult) -> TimeEvaluationResult:
    mask = result.density > 0
    return TimeEvaluationResult(
        result.total_time,
        result.gamma[mask],
        result.density[mask],
        remap_subgroups_density(mask_to_mapping(mask), result.density_subgroups),
        remap(mask, result.en_chg_rates),
        remap(mask, result.abs_inj_rates),
        remap(mask, result.rel_inj_rates))


def update_distribution(gamma_array, n_array, blob):
    gamma_sorted_and_merged, n_array_sorted_and_merged, _ = sort_and_merge_duplicates(gamma_array, n_array, None)
    if np.all(n_array_sorted_and_merged == 0):
        distribution = EmptyDistribution(gamma_min=gamma_sorted_and_merged[0], gamma_max=gamma_sorted_and_merged[-1])
    else:
        if len(gamma_sorted_and_merged[0]) == 1:
            raise DistributionToSinglePointCollapseError(gamma_sorted_and_merged[0])
        distribution = InterpolatedDistribution(gamma_sorted_and_merged[0], n_array_sorted_and_merged,
                                     interpolator=PchipInterpolator)
    blob.n_e = distribution
    return distribution


def energy_changes_lb(energy_changes_dict) -> dict:
    return {key: value[0] for key, value in energy_changes_dict.items()}


def to_total_energy_gev_sqr(gm_bins, density, total_volume):
    gev_per_gamma = mec2.to("GeV")
    gamma_as_gev = gm_bins * gev_per_gamma
    densities_per_gev = density / gev_per_gamma
    total_energy_per_gev = densities_per_gev * total_volume
    return total_energy_per_gev * gamma_as_gev ** 2
