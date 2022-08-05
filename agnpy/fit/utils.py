import numpy as np
import astropy.units as u
from astropy.table import Table
from sherpa.data import Data1D
from gammapy.estimators import FluxPoints
from gammapy.datasets import FluxPointsDataset, Datasets


def load_sherpa_flux_points(sed_path, E_min, E_max, systematics_dict=None):
    """Load the MWL SED at `sed_path` in a `~sherpa.data.Data1D` object.
    Note that low and high error on the flux are averaged. This is what is done
    in the chi2 computation of Gammapy, we adopt the same method in order to
    make crosschecks.
    
    Parameters
    ----------
    sed_path : str
        path to the .ecsv file to be used for fitting
    E_min : `~astropy.Quantity`
        minimum energy to be used in the fit
    E_max : `~astropy.Quantity`
        maximum energy to be used in the fit
    systematics_dict : dict
        dictionary containing the instrument name and the systematics error
        associated with it. The sys. error should be expressed as a relative
        error on the flux. For example:
        `systematics_dict = {"Fermi" : 0.1, "MAGIC" : 0.30, "Swift-XRT": 0.10}`

    Returns
    -------
    `~sherpa.data.Data1D`
    """
    table = Table.read(sed_path)
    table = table.group_by("instrument")

    x = np.asarray([])
    y = np.asarray([])
    y_err_stat = np.asarray([])
    y_err_syst = np.asarray([])

    for tab in table.groups:
        name = tab["instrument"][0]

        energy = tab["e_ref"]
        e2dnde = tab["e2dnde"]
        e2dnde_errn = tab["e2dnde_errn"]
        e2dnde_errp = tab["e2dnde_errp"]

        # add systematic errors
        if systematics_dict is not None:
            for instrument in systematics_dict.keys():
                if name == instrument:
                    syst_rel_error = systematics_dict[instrument]
                    e2dnde_err_syst = syst_rel_error * e2dnde

        x = np.append(x, energy)
        y = np.append(y, e2dnde)
        y_err_stat = np.append(y_err_stat, (e2dnde_errn + e2dnde_errp) / 2)
        y_err_syst = np.append(y_err_syst, e2dnde_err_syst)

    sed = Data1D("sed", x, y, staterror=y_err_stat, syserror=np.asarray(y_err_syst))

    # set the minimum energy to be used for the fit
    sed.notice(E_min.to_value("eV"), E_max.to_value("eV"))

    return sed


def add_systematic_errors_gammapy_flux_points(flux_points, syst_rel_error):
    """Add the systematic error on the flux points in a given energy range.
    We symply sum the systematic error in quadrature with the statystical one.
    The systematic error is expressed as a relative error on the flux.

    Parameters
    ----------
    flux_points : `~gammapy.estimators.FluxPoints`
        Gammapy's flux points
    syst : float
        systematic error as a percentage of the dnde flux
    """
    dnde_err_syst = syst_rel_error * flux_points.dnde.data
    # sum in quadrature with the stat error
    dnde_errn_tot = np.sqrt(flux_points.dnde_errn.data ** 2 + dnde_err_syst ** 2)
    dnde_errp_tot = np.sqrt(flux_points.dnde_errp.data ** 2 + dnde_err_syst ** 2)
    # the attributes we have to change is the norm_errn and norm_errp
    flux_points.norm_errn.data = dnde_errn_tot / flux_points.dnde_ref.data
    flux_points.norm_errp.data = dnde_errp_tot / flux_points.dnde_ref.data


def load_gammapy_flux_points(sed_path, E_min, E_max, systematics_dict=None):
    """Load the MWL SED at `sed_path` in a list of
    `~gammapy.datasets.FluxPointsDataset`. Add the systematic errors.
    
    Parameters
    ----------
    sed_path : str
        path to the .ecsv file to be used for fitting
    E_min : `~astropy.Quantity`
        minimum energy to be used in the fit
    E_max : `~astropy.Quantity`
        maximum energy to be used in the fit
    systematics_dict : dict
        dictionary containing the instrument name and the systematics error
        associated with it. The sys. error should be expressed as a relative
        error on the flux. For example:
        `systematics_dict = {"Fermi" : 0.1, "MAGIC" : 0.30, "Swift-XRT": 0.10}`

    Returns
    -------
    `~gammapy.dataset.Datasets`, list of flux points datasets
    """
    datasets = Datasets()

    table = Table.read(sed_path)
    table = table.group_by("instrument")

    for group in table.groups:
        name = group["instrument"][0]
        data = FluxPoints.from_table(group, sed_type="e2dnde", format="gadf-sed")

        if systematics_dict is not None:
            for instrument in systematics_dict.keys():
                if name == instrument:
                    syst_rel_error = systematics_dict[instrument]
                    add_systematic_errors_gammapy_flux_points(data, syst_rel_error)

        # load the flux points in a dataset
        dataset = FluxPointsDataset(data=data, name=name)

        # set the minimum energy to be used for the fit
        mask = (dataset.data.energy_ref >= E_min) * (dataset.data.energy_ref <= E_max)
        dataset.mask_fit = mask

        datasets.append(dataset)

    return datasets
