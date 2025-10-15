# compute the gamma-gamma absorption from EBL
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator


__all__ = ["ebl_files_dict", "EBL"]

agnpy_dir = Path(__file__).parent.parent
ebl_files_dict = {
    "franceschini_2008": f"{agnpy_dir}/data/ebl_models/ebl_franceschini.fits.gz",
    "franceschini_2017": f"{agnpy_dir}/data/ebl_models/ebl_franceschini_2017.fits.gz",
    "dominguez_2011": f"{agnpy_dir}/data/ebl_models/ebl_dominguez11.fits.gz",
    "finke_2010": f"{agnpy_dir}/data/ebl_models/frd_abs.fits.gz",
    "saldana_lopez_2021": f"{agnpy_dir}/data/ebl_models/ebl_saldana-lopez_2021.fits.gz",
}


class EBL:
    """Class representing for the Extragalactic Background Light absorption.
    Tabulated values of absorption as a function of redshift and energy according
    to the models of [Franceschini2008]_, [Finke2010]_, [Dominguez2011]_, [Saldana-Lopez2021]_ are available
    in `data/ebl_models`.
    They are interpolated by `agnpy` and can be later evaluated for a given redshift
    and range of frequencies.

    Parameters
    ----------
    model : ["franceschini", "dominguez", "finke", "saldana-lopez"]
        choose the reference for the EBL model
    """

    def __init__(self, model="dominguez_2011"):
        if model not in ebl_files_dict.keys():
            raise ValueError("No EBL model for the reference you specified")
        self.model_name = model
        self.model_file = ebl_files_dict[self.model_name]
        # load the absorption table
        self.load_absorption_table()
        self.interpolate_absorption_table()

    def load_absorption_table(self):
        """load the reference values from the table file to be interpolated later"""
        f = fits.open(self.model_file)
        # energies are in KeV, redshift is dimensionless
        self.energy_ref = np.sqrt(f["ENERGIES"].data["ENERG_LO"] * f["ENERGIES"].data["ENERG_HI"])
        self.z_ref = f["SPECTRA"].data["PARAMVAL"]
        self.values_ref = f["SPECTRA"].data["INTPSPEC"]
        # Franceschini 2008 file has two PARAMVAL rows repeated
        # at indexes 1001 and 1002, values of redshift = 1.001, eliminate them!
        if self.model_name == "franceschini_2008":
            self.z_ref = np.delete(self.z_ref, 1001)
            self.values_ref = np.delete(self.values_ref, 1001, axis=0)

    def interpolate_absorption_table(self):
        """interpolate the reference values, choose the kind of interpolation"""
        self.interpolated_model = RegularGridInterpolator(
            points=(self.energy_ref, self.z_ref),
            values=self.values_ref.T,
            method="linear",
            bounds_error=False
        )

    def absorption(self, nu, z):
        "This function returns the attenuation of the emission by EBL"
        energy = nu.to_value("keV", equivalencies=u.spectral())
        # for evaluation, reconstruct the shape of the input table
        z_array = np.full_like(energy, z)
        points = np.vstack([energy, z_array]).T
        absorption = self.interpolated_model(points)
        return absorption
