# standardise the MWL SED to a format readable by Gammapy's FluxPointsDataset
# SED format based on
# https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/flux_points/index.html
import numpy as np
import astropy.units as u
from astropy.table import Table


# - Mrk421 2011
sed_file = np.loadtxt("original/Mrk421_2011.txt")
# read
nu = sed_file[:, 0] * u.Hz
nuFnu = sed_file[:, 1] * u.Unit("erg cm-2 s-1")
nuFnu_err = sed_file[:, 2] * u.Unit("erg cm-2 s-1")
# store in table and remove points with orders of magnitude smaller error, they are upper limits
UL = nuFnu_err < (nuFnu * 1e-3)
sed_table = Table()
sed_table["e_ref"] = nu.to("eV", equivalencies=u.spectral())[~UL]
sed_table["e2dnde"] = nuFnu[~UL]
sed_table["e2dnde_err"] = nuFnu_err[~UL]
sed_table.meta["SED_TYPE"] = "e2dnde"
sed_table.meta["source"] = "Mrk421"
sed_table.meta["period"] = "2009"
sed_table.meta[
    "reference"
] = "https://ui.adsabs.harvard.edu/abs/2011ApJ...736..131A/abstract"
sed_table.meta["redshift"] = 0.03
sed_table.write("Mrk421_2011.ecsv", overwrite=True)

# - PKS1510-089 low state
sed_file = np.loadtxt("original/PKS1510-089_low.txt")
# read
E = sed_file[:, 0] * u.eV
nuFnu = sed_file[:, 1] * u.Unit("erg cm-2 s-1")
nuFnu_err = sed_file[:, 2] * u.Unit("erg cm-2 s-1")
# store in table and remove points with orders of magnitude smaller error, they are upper limits
UL = nuFnu_err < (nuFnu * 1e-3)
sed_table = Table()
sed_table["e_ref"] = E[~UL]
sed_table["e2dnde"] = nuFnu[~UL]
sed_table["e2dnde_err"] = nuFnu_err[~UL]
sed_table.meta["SED_TYPE"] = "e2dnde"
sed_table.meta["source"] = "PKS1510-089"
sed_table.meta["period"] = "2012-2017"
sed_table.meta[
    "reference"
] = "https://ui.adsabs.harvard.edu/abs/2018A%26A...619A.159M/abstract"
sed_table.meta["redshift"] = 0.361
sed_table.write("PKS1510-089_low.ecsv", overwrite=True)

# - PKS1510-089 2012 state
sed_file = np.loadtxt("original/PKS1510-089_2012.txt")
# read
nu = sed_file[:, 0] * u.Hz
nuFnu = sed_file[:, 1] * u.Unit("erg cm-2 s-1")
nuFnu_err_low = sed_file[:, 2] * u.Unit("erg cm-2 s-1")
nuFnu_err_high = sed_file[:, 3] * u.Unit("erg cm-2 s-1")
# store in table and remove points with orders of magnitude smaller error, they are upper limits
UL = nuFnu_err_low < (nuFnu * 1e-3)
sed_table = Table()
sed_table["e_ref"] = nu.to("eV", equivalencies=u.spectral())[~UL]
sed_table["e2dnde"] = nuFnu[~UL]
sed_table["e2dnde_errn"] = nuFnu_err_low[~UL]
sed_table["e2dnde_errp"] = nuFnu_err_high[~UL]
sed_table.meta["SED_TYPE"] = "e2dnde"
sed_table.meta["source"] = "PKS1510-089"
sed_table.meta["period"] = "2012"
sed_table.meta[
    "reference"
] = "https://ui.adsabs.harvard.edu/abs/2014A%26A...569A..46A/abstract"
sed_table.meta["redshift"] = 0.361
sed_table.write("PKS1510-089_2012.ecsv", overwrite=True)

# - PKS1510-089 2015, Period A
sed_file = np.loadtxt("original/PKS1510-089_2015a.txt")
# read
E = sed_file[:, 0] * u.eV
nuFnu = sed_file[:, 1] * u.Unit("erg cm-2 s-1")
nuFnu_err_low = sed_file[:, 2] * u.Unit("erg cm-2 s-1")
nuFnu_err_high = sed_file[:, 3] * u.Unit("erg cm-2 s-1")
# store in table and remove points with orders of magnitude smaller error, they are upper limits
UL = nuFnu_err_low < (nuFnu * 1e-3)
sed_table = Table()
sed_table["e_ref"] = E[~UL]
sed_table["e2dnde"] = nuFnu[~UL]
sed_table["e2dnde_errn"] = nuFnu_err_low[~UL]
sed_table["e2dnde_errp"] = nuFnu_err_high[~UL]
sed_table.meta["SED_TYPE"] = "e2dnde"
sed_table.meta["source"] = "PKS1510-089"
sed_table.meta["period"] = "2015, A"
sed_table.meta[
    "reference"
] = "https://ui.adsabs.harvard.edu/abs/2017A%26A...603A..29A/abstract"
sed_table.meta["redshift"] = 0.361
sed_table.write("PKS1510-089_2015a.ecsv", overwrite=True)

# - PKS1510-089 2015, Period B
sed_file = np.loadtxt("original/PKS1510-089_2015b.txt")
# read
E = sed_file[:, 0] * u.eV
nuFnu = sed_file[:, 1] * u.Unit("erg cm-2 s-1")
nuFnu_err_low = sed_file[:, 2] * u.Unit("erg cm-2 s-1")
nuFnu_err_high = sed_file[:, 3] * u.Unit("erg cm-2 s-1")
# store in table and remove points with orders of magnitude smaller error, they are upper limits
UL = nuFnu_err_low < (nuFnu * 1e-3)
sed_table = Table()
sed_table["e_ref"] = E[~UL]
sed_table["e2dnde"] = nuFnu[~UL]
sed_table["e2dnde_errn"] = nuFnu_err_low[~UL]
sed_table["e2dnde_errp"] = nuFnu_err_high[~UL]
sed_table.meta["SED_TYPE"] = "e2dnde"
sed_table.meta["source"] = "PKS1510-089"
sed_table.meta["period"] = "2015, B"
sed_table.meta[
    "reference"
] = "https://ui.adsabs.harvard.edu/abs/2017A%26A...603A..29A/abstract"
sed_table.meta["redshift"] = 0.361
sed_table.write("PKS1510-089_2015b.ecsv", overwrite=True)
