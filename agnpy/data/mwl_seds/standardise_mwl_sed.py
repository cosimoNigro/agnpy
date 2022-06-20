# standardise the MWL SED to a format readable by Gammapy's FluxPoints (at least until version 0.18.2)
# SED format based on
# https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/flux_points/index.html
import numpy as np
import astropy.units as u
from astropy.table import Table


# - Mrk421 2011
sed_file = np.loadtxt(
    "original/Mrk421_2011.txt",
    dtype={
        "names": ("nu", "sed", "sed_err_low", "sed_err_high", "instrument"),
        "formats": (float, float, float, float, "|S18"),
    },
)

# read
nu = sed_file["nu"] * u.Hz
nuFnu = sed_file["sed"] * u.Unit("erg cm-2 s-1")
nuFnu_err_low = sed_file["sed_err_low"] * u.Unit("erg cm-2 s-1")
nuFnu_err_high = sed_file["sed_err_high"] * u.Unit("erg cm-2 s-1")
instruments = sed_file["instrument"]

# store in table and remove points with orders of magnitude smaller error, they are upper limits
UL = nuFnu_err_low < (nuFnu * 1e-3)
sed_table = Table()
sed_table["e_ref"] = nu.to("eV", equivalencies=u.spectral())[~UL]
sed_table["e2dnde"] = nuFnu[~UL]
sed_table["e2dnde_errn"] = nuFnu_err_low[~UL]
sed_table["e2dnde_errp"] = nuFnu_err_high[~UL]
sed_table["instrument"] = instruments[~UL]
sed_table.meta["SED_TYPE"] = "e2dnde"
sed_table.meta["source"] = "Mrk421"
sed_table.meta["period"] = "2009"
sed_table.meta[
    "reference"
] = "https://ui.adsabs.harvard.edu/abs/2011ApJ...736..131A/abstract"
sed_table.meta["redshift"] = 0.03
# before writing sort in energy
sed_table.sort("e_ref")
sed_table.write("Mrk421_2011.ecsv", overwrite=True)


# - PKS1510-089 2015, Period B
sed_file = np.loadtxt(
    "original/PKS1510-089_2015b.txt",
    dtype={
        "names": ("E", "sed", "sed_err_low", "sed_err_high", "instrument"),
        "formats": (float, float, float, float, "|S15"),
    },
)

# read
E = sed_file["E"] * u.eV
nuFnu = sed_file["sed"] * u.Unit("TeV cm-2 s-1")
nuFnu_err_low = sed_file["sed_err_low"] * u.Unit("TeV cm-2 s-1")
nuFnu_err_high = sed_file["sed_err_high"] * u.Unit("TeV cm-2 s-1")
instruments = sed_file["instrument"]

# store in table and remove points with orders of magnitude smaller error, they are upper limits
UL = nuFnu_err_low < (nuFnu * 1e-3)
sed_table = Table()
sed_table["e_ref"] = E[~UL]
sed_table["e2dnde"] = nuFnu.to("erg cm-2 s-1")[~UL]
sed_table["e2dnde_errn"] = nuFnu_err_low.to("erg cm-2 s-1")[~UL]
sed_table["e2dnde_errp"] = nuFnu_err_high.to("erg cm-2 s-1")[~UL]
sed_table["instrument"] = instruments[~UL]
sed_table.meta["SED_TYPE"] = "e2dnde"
sed_table.meta["source"] = "PKS1510-089"
sed_table.meta["period"] = "2015, B"
sed_table.meta[
    "reference"
] = "https://ui.adsabs.harvard.edu/abs/2017A%26A...603A..29A/abstract"
sed_table.meta["redshift"] = 0.361
# before writing sort in energy
sed_table.sort("e_ref")
sed_table.write("PKS1510-089_2015b.ecsv", overwrite=True)
