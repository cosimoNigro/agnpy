import numpy as np
import astropy.units as u
from astropy.table import Table

data = np.loadtxt("sed.ecsv")
energies = data[:, 0] * u.eV
flux = data[:, 1] * u.Unit("erg cm-2 s-1")
flux_err = data[:, 2] * u.Unit("erg cm-2 s-1")
nu = energies.to("Hz", equivalencies=u.spectral())

sed_table = Table([nu, flux, flux_err], names=("x", "y", "dy"))
sed_table.meta = {
    "obj_name": "PKS 1510-089",
    "data_scale": "lin-lin",
    "restframe": "obs",
    "z": 0.361,
}

print(sed_table)
print(sed_table.meta)
sed_table.write("astropy_sed.ecsv", format="ascii.ecsv")
