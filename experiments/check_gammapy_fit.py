# import numpy, astropy and matplotlib for basic functionalities
import time
import pkg_resources
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt

# import agnpy classes
from agnpy.emission_regions import Blob
from agnpy.wrappers import SynchrotronSelfComptonSpectralModel

# import gammapy classes
from gammapy.modeling.models import SPECTRAL_MODEL_REGISTRY, SkyModel
from gammapy.estimators import FluxPoints
from gammapy.datasets import Datasets, FluxPointsDataset
from gammapy.modeling import Fit

# IMPORTANT: add the new custom model to the registry of spectral models recognised by gammapy
SPECTRAL_MODEL_REGISTRY.append(SynchrotronSelfComptonSpectralModel)

# total energy content of the electron distribution
spectrum_norm = 5e46 * u.Unit("erg")
# dictionary describing the electron distribution
spectrum_dict = {
    "type": "BrokenPowerLaw",
    "parameters": {
        "p1": 2.02,
        "p2": 3.43,
        "gamma_b": 9e4,
        "gamma_min": 500,
        "gamma_max": 1e6,
    },
}
R_b = 1.5e16 * u.cm
B = 0.1 * u.G
z = 0.0308
delta_D = 20
Gamma = 20
# emission region
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

# model
ssc_model = SynchrotronSelfComptonSpectralModel(blob)
ssc_model.parameters["gamma_b"].min = 1e3
ssc_model.parameters["gamma_b"].max = 5e5

print("loading flux points...")
# load the MWL flux points
datasets = Datasets()
flux_points = {}

sed_path = pkg_resources.resource_filename("agnpy", "data/mwl_seds/Mrk421_2011.ecsv")
table = Table.read(sed_path)
table = table.group_by("instrument")

# do not use frequency point below 1e11 Hz, affected by non-blazar emission
E_min_fit = (1e11 * u.Hz).to("eV", equivalencies=u.spectral())

for group in table.groups:
    name = group["instrument"][0]

    data = FluxPoints.from_table(group, sed_type="e2dnde", format="gadf-sed")
    dataset = FluxPointsDataset(data=data, name=name)

    flux_points.update({name: data})
    dataset.mask_fit = dataset.data.energy_ref > E_min_fit
    datasets.append(dataset)

# load the SSC model in the datasets
model = SkyModel(spectral_model=ssc_model, name="Mrk421")
datasets.models = [model]

# plot the starting model and the flux points
fig, ax = plt.subplots(figsize=(10, 6))

for key in flux_points.keys():
    flux_points[key].plot(ax=ax, label=key)

model.spectral_model.plot(
    energy_bounds=[1e-6, 1e14] * u.eV, energy_power=2, label="model"
)
plt.legend(ncol=4)
plt.xlim([1e-6, 1e14])
plt.show()
fig.savefig("initial_model.png")

# define the fitter and run it
fitter = Fit()
start = time.time()
print("fitting...")
results = fitter.run(datasets)
end = time.time()
delta_t = end - start
print(f"execution time {delta_t:.2f} s")

print(results)
print(model.spectral_model.parameters.to_table())

# plot the final model and the flux points
fig, ax = plt.subplots(figsize=(10, 6))

for key in flux_points.keys():
    flux_points[key].plot(ax=ax, label=key)

model.spectral_model.plot(
    energy_bounds=[1e-6, 1e14] * u.eV, energy_power=2, label="model"
)
plt.legend(ncol=4)
plt.xlim([1e-6, 1e14])
plt.show()
fig.savefig("final_model.png")
