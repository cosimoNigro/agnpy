# make EC on CMB SED with jetset to compare against agnpy, needs jetset to run
from jetset.jet_model import Jet
import numpy as np
import astropy.units as u
from agnpy.emission_regions import Blob
import matplotlib.pyplot as plt

# agnpy
# - blob
spectrum_norm = 6e42 * u.erg
parameters = {
    "p1": 2.0,
    "p2": 3.5,
    "gamma_b": 1e4,
    "gamma_min": 20,
    "gamma_max": 5e7,
}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 0.56 * u.G
z = 1
delta_D = 40
Gamma = 40
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

# jet(set) with broken power-law electron distribution
jet = Jet(name="", electron_distribution="bkn", electron_distribution_log_values=False)
jet.add_EC_component(["EC_CMB"])
# - blob
jet.set_par("N", val=blob.n_e_tot.value)
jet.set_par("p", val=blob.n_e.p1)
jet.set_par("p_1", val=blob.n_e.p2)
jet.set_par("gamma_break", val=blob.n_e.gamma_b)
jet.set_par("gmin", val=blob.n_e.gamma_min)
jet.set_par("gmax", val=blob.n_e.gamma_max)
jet.set_par("R", val=blob.R_b.value)
jet.set_par("B", val=blob.B.value)
jet.set_par("beam_obj", val=blob.Gamma)
jet.set_par("z_cosm", val=blob.z)
jet.electron_distribution.update()
jet.set_external_field_transf("disk")
jet.set_nu_grid(1e14, 1e30, 100)
jet.show_model()
jet.eval()

# plot and store the components
nu = jet.spectral_components.EC_CMB.SED.nu
ec_sed = jet.spectral_components.EC_CMB.SED.nuFnu
plt.loglog(nu, ec_sed)
plt.ylim([1e-30, 1e-20])
plt.show()
condition = ec_sed.value > 1e-40
nu = nu.value[condition]
sed = ec_sed.value[condition]
np.savetxt("data/ec_cmb_bpwl_jetset_1.1.2.txt", np.asarray([nu, sed]).T, delimiter=",")
