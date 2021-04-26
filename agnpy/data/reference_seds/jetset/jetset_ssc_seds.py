# generate a synchrotron + SSC SED to be confronted with the one produced by
# agnpy and Figure 7.4 of Dermer 2009
from jetset.jet_model import Jet
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

# jet with power-law electron distribution
pwl_jet = Jet(
    name="", electron_distribution="pl", electron_distribution_log_values=False
)
# set parameters according to Figure 7.4 of Dermer 2009
pwl_jet.set_par("N", 1298.13238394)
pwl_jet.set_par("p", 2.8)
pwl_jet.set_par("gmin", 1e2)
pwl_jet.set_par("gmax", 1e5)
pwl_jet.set_par("B", 1)
pwl_jet.set_par("R", 1e16)
pwl_jet.set_par("beam_obj", 10)
pwl_jet.set_par("z_cosm", 0.07)
# remove SSA
pwl_jet.spectral_components.Sync.state = "on"

# synchrotron emission
pwl_jet.set_nu_grid(1e9, 1e19, 100)
pwl_jet.show_model()
pwl_jet.eval()

synch_nu = pwl_jet.spectral_components.Sync.SED.nu
synch_sed = pwl_jet.spectral_components.Sync.SED.nuFnu
plt.loglog(synch_nu, synch_sed)
plt.ylim([1e-20, 1e-9])
plt.show()
condition = synch_sed.value > 1e-20
nu = synch_nu.value[condition]
sed = synch_sed.value[condition]
np.savetxt("synch_pwl_jetset_1.1.2.txt", np.asarray([nu, sed]).T, delimiter=",")

# SSC emission
pwl_jet.set_nu_grid(1e14, 1e26, 100)
pwl_jet.set_IC_nu_size(100)
pwl_jet.show_model()
pwl_jet.eval()

ssc_nu = pwl_jet.spectral_components.SSC.SED.nu
ssc_sed = pwl_jet.spectral_components.SSC.SED.nuFnu
plt.loglog(ssc_nu, ssc_sed)
plt.ylim([1e-20, 1e-9])
plt.show()
condition = ssc_sed.value > 1e-20
nu = ssc_nu.value[condition]
sed = ssc_sed.value[condition]
np.savetxt("ssc_pwl_jetset_1.1.2.txt", np.asarray([nu, sed]).T, delimiter=",")
