# generate a SSA synchrotron SED to be confronted with the one produced by agnpy
from jetset.jet_model import Jet
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

# jet with power-law electron distribution
pwl_jet = Jet(
    name="", electron_distribution="pl", electron_distribution_log_values=False
)
pwl_jet.set_nu_grid(1e9, 1e20, 50)
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
np.savetxt("synch_ssa_pwl_jetset_1.1.2.txt", np.asarray([nu, sed]).T, delimiter=",")

# jet with broken power-law electron distribution
bpl_jet = Jet(name="", electron_distribution="bkn")
bpl_jet.set_nu_grid(1e9, 1e20, 50)
bpl_jet.show_model()
bpl_jet.eval()
synch_nu = bpl_jet.spectral_components.Sync.SED.nu
synch_sed = bpl_jet.spectral_components.Sync.SED.nuFnu
plt.loglog(synch_nu, synch_sed)
plt.ylim([1e-20, 1e-9])
plt.show()
condition = synch_sed.value > 1e-20
nu = synch_nu.value[condition]
sed = synch_sed.value[condition]
np.savetxt("synch_ssa_bpwl_jetset_1.1.2.txt", np.asarray([nu, sed]).T, delimiter=",")

# jet with broken log-parabola electron distribution
lp_jet = Jet(name="", electron_distribution="lp")
lp_jet.set_nu_grid(1e9, 1e20, 50)
lp_jet.show_model()
lp_jet.eval()
synch_nu = lp_jet.spectral_components.Sync.SED.nu
synch_sed = lp_jet.spectral_components.Sync.SED.nuFnu
plt.loglog(synch_nu, synch_sed)
plt.ylim([1e-20, 1e-9])
plt.show()
condition = synch_sed.value > 1e-20
nu = synch_nu.value[condition]
sed = synch_sed.value[condition]
np.savetxt("synch_ssa_lp_jetset_1.1.2.txt", np.asarray([nu, sed]).T, delimiter=",")
