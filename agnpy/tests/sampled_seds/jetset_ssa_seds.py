# generate a SSA synchrotron SED to be confronted with the one produced by agnpy
from jetset.jet_model import Jet
import matplotlib.pyplot as plt

# jet with power-law electron distribution
pwl_jet = Jet(name="", electron_distribution="pl")
pwl_jet.show_model()
pwl_jet.electron_distribution.plot()
plt.show()
pwl_jet.eval()
pwl_synch_sed = pwl_jet.get_spectral_component_by_name("Sync").get_SED_points()
plt.loglog(pwl_synch_sed[0], pwl_synch_sed[1])
plt.ylim([1e-20, 1e-9])
plt.show()
print("SSA synch SED with power-law electron distribution:")
for (nu, sed) in zip(pwl_synch_sed[0], pwl_synch_sed[1]):
    if sed > 1e-120:
        print(f"{nu}, {sed}")

# jet with broken power-law electron distribution
bpl_jet = Jet(name="", electron_distribution="bkn")
bpl_jet.show_model()
bpl_jet.eval()
bpl_synch_sed = bpl_jet.get_spectral_component_by_name("Sync").get_SED_points()
plt.loglog(bpl_synch_sed[0], bpl_synch_sed[1])
plt.ylim([1e-20, 1e-9])
plt.show()
print("SSA synch SED with broken power-law electron distribution:")
for (nu, sed) in zip(bpl_synch_sed[0], bpl_synch_sed[1]):
    if sed > 1e-120:
        print(f"{nu}, {sed}")

# jet with log-parabola electron distribution
lp_jet = Jet(name="", electron_distribution="lp")
lp_jet.show_model()
lp_jet.eval()
lp_synch_sed = lp_jet.get_spectral_component_by_name("Sync").get_SED_points()
plt.loglog(lp_synch_sed[0], lp_synch_sed[1])
plt.ylim([1e-20, 1e-9])
plt.show()
print("SSA synch SED with log-parabola electron distribution:")
for (nu, sed) in zip(lp_synch_sed[0], lp_synch_sed[1]):
    if sed > 1e-120:
        print(f"{nu}, {sed}")
