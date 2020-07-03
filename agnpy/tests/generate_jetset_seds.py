# generate SEDs for comparison with jetset
from jetset.jet_model import Jet

# - SSC SED
jet = Jet(name="ssc", electron_distribution="pl",)
jet.parameters.par_table

jet.set_par("B", val=1)
# mind the normalisation in jetset is the total electron density (integrated along gamma)
jet.set_par("N", val=1299.9577)
jet.set_par("R", val=1e16)
jet.set_par("beam_obj", val=10)
jet.set_par("gmin", val=1e2)
jet.set_par("gmax", val=1e7)
jet.set_par("p", val=2.8)
jet.set_par("z_cosm", val=0.1)
# check the parameters have been updated
jet.set_gamma_grid_size(1000)
# use the same nu grid as agnpy
jet.nu_min = 1e8
jet.nu_max = 1e24
jet.nu_size = 50

# fetch and print unabsorbed and self-absorbed synchrotron points
jet.spectral_components.Sync.state = "on"
jet.eval()
synch_sed = jet.get_spectral_component_by_name("Sync").get_SED_points()[1]
print("jetset unabsorbed synchrotron SED spectral points")
print(synch_sed)

jet.spectral_components.Sync.state = "self-abs"
jet.eval()
synch_ssa_sed = jet.get_spectral_component_by_name("Sync").get_SED_points()[1]
print("jetset unabsorbed synchrotron SED spectral points")
print(synch_ssa_sed)
