"""profile and test synchrotron and synchrotron self Comton radiation"""
import sys

sys.path.append("../")
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton

# to profile
import cProfile, pstats
import timeit

# functions to profile and time
def profile(command, label):
    """function to profile a given command"""
    print(f"->{command} profiling section...")
    cProfile.run(command, f"Profile_{label}.prof")
    prof = pstats.Stats(f"Profile_{label}.prof")
    prof.strip_dirs().sort_stats("time").print_stats(10)


def timing(command, number):
    """function to time a given command, returns time in seconds"""
    return timeit.timeit(command, globals=globals(), number=number)


# define the blob
spectrum_norm = 1e48 * u.Unit("erg")
spectrum_dict = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e7},
}
R_b = 1e16 * u.cm
B = 1 * u.G
z = Distance(1e27, unit=u.cm).z
delta_D = 10
Gamma = 10
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
print("blob definition:")
print(blob)
synch = Synchrotron(blob)
synch_ssa = Synchrotron(blob, ssa=True)
ssc = SynchrotronSelfCompton(blob, synch)
ssc_ssa = SynchrotronSelfCompton(blob, synch_ssa)
nu_syn = np.logspace(8, 23) * u.Hz
nu_ssc = np.logspace(15, 30) * u.Hz

# commands to profile
syn_sed_command = "synch.sed_flux(nu_syn)"
syn_sed_ssa_command = "synch_ssa.sed_flux(nu_syn)"
ssc_sed_command = "ssc.sed_flux(nu_ssc)"
ssc_ssa_sed_command = "ssc_ssa.sed_flux(nu_ssc)"

n = 100
print("\nprofiling synchrotron sed computation:")
profile(syn_sed_command, "syn_sed")
time_syn = timing(syn_sed_command, n)
time_syn /= n
print(f"time: {time_syn:.2e} s")

print("\nprofiling synchrotron w/ SSA sed computation:")
profile(syn_sed_ssa_command, "syn_sed_ssa")
time_syn_ssa = timing(syn_sed_ssa_command, n)
time_syn_ssa /= n
print(f"time: {time_syn_ssa:.2e} s")

print("\nprofiling SSC sed computation:")
profile(ssc_sed_command, "ssc_sed")
time_ssc = timing(ssc_sed_command, n)
time_ssc /= n
print(f"time: {time_ssc:.2e} s")

print("\nprofiling SSC w/ SSA sed computation:")
profile(ssc_ssa_sed_command, "ssc_ssa_sed")
time_ssc_ssa = timing(ssc_ssa_sed_command, n)
time_ssc_ssa /= n
print(f"time: {time_ssc_ssa:.2e} s")
