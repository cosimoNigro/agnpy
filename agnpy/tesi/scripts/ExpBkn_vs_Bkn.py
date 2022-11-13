import numpy as np
import astropy.units as u
from agnpy.spectra import BrokenPowerLaw, ExpCutoffBrokenPowerLaw
from agnpy.emission_regions import Blob
#from agnpy.synchrotron import Synchrotron
from synchrotron_new import Synchrotron
from proton_synchrotron import ProtonSynchrotron

from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc
#from astropy.constants import e, h, c, m_e, m_p, sigma_T
from astropy.constants import m_p, m_e
from astropy.coordinates import Distance
from agnpy.absorption import EBL
import matplotlib.style

# Parameters for exp bkn and bkn: using the PKS 2155-304 parameters
# Exp Cut-off Broken Power Law
k=3.75e-5 * u.Unit("cm-3")
p1=2.0
p2=4.32
gamma_b=4e3
gamma_min=1
gamma_max=6e5
gamma_cutoff = 6e4

n_exp = ExpCutoffBrokenPowerLaw(
    k, p1, p2,gamma_cutoff, gamma_b, gamma_min, gamma_max
)

n_bkn = BrokenPowerLaw(
    k, p1, p2, gamma_b, gamma_min, gamma_max
)

gamma = np.logspace(np.log10(gamma_min),np.log10(gamma_max)) #just for exp

# the two particle distributions
n_ex = n_exp(gamma)
n_bk = n_bkn(gamma)


# Defining the blob
B = 80 * u.G
redshift = 0.117
distPKS = Distance(z=redshift)
doppler_s = 30
Gamma_bulk = 16
R = 5.2e14 * u.cm #radius of the blob

# Blob with exponential cut off broken power law
blob1 = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_e=n_exp,
)
# Blob with broken power law
blob2 = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_e=n_bkn,
)

#exp cut off bkn
e_synch = Synchrotron(blob1)
e_synch_ssa = Synchrotron(blob1, ssa = True)
# bkn
b_synch = Synchrotron(blob2)
b_synch_ssa = Synchrotron(blob2, ssa = True)

# Plotting distributions
plt.loglog(gamma,n_ex)
plt.loglog(gamma,n_bk)
plt.show()

# Plotting SED
nu = np.logspace(8, 23) * u.Hz
e_sed = e_synch.sed_flux(nu)
e_sed_ssa = e_synch_ssa.sed_flux(nu)
plot_sed(nu, e_sed, label="Synchrotron")
plot_sed(nu, e_sed_ssa, label="Synchrotron with SSA")
b_sed = b_synch.sed_flux(nu)
b_sed_ssa = b_synch_ssa.sed_flux(nu)
plot_sed(nu, b_sed, label="Synchrotron")
plot_sed(nu, b_sed_ssa, label="Synchrotron with SSA")
