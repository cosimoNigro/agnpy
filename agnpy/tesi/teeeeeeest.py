import numpy as np
import astropy.units as u
from agnpy.spectra import PowerLaw, BrokenPowerLaw, LogParabola, ExpCutoffPowerLaw, ExpCutoffBrokenPowerLaw
from agnpy.utils.conversion import mpc2
from agnpy.utils.plot import plot_sed
from agnpy.emission_regions import Blob
from agnpy.utils.plot import load_mpl_rc
from synchrotron_new import Synchrotron
from proton_synchrotron import ProtonSynchrotron
import matplotlib.pyplot as plt
from astropy.constants import m_p,m_e, h
from astropy.coordinates import Distance

#load_mpl_rc()  # adopt agnpy plotting style
plt.style.use('proton_synchrotron')

# Define source parameters PKS2155
B = 80 * u.G
redshift = 0.117
distPKS = Distance(z=redshift) # Already inside blob definition
doppler_s = 30
Gamma_bulk = 16
R = 5.2e14 * u.cm #radius of the blob
vol = (4. / 3) * np.pi * R ** 3

norm_p2 = 12e3 / u.Unit('cm3')
u_p = 3.7e2 * u.Unit('erg cm-3')


BPL_P= BrokenPowerLaw(k=12e3 * u.Unit("cm-3"),
        p1=2.0,
        p2=3.0,
        gamma_min=1,
        gamma_max=1e9,
        mass=m_p,
)

BPL_E = BrokenPowerLaw(k=12e3 * u.Unit("cm-3"),
        p1=2.0,
        p2=3.0,
        gamma_min=1,
        gamma_max=1e9,
        mass=m_e,
)


# distributions:
distributions = [BPL_E, BPL_P ]
labels = ['electrons', 'protons']

# compute the SED over an array of frequencies
nu = np.logspace(10,30, 100) * u.Hz

for i in distributions:

    label = labels[distributions.index(i)]
    blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_p= i
    )

    if label == 'electrons':
        synch = Synchrotron(blob)
    elif label == 'protons':
        synch= ProtonSynchrotron(blob)

    sed = synch.sed_flux(nu)

    #plt.loglog(nu, psed,  label = 'Proton Synctrotron')

    plot_sed(nu,  sed, label = label)
    plt.ylim(1e-28, 1e-6)
    plt.xlim(1e7, 1e30) # For frequencies

plt.show()
