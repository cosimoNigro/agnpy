import numpy as np
import astropy.units as u
from agnpy.spectra import PowerLaw, BrokenPowerLaw, LogParabola, ExpCutoffPowerLaw, ExpCutoffBrokenPowerLaw
from agnpy.utils.conversion import mpc2, mec2
from agnpy.utils.plot import plot_sed
from agnpy.emission_regions import Blob
from agnpy.utils.plot import load_mpl_rc
from agnpy.synchrotron import Synchrotron, ProtonSynchrotron
import matplotlib.pyplot as plt
from astropy.constants import m_p, h, m_e,m_p
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

# define the emission region and the radiative process
PL = PowerLaw(k=12e3* u.Unit("cm-3"),
        p=2.0,
        gamma_min=1,
        gamma_max=1e9,
        mass=m_p,
)

BPL = BrokenPowerLaw(k=1e-3 * u.Unit("cm-3"),
        p1=2.0,
        p2=3.0,
        gamma_b=1e6,
        gamma_min=1,
        gamma_max=1e9,
        mass=m_p,
)

LP = LogParabola(k=12e1 * u.Unit("cm-3"),
         p =2.,
         q=0.1,
         gamma_0=1e3,
         gamma_min=10,
         gamma_max=1e9,
         mass=m_p,
)

ECPL = ExpCutoffPowerLaw(k=12e6 * u.Unit("cm-3"),
        p=2.0,
        gamma_c = 1e7,
        gamma_min=10,
        gamma_max=1e9,
        mass=m_p,
)

ECBPL = ExpCutoffBrokenPowerLaw(k=12e5 * u.Unit("cm-3"),
         p1 =2.,
         p2 =3.,
         gamma_c = 1e5,
         gamma_b = 1e3,
         gamma_min=10,
         gamma_max=1e9,
         mass=m_p,
)

# distributions:
distributions = [PL, BPL, LP, ECBPL, ECPL ]
labels = ['PL', 'BPL', 'LP', 'ExpBPL', 'ExpPL']


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


    psynch= ProtonSynchrotron(blob)
    psed = psynch.sed_flux(nu)
    a = ['.']
    #plt.loglog(nu, psed,  label = 'Proton Synctrotron')

    plot_sed(nu,  psed, label = label)

    plt.ylim(1e-15, 1e-7)
    plt.xlim(1e10, 1e26) # For frequencies
plt.show()
