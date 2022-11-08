import numpy as np
import astropy.units as u
from agnpy.spectra import ExpCutoffPowerLaw, BrokenPowerLaw
from agnpy.emission_regions import Blob
#from agnpy.synchrotron import Synchrotron
from synchrotron_new import Synchrotron
#from proton_synchrotron import ProtonSynchrotron
from proton_synchrotron2 import ProtonSynchrotron2
from agnpy.utils.plot import plot_sed
import matplotlib.pyplot as plt
#from agnpy.utils.plot import load_mpl_rc
#from astropy.constants import e, h, c, m_e, m_p, sigma_T
from astropy.constants import m_p, m_e
from astropy.coordinates import Distance
from agnpy.absorption import EBL

# Extract data of PKS 2155-304
pks_sed = np.loadtxt('PKS2155-304_data_circa.txt')
lognu = pks_sed[:,0]
lognuFnu = pks_sed[:,1]
nu = 10**lognu
nuFnu = 10**lognuFnu

print(nu)
print(nuFnu)

# Define source parameters
B = 80 * u.G
redshift = 0.117
distPKS = Distance(z=redshift)
doppler_s = 30
Gamma_bulk = 16
R = 5.2e14 * u.cm #radius of the blob
vol = (4. / 3) * np.pi * R ** 3

# Proton distribution: Exponential cut-off power-law (ECPL)
# "power-law defined above gamma_p_min, with slope alpha_p and with an
# exponential cut-off at gamma_p_max" (from M. Cerruti 2012)
#
# Proton dostribution parameters
alpha_p = 2.0
gamma_pcut = 1e9 # Cut-off Lorentz factor
gamma_pmin = 1
norm_p = 12e3 / u.Unit('cm3')
u_p = 3.7e2 * u.Unit('erg cm-3')

# define the proton distribution
n_p = ExpCutoffPowerLaw(k=norm_p,
        p=alpha_p,
        gamma_c=gamma_pcut,
        gamma_min=gamma_pmin,
        gamma_max=1e12,
        mass=m_p,
                        )

# Define electron distribution
n_e = BrokenPowerLaw(k=6e2 * u.Unit("cm-3"),
        p1=2.0,
        p2=4.32,
        gamma_b=4e3,
        gamma_min=1,
        gamma_max=6e3,
        mass=m_e,
                     )

blob = Blob(R_b=R,
        z=redshift,
        delta_D=doppler_s,
        Gamma=Gamma_bulk,
        B=B,
        n_e=n_e,
        n_p=n_p,
            )

synch = Synchrotron(blob)
psynch2 = ProtonSynchrotron2(blob)

# compute the SED over an array of frequencies
nu = np.logspace(8, 28) * u.Hz
#sed = synch.sed_flux(nu)
psed2 = psynch2.sed_flux(nu)
psed2_obs = psed2 * doppler_s**4

ebl = EBL("saldana-lopez")
absorption = ebl.absorption(redshift, nu)
psed2_obs_abs = psed2_obs * absorption # Check if it is correct

nu2 = [3.63894599e+14, 4.81866605e+14, 5.70292370e+14, 1.56719108e+17, 2.19514602e+17, 3.25233387e+17, 4.07151555e+17, 4.81866605e+17,
 7.98801781e+17, 1.18350673e+18, 8.93757115e+17, 1.25187503e+18, 1.48160252e+18, 1.75348655e+18, 1.85478119e+18, 2.07526313e+18,
 2.59797009e+18, 3.07471508e+18, 3.63894599e+18, 6.74944857e+22, 1.32419279e+23, 2.45608788e+23, 5.09702863e+23, 9.45387283e+23,
 1.75348655e+24, 5.70292370e+24, 6.38084285e+25, 1.18350673e+26, 2.32195425e+26, 4.30671707e+26, 7.98801781e+26]
nuFnu2 = [9.34519215e-11, 1.02283109e-10, 1.40300372e-10, 7.45674033e-11, 6.51216642e-11, 6.08574464e-11, 5.31484001e-11, 4.85595339e-11,
 3.78818668e-11, 3.87467512e-11, 3.30832253e-11, 2.70005462e-11, 2.52325292e-11, 2.30539368e-11, 2.05932784e-11, 1.83952580e-11,
 1.68069998e-11, 1.53558729e-11, 1.37168662e-11, 2.52325292e-11, 3.38385515e-11, 4.33765395e-11, 4.64158883e-11, 4.64158883e-11,
 4.85595339e-11, 6.81292069e-11, 3.30832253e-11, 1.37168662e-11, 5.43618362e-12, 2.46693021e-12, 1.83952580e-12]
# plot it
plt.figure()
plt.scatter(nu2, nuFnu2, color = 'black')
#plot_sed(nu, sed, label="Synchrotron")
plot_sed(nu, psed2_obs_abs, label = 'ProtonSynchrotron2')
plt.ylim(1e-14, 1e-9)
plt.xlim(1e10, 1e28) # For frequencies

plt.show()
