import numpy as np
import astropy.units as u
from astropy.constants import m_e, c
import logging
from agnpy import Synchrotron, SynchrotronSelfCompton
from agnpy.spectra import PowerLaw
from agnpy.emission_regions import Blob
import matplotlib.pyplot as plt
from agnpy.time_evolution.time_evolution import TimeEvolution, synchrotron_loss, ssc_loss

logging.basicConfig(level=logging.INFO, force=True, format="%(message)s")

fig, axs = plt.subplots(2,1)

def plot_eed(**kwargs):
    blob.n_e.plot(ax=axs[0], gamma=gamma_logspace, gamma_power=2,  **kwargs)

def plot_sed(**kwargs):
    nu_hz = np.logspace(8, 30) * u.Hz
    sed = ssc.sed_flux(nu_hz) + synch.sed_flux(nu_hz)
    ax = axs[1]
    ax.set_xlabel(r"$\nu\,/\,\mathrm{TeV}$")
    ax.set_ylabel(r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$")
    nu = nu_hz.to("TeV", equivalencies=u.spectral())
    ax.loglog(nu, sed, **kwargs)

R_b = 100*u.s * c
B = 0.1 * u.G
V = (4 / 3) * np.pi * R_b ** 3
n_e_pl = PowerLaw.from_total_energy(1e45 * u.erg, V, p=2, gamma_min=1e3, gamma_max=1e6, mass=m_e)
blob = Blob(R_b=R_b, B=B, n_e=n_e_pl)
synch = Synchrotron(blob)
ssc = SynchrotronSelfCompton(blob)

gamma_logspace = np.logspace(3, 6, 200)
plot_eed(color="red", label="start")
plot_sed(color="red")

TimeEvolution(blob, 30 * u.s, energy_change_functions=[synchrotron_loss(synch), ssc_loss(ssc)]).evaluate()

plot_eed(color="black", label="end")
plot_sed(color="black")

axs[0].legend()
plt.show()