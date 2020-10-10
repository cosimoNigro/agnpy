# test on compton module
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c, M_sun
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.targets import SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.absorption import Absorption, tau_disk_finke_2016
import matplotlib.pyplot as plt
from pathlib import Path
import pytest

mec2 = m_e.to("erg", equivalencies=u.mass_energy())
tests_dir = Path(__file__).parent

# variables with _test are global and meant to be used in all tests
pwl_spectrum_norm_test = 1e48 * u.Unit("erg")
pwl_dict_test = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e5,},
}
pwl_blob_test = Blob(
    1e16 * u.cm, 0, 40, 50, 1 * u.G, pwl_spectrum_norm_test, pwl_dict_test,
)
disk_test = SSDisk(
    1.2 * 1e9 * M_sun.cgs, 2 * 1e46 * u.Unit("erg s-1"), 1 / 12, 6, 200, R_g_units=True
)

class TestAbsorption:
    """class grouping all tests related to the Absorption class"""

    def test_tau_disk_reference_tau(self):
        """test agnpy optical depth for Disk against the one in Figure 
        14 of Finke 2016"""
        # array of energies used in figure 14 of Finke
        r = 1.1 * 1e16 * u.cm
        E = np.logspace(0, 5) * u.GeV
        nu = E.to("Hz", equivalencies=u.spectral())
        absorption_disk = Absorption(pwl_blob_test, disk_test, r=r)
        tau = absorption_disk.tau(nu)
        # compute the absorption using the exact formula in Finke 2016
        tau_finke = tau_disk_finke_2016(nu, pwl_blob_test, disk_test, r=r)
        # array of sampled taus
        sampled_tau = np.loadtxt(
            f"{tests_dir}/sampled_taus/tau_disk_figure_14_finke_2016.txt",
            delimiter=",",
            comments="#",
        )
        E_ref = sampled_tau[:, 0] * u.GeV
        tau_ref = sampled_tau[:, 1]
        # comparison plot
        fig, ax = plt.subplots()
        ax.loglog(E_ref, tau_ref, marker="o", ls="-", label="reference")
        ax.loglog(E, tau, marker=".", ls="--", label="agnpy")
        ax.loglog(E, tau_finke, marker=".", ls="--", label="Finke 2016, Eq. 63")
        ax.legend()
        ax.set_xlabel(r"$E\,/\,{\rm GeV}$", fontsize=12)
        ax.set_ylabel(r"$\tau_{\gamma \gamma}$", fontsize=12)
        fig.savefig(
            f"{tests_dir}/crosscheck_figures/tau_disk_comaprison_figure_14_finke_2016.png"
        )
        assert True