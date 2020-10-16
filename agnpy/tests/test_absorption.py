# test on compton module
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c, M_sun
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.targets import PointSourceBehindJet, SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.absorption import Absorption, tau_disk_finke_2016
import matplotlib.pyplot as plt
from pathlib import Path
import pytest

mec2 = m_e.to("erg", equivalencies=u.mass_energy())
agnpy_dir = Path(__file__).parent.parent


class TestAbsorption:
    """class grouping all tests related to the Absorption class"""

    def test_absorption_disk_reference_tau(self):
        """test agnpy gamma-gamma optical depth for Disk against the one in 
        Figure 14 of Finke 2016"""
        # array of energies used in figure 14 of Finke
        blob = Blob()

        r = 1.1 * 1e16 * u.cm
        E = np.logspace(0, 5) * u.GeV
        nu = E.to("Hz", equivalencies=u.spectral())
        absorption_disk = Absorption(pwl_blob_test, disk_test, r=r)
        tau = absorption_disk.tau(nu)
        # compute the absorption using the exact formula in Finke 2016
        tau_finke = tau_disk_finke_2016(nu, pwl_blob_test, disk_test, r=r)
        # array of sampled taus
        sampled_tau = np.loadtxt(
            f"{agnpy_dir}/data/sampled_taus/tau_disk_figure_14_finke_2016.txt",
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
        ax.set_title("Absorption Shakura Sunyaev Disk")
        ax.set_xlabel(r"$E\,/\,{\rm GeV}$", fontsize=12)
        ax.set_ylabel(r"$\tau_{\gamma \gamma}$", fontsize=12)
        fig.savefig(
            f"{tests_dir}/crosscheck_figures/tau_disk_comaprison_figure_14_finke_2016.png"
        )
        assert True

    def test_absorption_blr_reference_tau(self):
        # array of energies used in figure 14 of Finke
        r = 1.1 * 1e16 * u.cm
        E = np.logspace(0, 5) * u.GeV
        nu = E.to("Hz", equivalencies=u.spectral())
        absorption_blr = Absorption(pwl_blob_test, blr_test, r=r)
        tau = absorption_blr.tau(nu)
        # array of sampled taus
        sampled_tau = np.loadtxt(
            f"{agnpy_dir}/data/sampled_taus/tau_blr_lyman_alpha_figure_14_finke_2016.txt",
            delimiter=",",
            comments="#",
        )
        E_ref = sampled_tau[:, 0] * u.GeV
        tau_ref = sampled_tau[:, 1]
        # comparison plot
        fig, ax = plt.subplots()
        ax.loglog(E_ref, tau_ref, marker="o", ls="-", label="reference")
        ax.loglog(E, tau, marker=".", ls="--", label="agnpy")
        ax.legend()
        ax.set_title("Absorption Broad Line Region")
        ax.set_xlabel(r"$E\,/\,{\rm GeV}$", fontsize=12)
        ax.set_ylabel(r"$\tau_{\gamma \gamma}$", fontsize=12)
        fig.savefig(
            f"{tests_dir}/crosscheck_figures/tau_blr_lyman_alpha_comaprison_figure_14_finke_2016.png"
        )
        assert True

    def test_absorption_blr_vs_point_source(self):
        """
        """
        # broad line region
        L_disk = 2e46 * u.Unit("erg s-1")
        xi_line = 0.024
        r = 100 * R_line
        # point like source approximating the blr
        blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
        ps_blr = PointSourceBehindJet(blr.xi_line * blr.L_disk, blr.epsilon_line)
        # absorption
        abs_blr = Absorption(pwl_blob_test, blr, r)
        abs_ps_blr = Absorption(pwl_blob_test, ps_blr, r)
        # seds
        E = np.logspace(0, 5) * u.GeV
        print(E)
        nu = E.to("Hz", equivalencies=u.spectral())
        tau_blr = abs_blr.tau(nu)
        tau_ps_blr = abs_ps_blr.tau(nu)
        # comparison plot
        fig, ax = plt.subplots()
        ax.loglog(E, tau_blr, marker="o", ls="-", label="BLR")
        ax.loglog(
            E, tau_ps_blr, marker=".", ls="--", label="point source apprximating BLR"
        )
        ax.legend()
        ax.set_title("Absorption BLR")
        ax.set_xlabel(r"$E\,/\,{\rm GeV}$", fontsize=12)
        ax.set_ylabel(r"$\tau_{\gamma \gamma}$", fontsize=12)
        fig.savefig(f"{tests_dir}/crosscheck_figures/tau_blr_vs_ps_comparison.png")
        assert True
