# test on compton module
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c, M_sun
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.targets import PointSourceBehindJet, SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.absorption import Absorption, tau_disk_finke_2016
from agnpy.utils import make_comparison_plot
import matplotlib.pyplot as plt
from pathlib import Path
import pytest

mec2 = m_e.to("erg", equivalencies=u.mass_energy())
agnpy_dir = Path(__file__).parent.parent.parent
data_dir = f"{agnpy_dir}/data"

# variables with _test are global and meant to be used in all tests
# here as a default we use the same parameters of Figure 7.4 in Dermer Menon 2009
spectrum_norm_test = 1e48 * u.Unit("erg")
p_test = 2.8
gamma_min_test = 1e2
gamma_max_test = 1e5
pwl_dict_test = {
    "type": "PowerLaw",
    "parameters": {
        "p": p_test,
        "gamma_min": gamma_min_test,
        "gamma_max": gamma_max_test,
    },
}
# blob parameters
R_b_test = 1e16 * u.cm
B_test = 1 * u.G
z_test = Distance(1e27, unit=u.cm).z
delta_D_test = 10
Gamma_test = 10
pwl_blob_test = Blob(
    R_b_test,
    z_test,
    delta_D_test,
    Gamma_test,
    B_test,
    spectrum_norm_test,
    pwl_dict_test,
)


class TestAbsorption:
    """class grouping all tests related to the Absorption class"""

    def test_absorption_disk_reference_tau(self):
        """test agnpy gamma-gamma optical depth for Disk against the one in 
        Figure 14 of Finke 2016"""
        # array of energies used in figure 14 of Finke
        M_BH = 1.2 * 1e9 * M_sun.cgs
        L_disk = 2 * 1e46 * u.Unit("erg s-1")
        eta = 1 / 12
        R_in = 6
        R_out = 200
        disk = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)
        E = np.logspace(0, 5) * u.GeV
        nu = E.to("Hz", equivalencies=u.spectral())
        absorption_disk = Absorption(pwl_blob_test, disk, r=r)
        tau = absorption_disk.tau(nu)
        # compute the absorption using the exact formula in Finke 2016
        tau_finke = tau_disk_finke_2016(nu, pwl_blob_test, disk, r=r)
        # array of sampled taus
        sampled_tau = np.loadtxt(
            f"{data_dir}/sampled_taus/tau_disk_figure_14_finke_2016.txt",
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
        ax.set_xlabel(r"$E\,/\,{\rm GeV}$")
        ax.set_ylabel(r"$\tau_{\gamma \gamma}$")
        fig.savefig(
            f"{data_dir}/crosscheck_figures/tau_disk_comaprison_figure_14_finke_2016.png"
        )
        assert True

    def test_absorption_blr_reference_tau(self):
        """test agnpy gamma-gamma optical depth for BLR against the one in 
        Figure 14 of Finke 2016"""
        # reference tau
        sampled_abs_blr_table = np.loadtxt(
            f"{data_dir}/sampled_taus/tau_blr_lyman_alpha_figure_14_finke_2016.txt",
            delimiter=",",
            comments="#",
        )
        sampled_abs_blr_E = sampled_abs_blr_table[:, 0] * u.GeV
        sampled_abs_blr_tau = sampled_abs_blr_table[:, 1]
        # agnpy tau
        L_disk = 2 * 1e46 * u.Unit("erg s-1")
        xi_line = 0.024
        R_line = 1e17 * u.cm
        blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
        r = 0.1 * R_line
        # recompute the tau at the same ordinates where the figure was sampled
        sampled_abs_blr_nu = sampled_abs_blr_E.to("Hz", equivalencies=u.spectral())
        abs_blr = Absorption(pwl_blob_test, blr, r)
        agnpy_abs_blr_tau = abs_blr.tau(sampled_abs_blr_nu)
        # sed comparison plot
        make_comparison_plot(
            sampled_abs_blr_nu,
            sampled_abs_blr_tau,
            agnpy_abs_blr_tau,
            "Figure 14, Finke (2016)",
            "agnpy",
            "External Compton on Spherical Shell Broad Line Region",
            f"{data_dir}/crosscheck_figures/tau_blr_lyman_alpha_comparison_figure_14_finke_2016.png",
            "sed",
        )
        # requires that the tau points deviate less than 30% from the figure
        assert u.allclose(agnpy_abs_blr_tau, sampled_abs_blr_tau, atol=0, rtol=0.3,)

    def test_absorption_blr_vs_point_source(self):
        """check if in the limit of large distances the absorption on the BLR 
        tends to the one of a point-like source approximating it"""
        # broad line region
        L_disk = 2e46 * u.Unit("erg s-1")
        xi_line = 0.024
        R_line = 1.1e17 * u.cm
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
        fig.savefig(f"{data_dir}/crosscheck_figures/tau_blr_vs_ps_comparison.png")
        assert True
