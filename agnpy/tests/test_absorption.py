# test on absorption module
import pytest
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c, M_sun
from astropy.coordinates import Distance
from pathlib import Path
from agnpy.emission_regions import Blob
from agnpy.targets import PointSourceBehindJet, SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.absorption import Absorption, tau_disk_finke_2016
from .utils import (
    make_comparison_plot,
    extract_columns_sample_file,
    check_deviation_within_bounds,
)
import matplotlib.pyplot as plt

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
pwl_blob_test = Blob(
    1e16 * u.cm, 0, 10, 10, 1 * u.G, spectrum_norm_test, pwl_dict_test,
)


class TestAbsorption:
    """class grouping all tests related to the Absorption class"""

    def test_absorption_disk_reference_tau(self):
        """test agnpy gamma-gamma optical depth for Disk against the one in 
        Figure 14 of Finke 2016"""
        # reference tau
        E_ref, tau_ref = extract_columns_sample_file(
            f"{data_dir}/sampled_taus/tau_disk_figure_14_finke_2016.txt", "GeV",
        )
        # target
        M_BH = 1.2 * 1e9 * M_sun.cgs
        L_disk = 2e46 * u.Unit("erg s-1")
        eta = 1 / 12
        R_in = 6
        R_out = 200
        disk = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)
        r = 1.1e16 * u.cm
        # array of energies used in figure 14 of Finke
        E = np.logspace(0, 5) * u.GeV
        nu = E.to("Hz", equivalencies=u.spectral())
        # compute the tau for agnpy
        abs_disk = Absorption(pwl_blob_test, disk, r)
        tau_agnpy = abs_disk.tau(nu)
        # compute the absorption using the exact formula in Finke 2016
        tau_finke = tau_disk_finke_2016(nu, pwl_blob_test, disk, r)
        # comparison plot
        fig, ax = plt.subplots()
        ax.loglog(E_ref, tau_ref, marker="o", ls="-", label="reference")
        ax.loglog(E, tau_agnpy, marker=".", ls="--", label="agnpy")
        ax.loglog(E, tau_finke, marker=".", ls="--", label="Finke 2016, Eq. 63")
        ax.legend()
        ax.set_title("Absorption Shakura Sunyaev Disk")
        ax.set_xlabel(r"$E\,/\,{\rm GeV}$")
        ax.set_ylabel(r"$\tau_{\gamma \gamma}$")
        fig.savefig(
            f"{data_dir}/crosscheck_figures/tau_disk_comaprison_figure_14_finke_2016.png"
        )
        assert True

    def test_abs_blr_reference_tau(self):
        """test agnpy gamma-gamma optical depth for BLR against the one in 
        Figure 14 of Finke 2016"""
        # reference tau
        E_ref, tau_ref = extract_columns_sample_file(
            f"{data_dir}/sampled_taus/tau_blr_lyman_alpha_figure_14_finke_2016.txt",
            "GeV",
        )
        nu_ref = E_ref.to("Hz", equivalencies=u.spectral())
        # target
        L_disk = 2e46 * u.Unit("erg s-1")
        xi_line = 0.024
        R_line = 1e17 * u.cm
        blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
        r = 1.1e16 * u.cm
        # recompute the tau, use the full energy range of figure 14
        ec_blr = Absorption(pwl_blob_test, blr, r)
        tau_agnpy = ec_blr.tau(nu_ref)
        # comparison plot
        make_comparison_plot(
            nu_ref,
            tau_ref,
            tau_agnpy,
            "Figure 14, Finke (2016)",
            "agnpy",
            "Absorption on Spherical Shell Broad Line Region",
            f"{data_dir}/crosscheck_figures/tau_blr_lyman_alpha_comparison_figure_14_finke_2016.png",
            "tau",
        )
        assert True

    def test_abs_dt_reference_tau(self):
        """test agnpy gamma-gamma optical depth for DT against the one in 
        Figure 14 of Finke 2016"""
        # reference tau
        E_ref, tau_ref = extract_columns_sample_file(
            f"{data_dir}/sampled_taus/tau_dt_figure_14_finke_2016.txt", "GeV"
        )
        nu_ref = E_ref.to("Hz", equivalencies=u.spectral())
        # target
        L_disk = 2e46 * u.Unit("erg s-1")
        T_dt = 1e3 * u.K
        csi_dt = 0.1
        dt = RingDustTorus(L_disk, csi_dt, T_dt)
        r = 1.1e16 * u.cm
        # recompute the tau, use the full energy range of figure 14
        ec_dt = Absorption(pwl_blob_test, dt, r)
        tau_agnpy = ec_dt.tau(nu_ref)
        # comparison plot
        make_comparison_plot(
            nu_ref,
            2 * tau_ref,
            tau_agnpy,
            "Figure 14, Finke (2016)",
            "agnpy",
            "Absorption on Ring Dust Torus",
            f"{data_dir}/crosscheck_figures/tau_dt_comparison_figure_14_finke_2016.png",
            "tau",
        )
        assert True

    def test_abs_blr_vs_point_source(self):
        """check if in the limit of large distances the gamma-gamma optical depth 
        on the BLR tends to the one of a point-like source approximating it"""
        # broad line region
        L_disk = 2e46 * u.Unit("erg s-1")
        xi_line = 0.024
        R_line = 1e17 * u.cm
        blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
        r = 1e20 * u.cm
        # point like source approximating the blr
        ps_blr = PointSourceBehindJet(blr.xi_line * L_disk, blr.epsilon_line)
        # absorption
        abs_blr = Absorption(pwl_blob_test, blr, r)
        abs_ps_blr = Absorption(pwl_blob_test, ps_blr, r)
        # taus
        E = np.logspace(2, 6) * u.GeV
        nu = E.to("Hz", equivalencies=u.spectral())
        tau_blr = abs_blr.tau(nu)
        tau_ps_blr = abs_ps_blr.tau(nu)
        # sed comparison plot
        make_comparison_plot(
            nu,
            tau_blr,
            tau_ps_blr,
            "spherical shell BLR",
            "point source approximating the BLR",
            "Absorption on Spherical Shell BLR, "
            + r"$r = 10^{20}\,{\rm cm} \gg R_{\rm line}$",
            f"{data_dir}/crosscheck_figures/tau_blr_point_source_comparison.png",
            "tau",
        )
        # requires a 10% deviation from the two SED points
        assert check_deviation_within_bounds(nu, tau_blr, tau_ps_blr, 0, 0.1)

    def test_abs_dt_vs_point_source(self):
        """check if in the limit of large distances the gamma-gamma optical depth 
        on the DT tends to the one of a point-like source approximating it"""
        # dust torus
        L_disk = 2e46 * u.Unit("erg s-1")
        T_dt = 1e3 * u.K
        csi_dt = 0.1
        dt = RingDustTorus(L_disk, csi_dt, T_dt)
        r = 1e22 * u.cm
        # point like source approximating the dt
        ps_dt = PointSourceBehindJet(dt.xi_dt * L_disk, dt.epsilon_dt)
        # absorption
        abs_dt = Absorption(pwl_blob_test, dt, r)
        abs_ps_dt = Absorption(pwl_blob_test, ps_dt, r)
        # taus
        E = np.logspace(2, 6) * u.GeV
        nu = E.to("Hz", equivalencies=u.spectral())
        tau_dt = abs_dt.tau(nu)
        tau_ps_dt = abs_ps_dt.tau(nu)
        make_comparison_plot(
            nu,
            tau_dt,
            tau_ps_dt,
            "ring dust torus",
            "point source approximating the DT",
            "Absorption on Ring Dust Torus, "
            + r"$r = 10^{22}\,{\rm cm} \gg R_{\rm dt}$",
            f"{data_dir}/crosscheck_figures/tau_dt_point_source_comparison.png",
            "tau",
        )
        # requires a 10% deviation from the two SED points
        assert check_deviation_within_bounds(nu, tau_dt, tau_ps_dt, 0, 0.1)
