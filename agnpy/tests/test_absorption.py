# test on absorption module
import pytest
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c, M_sun
from astropy.coordinates import Distance
from pathlib import Path
from agnpy.emission_regions import Blob
from agnpy.targets import PointSourceBehindJet, SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.absorption import Absorption, tau_disk_finke_2016, ebl_files_dict, EBL
from .utils import (
    make_comparison_plot,
    extract_columns_sample_file,
    check_deviation,
)
import matplotlib.pyplot as plt


agnpy_dir = Path(__file__).parent.parent
# where to read sampled files
data_dir = agnpy_dir / "data"
# where to save figures
figures_dir = agnpy_dir.parent / "crosschecks/figures/absorption"
figures_dir.mkdir(parents=True, exist_ok=True)

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
# blob parameters (use 3C 454.3 redshift z = 0.859)
pwl_blob_test = Blob(
    1e16 * u.cm, 0.859, 10, 10, 1 * u.G, spectrum_norm_test, pwl_dict_test,
)


class TestAbsorption:
    """class grouping all tests related to the Absorption class"""

    @pytest.mark.parametrize("r", ["1e-1", "1e0"])
    def test_absorption_disk_reference_tau(self, r):
        """test agnpy gamma-gamma optical depth for Disk against the one in 
        Figure 14 of Finke 2016"""
        # reference tau
        E_ref, tau_ref = extract_columns_sample_file(
            f"{data_dir}/reference_taus/finke_2016/figure_14_left/tau_SSdisk_r_{r}_R_Ly_alpha.txt",
            "GeV",
        )
        nu_ref = E_ref.to("Hz", equivalencies=u.spectral())
        # target
        M_BH = 1.2 * 1e9 * M_sun.cgs
        L_disk = 2e46 * u.Unit("erg s-1")
        eta = 1 / 12
        R_in = 6
        R_out = 200
        disk = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)
        R_Ly_alpha = 1.1e16 * u.cm
        _r = float(r) * R_Ly_alpha
        # recompute the tau, use the full energy range of figure 14
        abs_disk = Absorption(pwl_blob_test, disk, _r)
        tau_agnpy = abs_disk.tau(nu_ref)
        # comparison plot
        make_comparison_plot(
            nu_ref,
            tau_agnpy,
            tau_ref,
            "agnpy",
            "Figure 14, Finke (2016)",
            f"Absorption Shakura Sunyaev Disk, r = {r} R(Ly alpha)",
            f"{figures_dir}/tau_disk_comparison_r_{r}_R_Ly_alpha_figure_14_finke_2016.png",
            "tau",
            y_range=[1e-5, 1e5],
        )
        assert True

    @pytest.mark.parametrize("r", ["1e-1", "1e0", "1e0.5", "1e1"])
    def test_absorption_blr_reference_tau(self, r):
        """test agnpy gamma-gamma optical depth for a Lyman alpha BLR against 
        the one in Figure 14 of Finke 2016"""
        # reference tau
        E_ref, tau_ref = extract_columns_sample_file(
            f"{data_dir}/reference_taus/finke_2016/figure_14_left/tau_BLR_Ly_alpha_r_{r}_R_Ly_alpha.txt",
            "GeV",
        )
        nu_ref = E_ref.to("Hz", equivalencies=u.spectral())
        # target
        L_disk = 2e46 * u.Unit("erg s-1")
        xi_line = 0.024
        R_line = 1e17 * u.cm
        blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
        R_Ly_alpha = 1.1e16 * u.cm
        _r = float(r) * R_Ly_alpha if r != "1e0.5" else np.sqrt(10) * R_Ly_alpha
        # recompute the tau, use the full energy range of figure 14
        ec_blr = Absorption(pwl_blob_test, blr, _r)
        tau_agnpy = ec_blr.tau(nu_ref)
        # comparison plot
        make_comparison_plot(
            nu_ref,
            tau_agnpy,
            tau_ref,
            "agnpy",
            "Figure 14, Finke (2016)",
            f"Absorption on Spherical Shell BLR, r = {r} R(Ly alpha)",
            f"{figures_dir}/tau_BLR_Ly_alpha_comprison_r_{r}_R_Ly_alpha_figure_14_finke_2016.png",
            "tau",
            y_range=[1e-5, 1e3],
        )
        assert True

    @pytest.mark.parametrize("r", ["1e-1", "1e0", "1e1", "1e2"])
    def test_absorption_dt_reference_tau(self, r):
        """test agnpy gamma-gamma optical depth for DT against the one in 
        Figure 14 of Finke 2016"""
        # reference tau
        E_ref, tau_ref = extract_columns_sample_file(
            f"{data_dir}/reference_taus/finke_2016/figure_14_left/tau_DT_r_{r}_R_Ly_alpha.txt",
            "GeV",
        )
        nu_ref = E_ref.to("Hz", equivalencies=u.spectral())
        # target
        L_disk = 2e46 * u.Unit("erg s-1")
        T_dt = 1e3 * u.K
        csi_dt = 0.1
        dt = RingDustTorus(L_disk, csi_dt, T_dt)
        R_Ly_alpha = 1.1e16 * u.cm
        _r = float(r) * R_Ly_alpha
        # recompute the tau, use the full energy range of figure 14
        ec_dt = Absorption(pwl_blob_test, dt, _r)
        tau_agnpy = ec_dt.tau(nu_ref)
        # comparison plot
        make_comparison_plot(
            nu_ref,
            tau_agnpy,
            2 * tau_ref,
            "agnpy",
            "Figure 14, Finke (2016)",
            f"Absorption on Dust Torus, r = {r} R(Ly alpha)",
            f"{figures_dir}/tau_DT_comprison_r_{r}_R_Ly_alpha_figure_14_finke_2016.png",
            "tau",
            y_range=[1e-5, 1e3],
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
            tau_ps_blr,
            tau_blr,
            "point source approximating the BLR",
            "spherical shell BLR",
            "Absorption on Spherical Shell BLR, "
            + r"$r = 10^{20}\,{\rm cm} \gg R_{\rm line}$",
            f"{figures_dir}/tau_blr_point_source_comparison.png",
            "tau",
        )
        # requires a 10% deviation from the two SED points
        assert check_deviation(nu, tau_blr, tau_ps_blr, 0.1)

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
            tau_ps_dt,
            tau_dt,
            "point source approximating the DT",
            "ring dust torus",
            "Absorption on Ring Dust Torus, "
            + r"$r = 10^{22}\,{\rm cm} \gg R_{\rm dt}$",
            f"{figures_dir}/tau_dt_point_source_comparison.png",
            "tau",
        )
        # requires a 10% deviation from the two SED points
        assert check_deviation(nu, tau_dt, tau_ps_dt, 0.1)


class TestEBL:
    """class grouping all tests related to the EBL class"""

    @pytest.mark.parametrize("model", ["franceschini", "finke", "dominguez"])
    @pytest.mark.parametrize("z", [0.5, 1.5])
    def test_correct_interpolation(self, model, z):
        # define the ebl model, evaluate it at the reference energies
        ebl = EBL(model)
        nu_ref = ebl.energy_ref.to("Hz", equivalencies=u.spectral())
        absorption = ebl.absorption(z, nu_ref)
        # find in the reference values the spectra for this redshift
        z_idx = np.abs(z - ebl.z_ref).argmin()
        absorption_ref = ebl.values_ref[z_idx]
        make_comparison_plot(
            nu_ref,
            absorption,
            absorption_ref,
            "agnpy interpolation",
            "data",
            f"EBL absorption, {model} model, z = {z}",
            f"{figures_dir}/ebl_abs_interp_comparison_{model}_z_{z}.png",
            "abs.",
        )
        # requires a 1% deviation from the two SED points
        assert check_deviation(nu_ref, absorption_ref, absorption, 0.01)
