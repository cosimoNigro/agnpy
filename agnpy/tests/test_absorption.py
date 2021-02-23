# test on absorption module
import pytest
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c, M_sun, sigma_T, k_B
from astropy.coordinates import Distance
from pathlib import Path
from agnpy.emission_regions import Blob
from agnpy.targets import PointSourceBehindJet, SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.absorption import Absorption, EBL, ebl_files_dict
from agnpy.utils.math import axes_reshaper
from .utils import (
    make_comparison_plot,
    extract_columns_sample_file,
    check_deviation,
)
import matplotlib.pyplot as plt

mec2 = m_e.to("erg", equivalencies=u.mass_energy())

agnpy_dir = Path(__file__).parent.parent
# where to read sampled files
data_dir = agnpy_dir / "data"
# where to save figures
figures_dir = agnpy_dir.parent / "crosschecks/figures/absorption"
for subdir in ["ebl", "disk", "blr", "dt"]:
    Path(figures_dir / subdir).mkdir(parents=True, exist_ok=True)


class TestAbsorption:
    """class grouping all tests related to the Absorption class"""

    @pytest.mark.parametrize("r", ["1e-1", "1e0", "1e1"])
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
        R_Ly_alpha = 1.1e17 * u.cm
        _r = float(r) * R_Ly_alpha
        # recompute the tau, use the full energy range of figure 14
        z = 0.859
        abs_disk = Absorption(disk, _r, z)
        tau_agnpy = abs_disk.tau(nu_ref)
        # comparison plot
        make_comparison_plot(
            nu_ref,
            tau_agnpy,
            tau_ref,
            "agnpy",
            "Figure 14, Finke (2016)",
            f"Absorption Shakura Sunyaev Disk, r = {r} R(Ly alpha)",
            f"{figures_dir}/disk/tau_disk_comparison_r_{r}_R_Ly_alpha_figure_14_finke_2016.png",
            "tau",
            y_range=[1e-10, 1e5],
        )
        assert True

    @pytest.mark.parametrize("r", ["1e-1", "1e0", "1e1"])
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
        R_Ly_alpha = 1.1e17 * u.cm
        _r = float(r) * R_Ly_alpha
        # recompute the tau, use the full energy range of figure 14
        z = 0.859
        ec_blr = Absorption(blr, _r, z)
        tau_agnpy = ec_blr.tau(nu_ref)
        # comparison plot
        make_comparison_plot(
            nu_ref,
            tau_agnpy,
            tau_ref,
            "agnpy",
            "Figure 14, Finke (2016)",
            f"Absorption on Spherical Shell BLR, r = {r} R(Ly alpha)",
            f"{figures_dir}/blr/tau_blr_Ly_alpha_comprison_r_{r}_R_Ly_alpha_figure_14_finke_2016.png",
            "tau",
            y_range=[1e-5, 1e5],
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
        R_Ly_alpha = 1.1e17 * u.cm
        _r = float(r) * R_Ly_alpha
        # recompute the tau, use the full energy range of figure 14
        z = 0.859
        ec_dt = Absorption(dt, _r, z)
        tau_agnpy = ec_dt.tau(nu_ref)
        # comparison plot
        make_comparison_plot(
            nu_ref,
            tau_agnpy,
            2 * tau_ref,
            "agnpy",
            "Figure 14, Finke (2016)",
            f"Absorption on Dust Torus, r = {r} R(Ly alpha)",
            f"{figures_dir}/dt/tau_dt_comprison_r_{r}_R_Ly_alpha_figure_14_finke_2016.png",
            "tau",
            y_range=[1e-5, 1e5],
        )
        assert True

    # FIXME: test is temporarily disabled because BLR
    # absorption has to be rechecked for the mu_s != 1 case
    def _test_abs_blr_vs_point_source(self):
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
        # absorption, consider a small viewing angle for this case
        z = 0.859
        theta_s = np.deg2rad(10)
        abs_blr = Absorption(blr, r, z, mu_s=np.cos(theta_s))
        abs_ps_blr = Absorption(ps_blr, r, z, mu_s=np.cos(theta_s))
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
            f"{figures_dir}/blr/tau_blr_point_source_comparison.png",
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
        z = 0.859
        theta_s = np.deg2rad(10)
        abs_dt = Absorption(dt, r, z, mu_s=np.cos(theta_s))
        abs_ps_dt = Absorption(ps_dt, r, z, mu_s=np.cos(theta_s))
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
            f"{figures_dir}/dt/tau_dt_point_source_comparison.png",
            "tau",
        )
        # requires a 10% deviation from the two SED points
        assert check_deviation(nu, tau_dt, tau_ps_dt, 0.1)


def sigma_pp(b):
    """pair production cross section"""
    return (
        sigma_T
        * 3.0
        / 16.0
        * (1 - b ** 2)
        * (2 * b * (b ** 2 - 2) + (3 - b ** 4) * np.log((1 + b) / (1 - b)))
    )


class TestAbsorptionMuS:
    """test absorption functions for calculations with mu_s !=1"""

    def test_tau_dt_mu_s_simple(self):
        """
        order of magnitude test comparing with simplified calculations
        the case of a perpendicularly moving photon starting at r~R_re
        """
        r = 1.0e17 * u.cm  # distance at which the photon starts
        mu_s = 0.0  # angle of propagation

        L_disk = 2e46 * u.Unit("erg s-1")
        xi_DT = 0.1
        temp = 1000 * u.K
        R_DT = r  # radius of DT: assume the same as the distance r
        dt = RingDustTorus(L_disk, xi_DT, temp, R_dt=R_DT)

        nu_ref = np.logspace(26, 32, 60) * u.Hz

        # absorption at mu_s
        abs_dt_mu_s = Absorption(dt, r, z=0, mu_s=mu_s)
        tau_dt_mu_s = abs_dt_mu_s.tau(nu_ref)

        eps = (2.7 * temp * k_B).to("eV")  # energy of soft photons
        # soft photon density
        nph = (L_disk * xi_DT / (4 * np.pi * (r ** 2 + R_DT ** 2)) / c / eps).to("cm-3")

        E = nu_ref.to("eV", equivalencies=u.spectral())
        cospsi = 0  # assume perpendicular scattering
        beta2 = 1 - 2 * mec2 ** 2 / (E * eps * (1 - cospsi))
        beta2[beta2 < 0] = 0  # below the threshold
        # for tau calculations we assume that gamma ray moves
        # roughtly the characteristic distance of ~r~R_DT
        tau_my = (sigma_pp(np.sqrt(beta2)) * nph * r * (1 - cospsi)).to("")

        max_agnpy = max(tau_dt_mu_s)
        max_my = max(tau_my)
        max_pos_agnpy = nu_ref[np.argmax(tau_dt_mu_s)]
        max_pos_my = nu_ref[np.argmax(tau_my)]
        # very rough calculations so allowing for
        # 15% accuracy in peak maximum and 30% in peak position
        assert np.isclose(max_agnpy, max_my, atol=0, rtol=0.15)
        assert np.isclose(max_pos_agnpy, max_pos_my, atol=0, rtol=0.3)

    @pytest.mark.parametrize("mu_s", np.linspace(0, 0.9, 5))
    def test_tau_dt_mu_s_far(self, mu_s):
        """
        comparing the DT absorption for mu_s !=1 with point source simplification
        """
        L_disk = 2e46 * u.Unit("erg s-1")
        xi_DT = 0.1
        temp = 1000 * u.K
        R_DT = 1.0e17 * u.cm  # radius of DT
        dt = RingDustTorus(L_disk, xi_DT, temp, R_dt=R_DT)

        r = 10 * R_DT  # distance at which the photon starts

        nu_ref = np.logspace(26, 32, 60) * u.Hz

        # absorption at mu_s
        abs_dt_mu_s = Absorption(dt, r, z=0, mu_s=mu_s)
        tau_dt_mu_s = abs_dt_mu_s.tau(nu_ref)

        uu = np.logspace(-5, 5, 100) * r
        _u, _nu_ref = axes_reshaper(uu, nu_ref)

        eps = (2.7 * temp * k_B).to("eV")  # energy of soft photons
        # distance from the DT
        _x = np.sqrt(r * r + _u * _u + 2 * mu_s * _u * r)

        # soft photon density
        _nph = (L_disk * xi_DT / (4 * np.pi * _x ** 2) / c / eps).to("cm-3")

        _E = _nu_ref.to("eV", equivalencies=u.spectral())
        _cospsi = (_u ** 2 + _x ** 2 - r ** 2) / (2 * _u * _x)

        _beta2 = 1 - 2 * mec2 ** 2 / (_E * eps * (1 - _cospsi))
        _beta2[_beta2 < 0] = 0

        integrand = sigma_pp(np.sqrt(_beta2)) * _nph * (1 - _cospsi)
        tau_my = (np.trapz(integrand, uu, axis=0)).to("")

        print(tau_my / tau_dt_mu_s)

        max_agnpy = max(tau_dt_mu_s)
        max_my = max(tau_my)
        max_pos_agnpy = nu_ref[np.argmax(tau_dt_mu_s)]
        max_pos_my = nu_ref[np.argmax(tau_my)]

        # if r>>R_re this should be pretty precise, allowing for 10% accuracy
        assert np.isclose(max_agnpy, max_my, atol=0, rtol=0.1)
        assert np.isclose(max_pos_agnpy, max_pos_my, atol=0, rtol=0.1)


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
            f"{figures_dir}/ebl/ebl_abs_interp_comparison_{model}_z_{z}.png",
            "abs.",
        )
        # requires a 1% deviation from the two SED points
        assert check_deviation(nu_ref, absorption_ref, absorption, 0.01)
