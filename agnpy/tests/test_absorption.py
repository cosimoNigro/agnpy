# test on absorption module
from pathlib import Path
import pytest
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c, M_sun, sigma_T, k_B
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
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

    @pytest.mark.parametrize("r", ["1e-1", "1e0", "1e1", "1e2"])
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

    @pytest.mark.parametrize(
        "r,nu_min", [("1e-1", 5e24), ("1e0.5", 2e26), ("1e1", 2.3e27)]
    )
    def test_absorption_blr_reference_tau(self, r, nu_min):
        """test agnpy gamma-gamma optical depth for a Lyman alpha BLR against
        the one in Figure 14 of Finke 2016"""
        # reference tau
        E_ref, tau_ref = extract_columns_sample_file(
            f"{data_dir}/reference_taus/finke_2016/figure_14_left/tau_BLR_Ly_alpha_r_{r}_R_Ly_alpha.txt",
            "GeV",
        )
        # Finke's frequencies are not corrected for the redshift
        z = 0.859
        nu_ref = E_ref.to("Hz", equivalencies=u.spectral()) / (1 + z)
        # target
        L_disk = 2 * 1e46 * u.Unit("erg s-1")
        xi_line = 0.024
        R_Ly_alpha = 1.1 * 1e17 * u.cm
        blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_Ly_alpha)
        # parse the string providing the distance in units of R_Ly_alpha
        # numbers as 1e1.5 cannot be directly converted to float
        num = r.split("e")
        Ly_alpha_units = (float(num[0]) * 10) ** (float(num[-1]))
        _r = Ly_alpha_units * R_Ly_alpha
        # recompute the tau, use the full energy range of figure 14
        abs_blr = Absorption(blr, _r, z)
        tau_agnpy = abs_blr.tau(nu_ref)
        # check in a restricted energy range
        nu_range = [nu_min, 3e28] * u.Hz
        make_comparison_plot(
            nu_ref,
            tau_agnpy,
            tau_ref,
            "agnpy",
            "Figure 14, Finke (2016)",
            f"Absorption on Spherical Shell BLR, r = {Ly_alpha_units:.2e} R(Ly alpha)",
            f"{figures_dir}/blr/tau_blr_Ly_alpha_comprison_r_{r}_R_Ly_alpha_figure_14_finke_2016.png",
            "tau",
            comparison_range=nu_range.to_value("Hz"),
            y_range=[1e-6, 1e3],
        )
        # requires that the SED points deviate less than 35 % from those of the reference figure
        assert check_deviation(nu_ref, tau_agnpy, tau_ref, 0.35, nu_range)

    @pytest.mark.parametrize(
        "r,nu_min", [("1e-1", 3.3e26), ("1e0", 3.3e26), ("1e1", 3.3e26), ("1e2", 8e26)]
    )
    def test_absorption_dt_reference_tau(self, r, nu_min):
        """test agnpy gamma-gamma optical depth for DT against the one in
        Figure 14 of Finke 2016"""
        # reference tau
        E_ref, tau_ref = extract_columns_sample_file(
            f"{data_dir}/reference_taus/finke_2016/figure_14_left/tau_DT_r_{r}_R_Ly_alpha.txt",
            "GeV",
        )
        nu_ref = E_ref.to("Hz", equivalencies=u.spectral())
        # target
        L_disk = 2 * 1e46 * u.Unit("erg s-1")
        T_dt = 1e3 * u.K
        csi_dt = 0.1
        dt = RingDustTorus(L_disk, csi_dt, T_dt)
        R_Ly_alpha = 1.1 * 1e17 * u.cm
        # parse the string providing the distance in units of R_Ly_alpha
        # numbers as 1e1.5 cannot be directly converted to float
        num = r.split("e")
        Ly_alpha_units = (float(num[0]) * 10) ** (float(num[-1]))
        _r = Ly_alpha_units * R_Ly_alpha
        # recompute the tau, use the full energy range of figure 14
        z = 0.859
        abs_dt = Absorption(dt, _r, z)
        tau_agnpy = abs_dt.tau(nu_ref)
        # check in a restricted energy range
        nu_range = [nu_min, 3e28] * u.Hz
        make_comparison_plot(
            nu_ref,
            tau_agnpy,
            2 * tau_ref,
            "agnpy",
            "Figure 14, Finke (2016)",
            f"Absorption on Dust Torus, r = {Ly_alpha_units:.2e} R(Ly alpha)",
            f"{figures_dir}/dt/tau_dt_comprison_r_{r}_R_Ly_alpha_figure_14_finke_2016.png",
            "tau",
            comparison_range=nu_range.to_value("Hz"),
            y_range=[1e-6, 1e3],
        )
        # requires that the SED points deviate less than 20 % from those of the reference figure
        assert check_deviation(nu_ref, tau_agnpy, 2 * tau_ref, 0.20, nu_range)

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
            + r"$r = 10^{20}\,{\rm cm} \gg R({\rm Ly\alpha}),\,\theta_s=10^{\circ}$",
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
            + r"$r = 10^{22}\,{\rm cm} \gg R_{\rm dt},\,\theta_s=10^{\circ}$",
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
        the case of a perpendicularly moving photon starting at r~0.5 * R_re
        """
        r = 1.0e17 * u.cm  # distance at which the photon starts
        mu_s = 0.0  # angle of propagation

        L_disk = 2e46 * u.Unit("erg s-1")
        xi_DT = 0.1
        temp = 1000 * u.K
        R_DT = 2 * r  # radius of DT: assume the same as the distance r
        dt = RingDustTorus(L_disk, xi_DT, temp, R_dt=R_DT)

        nu_ref = np.logspace(26, 32, 120) * u.Hz

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
        # roughtly the characteristic distance of ~R_DT
        tau_my = (sigma_pp(np.sqrt(beta2)) * nph * R_DT * (1 - cospsi)).to("")

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

    @pytest.mark.parametrize("r_to_R", ["0.11", "10."])
    def test_abs_blr_mu_s_vs_on_axis(self, r_to_R):
        """check if the codes computing absorption on BLR for mu_s = 1 and !=1 cases are consistent """
        # broad line region
        L_disk = 2e46 * u.Unit("erg s-1")
        xi_line = 0.024
        R_line = 1e17 * u.cm
        blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
        r = r_to_R * R_line
        z = 0.859

        abs_blr = Absorption(blr, r, z, mu_s=0.9999)
        abs_blr_on_axis = Absorption(blr, r, z)
        abs_blr.set_l(50)
        abs_blr_on_axis.set_l(50)

        # taus
        E = np.logspace(0, 6) * u.GeV
        nu = E.to("Hz", equivalencies=u.spectral())
        tau_blr = abs_blr.tau(nu)
        tau_blr_on_axis = abs_blr_on_axis.tau(nu)
        # sed comparison plot
        make_comparison_plot(
            nu,
            tau_blr_on_axis,
            tau_blr,
            "on-axis calculations",
            "general",
            "Absorption on Spherical Shell BLR, " + r"$r/R_{\rm line}=$" + f"{r_to_R}",
            f"{figures_dir}/blr/tau_blr_on_axis_vs_general_r_{r_to_R}_R_line_comparison.png",
            "tau",
        )

        # only check in there range with measurable absorption
        xmin = min(nu[tau_blr_on_axis > 1.0e-4])
        xmax = max(nu[tau_blr_on_axis > 1.0e-4])
        xrange = (xmin, xmax)
        print(xrange)

        # close to the threshold there are some differences up to ~25%
        # which are probably due to numerical uncertainties in the integrals
        assert check_deviation(nu, tau_blr, tau_blr_on_axis, 0.25, x_range=xrange)

    @pytest.mark.parametrize("r_R_line", [1, 1.0e5 ** (-10.0 / 49)])
    def test_tau_blr_Rline(self, r_R_line):
        """
        Checks if absorption on BLR works also fine
        if one of the integration points falls on R_line
        """
        L_disk = 2 * 1e46 * u.Unit("erg s-1")
        xi_line = 0.024
        R_line = 1.1 * 1e17 * u.cm
        blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)

        abs_blr = Absorption(blr, r=r_R_line * R_line)

        # reference (without problem)
        abs_blr0 = Absorption(blr, r=r_R_line * R_line * 1.001)

        nu = np.logspace(22, 28) * u.Hz
        tau_blr = abs_blr.tau(nu)
        tau_blr0 = abs_blr0.tau(nu)

        # the current integrals are not very precise, and this includes
        # tricky part to integrate, so allowing for rather large error margin
        assert np.allclose(tau_blr, tau_blr0, atol=0.01, rtol=0.04)

    def find_r_for_x_cross(mu_s, ipoint, npoints):
        """
        Finding which r should be used to fall with one of the points on top of BLR sphere.
        """
        # u = alpha * r
        alpha = 1.0e-5 * 1.0e10 ** (ipoint / (npoints - 1.0))
        return 1 / np.sqrt(alpha ** 2 + 2 * alpha * mu_s + 1)

    @pytest.mark.parametrize("r_R_line", [1, find_r_for_x_cross(0.99, 60, 100)])
    def test_tau_blr_mus_Rline(self, r_R_line):
        """
        Checks if absorption on BLR works also fine
        if one of the integration points falls on R_line
        """
        L_disk = 2 * 1e46 * u.Unit("erg s-1")
        xi_line = 0.024
        R_line = 1.1 * 1e17 * u.cm
        blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)

        abs_blr = Absorption(blr, r=r_R_line * R_line, mu_s=0.99)

        # reference (without problem)
        abs_blr0 = Absorption(blr, r=r_R_line * R_line * 1.001, mu_s=0.99)

        nu = np.logspace(22, 28) * u.Hz
        tau_blr = abs_blr.tau(nu)
        tau_blr0 = abs_blr0.tau(nu)

        # the current integrals are not very precise, and this includes
        # tricky part to integrate, so allowing for rather large error margin
        assert np.allclose(tau_blr, tau_blr0, atol=0.01, rtol=1.04)

    def test_absorption_failing_without_r(self):
        """Tests that you cannot run absorption on "fixed" targets
        (like BLR, DT or the disk)  without specifying distance"""

        L_disk = 2 * 1e46 * u.Unit("erg s-1")
        xi_line = 0.024
        R_line = 1.1 * 1e17 * u.cm
        blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)

        # no distance specified, should fail
        with pytest.raises(ValueError):
            abs_blr = Absorption(blr)

    def test_tau_on_synchrotron_compare_with_delta_approximation(self):
        """Compare the output of the calculation of absorption of gamma rays
        in synchrotron radiation and compare it with simplified delta approximation.
        """

        # create a test blob
        r_b = 1.0e16 * u.cm
        z = 2.01
        delta_D = 10.1
        Gamma = 10.05
        B0 = 1.1 * u.G
        gmin = 10
        gbreak = 1.0e3
        gmax = 1.0e5
        spectrum_norm = 0.1 * u.Unit("erg cm-3")
        parameters = {
            "p1": 1.5,
            "p2": 2.5,
            "gamma_b": gbreak,
            "gamma_min": gmin,
            "gamma_max": gmax,
        }
        spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
        blob = Blob(r_b, z, delta_D, Gamma, B0, spectrum_norm, spectrum_dict)

        nu_tau = np.logspace(22, 34, 100) * u.Hz  # for absorption calculations
        e_tau = nu_tau.to("eV", equivalencies=u.spectral())

        # full calculations
        absorb = Absorption(blob)
        tau = absorb.tau(nu_tau)

        # now simplified calculations using Eq. 37 of Finke 2008
        mec2 = (m_e * c ** 2).to("eV")
        eps1 = e_tau / mec2
        eps1p = eps1 * (1 + z) / blob.delta_D
        eps_bar = 2 * blob.delta_D ** 2 / (1 + z) ** 2 / eps1
        nu_bar = (eps_bar * mec2).to("Hz", equivalencies=u.spectral())
        synch = Synchrotron(blob, ssa=True)
        synch_sed_ebar = synch.sed_flux(nu_bar)

        tau_delta = (
            synch_sed_ebar
            * sigma_T
            * Distance(z=z) ** 2
            * eps1p
            / (2 * m_e * c ** 3 * blob.R_b * blob.delta_D ** 4)
        )
        tau_delta = tau_delta.to("")

        # the delta approximation does not work well with sharp cut-offs of synchrotron SED.
        # We first find the peak of the synchr SED
        maxidx = np.where(synch_sed_ebar == np.amax(synch_sed_ebar))[0][0]
        # and only take energies below the peak, (note that energies are ordered
        # in reversed order), and  take only those that are at least 1% of the peak
        idxs = synch_sed_ebar[maxidx:] > 1.0e-2 * synch_sed_ebar[maxidx]

        # if there are not many points something went wrong with the test
        assert sum(idxs) > 10

        # the agreement in the middle range is pretty good, but at the edges
        # of energy range it spoils very fast, so only rough agreement is checked
        # (0.6 in the log space)
        assert np.allclose(
            np.log(tau)[maxidx:][idxs], np.log(tau_delta)[maxidx:][idxs], atol=0.6
        )

    def test_absorption_pointlike_and_homogeneous(self):
        """Simple test for checking the attenuation factors in both considered cases."""

        blob = Blob()
        absorb = Absorption(blob)

        nu_tau = np.logspace(22, 34, 100) * u.Hz  # for absorption calculations

        tau = absorb.tau(nu_tau)
        abs_pointlike = absorb.absorption(nu_tau)
        abs_homogeneous = absorb.absorption_homogeneous(nu_tau)
        # select good points (to avoid division by zero)
        idxs = tau > 1.0e-3
        # if this fails something is wrong with the test itself (e.g. default parameters of the blob)
        assert sum(idxs) > 10

        # the formulas reproduce what should be in the function, so the agreement should be down to numerical accuracy
        assert np.allclose(np.exp(-tau), abs_pointlike, atol=1.0e-5)
        assert np.allclose((1 - np.exp(-tau)) / tau, abs_homogeneous, atol=1.0e-5)


class TestEBL:
    """class grouping all tests related to the EBL class"""

    @pytest.mark.parametrize("model", ["franceschini", "finke", "dominguez", "saldana-lopez"])
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
