# test the compton module
from pathlib import Path
import pytest
import numpy as np
import astropy.units as u
from astropy.constants import M_sun, m_e
from astropy.coordinates import Distance
from agnpy.spectra import PowerLaw, BrokenPowerLaw
from agnpy.emission_regions import Blob
from agnpy.targets import (
    PointSourceBehindJet,
    SSDisk,
    SphericalShellBLR,
    RingDustTorus,
    CMB,
)
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.utils.math import trapz_loglog
from .utils import (
    make_comparison_plot,
    extract_columns_sample_file,
    check_deviation,
    clean_and_make_dir,
)

agnpy_dir = Path(__file__).parent.parent.parent  # go to the agnpy root
# where to read sampled files
data_dir = agnpy_dir / "agnpy/data"
# where to save figures, clean-up before making the new
figures_dir_ssc = clean_and_make_dir(agnpy_dir, "crosschecks/figures/compton/ssc")
figures_dir_ec_disk = clean_and_make_dir(
    agnpy_dir, "crosschecks/figures/compton/ec_disk"
)
figures_dir_ec_blr = clean_and_make_dir(agnpy_dir, "crosschecks/figures/compton/ec_blr")
figures_dir_ec_dt = clean_and_make_dir(agnpy_dir, "crosschecks/figures/compton/ec_dt")
figures_dir_ec_cmb = clean_and_make_dir(agnpy_dir, "crosschecks/figures/compton/ec_cmb")

# blob reproducing Figure 7.4 of Dermer Menon 2009
W_e = 1e48 * u.Unit("erg")
R_b = 1e16 * u.cm
V_b = 4 / 3 * np.pi * R_b ** 3

PWL = PowerLaw.from_total_energy(W_e, V_b, m_e, p=2.8, gamma_min=1e2, gamma_max=1e5)
PWL_BLOB = Blob(
    R_b=R_b, z=Distance(1e27, unit=u.cm).z, delta_D=10, Gamma=10, B=1 * u.G, n_e=PWL
)

# blob reproducing the EC scenarios in Finke 2016
W_e = 6e42 * u.Unit("erg")
BPWL = BrokenPowerLaw.from_total_energy(
    W_e, V_b, m_e, p1=2.0, p2=3.5, gamma_b=1e4, gamma_min=20, gamma_max=5e7
)
BPWL_BLOB = Blob(R_b=R_b, z=1, delta_D=40, Gamma=40, B=0.56 * u.G, n_e=BPWL)
BPWL_BLOB.set_gamma_e(gamma_size=400)

# targets used for the EC scenario in Finke 2016
L_disk = 2e46 * u.Unit("erg s-1")
DISK = SSDisk(
    M_BH=1.2 * 1e9 * M_sun.cgs,
    L_disk=L_disk,
    eta=1 / 12,
    R_in=6,
    R_out=200,
    R_g_units=True,
)
BLR = SphericalShellBLR(L_disk, xi_line=0.024, line="Lyalpha", R_line=1e17 * u.cm)
DT = RingDustTorus(L_disk, xi_dt=0.1, T_dt=1e3 * u.K)


class TestSynchrotronSelfCompton:
    """Class grouping all tests related to the Synchrotron Slef Compton class."""

    @pytest.mark.parametrize("gamma_max, nu_range_max", [("1e5", 1e25), ("1e7", 1e27)])
    def test_ssc_reference_sed(self, gamma_max, nu_range_max):
        """Test agnpy SSC SED against the ones in Figure 7.4 of Dermer Menon
        2009."""

        # reference SED
        nu_ref, sed_ref = extract_columns_sample_file(
            f"{data_dir}/reference_seds/dermer_menon_2009/figure_7_4/ssc_gamma_max_{gamma_max}.txt",
            "Hz",
            "erg cm-2 s-1",
        )

        # agnpy
        PWL_BLOB.n_e.gamma_max = float(gamma_max)
        PWL_BLOB.set_gamma_e(gamma_size=200, gamma_max=float(gamma_max))
        ssc = SynchrotronSelfCompton(PWL_BLOB)
        sed_agnpy = ssc.sed_flux(nu_ref)

        # sed comparison plot
        nu_range = [1e14, nu_range_max] * u.Hz
        make_comparison_plot(
            nu_ref,
            sed_agnpy,
            sed_ref,
            "agnpy",
            "Figure 7.4, Dermer and Menon (2009)",
            "Synchrotron Self Compton, " + r"$\gamma_{max} = $" + gamma_max,
            f"{figures_dir_ssc}/ssc_comparison_gamma_max_{gamma_max}_figure_7_4_dermer_menon_2009.png",
            "sed",
            y_range=[1e-13, 1e-9],
            comparison_range=nu_range.to_value("Hz"),
        )
        # requires that the SED points deviate less than 20% from the figure
        assert check_deviation(nu_ref, sed_agnpy, sed_ref, 0.2, nu_range)

    def test_ssc_integration_methods(self):
        """Test SSC SED for different integration methods against each other
        """
        nu = np.logspace(15, 28) * u.Hz
        ssc_trapz = SynchrotronSelfCompton(PWL_BLOB, integrator=np.trapz)
        ssc_trapz_loglog = SynchrotronSelfCompton(PWL_BLOB, integrator=trapz_loglog)
        sed_ssc_trapz = ssc_trapz.sed_flux(nu)
        sed_ssc_trapz_loglog = ssc_trapz_loglog.sed_flux(nu)

        # compare in a restricted energy range
        nu_range = [1e15, 1e27] * u.Hz
        make_comparison_plot(
            nu,
            sed_ssc_trapz_loglog,
            sed_ssc_trapz,
            "trapezoidal log-log integration",
            "trapezoidal integration",
            "Synchrotron Self Compton",
            f"{figures_dir_ssc}/ssc_comparison_integration_methods.png",
            "sed",
            comparison_range=nu_range.to_value("Hz"),
        )

        # requires that the SED points deviate less than 15%
        assert check_deviation(nu, sed_ssc_trapz_loglog, sed_ssc_trapz, 0.15, nu_range)


class TestExternalCompton:
    """class grouping all tests related to the External Compton class"""

    # tests for EC on SSDisk
    @pytest.mark.parametrize("r", ["1e17", "1e18"])
    def test_ec_disk_reference_sed(self, r):
        """test agnpy SED for EC on Disk against the one in Figure 8 of Finke 2016"""
        # reference SED
        nu_ref, sed_ref = extract_columns_sample_file(
            f"{data_dir}/reference_seds/finke_2016/figure_8/ec_disk_r_{r}.txt",
            "Hz",
            "erg cm-2 s-1",
        )
        # recompute the SED at the same ordinates where the figure was sampled
        ec_disk = ExternalCompton(bpwl_blob_test, disk_test, float(r) * u.cm)
        sed_agnpy = ec_disk.sed_flux(nu_ref)
        # check in a restricted energy range
        nu_range = [1e18, 1e28] * u.Hz
        make_comparison_plot(
            nu_ref,
            sed_agnpy,
            sed_ref,
            "agnpy",
            "Figure 8, Finke (2016)",
            f"External Compton on Shakura Sunyaev Disk, r = {r} cm",
            f"{figures_dir}/ec/disk/comparison_r_{r}_cm_figure_8_finke_2016.png",
            "sed",
            comparison_range=nu_range.to_value("Hz"),
        )
        # requires that the SED points deviate less than 35% from the figure
        assert check_deviation(nu_ref, sed_agnpy, sed_ref, 0.35, nu_range)

    def test_ec_disk_integration_methods(self):
        """test EC on Disk SED for different integration methods against each other
        """
        nu = np.logspace(15, 28) * u.Hz
        r = 1e18 * u.cm
        ec_disk_trapz = ExternalCompton(
            bpwl_blob_test, disk_test, r, integrator=np.trapz
        )
        ec_disk_trapz_loglog = ExternalCompton(
            bpwl_blob_test, disk_test, r, integrator=trapz_loglog
        )
        sed_ec_disk_trapz = ec_disk_trapz.sed_flux(nu)
        sed_ec_disk_trapz_loglog = ec_disk_trapz_loglog.sed_flux(nu)
        make_comparison_plot(
            nu,
            sed_ec_disk_trapz_loglog,
            sed_ec_disk_trapz,
            "trapezoidal log-log integration",
            "trapezoidal integration",
            "External Compton on Shakura Sunyaev Disk",
            f"{figures_dir}/ec/disk/comparison_integration_methods.png",
            "sed",
        )
        # requires that the SED points deviate less than 20%
        assert check_deviation(nu, sed_ec_disk_trapz_loglog, sed_ec_disk_trapz, 0.2)

    # tests for EC on BLR
    @pytest.mark.parametrize("r", ["1e18", "1e19"])
    def test_ec_blr_reference_sed(self, r):
        """test agnpy SED for EC on BLR against the one in Figure 10 of Finke 2016"""
        # reference SED
        nu_ref, sed_ref = extract_columns_sample_file(
            f"{data_dir}/reference_seds/finke_2016/figure_10/ec_blr_r_{r}.txt",
            "Hz",
            "erg cm-2 s-1",
        )
        # recompute the SED at the same ordinates where the figure was sampled
        ec_blr = ExternalCompton(bpwl_blob_test, blr_test, float(r) * u.cm)
        sed_agnpy = ec_blr.sed_flux(nu_ref)
        # check in a restricted energy range
        nu_range = [1e18, 1e28] * u.Hz
        make_comparison_plot(
            nu_ref,
            sed_agnpy,
            sed_ref,
            "agnpy",
            "Figure 10, Finke (2016)",
            f"External Compton on Spherical Shell BLR, r = {r} cm",
            f"{figures_dir}/ec/blr/comparison_r_{r}_cm_figure_10_finke_2016.png",
            "sed",
            comparison_range=nu_range.to_value("Hz"),
        )
        # requires that the SED points deviate less than 30% from the figure
        assert check_deviation(nu_ref, sed_agnpy, sed_ref, 0.3, nu_range)

    def test_ec_blr_integration_methods(self):
        """test EC on BLR SED for different integration methods
        """
        nu = np.logspace(15, 28) * u.Hz
        r = 1e18 * u.cm
        ec_blr_trapz = ExternalCompton(bpwl_blob_test, blr_test, r, integrator=np.trapz)
        ec_blr_trapz_loglog = ExternalCompton(
            bpwl_blob_test, blr_test, r, integrator=trapz_loglog
        )
        sed_ec_blr_trapz = ec_blr_trapz.sed_flux(nu)
        sed_ec_blr_trapz_loglog = ec_blr_trapz_loglog.sed_flux(nu)
        # check in a restricted energy range
        make_comparison_plot(
            nu,
            sed_ec_blr_trapz_loglog,
            sed_ec_blr_trapz,
            "trapezoidal log-log integration",
            "trapezoidal integration",
            "External Compton on Spherical Shell Broad Line Region",
            f"{figures_dir}/ec/blr/comparison_integration_methods.png",
            "sed",
        )
        # requires that the SED points deviate less than 30%
        assert check_deviation(nu, sed_ec_blr_trapz_loglog, sed_ec_blr_trapz, 0.3)

    # tests for EC on DT
    @pytest.mark.parametrize("r", ["1e20", "1e21"])
    def test_ec_dt_reference_sed(self, r):
        """test agnpy SED for EC on DT against the one in Figure 11 of Finke 2016"""
        # reference SED
        nu_ref, sed_ref = extract_columns_sample_file(
            f"{data_dir}/reference_seds/finke_2016/figure_11/ec_dt_r_{r}.txt",
            "Hz",
            "erg cm-2 s-1",
        )
        # recompute the SED at the same ordinates where the figure was sampled
        ec_dt = ExternalCompton(bpwl_blob_test, dt_test, float(r) * u.cm)
        sed_agnpy = ec_dt.sed_flux(nu_ref)
        # check in a restricted energy range
        nu_range = [1e18, 1e28] * u.Hz
        make_comparison_plot(
            nu_ref,
            sed_agnpy,
            sed_ref,
            "agnpy",
            "Figure 11, Finke (2016)",
            f"External Compton on Ring Dust Torus, r = {r} cm",
            f"{figures_dir}/ec/dt/comparison_r_{r}_cm_figure_11_finke_2016.png",
            "sed",
            comparison_range=nu_range.to_value("Hz"),
        )
        # requires that the SED points deviate less than 30% from the figure
        assert check_deviation(nu_ref, sed_agnpy, sed_ref, 0.3, nu_range)

    def test_ec_dt_integration_methods(self):
        """test EC on DT SED for different integration methods
        """
        nu = np.logspace(15, 28) * u.Hz
        r = 1e20 * u.cm
        ec_dt_trapz = ExternalCompton(bpwl_blob_test, dt_test, r, integrator=np.trapz)
        ec_dt_trapz_loglog = ExternalCompton(
            bpwl_blob_test, dt_test, r, integrator=trapz_loglog
        )
        sed_ec_dt_trapz = ec_dt_trapz.sed_flux(nu)
        sed_ec_dt_trapz_loglog = ec_dt_trapz_loglog.sed_flux(nu)
        make_comparison_plot(
            nu,
            sed_ec_dt_trapz_loglog,
            sed_ec_dt_trapz,
            "trapezoidal log-log integration",
            "trapezoidal integration",
            "External Compton on Ring Dust Torus",
            f"{figures_dir}/ec/dt/comparison_integration_methods.png",
            "sed",
        )
        # requires that the SED points deviate less than 20%
        assert check_deviation(nu, sed_ec_dt_trapz_loglog, sed_ec_dt_trapz, 0.2)

    # tests against point-like sources approximating the targets
    def test_ec_blr_vs_point_source(self):
        """check if in the limit of large distances the EC on the BLR tends to
        the one of a point-like source approximating it"""
        r = 1e22 * u.cm
        # point like source approximating the blr
        ps_blr = PointSourceBehindJet(
            blr_test.xi_line * blr_test.L_disk, blr_test.epsilon_line
        )
        # external Compton
        ec_blr = ExternalCompton(bpwl_blob_test, blr_test, r)
        ec_ps_blr = ExternalCompton(bpwl_blob_test, ps_blr, r)
        # seds
        nu = np.logspace(15, 30) * u.Hz
        sed_ec_blr = ec_blr.sed_flux(nu)
        sed_ec_ps_blr = ec_ps_blr.sed_flux(nu)
        # sed comparison plot
        make_comparison_plot(
            nu,
            sed_ec_ps_blr,
            sed_ec_blr,
            "point source approximating the BLR",
            "spherical shell BLR",
            "External Compton on Spherical Shell BLR, "
            + r"$r = 10^{22}\,{\rm cm} \gg R_{\rm line}$",
            f"{figures_dir}/ec/blr/comparison_point_source.png",
            "sed",
        )
        # requires a 20% deviation from the two SED points
        assert check_deviation(nu, sed_ec_ps_blr, sed_ec_blr, 0.2)

    def test_ec_dt_vs_point_source(self):
        """check if in the limit of large distances the EC on the DT tends to
        the one of a point-like source approximating it"""
        r = 1e22 * u.cm
        # point like source approximating the dt
        ps_dt = PointSourceBehindJet(dt_test.xi_dt * dt_test.L_disk, dt_test.epsilon_dt)
        # external Compton
        ec_dt = ExternalCompton(bpwl_blob_test, dt_test, r)
        ec_ps_dt = ExternalCompton(bpwl_blob_test, ps_dt, r)
        # seds
        nu = np.logspace(15, 28) * u.Hz
        sed_ec_dt = ec_dt.sed_flux(nu)
        sed_ec_ps_dt = ec_ps_dt.sed_flux(nu)
        make_comparison_plot(
            nu,
            sed_ec_ps_dt,
            sed_ec_dt,
            "point source approximating the DT",
            "ring Dust Torus",
            "External Compton on Ring Dust Torus, "
            + r"$r = 10^{22}\,{\rm cm} \gg R_{\rm dt}$",
            f"{figures_dir}/ec/dt/comparison_point_source.png",
            "sed",
        )
        # requires a 20% deviation from the two SED points
        assert check_deviation(nu, sed_ec_dt, sed_ec_ps_dt, 0.2)

    def test_ec_cmb_vs_jetset(self):
        """check the SED for EC on CMB against jetset"""
        # reference SED
        file_ref = f"{data_dir}/reference_seds/jetset/data/ec_cmb_bpwl_jetset_1.1.2.txt"
        nu_ref, sed_ref = extract_columns_sample_file(file_ref, "Hz", "erg cm-2 s-1")
        # recompute the SED at the same ordinates where the figure was sampled
        cmb = CMB(z=bpwl_blob_test.z)
        ec_cmb = ExternalCompton(bpwl_blob_test, cmb)
        sed_agnpy = ec_cmb.sed_flux(nu_ref)
        # sed comparison plot, we will check between 10^(11) and 10^(19) Hz
        nu_range = [1e16, 5e27] * u.Hz
        make_comparison_plot(
            nu_ref,
            sed_agnpy,
            sed_ref,
            "agnpy",
            "jetset 1.1.2",
            "EC on CMB, z = 1",
            f"{figures_dir}/ec/cmb/ec_cmb_bpwl_comparison_jetset_1.1.2.png",
            "sed",
            comparison_range=nu_range.to_value("Hz"),
        )
        # requires that the SED points deviate less than 35% from the figure
        assert check_deviation(nu_ref, sed_agnpy, sed_ref, 0.35, nu_range)
