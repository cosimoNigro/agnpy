# test on compton module
import numpy as np
import astropy.units as u
from astropy.constants import m_e, c, M_sun
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.targets import PointSourceBehindJet, SSDisk, SphericalShellBLR, RingDustTorus
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
import matplotlib.pyplot as plt
from pathlib import Path
import pytest
from .utils import make_sed_comparison_plot


mec2 = m_e.to("erg", equivalencies=u.mass_energy())
agnpy_dir = Path(__file__).parent.parent.parent
data_dir = f"{agnpy_dir}/data"

# variables with _test are global and meant to be used in all tests
pwl_spectrum_norm_test = 1e48 * u.Unit("erg")
pwl_dict_test = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e5,},
}
bpwl_spectrum_norm_test = 6e42 * u.Unit("erg")
bpwl_dict_test = {
    "type": "BrokenPowerLaw",
    "parameters": {
        "p1": 2.0,
        "p2": 3.5,
        "gamma_b": 1e4,
        "gamma_min": 20,
        "gamma_max": 5e7,
    },
}
# blob reproducing Figure 7.4 of Dermer Menon 2009
pwl_blob_test = Blob(
    1e16 * u.cm,
    Distance(1e27, unit=u.cm).z,
    10,
    10,
    1 * u.G,
    pwl_spectrum_norm_test,
    pwl_dict_test,
)
# blob reproducing the EC scenarios in Finke 2016
bpwl_blob_test = Blob(
    1e16 * u.cm, 1, 40, 40, 0.56 * u.G, bpwl_spectrum_norm_test, bpwl_dict_test,
)
bpwl_blob_test.set_gamma_size(500)


class TestSynchrotronSelfCompton:
    """class grouping all tests related to the Synchrotron Slef Compton class"""

    def test_ssc_reference_sed(self):
        """test agnpy SSC SED against the one in Figure 7.4 of Dermer Menon"""
        sampled_ssc_table = np.loadtxt(
            f"{data_dir}/sampled_seds/ssc_figure_7_4_dermer_menon_2009.txt",
            delimiter=",",
            comments="#",
        )
        sampled_ssc_nu = sampled_ssc_table[:, 0] * u.Hz
        sampled_ssc_sed = sampled_ssc_table[:, 1] * u.Unit("erg cm-2 s-1")
        # agnpy
        synch = Synchrotron(pwl_blob_test)
        ssc = SynchrotronSelfCompton(pwl_blob_test, synch)
        # recompute the SED at the same ordinates where the figure was sampled
        agnpy_ssc_sed = ssc.sed_flux(sampled_ssc_nu)
        # sed comparison plot
        make_sed_comparison_plot(
            sampled_ssc_nu,
            sampled_ssc_sed,
            agnpy_ssc_sed,
            "Figure 7.4, Dermer and Menon (2009)",
            "agnpy",
            "Synchrotron Self Compton",
            f"{data_dir}/crosscheck_figures/ssc_comparison_figure_7_4_dermer_menon_2009.png",
        )
        # requires that the SED points deviate less than 15% from the figure
        assert u.allclose(
            agnpy_ssc_sed, sampled_ssc_sed, atol=0 * u.Unit("erg cm-2 s-1"), rtol=0.15
        )


class TestExternalCompton:
    """class grouping all tests related to the Synchrotron Slef Compton class"""

    def test_ec_disk_reference_sed(self):
        """test agnpy SED for EC on Disk against the one in Figure 8 of Finke 2016"""
        # reference SED
        sampled_ec_disk_table = np.loadtxt(
            f"{data_dir}/sampled_seds/ec_disk_figure_8_finke_2016.txt",
            delimiter=",",
            comments="#",
        )
        sampled_ec_disk_nu = sampled_ec_disk_table[:, 0] * u.Hz
        sampled_ec_disk_sed = sampled_ec_disk_table[:, 1] * u.Unit("erg cm-2 s-1")
        # agnpy SED
        M_BH = 1.2 * 1e9 * M_sun.cgs
        L_disk = 2 * 1e46 * u.Unit("erg s-1")
        eta = 1 / 12
        R_in = 6
        R_out = 200
        disk = SSDisk(M_BH, L_disk, eta, R_in, R_out, R_g_units=True)
        # recompute the SED at the same ordinates where the figure was sampled
        ec_disk = ExternalCompton(bpwl_blob_test, disk, r=1e17 * u.cm)
        agnpy_ec_disk_sed = ec_disk.sed_flux(sampled_ec_disk_nu)
        # sed comparison plot
        make_sed_comparison_plot(
            sampled_ec_disk_nu,
            sampled_ec_disk_sed,
            agnpy_ec_disk_sed,
            "Figure 8, Finke (2016)",
            "agnpy",
            "External Compton on Shakura Sunyaev Disk",
            f"{data_dir}/crosscheck_figures/ec_disk_comparison_figure_8_finke_2016.png",
        )
        # requires that the SED points deviate less than 40% from the figure
        assert u.allclose(
            agnpy_ec_disk_sed,
            sampled_ec_disk_sed,
            atol=0 * u.Unit("erg cm-2 s-1"),
            rtol=0.4,
        )

    def test_ec_blr_reference_sed(self):
        """test agnpy SED for EC on BLR against the one in Figure 10 of Finke 2016"""
        # reference SED
        sampled_ec_blr_table = np.loadtxt(
            f"{data_dir}/sampled_seds/ec_blr_figure_10_finke_2016.txt",
            delimiter=",",
            comments="#",
        )
        sampled_ec_blr_nu = sampled_ec_blr_table[:, 0] * u.Hz
        sampled_ec_blr_sed = sampled_ec_blr_table[:, 1] * u.Unit("erg cm-2 s-1")
        # agnpy SED
        L_disk = 2 * 1e46 * u.Unit("erg s-1")
        xi_line = 0.024
        R_line = 1e17 * u.cm
        blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
        # recompute the SED at the same ordinates where the figure was sampled
        ec_blr = ExternalCompton(bpwl_blob_test, blr, r=1e18 * u.cm)
        agnpy_ec_blr_sed = ec_blr.sed_flux(sampled_ec_blr_nu)
        # sed comparison plot
        make_sed_comparison_plot(
            sampled_ec_blr_nu,
            sampled_ec_blr_sed,
            agnpy_ec_blr_sed,
            "Figure 10, Finke (2016)",
            "agnpy",
            "External Compton on Spherical Shell Broad Line Region",
            f"{data_dir}/crosscheck_figures/ec_blr_comparison_figure_10_finke_2016.png",
        )
        # requires that the SED points deviate less than 30% from the figure
        assert u.allclose(
            agnpy_ec_blr_sed,
            sampled_ec_blr_sed,
            atol=0 * u.Unit("erg cm-2 s-1"),
            rtol=0.3,
        )

    def test_ec_dt_reference_sed(self):
        """test agnpy SED for EC on DT against the one in Figure 11 of Finke 2016"""
        # reference SED
        sampled_ec_dt_table = np.loadtxt(
            f"{data_dir}/sampled_seds/ec_dt_figure_11_finke_2016.txt",
            delimiter=",",
            comments="#",
        )
        sampled_ec_dt_nu = sampled_ec_dt_table[:, 0] * u.Hz
        # multiply the reference SED for 2 as this is the missing factor
        # in the emissivity expression in Eq. 90 of Finke 2016
        sampled_ec_dt_sed = 2 * sampled_ec_dt_table[:, 1] * u.Unit("erg cm-2 s-1")
        # agnpy SED
        L_disk = 2 * 1e46 * u.Unit("erg s-1")
        T_dt = 1e3 * u.K
        csi_dt = 0.1
        dt = RingDustTorus(L_disk, csi_dt, T_dt)
        # recompute the SED at the same ordinates where the figure was sampled
        ec_dt = ExternalCompton(bpwl_blob_test, dt, r=1e20 * u.cm)
        agnpy_ec_dt_sed = ec_dt.sed_flux(sampled_ec_dt_nu)
        # sed comparison plot
        make_sed_comparison_plot(
            sampled_ec_dt_nu,
            sampled_ec_dt_sed,
            agnpy_ec_dt_sed,
            "Figure 11, Finke (2016)",
            "agnpy",
            "External Compton on Ring Dust Torus",
            f"{data_dir}/crosscheck_figures/ec_dt_comparison_figure_11_finke_2016.png",
        )
        # requires that the SED points deviate less than 30% from the figure
        assert u.allclose(
            agnpy_ec_dt_sed,
            sampled_ec_dt_sed,
            atol=0 * u.Unit("erg cm-2 s-1"),
            rtol=0.3,
        )

    def test_ec_blr_vs_point_source(self):
        """check if in the limit of large distances the EC on the BLR tends to
        the one of a point-like source approximating it"""
        # broad line region
        L_disk = 2 * 1e46 * u.Unit("erg s-1")
        xi_line = 0.024
        R_line = 1e17 * u.cm
        blr = SphericalShellBLR(L_disk, xi_line, "Lyalpha", R_line)
        # point like source approximating the blr
        ps_blr = PointSourceBehindJet(blr.xi_line * L_disk, blr.epsilon_line)
        # external Compton
        ec_blr = ExternalCompton(bpwl_blob_test, blr, r=1e22 * u.cm)
        ec_ps_blr = ExternalCompton(bpwl_blob_test, ps_blr, r=1e22 * u.cm)
        # seds
        nu = np.logspace(15, 28) * u.Hz
        ec_blr_sed = ec_blr.sed_flux(nu)
        ec_ps_blr_sed = ec_ps_blr.sed_flux(nu)
        # sed comparison plot
        make_sed_comparison_plot(
            nu,
            ec_blr_sed,
            ec_ps_blr_sed,
            "spherical shell BLR",
            "point source behind the jet",
            "External Compton " + r"$r = 10^{22}\,{\rm cm} \gg R_{\rm line}$",
            f"{data_dir}/crosscheck_figures/ec_blr_point_source_comparison.png",
        )
        # requires a 20% deviation from the two SED points
        assert u.allclose(
            ec_blr_sed, ec_ps_blr_sed, atol=0 * u.Unit("erg cm-2 s-1"), rtol=0.2
        )

    def test_ec_dt_vs_point_source(self):
        """check if in the limit of large distances the EC on the DT tends to
        the one of a point-like source approximating it"""
        # dust torus
        L_disk = 2 * 1e46 * u.Unit("erg s-1")
        T_dt = 1e3 * u.K
        csi_dt = 0.1
        dt = RingDustTorus(L_disk, csi_dt, T_dt)
        # point like source approximating the dt
        ps_dt = PointSourceBehindJet(dt.xi_dt * L_disk, dt.epsilon_dt)
        # external Compton
        ec_dt = ExternalCompton(bpwl_blob_test, dt, r=1e22 * u.cm)
        ec_ps_dt = ExternalCompton(bpwl_blob_test, ps_dt, r=1e22 * u.cm)
        # seds
        nu = np.logspace(15, 28) * u.Hz
        ec_dt_sed = ec_dt.sed_flux(nu)
        ec_ps_dt_sed = ec_ps_dt.sed_flux(nu)
        make_sed_comparison_plot(
            nu,
            ec_dt_sed,
            ec_ps_dt_sed,
            "ring dust torus",
            "point source behind the jet",
            "External Compton " + r"$r = 10^{22}\,{\rm cm} \gg R_{\rm dt}$",
            f"{data_dir}/crosscheck_figures/ec_dt_point_source_comparison.png",
        )
        # requires a 20% deviation from the two SED points
        assert u.allclose(
            ec_dt_sed, ec_ps_dt_sed, atol=0 * u.Unit("erg cm-2 s-1"), rtol=0.2
        )
