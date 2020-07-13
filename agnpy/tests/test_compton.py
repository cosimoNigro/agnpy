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


mec2 = m_e.to("erg", equivalencies=u.mass_energy())
tests_dir = Path(__file__).parent


def make_sed_comparison_plot(nu, reference_sed, agnpy_sed, fig_title, fig_name):
    """make a SED comparison plot for visual inspection"""
    fig, ax = plt.subplots(
        2, sharex=True, gridspec_kw={"height_ratios": [2, 1]}, figsize=(8, 6)
    )
    # plot the SEDs in the upper panel
    ax[0].loglog(nu, reference_sed, marker=".", ls="-", lw=1.5, label="reference")
    ax[0].loglog(nu, agnpy_sed, marker=".", ls="--", lw=1.5, label="agnpy")
    ax[0].legend()
    ax[0].set_xlabel(r"$\nu\,/\,{\rm Hz}$")
    ax[0].set_ylabel(r"$\nu F_{\nu}\,/\,({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$")
    ax[0].set_title(fig_title)
    # plot the deviation in the bottom panel
    deviation = 1 - agnpy_sed / reference_sed
    ax[1].semilogx(
        nu,
        deviation,
        lw=1.5,
        label=r"$|1 - \nu F_{\nu, \rm agnpy} \, / \,\nu F_{\nu, \rm reference}|$",
    )
    ax[1].legend(loc=2)
    ax[1].axhline(0, ls="-", lw=1.5, color="dimgray")
    ax[1].axhline(0.2, ls="--", lw=1.5, color="dimgray")
    ax[1].axhline(-0.2, ls="--", lw=1.5, color="dimgray")
    ax[1].axhline(0.3, ls=":", lw=1.5, color="dimgray")
    ax[1].axhline(-0.3, ls=":", lw=1.5, color="dimgray")
    ax[1].set_ylim([-0.5, 0.5])
    ax[1].set_xlabel(r"$\nu / Hz$")
    fig.savefig(f"{tests_dir}/crosscheck_figures/{fig_name}.png")


# global PWL blob, same parameters of Figure 7.4 in Dermer Menon 2009
PWL_SPECTRUM_NORM = 1e48 * u.Unit("erg")
PWL_DICT = {
    "type": "PowerLaw",
    "parameters": {"p": 2.8, "gamma_min": 1e2, "gamma_max": 1e5},
}
R_B_PWL = 1e16 * u.cm
B_PWL = 1 * u.G
Z_PWL = Distance(1e27, unit=u.cm).z
DELTA_D_PWL = 10
GAMMA_PWL = 10
PWL_BLOB = Blob(
    R_B_PWL, Z_PWL, DELTA_D_PWL, GAMMA_PWL, B_PWL, PWL_SPECTRUM_NORM, PWL_DICT
)

# global BPL blob, same parameters of the examples in Finke 2016
BPL_SPECTRUM_NORM = 6e42 * u.Unit("erg")
BPL_DICT = {
    "type": "BrokenPowerLaw",
    "parameters": {
        "p1": 2.0,
        "p2": 3.5,
        "gamma_b": 1e4,
        "gamma_min": 20,
        "gamma_max": 5e7,
    },
}
R_B_BPL = 1e16 * u.cm
B_BPL = 0.56 * u.G
Z_BPL = 1
DELTA_D_BPL = 40
GAMMA_BPL = 40
BPL_BLOB = Blob(
    R_B_BPL, Z_BPL, DELTA_D_BPL, GAMMA_BPL, B_BPL, BPL_SPECTRUM_NORM, BPL_DICT
)
BPL_BLOB.set_gamma_size(500)


class TestSynchrotronSelfCompton:
    """class grouping all tests related to the Synchrotron Slef Compton class"""

    def test_ssc_reference_sed(self):
        """test agnpy SSC SED against the one in Figure 7.4 of Dermer Menon"""
        sampled_ssc_table = np.loadtxt(
            f"{tests_dir}/sampled_seds/ssc_figure_7_4_dermer_menon_2009.txt",
            delimiter=",",
            comments="#",
        )
        sampled_ssc_nu = sampled_ssc_table[:, 0] * u.Hz
        sampled_ssc_sed = sampled_ssc_table[:, 1] * u.Unit("erg cm-2 s-1")
        # agnpy
        synch = Synchrotron(PWL_BLOB)
        ssc = SynchrotronSelfCompton(PWL_BLOB, synch)
        # recompute the SED at the same ordinates where the figure was sampled
        agnpy_ssc_sed = ssc.sed_flux(sampled_ssc_nu)
        # sed comparison plot
        make_sed_comparison_plot(
            sampled_ssc_nu,
            sampled_ssc_sed,
            agnpy_ssc_sed,
            "Synchrotron Self Compton",
            "ssc_comparison_figure_7_4_dermer_menon_2009",
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
            f"{tests_dir}/sampled_seds/ec_disk_figure_8_finke_2016.txt",
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
        ec_disk = ExternalCompton(BPL_BLOB, disk, r=1e17 * u.cm)
        agnpy_ec_disk_sed = ec_disk.sed_flux(sampled_ec_disk_nu)
        # sed comparison plot
        make_sed_comparison_plot(
            sampled_ec_disk_nu,
            sampled_ec_disk_sed,
            agnpy_ec_disk_sed,
            "External Compton on Shakura Sunyaev Disk",
            "ec_disk_comparison_figure_8_finke_2016",
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
            f"{tests_dir}/sampled_seds/ec_blr_figure_10_finke_2016.txt",
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
        ec_blr = ExternalCompton(BPL_BLOB, blr, r=1e18 * u.cm)
        agnpy_ec_blr_sed = ec_blr.sed_flux(sampled_ec_blr_nu)
        # sed comparison plot
        make_sed_comparison_plot(
            sampled_ec_blr_nu,
            sampled_ec_blr_sed,
            agnpy_ec_blr_sed,
            "External Compton on Spherical Shell Broad Line Region",
            "ec_blr_comparison_figure_10_finke_2016",
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
            f"{tests_dir}/sampled_seds/ec_dt_figure_11_finke_2016.txt",
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
        ec_dt = ExternalCompton(BPL_BLOB, dt, r=1e20 * u.cm)
        agnpy_ec_dt_sed = ec_dt.sed_flux(sampled_ec_dt_nu)
        # sed comparison plot
        make_sed_comparison_plot(
            sampled_ec_dt_nu,
            sampled_ec_dt_sed,
            agnpy_ec_dt_sed,
            "External Compton on Ring Dust Torus",
            "ec_dt_comparison_figure_11_finke_2016",
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
        ec_blr = ExternalCompton(BPL_BLOB, blr, r=1e22 * u.cm)
        ec_ps_blr = ExternalCompton(BPL_BLOB, ps_blr, r=1e22 * u.cm)
        # seds
        nu = np.logspace(15, 28) * u.Hz
        ec_blr_sed = ec_blr.sed_flux(nu)
        ec_ps_blr_sed = ec_ps_blr.sed_flux(nu)
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
        ec_dt = ExternalCompton(BPL_BLOB, dt, r=1e22 * u.cm)
        ec_ps_dt = ExternalCompton(BPL_BLOB, ps_dt, r=1e22 * u.cm)
        # seds
        nu = np.logspace(15, 28) * u.Hz
        ec_dt_sed = ec_dt.sed_flux(nu)
        ec_ps_dt_sed = ec_ps_dt.sed_flux(nu)
        # requires a 20% deviation from the two SED points
        assert u.allclose(
            ec_dt_sed, ec_ps_dt_sed, atol=0 * u.Unit("erg cm-2 s-1"), rtol=0.2
        )
