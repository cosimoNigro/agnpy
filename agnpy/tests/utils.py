# utils for the test scripts
import matplotlib.pyplot as plt


def make_sed_comparison_plot(nu, reference_sed, agnpy_sed, fig_title, fig_path):
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
        label=r"$1 - \nu F_{\nu, \rm agnpy} \, / \,\nu F_{\nu, \rm reference}$",
    )
    ax[1].legend(loc=2)
    ax[1].axhline(0, ls="-", lw=1.5, color="dimgray")
    ax[1].axhline(0.2, ls="--", lw=1.5, color="dimgray")
    ax[1].axhline(-0.2, ls="--", lw=1.5, color="dimgray")
    ax[1].axhline(0.3, ls=":", lw=1.5, color="dimgray")
    ax[1].axhline(-0.3, ls=":", lw=1.5, color="dimgray")
    ax[1].set_ylim([-0.5, 0.5])
    ax[1].set_xlabel(r"$\nu / Hz$")
    fig.savefig(f"{fig_path}")
