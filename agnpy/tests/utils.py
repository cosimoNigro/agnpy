# utils for testing
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

SED_X_LABEL = r"$\nu\,/\,{\rm Hz}$"
SED_Y_LABEL = r"$\nu F_{\nu}\,/\,({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$"
SED_DEVIATION_LABEL = r"$\nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm reference} - 1$"

TAU_X_LABEL = r"$\nu\,/\,{\rm Hz}$"
TAU_Y_LABEL = r"$\tau_{\gamma\gamma}$"
TAU_DEVIATION_LABEL = (
    r"$\tau_{\gamma\gamma, \rm agnpy}\,/\,\tau_{\gamma\gamma, \rm reference}$ - 1"
)


def extract_columns_sample_file(sample_file, x_unit, y_unit=None):
    """return two arrays of quantities from a sample file"""
    sample_table = np.loadtxt(sample_file, delimiter=",", comments="#")
    x = sample_table[:, 0] * u.Unit(x_unit)
    y = sample_table[:, 1] if y_unit is None else sample_table[:, 1] * u.Unit(y_unit)
    return x, y


def check_deviation(x, y_comp, y_ref, rtol, x_range=None):
    """check the deviation of two quantities within a given range of x
    when setting atol = 0 in np.allclose it will check that
    |a - b| <= rtol * |b|, that is |a / b - 1| <= rtol. 
    If we choose the agnpy values to be a and the reference (code ro figure from 
    the literature) to be b then |a / b - 1| will be positive when agnpy
    overestimates the reference (a > b) and negative when agnpy underestimates 
    the reference (a < b). 
    """
    if x_range is not None:
        condition = (x >= x_range[0]) * (x <= x_range[1])
        y_ref = y_ref[condition]
        y_comp = y_comp[condition]
    return np.allclose(y_comp, y_ref, atol=0, rtol=rtol)


def make_comparison_plot(
    nu,
    y_comp,
    y_ref,
    comp_label,
    ref_label,
    fig_title,
    fig_path,
    plot_type,
    y_range=None,
    comparison_range=None,
    x_scale="log",
    y_scale="log",
):
    """make a comparison plot, for SED or gamma-gamma absorption 
    between two different sources: a reference (literature or another code)
    and a comparison (usually the agnpy result)

    Parameters
    ----------
    nu: :class:`~astropy.units.Quantity`
        frequencies over which the comparison plot has to be plotted
    y_comp: :class:`~astropy.units.Quantity` or :class:`~numpy.ndarray`
        SED or gamma-gamma absorption to be compared (usually agnpy)
    y_ref: :class:`~astropy.units.Quantity` or :class:`~numpy.ndarray`
        reference SED or gamma-gamma absorption (from literature or another code)
    ref_label : `string`
        label of the reference model
    comp_label : `string`
        label of the comparison model
    fig_title : `string`
        upper title of the figure    
    fig_path : `string`
        path to save the figure
    plot_type : `{"sed", "tau", ...}`
        whether we are doing a comparison plot for a SED or an optical depth 
        if another string is specified it will be used for the y axis
    y_range : list of float
        lower and upper limit of the y axis limt
    comparison_range : list of float
        plot the range over which the residuals were checked 
    """
    if plot_type == "sed":
        # set the axes labels for an SED plot
        x_label = SED_X_LABEL
        y_label = SED_Y_LABEL
        deviation_label = SED_DEVIATION_LABEL
    elif plot_type == "tau":
        # set the axes labels for a tau plot
        x_label = TAU_X_LABEL
        y_label = TAU_Y_LABEL
        deviation_label = TAU_DEVIATION_LABEL
    else:
        # set a custom y label, keep the x-axis in frequency
        x_label = SED_X_LABEL
        y_label = plot_type
        deviation_label = f"({plot_type} agnpy / {plot_type} ref.) - 1"
    # make the plot
    fig, ax = plt.subplots(
        2,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
        figsize=(8, 6),
    )
    # plot the SEDs or TAUs in the upper panel
    # plot the reference sed with a continuous line and agnpy sed with a dashed one
    ax[0].loglog(nu, y_ref, marker=".", ls="-", color="k", lw=1.5, label=ref_label)
    ax[0].loglog(
        nu, y_comp, marker=".", ls="--", color="crimson", lw=1.5, label=comp_label
    )
    ax[0].set_ylabel(y_label)
    ax[0].set_title(fig_title)
    ax[0].legend(loc="best")
    if y_range is not None:
        ax[0].set_ylim(y_range)
    # plot the deviation in the bottom panel
    deviation = y_comp / y_ref - 1
    ax[1].axhline(0, ls="-", color="darkgray")
    ax[1].axhline(0.2, ls="--", color="darkgray")
    ax[1].axhline(-0.2, ls="--", color="darkgray")
    ax[1].axhline(0.3, ls=":", color="darkgray")
    ax[1].axhline(-0.3, ls=":", color="darkgray")
    ax[1].set_ylim([-0.5, 0.5])
    ax[1].semilogx(
        nu,
        deviation,
        marker=".",
        ls="--",
        color="crimson",
        lw=1.5,
        label=deviation_label,
    )
    ax[1].set_xlabel(x_label)
    ax[1].legend(loc="best")
    if comparison_range is not None:
        ax[1].axvline(comparison_range[0], ls="--", color="k")
        ax[1].axvline(comparison_range[1], ls="--", color="k")
    fig.savefig(f"{fig_path}")
    # avoid RuntimeWarning: More than 20 figures have been opened.
    plt.close(fig)
