# utils for testing
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

SED_X_LABEL = r"$\nu\,/\,{\rm Hz}$"
SED_Y_LABEL = r"$\nu F_{\nu}\,/\,({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$"
SED_DEVIATION_LABEL = r"$1 - \nu F_{\nu, \rm agnpy}\,/\,\nu F_{\nu, \rm reference}$"

TAU_X_LABEL = r"$\nu\,/\,{\rm Hz}$"
TAU_Y_LABEL = r"$\tau_{\gamma\gamma}$"
TAU_DEVIATION_LABEL = (
    r"$1 - \tau_{\gamma\gamma, \rm agnpy}\,/\,\tau_{\gamma\gamma, \rm reference}$"
)


def extract_columns_sample_file(sample_file, x_unit, y_unit=None):
    """return two arrays of quantities from a sample file"""
    sample_table = np.loadtxt(sample_file, delimiter=",", comments="#")
    x = sample_table[:, 0] * u.Unit(x_unit)
    y = (
        sample_table[:, 1]
        if y_unit is None
        else sample_table[:, 1] * u.Unit(y_unit)
    )

    return x, y


def check_deviation_within_bounds(x, y_ref, y_comp, atol, rtol, x_range=None):
    """check the deviation of two quantities within a given range of x"""
    if x_range is not None:
        condition = (x >= x_range[0]) * (x <= x_range[1])
        y_ref = y_ref[condition]
        y_comp = y_comp[condition]
    # have the quantities to be compared units?
    try:
        y_ref.unit
        atol *= y_ref.unit
        comparison = u.allclose(y_ref, y_comp, atol=atol, rtol=rtol)
    # dimensionless quantities to be compared
    except AttributeError:
        comparison = np.allclose(y_ref, y_comp, atol=atol, rtol=rtol)
    return comparison


def make_comparison_plot(
    nu, y_ref, y_comp, ref_label, comp_label, fig_title, fig_path, plot_type
):
    """make a comparison plot, for SED or gamma-gamma absorption 
    between two different sources: a reference (literature or another code)
    and a comparison (usually the agnpy result)

    Parameters
    ----------
    nu: :class:`~astropy.units.Quantity`
        frequencies over which the comparison plot has to be plotted
    y_ref: :class:`~astropy.units.Quantity` or :class:`~numpy.ndarray`
        SED or gamma-gamma absorption to be compare with (from literature or
        another code)
    y_comp: :class:`~astropy.units.Quantity` or :class:`~numpy.ndarray`
        SED or gamma-gamma absorption to compare with (usually agnpy)
    ref_label : `string`
        label of the reference model
    comp_label : `string`
        label of the comparison model
    fig_title : `string`
        upper title of the figure    
    fig_path : `string`
        path to save the figure
    plot_type : `{"sed", "tau"}`
        whether we are doing a comparison plot for a SED or an optical depth 
    """
    if plot_type == "sed":
        x_label = SED_X_LABEL
        y_label = SED_Y_LABEL
        deviation_label = SED_DEVIATION_LABEL
    elif plot_type == "tau":
        x_label = TAU_X_LABEL
        y_label = TAU_Y_LABEL
        deviation_label = TAU_DEVIATION_LABEL
    else:
        raise ValueError("plot_type can have either SED or TAU values")

    fig, ax = plt.subplots(
        2,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
        figsize=(8, 6),
    )
    # plot the SEDs or TAUs in the upper panel
    ax[0].loglog(nu, y_ref, marker="o", ls="-", lw=1.5, label=ref_label)
    ax[0].loglog(nu, y_comp, marker=".", ls="--", lw=1.5, label=comp_label)
    ax[0].legend(loc="best")
    ax[0].set_ylabel(y_label)
    ax[0].set_title(fig_title)
    # plot the deviation in the bottom panel
    deviation = 1 - y_comp / y_ref
    ax[1].axhline(0, ls="-", color="darkgray")
    ax[1].axhline(0.2, ls="--", color="darkgray")
    ax[1].axhline(-0.2, ls="--", color="darkgray")
    ax[1].axhline(0.3, ls=":", color="darkgray")
    ax[1].axhline(-0.3, ls=":", color="darkgray")
    ax[1].set_ylim([-0.5, 0.5])
    ax[1].semilogx(
        nu, deviation, marker=".", ls="--", color="C1", lw=1.5, label=deviation_label
    )
    ax[1].set_xlabel(x_label)
    ax[1].legend(loc="best")
    fig.savefig(f"{fig_path}")
