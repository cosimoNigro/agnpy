# plotting utilities for agnpy
from pathlib import Path
import matplotlib.pyplot as plt

# axes labels
sed_x_label = r"$\nu\,/\,Hz$"
sed_y_label = r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$"

mpl_linewidth = 1.6
mpl_rc = {
    "figure.autolayout": True,
    "figure.dpi": 150,
    "font.size": 12,
    "lines.linewidth": mpl_linewidth,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 1.4,
    "axes.linewidth": mpl_linewidth,
    "xtick.major.size": 7,
    "xtick.major.width": mpl_linewidth,
    "xtick.minor.size": 4,
    "xtick.minor.width": mpl_linewidth,
    "xtick.minor.visible": True,
    "ytick.major.size": 7,
    "ytick.major.width": mpl_linewidth,
    "ytick.minor.size": 4,
    "ytick.minor.width": mpl_linewidth,
    "ytick.minor.visible": True,
}


def load_mpl_rc():
    """use the custom matplotlib rc"""
    plt.style.use(mpl_rc)


def plot_sed(nu, sed, ax=None, **kwargs):
    """plot an SED

    Parameters
    ----------
    nu: :class:`~astropy.units.Quantity`
        frequencies over which the comparison plot has to be plotted
    y_range : list of float
        lower and upper limit of the y axis limt
    comparison_range : list of float
        plot the range over which the residuals were checked 
    """
    ax = plt.gca() if ax is None else ax

    ax.loglog(nu, sed, **kwargs)
    ax.set_xlabel(sed_x_label)
    ax.set_ylabel(sed_y_label)

    if "label" in kwargs:
        ax.legend()

    return ax
