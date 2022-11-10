# plotting utilities for agnpy
import numpy as np
import importlib.resources
import matplotlib.pyplot as plt

# axes labels
sed_x_label = r"$\nu\,/\,\mathrm{Hz}$"
sed_y_label = r"$\nu F_{\nu}\,/\,(\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1})$"


def load_mpl_rc():
    """use the custom matplotlibrc in this subdirectory"""
    with importlib.resources.path("agnpy.utils", "matplotlibrc") as mpl_rc:
        plt.style.use(mpl_rc)


def plot_sed(nu, sed, ax=None, **kwargs):
    """plot an SED

    Parameters
    ----------
    nu: :class:`~astropy.units.Quantity`
        frequency values
    sed : :class:`~astropy.units.Quantity`
        SED values
    """
    ax = plt.gca() if ax is None else ax

    ax.loglog(nu, sed, **kwargs)
    ax.set_xlabel(sed_x_label)
    ax.set_ylabel(sed_y_label)

    if "label" in kwargs:
        ax.legend()

    return ax
