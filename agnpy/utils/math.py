# math utilities for agnpy
import numpy as np
import astropy.units as u

# default arrays to be used for integration
gamma_to_integrate = np.logspace(1, 9, 200)
nu_to_integrate = np.logspace(5, 30, 200) * u.Hz  # used for SSC
mu_to_integrate = np.linspace(-1, 1, 100)
phi_to_integrate = np.linspace(0, 2 * np.pi, 50)

# minimum relative distance to the absorber (to avoid infinite integrals)
min_rel_distance = 1.0e-4

# type of float to be used for math operation
numpy_type = np.float64
# smallest positive float
ftiny = np.finfo(numpy_type).tiny
# largest positive float
fmax = np.finfo(numpy_type).max


def axes_reshaper(*args):
    """reshape 1-dimensional arrays of different lengths in order for them to be
    broadcastable in multi-dimensional operations

    the rearrangement scheme for a list of n arrays is the following:
    `args[0]` is reshaped as `(args[0].size, 1, 1, ..., 1)` -> axis 0
    `args[1]` is reshaped as `(1, args[1].size, 1, ..., 1)` -> axis 1
    `args[2]` is reshaped as `(1, 1, args[2].size ..., 1)` -> axis 2
        .
        .
        .
    `args[n-1]` is reshaped as `(1, 1, 1, ..., args[n-1].size)` -> axis n-1
    Parameters
    ----------
    args: 1-dimensional `~numpy.ndarray`s to be reshaped
    """
    n = len(args)
    dim = (1,) * n
    reshaped_arrays = []
    for i, arg in enumerate(args):
        reshaped_dim = list(dim)  # the tuple is copied in the list
        reshaped_dim[i] = arg.size
        reshaped_array = np.reshape(arg, reshaped_dim)
        reshaped_arrays.append(reshaped_array)
    return reshaped_arrays


def log(x):
    """clipped log to avoid RuntimeWarning: divide by zero encountered in log"""
    values = np.clip(x, ftiny, fmax)
    return np.log(values)


def trapz_loglog(y, x, axis=0):
    """
    Integrate a function approximating its discrete intervals as power-laws
    (straight lines in log-log space), largely copied from naima

    Parameters
    ----------
    y : array_like
        array to integrate
    x : array_like, optional
        independent variable to integrate over
    axis : int, optional
        along which axis the integration has to be performed

    Returns
    -------
    trapz : float
        Definite integral as approximated by trapezoidal rule in loglog space.
    """
    try:
        y_unit = y.unit
        y = y.value
    except AttributeError:
        y_unit = 1.0
    try:
        x_unit = x.unit
        x = x.value
    except AttributeError:
        x_unit = 1.0

    slice_low = [slice(None)] * y.ndim
    slice_up = [slice(None)] * y.ndim
    # multi-dimensional equivalent of x_low = x[:-1]
    slice_low[axis] = slice(None, -1)
    # multi-dimensional equivalent of x_up = x[1:]
    slice_up[axis] = slice(1, None)

    # reshape x to be broadcastable with y
    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    x_low = x[tuple(slice_low)]
    x_up = x[tuple(slice_up)]
    y_low = y[tuple(slice_low)]
    y_up = y[tuple(slice_up)]

    # slope in the given logarithmic bin
    m = (log(y_low) - log(y_up)) / (log(x_low) - log(x_up))

    vals = np.where(
        np.abs(m + 1) > 1e-10,
        y_low / (m + 1) * (x_up * np.power(x_up / x_low, m) - x_low),
        x_low * y_low * log(x_up / x_low),
    )

    tozero = (
        (y[tuple(slice_low)] == 0.0)
        + (y[tuple(slice_up)] == 0.0)
        + (x[tuple(slice_low)] == x[tuple(slice_up)])
    )
    vals[tozero] = 0.0

    return np.add.reduce(vals, axis) * x_unit * y_unit
