# math utilities for agnpy
import numpy as np
import warnings

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
    `arrays[0]` is reshaped as `(arrays[0].size, 1, 1, ..., 1)` -> axis 0
    `arrays[1]` is reshaped as `(1, arrays[1].size, 1, ..., 1)` -> axis 1
    `arrays[2]` is reshaped as `(1, 1, arrays[2].size ..., 1)` -> axis 2
        .
        .
        .
    `arrays[n-1]` is reshaped as `(1, 1, 1, ..., arrays[n-1].size)` -> axis n-1
    
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
    (straight lines in log-log space)

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
    m = np.where(
        (y_low == 0) + (y_up == 0),
        -np.inf,
        (log(y_low) - log(y_up)) / (log(x_low) - log(x_up)),
    )

    vals = np.where(
        np.isclose(m, -1, atol=0, rtol=1e-6),
        x_low * y_low * log(x_up / x_low),
        y_low / (m + 1) * (x_up * np.power(x_up / x_low, m) - x_low),
    )

    return np.add.reduce(vals, axis) * x_unit * y_unit


def trapz_loglog_naima(y, x, axis=-1, intervals=False):
    """
    Integrate along the given axis using the composite trapezoidal rule in
    loglog space.

    Integrate `y` (`x`) along given axis in loglog space.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        Independent variable to integrate over.
    axis : int, optional
        Specify the axis.

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

    y = np.asanyarray(y)
    x = np.asanyarray(x)

    slice1 = [slice(None)] * y.ndim
    slice2 = [slice(None)] * y.ndim
    slice1[axis] = slice(None, -1)
    slice2[axis] = slice(1, None)

    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Compute the power law indices in each integration bin
        b = np.log10(y[slice2] / y[slice1]) / np.log10(x[slice2] / x[slice1])

        # if local powerlaw index is -1, use \int 1/x = log(x); otherwise use
        # normal powerlaw integration
        trapzs = np.where(
            np.abs(b + 1.0) > 1e-10,
            (y[slice1] * (x[slice2] * (x[slice2] / x[slice1]) ** b - x[slice1]))
            / (b + 1),
            x[slice1] * y[slice1] * np.log(x[slice2] / x[slice1]),
        )

    tozero = (y[slice1] == 0.0) + (y[slice2] == 0.0) + (x[slice1] == x[slice2])
    trapzs[tozero] = 0.0

    if intervals:
        return trapzs * x_unit * y_unit

    ret = np.add.reduce(trapzs, axis) * x_unit * y_unit

    return ret
