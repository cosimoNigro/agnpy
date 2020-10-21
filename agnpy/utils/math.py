# math utilities for agnpy
import numpy as np

# type of float to be used for math operation
numpy_type = np.float64
# smallest positive float
ftiny = np.finfo(numpy_type).tiny
# largest positive float
fmax = np.finfo(numpy_type).max
# the smallest positive power of the base 10 that causes overflow.
fminexp = np.finfo(numpy_type).minexp
# the smallest negative power of the base 10 that causes overflow.
fmaxexp = np.finfo(numpy_type).maxexp


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
    arg: 1-dimensional `~numpy.ndarray`s to be reshaped
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


def power(x, m):
    """clipped power to avoid RuntimeWarning: overflow in power"""
    m = np.clip(m, fminexp, fmaxexp)
    return x ** m


def trapz_loglog(y, x, axis=0):
    """
    Integrate a function approximating its discrete intervals as power-laws
    (straight lines in log-log space)

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

    slice_low = [slice(None)] * y.ndim
    slice_up = [slice(None)] * y.ndim
    # multi-dimensional equivalent of x_low = x[:-1]
    slice_low[axis] = slice(None, -1)
    # multi-dimensional equivalent of x_up = x[1:]
    slice_up[axis] = slice(1, None)

    slice_low = tuple(slice_low)
    slice_up = tuple(slice_up)

    # reshape x to be broadcastable with y
    if x.ndim == 1:
        shape = [1] * y.ndim
        shape[axis] = x.shape[0]
        x = x.reshape(shape)

    x_low = x[slice_low]
    x_up = x[slice_up]
    y_low = y[slice_low]
    y_up = y[slice_up]

    log_x_low = log(x_low)
    log_x_up = log(x_up)
    log_y_low = log(y_low)
    log_y_up = log(y_up)

    # index in the bin
    m = (log_y_low - log_y_up) / (log_x_low - log_x_up)
    vals = y_low / (m + 1) * (x_up * power(x_up / x_low, m) - x_low)

    # value of y very close to zero will make m large and explode the exponential
    tozero = (
        np.isclose(y_low, 0, atol=0, rtol=1e-10)
        + np.isclose(y_up, 0, atol=0, rtol=1e-10)
        + np.isclose(x_low, x_up, atol=0, rtol=1e-10)
    )
    vals[tozero] = 0.0

    return np.add.reduce(vals, axis) * x_unit * y_unit
