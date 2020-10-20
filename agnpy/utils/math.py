# math utils for agnpy
import numpy as np
import warnings

def log(x):
    """log version clipping 0 values"""
    # smallest positive float (before 0)
    float_tiny = np.finfo(np.float64).tiny
    # largest positive float
    float_max = np.finfo(np.float64).max
    values = np.clip(x, float_tiny, float_max)
    return np.log(values)

def trapz_loglog(y, x, axis=0):
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
    
    slice_low = [slice(None)] * y.ndim
    slice_up = [slice(None)] * y.ndim
    # multi-dimensional equivalent of x_low = x[:-1]
    slice_low[axis] = slice(None, -1)
    # multi-dimensional equivalent of x_up = x[1:]
    slice_up[axis] = slice(1, None)

    slice_low = tuple(slice_low)
    slice_up = tuple(slice_up)

    # reshape x to be broadcasted with y
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
    vals = y_low / (m + 1) * (x_up * (x_up / x_low) ** m - x_low)
    
    # value of y very close to zero will make m large and explode the exponential
    tozero = (
        np.isclose(y_low, 0, atol=0, rtol=1e-10) +
        np.isclose(y_up, 0, atol=0, rtol=1e-10) +
        np.isclose(x_low, x_up, atol=0, rtol=1e-10)
    )
    vals[tozero] = 0.0
    
    return np.add.reduce(vals, axis) * x_unit * y_unit