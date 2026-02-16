import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import convolve1d


def savgol_smoothing(data, window_length=11, polyorder=2):
    """
    Apply the Savitzky–Golay filter to spectral data (row-wise).

    This function performs smoothing using a local polynomial fit within a moving window.
    In vibrational spectroscopy (e.g., FTIR), Savitzky–Golay filtering is commonly used
    to reduce noise, remove baseline effects, and enhance peak resolution.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_samples, n_features)
        Spectral dataset where each row corresponds to one spectrum and
        each column corresponds to a wavenumber (or feature).

    window_length : int, default=11
        Length of the moving window (must be a positive odd integer).
        The window defines how many neighboring points are used to fit
        the local polynomial.

    polyorder : int, default=2
        Order of the polynomial used in the local least-squares fit.
        Must be less than `window_length`.

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_features)
        Filtered (or differentiated) spectral data.
    """
    return savgol_filter(
        data,
        window_length=window_length,
        polyorder=polyorder,
        deriv=0,
        axis=1
    )


def moving_average_smoothing(data, window_size):
    """
    Apply uniform moving average smoothing to spectral data (row-wise).

    This function performs smoothing by replacing each data point with the
    average of its neighboring points within a specified window. In FTIR
    spectroscopy, moving average smoothing can reduce high-frequency noise,
    though it may broaden spectral peaks more than Savitzky–Golay filtering.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_samples, n_features)
        Spectral dataset where each row corresponds to one spectrum and
        each column corresponds to a wavenumber (or feature).

    window_size : int
        Size of the smoothing window. Defines the total number of neighboring
        points used in the average. Larger values produce stronger smoothing
        but may distort peak shapes.

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_features)
        Smoothed spectral data.
    """
    weights = np.ones(window_size) / window_size
    return convolve1d(data, weights, axis=1, mode='reflect')
