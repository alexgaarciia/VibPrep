from scipy.signal import savgol_filter


def savgol_derivative(data, window_length=11, polyorder=2, deriv=1, delta=1.0):
    """
    Compute spectral derivatives using the Savitzky–Golay filter (row-wise).

    This function applies a local polynomial fit within a moving window and
    computes the specified order derivative of the spectral signal. In FTIR/Raman
    spectroscopy, Savitzky–Golay derivatives are commonly used to reduce
    baseline effects, enhance resolution of overlapping peaks, and highlight
    subtle spectral features.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_samples, n_features)
        Spectral dataset where each row corresponds to one spectrum and
        each column corresponds to a wavenumber (or feature).

    window_length : int, default=11
        Length of the moving window (must be a positive odd integer).
        Defines the number of neighboring points used for the local polynomial fit.

    polyorder : int, default=2
        Order of the polynomial used in the local least-squares fit.
        Must be less than `window_length`.

    deriv : int, default=1
        Order of the derivative to compute.
        Typically 1 (first derivative) or 2 (second derivative).

    delta : float, default=1.0
        Spacing between data points along the spectral axis.
        If wavenumbers are uniformly spaced, this can remain 1.0.
        For physically meaningful derivatives, use the actual spacing.

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_features)
        Spectral derivatives with the same shape as the input.
    """

    return savgol_filter(
        data,
        window_length=window_length,
        polyorder=polyorder,
        deriv=deriv,
        delta=delta,
        axis=-1
    )
