import numpy as np
from pybaselines.whittaker import asls


def polynomial_baseline_correction(data, wavelength, degree=2):
    """
    Perform polynomial baseline correction on 2D spectral data.

    This function fits a polynomial of specified degree to each spectrum
    (row-wise) using least-squares regression and subtracts the fitted
    baseline from the original signal.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_samples, n_features)
        Spectral intensity matrix where each row corresponds to one spectrum.

    wavelength : numpy.ndarray of shape (n_features,)
        Spectral axis (e.g., wavenumbers). Must match the number of features.

    degree : int, default=2
        Degree of the polynomial used to model the baseline.
        Lower degrees model simple linear or quadratic trends,
        while higher degrees allow more flexible baseline shapes.

    Returns
    -------
    corrected : numpy.ndarray of shape (n_samples, n_features)
        Baseline-corrected spectra.

    baseline : numpy.ndarray of shape (n_samples, n_features)
        Estimated polynomial baseline for each spectrum.
    """

    data = np.asarray(data)
    wavelength = np.asarray(wavelength)

    n_samples, n_features = data.shape

    if wavelength.shape[0] != n_features:
        raise ValueError("wavelength length must match number of features")

    baseline = np.zeros_like(data)

    for i in range(n_samples):
        coeffs = np.polyfit(wavelength, data[i], degree)
        baseline[i] = np.polyval(coeffs, wavelength)

    corrected = data - baseline

    return corrected, baseline


def als_baseline_correction(data, lam=1e5, p=0.01, niter=10):
    """
    Perform Asymmetric Least Squares (AsLS) baseline correction
    on 2D spectral data.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_samples, n_features)
        Spectral intensity matrix where each row corresponds to one spectrum.

    lam : float, default=1e5
        Smoothness parameter. Higher values produce smoother baselines.

    p : float, default=0.01
        Asymmetry parameter. Small values force the baseline to stay
        below peaks (typical range: 0.001–0.01).

    niter : int, default=10
        Number of iterations.

    Returns
    -------
    corrected : numpy.ndarray of shape (n_samples, n_features)
        Baseline-corrected spectra.

    baseline : numpy.ndarray of shape (n_samples, n_features)
        Estimated AsLS baseline for each spectrum.
    """
    data = np.asarray(data)
    baseline = np.zeros_like(data)

    for i in range(data.shape[0]):
        bline, _ = asls(data[i], lam=lam, p=p, max_iter=niter)
        baseline[i] = bline

    corrected = data - baseline

    return corrected, baseline
