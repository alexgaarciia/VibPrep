import numpy as np


def standard_normal_variate(data):
    """
    Apply Standard Normal Variate (SNV) normalization to spectral data.

    SNV is performed independently on each spectrum (row-wise). For every
    spectrum, the mean is subtracted and the result is divided by its
    standard deviation:

        x_snv = (x - mean(x)) / std(x)

    This operation corrects for multiplicative scatter effects and
    intensity variations caused by differences in sample thickness,
    concentration, or optical path length. SNV is widely used in
    vibrational spectroscopy (e.g., FTIR, Raman).

    Parameters
    ----------
    data : numpy.ndarray of shape (n_samples, n_features)
        Spectral dataset where each row corresponds to a spectrum and
        each column corresponds to a wavenumber (or feature).

    Returns
    -------
    numpy.ndarray of shape (n_samples, n_features)
        SNV-normalized spectral data.
    """

    data = np.asarray(data)

    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    snv = (data - mean) / (std + 1e-8)
    return snv
