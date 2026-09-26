import numpy as np
from vibprep.replicates import average_replicates
from vibprep.trimmer import trim_spectral_region
from vibprep.baseline import (
    polynomial_baseline_correction,
    als_baseline_correction,
    aspls_baseline_correction,
)
from vibprep.scatter import standard_normal_variate
from vibprep.smoothing import savgol_smoothing, moving_average_smoothing
from vibprep.derivatives import savgol_derivative
from vibprep.normalization import intensity_normalization


VALID_STEPS = {"baseline", "scatter", "smoothing", "derivative", "normalization"}


class PreprocessingPipeline():
    """
    Vibrational spectroscopy (FTIR, Raman) preprocessing pipeline.

    This class applies a configurable sequence of preprocessing
    operations to FTIR or Raman spectra, including replicate averaging,
    spectral region selection, baseline correction, scatter correction,
    smoothing, derivative computation, and normalization.

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary with the following keys:

        - ``average`` : bool
            Whether to average replicate spectra.
        - ``region`` : str
            Spectral region to retain (e.g. ``"full"``,
            ``"fingerprint"``, ``"amide"``, ``"lipid"``,
            ``"nucleic"``).
        - ``steps`` : list[tuple]
            Ordered list of preprocessing steps. Each element must be
            a tuple ``(step_name, step_value)``.

    Examples
    --------
    >>> config = {
    ...     "average": True,
    ...     "region": "fingerprint",
    ...     "steps": [
    ...         ("baseline", "als"),
    ...         ("scatter", "snv"),
    ...         ("normalization", "area"),
    ...     ],
    ... }
    >>> pipeline = PreprocessingPipeline(config)
    >>> X_proc, wn_proc, meta_proc = pipeline.transform(
    ...     X, wavelengths, metadata
    ... )
    """

    def __init__(self, config):
        self.config = config
        self.average = config["average"]
        self.region = config["region"]
        self.steps = config["steps"]
        self._validate_steps()

    def transform(self, X, wavelengths, metadata=None, groupby_cols=None):
        """
        Apply the preprocessing pipeline to a set of spectra.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Spectral intensity matrix.
        wavelengths : ndarray of shape (n_features,)
            Wavenumber axis corresponding to the spectra.
        metadata : pandas.DataFrame, optional
            Metadata associated with each spectrum. Required if
            replicate averaging is enabled.
        groupby_cols : str or list[str], optional
            Metadata columns used to identify replicate groups when
            averaging spectra.

        Returns
        -------
        X : ndarray
            Preprocessed spectra.
        wavelengths : ndarray
            Processed wavenumber axis.
        metadata : pandas.DataFrame or None
            Updated metadata after preprocessing.

        Notes
        -----
        The preprocessing workflow is applied in the following order:

        1. Ensure ascending wavenumber order.
        2. Average replicates (optional).
        3. Trim the selected spectral region.
        4. Apply configured preprocessing steps sequentially.
        """
        X, wavelengths = self._ensure_ascending(X, wavelengths)
        X, metadata = self._apply_average(X, metadata, groupby_cols)
        X, wavelengths = self._apply_trim(X, wavelengths)

        for step_name, step_value in self.steps:
            if step_name == "baseline":
                X = self._apply_baseline(X, wavelengths, step_value)
            elif step_name == "scatter":
                X = self._apply_scatter(X, step_value)
            elif step_name == "smoothing":
                X = self._apply_smoothing(X, step_value)
            elif step_name == "derivative":
                X = self._apply_derivative(X, wavelengths, step_value)
            elif step_name == "normalization":
                X = self._apply_normalization(X, step_value)

        return X, wavelengths, metadata
    
    def _validate_steps(self):
        for step_name, _ in self.steps:
            if step_name not in VALID_STEPS:
                raise ValueError(
                    f"Paso desconocido: '{step_name}'. "
                    f"Válidos: {VALID_STEPS}"
                )
            
    def _ensure_ascending(self, X, wavelengths):
        if wavelengths[0] > wavelengths[-1]:
            wavelengths = wavelengths[::-1]
            X = X[:, ::-1]
        return X, wavelengths
    
    def _apply_average(self, X, metadata, groupby_cols):
        if not self.average:
            return X, metadata
        X_agg, metadata_agg = average_replicates(X, metadata, groupby_cols)
        return X_agg, metadata_agg

    def _apply_trim(self, X, wavelengths):
        return trim_spectral_region(X, wavelengths, self.region)

    def _apply_baseline(self, X, wavelengths, method):
        if method == "none":
            return X
        elif method == "polynomial":
            corrected, _ = polynomial_baseline_correction(X, wavelengths, degree=2)
            return corrected
        elif method == "als":
            corrected, _ = als_baseline_correction(X, lam=1e5, p=0.01, niter=10)
            return corrected
        elif method == "aspls":
            corrected, _ = aspls_baseline_correction(X, lam=1e5, max_iter=10)
            return corrected
        else:
            raise ValueError(f"Baseline desconocido: '{method}'")

    def _apply_scatter(self, X, method):
        if method == "none":
            return X
        elif method == "snv":
            return standard_normal_variate(X)
        else:
            raise ValueError(f"Scatter desconocido: '{method}'")

    def _apply_smoothing(self, X, method):
        if method == "none":
            return X
        elif method == "savitzky_golay":
            return savgol_smoothing(X, window_length=11, polyorder=2)
        elif method == "moving_average":
            return moving_average_smoothing(X, window_size=5)
        else:
            raise ValueError(f"Smoothing desconocido: '{method}'")

    def _apply_derivative(self, X, wavelengths, order):
        real_delta = abs(wavelengths[1] - wavelengths[0])

        if order in ("none", 0):
            return X
        elif order in (1, 2):
            return savgol_derivative(
                X, window_length=11, polyorder=2, deriv=order, delta=real_delta)
        elif order == "1+2":
            deriv1 = savgol_derivative(X, window_length=11, polyorder=2, deriv=1, delta=real_delta)
            deriv2 = savgol_derivative(X, window_length=11, polyorder=2, deriv=2, delta=real_delta)
            return np.concatenate([deriv1, deriv2], axis=1)
        else:
            raise ValueError(f"Orden de derivada desconocido: '{order}'")

    def _apply_normalization(self, X, method):
        return intensity_normalization(X, method=method)
