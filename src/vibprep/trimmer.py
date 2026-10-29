import numpy as np


REGIONS = {
    "full": (None, None),
    "fingerprint": (900, 1800),
    "amide": (1500, 1700),
    "lipid": (2800, 3000),
    "nucleic": (1000, 1250),
    "polysaccharide": (900, 1200)
}


def trim_spectral_region(data, wavelength, region, exclude=None):
    """
    Select (trim) a specific spectral region from FTIR or Raman data.

    The predefined regions are FTIR (infrared) bands. For Raman spectra,
    pass a custom ``(min, max)`` tuple in Raman shift units (cm⁻¹).

    This function extracts a predefined wavenumber interval from a
    spectral dataset. The trimming is performed column-wise using
    a boolean mask on the spectral axis.

    Parameters
    ----------
    data : numpy.ndarray of shape (n_samples, n_features)
        Spectral intensity matrix where each row corresponds to one spectrum
        and each column corresponds to a wavenumber.

    wavelength : numpy.ndarray of shape (n_features,)
        Spectral axis (wavenumbers). Must be aligned with the columns of `data`.
        It is recommended that the axis is sorted in ascending order.

    region : str, tuple of (float, float) or None
        Name of the predefined spectral region to extract, or a custom
        ``(min, max)`` interval. Available predefined options:

        - "full": return full spectrum (no trimming)
        - "fingerprint": 900–1800 cm⁻¹
        - "amide": 1500–1700 cm⁻¹
        - "lipid": 2800–3000 cm⁻¹
        - "nucleic": 1000–1250 cm⁻¹
        - "polysaccharide": 900-1200 cm⁻¹

        If None or "full", the original data is returned unchanged.

    exclude : tuple of (float, float) or list of such tuples, optional
        Sub-region(s) to remove from the selected region, as inclusive
        ``(min, max)`` intervals. For example, ``region=(1000, 3000)`` with
        ``exclude=(2000, 2500)`` keeps 1000-2000 and 2500-3000 (the
        boundaries 2000 and 2500 are removed as well). Several intervals can
        be given as a list. Applied after ``region``, also when
        ``region`` is "full" or None.

    Returns
    -------
    data_trimmed : numpy.ndarray of shape (n_samples, n_selected_features)
        Spectral data restricted to the selected region.

    wavelength_trimmed : numpy.ndarray of shape (n_selected_features,)
        Trimmed spectral axis corresponding to the selected region.

    Raises
    ------
    ValueError
        If the specified region is not defined in REGIONS, or if an
        ``exclude`` interval is not a valid ``(min, max)`` pair.
    """

    if region is None or region == "full":
        min_wavenumber, max_wavenumber = REGIONS["full"]
    elif isinstance(region, tuple):
        min_wavenumber, max_wavenumber = region
    elif region in REGIONS:
        min_wavenumber, max_wavenumber = REGIONS[region]
    else:
        raise ValueError(
            f"Unknown region: '{region}'. "
            f"Valid: {list(REGIONS.keys())} or tuple (min, max)"
        )

    mask = np.ones(len(wavelength), dtype=bool)
    if min_wavenumber is not None:
        mask &= wavelength >= min_wavenumber
    if max_wavenumber is not None:
        mask &= wavelength <= max_wavenumber

    for low, high in _normalize_exclude(exclude):
        mask &= ~((wavelength >= low) & (wavelength <= high))

    if mask.all():
        return data, wavelength
    return data[:, mask], wavelength[mask]

def _normalize_exclude(exclude):
    """Return ``exclude`` as a list of validated ``(min, max)`` tuples."""
    if exclude is None:
        return []
    if (
        isinstance(exclude, (tuple, list))
        and len(exclude) == 2
        and not isinstance(exclude[0], (tuple, list))
    ):
        exclude = [exclude]
    elif not isinstance(exclude, (tuple, list)):
        raise ValueError(
            f"Invalid exclude: {exclude!r}. "
            "Expected (min, max) or a list of (min, max)."
        )

    intervals = []
    for interval in exclude:
        try:
            low, high = interval
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid exclude interval: {interval!r}. "
                "Expected (min, max) or a list of (min, max)."
            )
        if low > high:
            raise ValueError(f"Invalid exclude interval {interval!r}: min > max.")
        intervals.append((low, high))
    return intervals
