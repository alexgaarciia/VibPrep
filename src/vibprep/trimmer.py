REGIONS = {
    "full": (None, None),
    "fingerprint": (900, 1800),
    "amide": (1500, 1700),
    "lipid": (2800, 3000),
    "nucleic": (1000, 1250),
    "polysaccharide": (900, 1200)
}


def trim_spectral_region(data, wavelength, region):
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

    Returns
    -------
    data_trimmed : numpy.ndarray of shape (n_samples, n_selected_features)
        Spectral data restricted to the selected region.

    wavelength_trimmed : numpy.ndarray of shape (n_selected_features,)
        Trimmed spectral axis corresponding to the selected region.

    Raises
    ------
    ValueError
        If the specified region is not defined in REGIONS.
    """

    if region is None or region == "full":
        return data, wavelength

    if isinstance(region, tuple):
        min_wavenumber, max_wavenumber = region
        mask = (wavelength >= min_wavenumber) & (wavelength <= max_wavenumber)
        return data[:, mask], wavelength[mask]

    if region not in REGIONS:
        raise ValueError(
            f"Unknown region: '{region}'. "
            f"Valid: {list(REGIONS.keys())} or tuple (min, max)"
        )

    min_wavenumber, max_wavenumber = REGIONS[region]
    mask = (wavelength >= min_wavenumber) & (wavelength <= max_wavenumber)
    return data[:, mask], wavelength[mask]
