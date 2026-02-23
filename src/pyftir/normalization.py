import numpy as np

def intensity_normalization(data, method="none"):
    """
    Apply intensity normalization to spectral data.

    Parameters
    ----------
    data : ndarray (n_samples, n_features)
    method : str
        One of:
        - "none"
        - "area"
        - "l2"
        - "max"

    Returns
    -------
    normalized : ndarray
    """

    data = np.asarray(data)

    if method == "none":
        return data
    
    elif method == "area":
        area = np.sum(data, axis=1, keepdims=True)
        return data / (area + 1e-8)

    elif method == "l2":
        norm = np.linalg.norm(data, axis=1, keepdims=True)
        return data / (norm + 1e-8)

    elif method == "max":
        max_val = np.max(data, axis=1, keepdims=True)
        return data / (max_val + 1e-8)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    