import numpy as np
import pandas as pd

def average_replicates(X, metadata, groupby_cols, method="mean"):
    """
    Aggregate replicate spectra based on metadata grouping.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        Spectral intensity matrix.

    metadata : pandas.DataFrame of shape (n_samples, n_metadata)
        Metadata corresponding to each spectrum.

    groupby_cols : str or list of str
        Column(s) in metadata that define biological replicates.
        Spectra sharing the same values in these columns
        will be aggregated.

    method : {"mean", "median"}, default="mean"
        Aggregation method used to combine replicate spectra.

    Returns
    -------
    X_agg : numpy.ndarray of shape (n_groups, n_features)
        Aggregated spectral matrix.

    metadata_agg : pandas.DataFrame of shape (n_groups, n_metadata)
        Metadata for aggregated spectra (first entry per group).
    """

    if isinstance(groupby_cols, str):
        groupby_cols = [groupby_cols]

    if len(X) != len(metadata):
        raise ValueError("X and metadata must have the same number of rows")

    df_meta = metadata.copy()
    df_meta["_row_index"] = np.arange(len(df_meta))

    grouped = df_meta.groupby(groupby_cols)

    X_agg = []
    metadata_agg = []

    for _, group in grouped:
        idx = group["_row_index"].values
        subset = X[idx]

        if method == "mean":
            X_agg.append(subset.mean(axis=0))
        elif method == "median":
            X_agg.append(np.median(subset, axis=0))
        else:
            raise ValueError("method must be 'mean' or 'median'")

        metadata_agg.append(group.iloc[0].drop("_row_index"))

    X_agg = np.vstack(X_agg)
    metadata_agg = pd.DataFrame(metadata_agg).reset_index(drop=True)

    return X_agg, metadata_agg
