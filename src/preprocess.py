"""Truncation / padding, leakage-safe standardisation, and static FC helpers."""
import numpy as np


def cut_or_pad(x, t):
    """Keep the first `t` timepoints; right-pad with zeros when shorter."""
    n = x.shape[0]
    if n >= t:
        return x[:t]
    out = np.zeros((t, x.shape[1]), dtype=x.dtype)
    out[:n] = x
    return out


def fit_zscore(x_train):
    """Per-ROI mean/std fitted on TRAINING data only.

    x_train: (N, T, R) -> mu, sd: (R,).
    """
    flat = x_train.reshape(-1, x_train.shape[-1])
    return flat.mean(0), flat.std(0)


def apply_zscore(x, mu, sd):
    """Standardise with pre-fitted (train-only) per-ROI statistics."""
    return (x - mu) / (sd + 1e-8)


def pearson_fc(x):
    """Static functional connectivity of a (T, R) series -> (R, R) correlation.

    Pearson correlation is invariant to per-ROI affine scaling, so it does not
    depend on whether the series was z-scored first.

    ROIs that are constant over the window have an undefined correlation. Rather
    than propagating NaN they are returned as 0 against every ROI (including
    themselves). The audit found this affects 0.21% of subject-ROI pairs.
    """
    z = np.asarray(x, dtype=np.float64)
    z = z - z.mean(0, keepdims=True)
    sd = z.std(0, keepdims=True)
    varying = (sd > 0).ravel()

    out = np.zeros((z.shape[1], z.shape[1]), dtype=np.float64)
    if varying.any():
        zv = z[:, varying] / sd[:, varying]
        out[np.ix_(varying, varying)] = (zv.T @ zv) / z.shape[0]
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def fc_vector(x):
    """Upper-triangle of the static FC matrix, as a flat feature vector."""
    fc = pearson_fc(x)
    iu = np.triu_indices(fc.shape[0], k=1)
    return fc[iu]
