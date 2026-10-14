"""Temporal tensor construction: full sequences and optional sliding windows.

Decision 2026-10-13: the cache already fixes the series length to T = 78, so the
default input for the sequence models is the whole (T, R) series. Sliding windows
are provided for models that prefer shorter segments; they only reshape a single
subject's data and can never cross the train/test split.
"""
import numpy as np


def full_sequence(x):
    """Identity: the cached (T, R) series already is the sequence input."""
    return x


def sliding_windows(x, window, stride=1):
    """(T, R) -> (n_windows, window, R)."""
    t = x.shape[0]
    if window > t:
        raise ValueError(f"window {window} > series length {t}")
    starts = range(0, t - window + 1, stride)
    return np.stack([x[s:s + window] for s in starts]).astype(np.float32)


def window_mean_features(x, window, stride=1):
    """Mean over each window -> (n_windows, R)."""
    return sliding_windows(x, window, stride).mean(axis=1)
