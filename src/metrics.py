"""Binary classification metrics used by the stage-2 experiments."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_score, threshold=0.5):
    """Return threshold-free and fixed-threshold binary metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true and y_score must be one-dimensional")
    if len(y_true) != len(y_score) or len(y_true) == 0:
        raise ValueError("y_true and y_score must have the same non-zero length")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(y_score).all():
        raise ValueError("y_score must contain only finite values")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    y_pred = (y_score >= threshold).astype(np.int64)

    auc = float("nan")
    pr_auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    return {
        "n": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "auc": auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float((y_pred == y_true).mean()),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def _nanstd(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0


def mean_std(values):
    """Mean and sample standard deviation, ignoring NaNs."""
    return _nanmean(values), _nanstd(values)
