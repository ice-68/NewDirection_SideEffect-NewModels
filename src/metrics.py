"""Binary classification metrics used by the stage-2 experiments."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
)


def binary_metrics(y_true, y_score, threshold=0.5):
    """Return threshold-free and fixed-threshold binary metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)

    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, y_score))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    return {
        "n": int(len(y_true)),
        "auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float((y_pred == y_true).mean()),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=[0, 1])),
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
