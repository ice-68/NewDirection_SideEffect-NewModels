"""Shared metric summaries for stage-2 experiments."""
from __future__ import annotations

import pandas as pd

from .metrics import binary_metrics, mean_std


METRIC_NAMES = (
    "accuracy",
    "auc",
    "pr_auc",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "f1",
    "kappa",
)


def summarize_predictions(predictions, selections, threshold=0.5):
    """Aggregate predictions by site, seed and overall experiment."""
    metric_rows = []
    for (seed, site), group in predictions.groupby(["seed", "test_site"], sort=True):
        metric_rows.append(
            {
                "seed": int(seed),
                "test_site": site,
                **binary_metrics(
                    group["label"].to_numpy(),
                    group["score"].to_numpy(),
                    threshold=threshold,
                ),
            }
        )
    site_metrics = pd.DataFrame(metric_rows)

    seed_rows = []
    for seed, group in site_metrics.groupby("seed", sort=True):
        row = {"seed": int(seed), "n_sites": int(len(group))}
        for name in METRIC_NAMES:
            row[f"macro_{name}"] = float(group[name].mean())
        row["worst_site_auc"] = float(group["auc"].min())
        row["std_site_auc"] = float(group["auc"].std(ddof=1))
        seed_rows.append(row)
    seed_summary = pd.DataFrame(seed_rows)

    site_rows = []
    for site, group in site_metrics.groupby("test_site", sort=True):
        row = {
            "test_site": site,
            "n_subjects": int(group["n"].iloc[0]),
            "n_seeds": int(len(group)),
        }
        for name in METRIC_NAMES:
            value_mean, value_std = mean_std(group[name])
            row[f"{name}_mean"] = value_mean
            row[f"{name}_std"] = value_std
        row["worst_seed_auc"] = float(group["auc"].min())
        site_rows.append(row)
    site_summary = pd.DataFrame(site_rows)

    worst_site_auc_mean, worst_site_auc_std = mean_std(seed_summary["worst_site_auc"])
    summary = {
        "n_predictions": int(len(predictions)),
        "n_folds": int(len(selections)),
        "n_seeds": int(predictions["seed"].nunique()),
        "n_sites": int(predictions["test_site"].nunique()),
        "worst_site_auc_mean": worst_site_auc_mean,
        "worst_site_auc_std": worst_site_auc_std,
        "primary_metric": "site_macro_auc",
        "binary_threshold": float(threshold),
        "metric_names": list(METRIC_NAMES),
    }
    for column in ("best_epoch", "epochs_run", "wall_seconds"):
        if column in selections:
            if column == "wall_seconds":
                summary["total_wall_seconds"] = float(selections[column].sum())
            else:
                summary[f"mean_{column}"] = float(selections[column].mean())
    for name in METRIC_NAMES:
        value_mean, value_std = mean_std(seed_summary[f"macro_{name}"])
        summary[f"macro_{name}_mean"] = value_mean
        summary[f"macro_{name}_std"] = value_std
    return site_metrics, seed_summary, site_summary, summary
