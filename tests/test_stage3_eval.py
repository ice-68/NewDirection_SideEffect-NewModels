"""Checks for Stage 3 site-cluster uncertainty summaries."""
import numpy as np
import pandas as pd
import pytest

from src.stage3_eval import site_cluster_bootstrap_ci, site_model_tables


def _site_metrics():
    rows = []
    auc_by_site = {"A": (0.2, 0.3), "B": (0.6, 0.7), "C": (0.9, 1.0)}
    for model, shift in (("B0", 0.0), ("B1", -0.1)):
        for site, values in auc_by_site.items():
            for seed, value in enumerate(values):
                rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "test_site": site,
                        "n": 10 + seed,
                        "auc": value + shift,
                    }
                )
    return pd.DataFrame(rows)


def test_site_cluster_bootstrap_is_deterministic():
    metrics = _site_metrics()
    first = site_cluster_bootstrap_ci(metrics, "B0", n_replicates=200, seed=17)
    second = site_cluster_bootstrap_ci(metrics, "B0", n_replicates=200, seed=17)
    assert first == second
    assert first["estimate"] == pytest.approx(0.6166666667)
    assert first["ci_low"] <= first["estimate"] <= first["ci_high"]
    assert first["n_sites"] == 3
    assert first["n_seeds"] == 2


def test_bootstrap_changes_with_seed():
    metrics = _site_metrics()
    first = site_cluster_bootstrap_ci(metrics, "B0", n_replicates=200, seed=1)
    second = site_cluster_bootstrap_ci(metrics, "B0", n_replicates=200, seed=2)
    assert first["ci_low"] != second["ci_low"]
    assert first["bootstrap_mean"] != second["bootstrap_mean"]


def test_site_model_tables_cover_design():
    metrics = _site_metrics()
    long, wide = site_model_tables(metrics)
    assert len(long) == 6
    assert wide.shape == (3, 3)
    assert list(wide.columns) == ["test_site", "B0", "B1"]
    b0 = long.loc[long["model"] == "B0"].set_index("test_site")
    assert b0.loc["A", "auc_mean"] == pytest.approx(0.25)
    assert b0.loc["C", "auc_mean"] == pytest.approx(0.95)


def test_incomplete_site_seed_matrix_is_rejected():
    metrics = _site_metrics()
    incomplete = metrics.drop(index=0)
    with pytest.raises(ValueError, match="complete"):
        site_cluster_bootstrap_ci(incomplete, "B0", n_replicates=10, seed=1)


def test_non_finite_auc_is_rejected():
    metrics = _site_metrics()
    metrics.loc[0, "auc"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        site_cluster_bootstrap_ci(metrics, "B0", n_replicates=10, seed=1)
