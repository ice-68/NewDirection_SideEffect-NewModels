"""Unit checks for the stage-2 baseline protocol."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.b0 import fc_upper_features, run_loso, summarize  # noqa: E402
from src.metrics import binary_metrics  # noqa: E402
from src.splits import loso_folds  # noqa: E402


def test_fc_upper_features():
    fc = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = fc_upper_features(fc)
    assert out.shape == (2, 1)
    assert np.array_equal(out[:, 0], np.array([1.0, 5.0], dtype=np.float32))


def test_binary_metrics():
    out = binary_metrics([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9])
    assert out["n"] == 4
    assert np.isclose(out["auc"], 0.75)
    assert np.isclose(out["balanced_accuracy"], 0.5)
    assert np.isclose(out["sensitivity"], 0.5)
    assert np.isclose(out["specificity"], 0.5)
    assert np.isclose(out["pr_auc"], 5.0 / 6.0)
    assert np.isclose(out["f1"], 0.5)
    assert np.isclose(out["kappa"], 0.0)
    assert (out["tn"], out["fp"], out["fn"], out["tp"]) == (1, 1, 1, 1)


def test_b0_loso_smoke_and_summary():
    rng = np.random.default_rng(7)
    n_sites, per_site, n_features = 3, 8, 6
    site_idx = np.repeat(np.arange(n_sites), per_site)
    labels = np.tile(np.array([0, 1, 0, 1, 0, 1, 0, 1]), n_sites)
    subject_ids = np.arange(100, 100 + len(labels))
    signal = labels * 0.8
    features = rng.normal(size=(len(labels), n_features))
    features[:, 0] += signal
    folds = {
        str(seed): loso_folds(site_idx, labels, n_sites, 0.25, seed)
        for seed in (0, 1)
    }

    predictions, selections = run_loso(
        features,
        labels,
        subject_ids,
        site_idx,
        ["A", "B", "C"],
        folds,
        seeds=(0, 1),
        c_grid=(0.1, 1.0),
        max_iter=500,
        verbose=False,
    )
    assert len(selections) == 6
    assert len(predictions) == len(labels) * 2
    assert predictions["prediction"].isin([0, 1]).all()
    assert (predictions["score"] >= 0).all() and (predictions["score"] <= 1).all()

    site_metrics, seed_summary, site_summary, summary = summarize(
        predictions, selections
    )
    assert len(site_metrics) == 6
    assert len(seed_summary) == 2
    assert len(site_summary) == 3
    assert np.isfinite(summary["macro_auc_mean"])
    assert np.isfinite(summary["macro_pr_auc_mean"])
    assert np.isfinite(summary["macro_f1_mean"])


def main():
    tests = [
        test_fc_upper_features,
        test_binary_metrics,
        test_b0_loso_smoke_and_summary,
    ]
    print("stage-2 checks")
    print("-" * 62)
    for test in tests:
        test()
        print(f"  PASS  {test.__name__}")
    print("-" * 62)
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
