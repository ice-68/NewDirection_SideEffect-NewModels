"""Verify completeness and provenance of the Stage 2 formal runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config
from .provenance import sha256_file
from .stage2_summary import METRIC_NAMES


REQUIRED_FILES = (
    "predictions.csv",
    "fold_selection.csv",
    "site_seed_metrics.csv",
    "seed_metrics.csv",
    "site_summary.csv",
    "summary.json",
    "report.md",
)


def verify_output(name, output_dir, protocol_version, require_checkpoints):
    output_dir = Path(output_dir)
    for filename in REQUIRED_FILES:
        path = output_dir / filename
        if not path.exists():
            raise AssertionError(f"{name}: missing {path}")

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output_dir / "predictions.csv")
    selections = pd.read_csv(output_dir / "fold_selection.csv")
    site_metrics = pd.read_csv(output_dir / "site_seed_metrics.csv")

    assert summary["n_folds"] == 100, f"{name}: expected 100 folds"
    assert summary["n_seeds"] == 5, f"{name}: expected 5 seeds"
    assert summary["n_sites"] == 20, f"{name}: expected 20 sites"
    assert len(predictions) == 4385, f"{name}: expected 4385 predictions"
    assert len(selections) == 100, f"{name}: expected 100 selection rows"
    assert len(site_metrics) == 100, f"{name}: expected 100 site/seed rows"
    assert sorted(predictions["seed"].unique()) == [0, 1, 2, 3, 4]
    assert summary["protocol_version"] == protocol_version

    for seed, group in predictions.groupby("seed"):
        assert len(group) == 877, f"{name}: seed {seed} has {len(group)} rows"
        assert group["subject_id"].nunique() == 877
        assert group["test_site"].nunique() == 20

    required_metrics = set(METRIC_NAMES) | {"tn", "fp", "fn", "tp"}
    assert required_metrics <= set(site_metrics.columns)

    provenance = summary["provenance"]
    assert provenance["cache_sha256"] == sha256_file(summary["cache"])
    assert provenance["folds_sha256"] == sha256_file(summary["folds"])
    assert provenance["git_commit"]
    assert provenance["environment"]["python"]
    assert provenance["environment"]["torch"]

    if name == "B1":
        assert (selections["inference_mode"] == "inductive_star").all()
        assert (selections["n_graph_train_nodes"] == selections["n_fit"]).all()
        assert (selections["n_fit"] + selections["n_test"] == 877).all()

    if require_checkpoints:
        checkpoints = list((output_dir / "checkpoints").glob("seed*_fold*.json"))
        assert len(checkpoints) == 100, f"{name}: expected 100 checkpoints"

    print(
        f"PASS {name}: folds={summary['n_folds']}, seeds={summary['n_seeds']}, "
        f"AUC={summary['macro_auc_mean']:.4f}+/-{summary['macro_auc_std']:.4f}, "
        f"PR-AUC={summary['macro_pr_auc_mean']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Verify formal Stage 2 outputs")
    parser.add_argument(
        "--b0", default=str(config.PROJECT_ROOT / "outputs" / "stage2_b0_v2")
    )
    parser.add_argument(
        "--b1",
        default=str(config.PROJECT_ROOT / "outputs" / "stage2_b1_inductive"),
    )
    parser.add_argument(
        "--b2", default=str(config.PROJECT_ROOT / "outputs" / "stage2_b2_v2")
    )
    args = parser.parse_args()

    verify_output("B0", args.b0, "b0_logistic_fc_v2", False)
    verify_output("B1", args.b1, "b1_inductive_star_v1", True)
    verify_output("B2", args.b2, "b2_gru_v2", True)


if __name__ == "__main__":
    main()
