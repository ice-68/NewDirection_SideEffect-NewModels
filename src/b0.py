"""Stage-2 B0: logistic regression on static FC upper-triangle features.

Each LOSO fold is handled independently:

1. Fit a scalar standardiser on the inner training partition only.
2. Choose L2 strength C on the inner validation partition.
3. Refit on training + validation and evaluate the held-out site.

The held-out test site is never used for fitting or model selection.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import config
from .metrics import binary_metrics
from .provenance import run_fingerprint
from .stage2_summary import summarize_predictions


DEFAULT_C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def fc_upper_features(fc):
    """(N, R, R) FC matrices -> (N, R*(R-1)/2) strict upper-triangle features."""
    fc = np.asarray(fc)
    if fc.ndim != 3 or fc.shape[1] != fc.shape[2]:
        raise ValueError(f"expected (N, R, R) FC tensor, got {fc.shape}")
    iu = np.triu_indices(fc.shape[1], k=1)
    return fc[:, iu[0], iu[1]]


def _new_model(c, max_iter, seed):
    return LogisticRegression(
        C=float(c),
        solver="liblinear",
        dual=True,
        max_iter=int(max_iter),
        random_state=int(seed),
    )


def fit_fold(features, labels, train_idx, val_idx, test_idx, c_grid, max_iter, seed):
    """Fit one leakage-safe fold and return test scores plus selection details."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    c_grid = tuple(float(c) for c in c_grid)
    if not c_grid:
        raise ValueError("c_grid must not be empty")

    train_scaler = StandardScaler().fit(features[train_idx])
    z_train = train_scaler.transform(features[train_idx])
    z_val = train_scaler.transform(features[val_idx])

    val_aucs = []
    for c in c_grid:
        model = _new_model(c, max_iter=max_iter, seed=seed)
        model.fit(z_train, labels[train_idx])
        val_score = model.predict_proba(z_val)[:, 1]
        val_aucs.append(binary_metrics(labels[val_idx], val_score)["auc"])

    finite = np.isfinite(val_aucs)
    if not finite.any():
        raise RuntimeError("no candidate produced a finite validation AUC")
    best_pos = int(np.nanargmax(np.asarray(val_aucs, dtype=np.float64)))
    best_c = c_grid[best_pos]

    fit_idx = np.concatenate([train_idx, val_idx])
    final_scaler = StandardScaler().fit(features[fit_idx])
    z_fit = final_scaler.transform(features[fit_idx])
    z_test = final_scaler.transform(features[test_idx])
    model = _new_model(best_c, max_iter=max_iter, seed=seed)
    model.fit(z_fit, labels[fit_idx])
    score = model.predict_proba(z_test)[:, 1]

    details = {
        "selected_c": float(best_c),
        "val_auc": float(val_aucs[best_pos]),
        "candidate_c": json.dumps(c_grid),
        "candidate_val_auc": json.dumps(
            [None if not np.isfinite(v) else float(v) for v in val_aucs]
        ),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_fit": int(len(fit_idx)),
        "n_test": int(len(test_idx)),
    }
    return np.asarray(score, dtype=np.float64), details


def run_loso(
    features,
    labels,
    subject_ids,
    site_idx,
    sites,
    folds,
    seeds,
    c_grid=DEFAULT_C_GRID,
    fixed_c=None,
    max_iter=2000,
    threshold=0.5,
    verbose=True,
):
    """Run B0 over precomputed subject-level features and LOSO folds."""
    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    subject_ids = np.asarray(subject_ids)
    site_idx = np.asarray(site_idx, dtype=np.int64)
    sites = list(sites)
    prediction_rows = []
    selection_rows = []

    run_grid = (float(fixed_c),) if fixed_c is not None else tuple(c_grid)
    started = time.time()
    total = sum(len(folds[str(seed)]) for seed in seeds)
    completed = 0

    for seed in seeds:
        for fold in folds[str(seed)]:
            fold_started = time.time()
            test_site_idx = int(fold["test_site_idx"])
            test_site = sites[test_site_idx]
            test_idx = np.asarray(fold["test_idx"], dtype=np.int64)
            scores, details = fit_fold(
                features,
                labels,
                fold["train_idx"],
                fold["val_idx"],
                test_idx,
                run_grid,
                max_iter=max_iter,
                seed=seed,
            )

            selection_rows.append(
                {
                    "seed": int(seed),
                    "fold": int(fold["fold"]),
                    "test_site": test_site,
                    **details,
                    "wall_seconds": float(time.time() - fold_started),
                }
            )
            for idx, score in zip(test_idx, scores):
                prediction_rows.append(
                    {
                        "seed": int(seed),
                        "fold": int(fold["fold"]),
                        "test_site": test_site,
                        "subject_id": int(subject_ids[idx]),
                        "site": sites[int(site_idx[idx])],
                        "label": int(labels[idx]),
                        "score": float(score),
                        "prediction": int(score >= threshold),
                    }
                )

            completed += 1
            if verbose and (completed == total or completed % 10 == 0):
                elapsed = time.time() - started
                print(
                    f"[B0] {completed}/{total} folds | seed={seed} "
                    f"site={test_site} C={details['selected_c']:g} "
                    f"val_auc={details['val_auc']:.4f} | {elapsed:.1f}s",
                    flush=True,
                )

    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    return predictions, selections


def summarize(predictions, selections, threshold=0.5):
    """Build seed-level, site-level and overall summaries."""
    site_metrics, seed_summary, site_summary, summary = summarize_predictions(
        predictions, selections, threshold=threshold
    )
    selected_counts = selections["selected_c"].value_counts().sort_index()
    summary["selected_c_counts"] = {
        f"{float(c):g}": int(n) for c, n in selected_counts.items()
    }
    return site_metrics, seed_summary, site_summary, summary


def _write_report(path, summary, site_summary, seed_summary):
    metric_labels = {
        "accuracy": "ACC",
        "auc": "AUC",
        "pr_auc": "PR-AUC",
        "sensitivity": "SEN",
        "specificity": "SPE",
        "balanced_accuracy": "Balanced ACC",
        "f1": "F1",
        "kappa": "Cohen's kappa",
    }
    lines = [
        "# Stage 2 B0 report",
        "",
        "Model: logistic regression on static FC upper-triangle features.",
        "",
        "Protocol: per-fold training-only standardisation; C selected on the",
        "inner validation split from training sites; final fit on train + val;",
        "held-out site used for testing only.",
        "",
        "## Overall",
        "",
    ]
    for name, label in metric_labels.items():
        lines.append(
            f"- site-macro {label}: **{summary[f'macro_{name}_mean']:.4f} "
            f"± {summary[f'macro_{name}_std']:.4f}**"
        )
    lines += [
        f"- mean worst-site AUC: {summary['worst_site_auc_mean']:.4f} "
        f"± {summary['worst_site_auc_std']:.4f}",
        f"- folds: {summary['n_folds']} ({summary['n_seeds']} seeds x "
        f"{summary['n_sites']} sites)",
        "",
        "## Per-seed",
        "",
        "| seed | ACC | AUC | PR-AUC | SEN | SPE | Balanced ACC | F1 | Kappa |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_summary.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.macro_accuracy:.4f} | "
            f"{row.macro_auc:.4f} | {row.macro_pr_auc:.4f} | "
            f"{row.macro_sensitivity:.4f} | {row.macro_specificity:.4f} | "
            f"{row.macro_balanced_accuracy:.4f} | {row.macro_f1:.4f} | "
            f"{row.macro_kappa:.4f} |"
        )

    lines += [
        "",
        "## Per-site",
        "",
        "| site | n | ACC | AUC | PR-AUC | SEN | SPE | Balanced ACC | F1 | Kappa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in site_summary.itertuples(index=False):
        lines.append(
            f"| {row.test_site} | {row.n_subjects} | {row.accuracy_mean:.4f} | "
            f"{row.auc_mean:.4f} | {row.pr_auc_mean:.4f} | "
            f"{row.sensitivity_mean:.4f} | {row.specificity_mean:.4f} | "
            f"{row.balanced_accuracy_mean:.4f} | {row.f1_mean:.4f} | "
            f"{row.kappa_mean:.4f} |"
        )
    lines += [
        "",
        "CMU has only five subjects and is reported for completeness; its",
        "fold-level estimates are substantially less stable than the other sites.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_from_cache(cache_path, folds_path, output_dir, seeds, c_grid, fixed_c, max_iter):
    cache_path = Path(cache_path)
    folds_path = Path(folds_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(cache_path)
    fold_payload = json.loads(folds_path.read_text(encoding="utf-8"))
    sites = list(fold_payload["sites"])
    cached_sites = json.loads(str(data["sites_json"]))
    if cached_sites != sites:
        raise ValueError("cache and fold files disagree on site ordering")

    provenance = run_fingerprint(cache_path, folds_path)
    features = fc_upper_features(data["FC"])
    predictions, selections = run_loso(
        features=features,
        labels=data["label"],
        subject_ids=data["subject_id"],
        site_idx=data["site_idx"],
        sites=sites,
        folds=fold_payload["folds"],
        seeds=seeds,
        c_grid=c_grid,
        fixed_c=fixed_c,
        max_iter=max_iter,
    )
    site_metrics, seed_summary, site_summary, summary = summarize(predictions, selections)
    summary.update(
        {
            "model": "B0_logistic_regression_fc_upper",
            "cache": str(cache_path),
            "folds": str(folds_path),
            "seeds": [int(s) for s in seeds],
            "c_grid": [float(c) for c in (c_grid if fixed_c is None else (fixed_c,))],
            "max_iter": int(max_iter),
            "binary_threshold": 0.5,
            "provenance": provenance,
        }
    )

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    selections.to_csv(output_dir / "fold_selection.csv", index=False)
    site_metrics.to_csv(output_dir / "site_seed_metrics.csv", index=False)
    seed_summary.to_csv(output_dir / "seed_metrics.csv", index=False)
    site_summary.to_csv(output_dir / "site_summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(output_dir / "report.md", summary, site_summary, seed_summary)

    print(
        f"[B0] site-macro AUC {summary['macro_auc_mean']:.4f} "
        f"± {summary['macro_auc_std']:.4f}",
        flush=True,
    )
    print(f"[B0] results: {output_dir}", flush=True)
    return summary


def main():
    tag = f"{config.ATLAS.lower()}_T{config.T}"
    ap = argparse.ArgumentParser(description="Stage-2 B0 FC logistic-regression baseline")
    ap.add_argument("--cache", default=str(config.CACHE_DIR / f"{tag}.npz"))
    ap.add_argument("--folds", default=str(config.CACHE_DIR / f"{tag}_folds.json"))
    ap.add_argument("--out", default=str(config.PROJECT_ROOT / "outputs" / "stage2_b0_v2"))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    ap.add_argument("--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    ap.add_argument("--fixed-c", type=float, default=None)
    ap.add_argument("--max-iter", type=int, default=2000)
    args = ap.parse_args()

    run_from_cache(
        cache_path=args.cache,
        folds_path=args.folds,
        output_dir=args.out,
        seeds=args.seeds,
        c_grid=tuple(args.c_grid),
        fixed_c=args.fixed_c,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
