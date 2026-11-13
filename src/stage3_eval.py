"""Post-process frozen Stage 2 outputs for site-level uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .provenance import environment_info, git_commit, sha256_file


MODEL_DIRS = {
    "B0": "stage2_b0_v2",
    "B1": "stage2_b1_inductive",
    "B2": "stage2_b2_v2",
    "B3": "stage2_b3_v1",
}

PROTOCOL_VERSION = "stage3_uncertainty_v1"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20261113
CI_LEVEL = 0.95
REQUIRED_COLUMNS = ("seed", "test_site", "auc")


def _validate_site_seed_matrix(model, group):
    group = group.loc[:, REQUIRED_COLUMNS]
    if group.isna().any().any():
        raise ValueError(f"{model}: site-by-seed metrics contain missing values")
    if not np.isfinite(group["auc"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{model}: AUC values must be finite")
    if group.duplicated(["seed", "test_site"]).any():
        raise ValueError(f"{model}: duplicate seed/site cells")

    sites = sorted(group["test_site"].unique())
    seeds = sorted(group["seed"].unique())
    expected = len(sites) * len(seeds)
    if len(group) != expected:
        raise ValueError(
            f"{model}: expected a complete {len(sites)}x{len(seeds)} "
            f"site/seed matrix, found {len(group)} rows"
        )
    if group.groupby("seed")["test_site"].nunique().ne(len(sites)).any():
        raise ValueError(f"{model}: at least one seed is missing a site")
    if group.groupby("test_site")["seed"].nunique().ne(len(seeds)).any():
        raise ValueError(f"{model}: at least one site is missing a seed")
    return group, sites, seeds


def load_site_metrics(output_root=None):
    """Load and validate the frozen site-by-seed metrics for every model."""
    root = Path(output_root or config.PROJECT_ROOT / "outputs")
    frames = []
    for model, directory in MODEL_DIRS.items():
        path = root / directory / "site_seed_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"{model}: missing {path}")
        frame = pd.read_csv(path)
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{model}: missing columns {sorted(missing)}")
        _validate_site_seed_matrix(model, frame)
        frame = frame.copy()
        frame.insert(0, "model", model)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def site_model_tables(site_metrics):
    """Return long and wide site-by-model AUC summaries."""
    rows = []
    for (model, site), group in site_metrics.groupby(
        ["model", "test_site"], sort=True
    ):
        rows.append(
            {
                "model": model,
                "test_site": site,
                "n_subjects": int(group["n"].iloc[0]) if "n" in group else None,
                "n_seeds": int(group["seed"].nunique()),
                "auc_mean": float(group["auc"].mean()),
                "auc_std": float(group["auc"].std(ddof=1)),
                "worst_seed_auc": float(group["auc"].min()),
            }
        )
    long = pd.DataFrame(rows)
    wide = long.pivot(index="test_site", columns="model", values="auc_mean")
    wide = wide.loc[:, [model for model in MODEL_DIRS if model in wide.columns]]
    wide.index.name = "test_site"
    return long, wide.reset_index()


def site_cluster_bootstrap_ci(
    site_metrics,
    model,
    n_replicates=BOOTSTRAP_REPLICATES,
    seed=BOOTSTRAP_SEED,
    ci_level=CI_LEVEL,
):
    """Bootstrap the seed-averaged site-macro AUC by resampling whole sites."""
    if not 0.0 < float(ci_level) < 1.0:
        raise ValueError("ci_level must be between 0 and 1")
    if int(n_replicates) < 1:
        raise ValueError("n_replicates must be positive")

    group = site_metrics.loc[site_metrics["model"] == model]
    if group.empty:
        raise ValueError(f"unknown model: {model}")
    _, sites, seeds = _validate_site_seed_matrix(model, group)
    matrix = (
        group.pivot(index="seed", columns="test_site", values="auc")
        .reindex(index=seeds, columns=sites)
        .to_numpy(dtype=np.float64)
    )

    observed = float(matrix.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(int(n_replicates), dtype=np.float64)
    n_sites = len(sites)
    for start in range(0, int(n_replicates), 1_000):
        stop = min(start + 1_000, int(n_replicates))
        sampled = rng.integers(0, n_sites, size=(stop - start, n_sites))
        draws[start:stop] = matrix[:, sampled].mean(axis=(0, 2))

    alpha = (1.0 - float(ci_level)) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    return {
        "model": model,
        "estimate": observed,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "bootstrap_mean": float(draws.mean()),
        "bootstrap_std": float(draws.std(ddof=1)),
        "ci_level": float(ci_level),
        "n_sites": int(len(sites)),
        "n_seeds": int(len(seeds)),
        "n_replicates": int(n_replicates),
        "bootstrap_seed": int(seed),
        "resampling_unit": "held_out_site",
    }


def _markdown_table(frame):
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path, intervals, site_long):
    interval_table = intervals[
        ["model", "estimate", "ci_low", "ci_high", "n_sites", "n_seeds"]
    ]
    site_table = site_long.pivot(
        index=["test_site", "n_subjects"],
        columns="model",
        values="auc_mean",
    ).reset_index()
    lines = [
        "# Stage 3 Site-Level Evaluation",
        "",
        "Protocol: `stage3_uncertainty_v1`",
        "",
        "## Site-Macro AUC Uncertainty",
        "",
        _markdown_table(interval_table),
        "",
        "The 95% intervals use 10,000 site-cluster bootstrap replicates. All",
        "five seed results for a sampled site are retained together.",
        "",
        "## Site By Model AUC",
        "",
        _markdown_table(site_table),
        "",
        "Each cell is the mean AUC across seeds 0-4 for that held-out site.",
        "",
        "## Interpretation Boundary",
        "",
        "These intervals describe between-site heterogeneity conditional on the",
        "frozen predictions. They do not include model refitting or subject",
        "resampling within sites. Any comparative or scientific interpretation",
        "requires human review. [待人工复核]",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run(output_root=None, output_dir=None):
    output_root = Path(output_root or config.PROJECT_ROOT / "outputs")
    output_dir = Path(
        output_dir or config.PROJECT_ROOT / "results" / "stage3_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    site_metrics = load_site_metrics(output_root)
    site_long, site_wide = site_model_tables(site_metrics)
    intervals = pd.DataFrame(
        [
            site_cluster_bootstrap_ci(site_metrics, model)
            for model in MODEL_DIRS
        ]
    )

    site_long.to_csv(output_dir / "site_model_auc_long.csv", index=False)
    site_wide.to_csv(output_dir / "site_model_auc_wide.csv", index=False)
    intervals.to_csv(output_dir / "macro_auc_bootstrap_ci.csv", index=False)
    write_report(output_dir / "report.md", intervals, site_long)

    input_hashes = {
        model: sha256_file(output_root / directory / "site_seed_metrics.csv")
        for model, directory in MODEL_DIRS.items()
    }
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": git_commit(),
        "environment": environment_info(),
        "bootstrap": {
            "n_replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "ci_level": CI_LEVEL,
            "resampling_unit": "held_out_site",
            "seed_preservation": "all_seeds_for_sampled_site",
        },
        "models": list(MODEL_DIRS),
        "input_sha256": input_hashes,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return intervals, site_long, site_wide


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap site-macro AUC and build a site-by-model matrix"
    )
    parser.add_argument(
        "--output-root",
        default=str(config.PROJECT_ROOT / "outputs"),
        help="Directory containing the four frozen Stage 2 output directories",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.PROJECT_ROOT / "results" / "stage3_evaluation"),
        help="Directory for Stage 3 tables and report",
    )
    args = parser.parse_args()
    intervals, _, _ = run(args.output_root, args.output_dir)
    for row in intervals.itertuples(index=False):
        print(
            f"[{row.model}] site-macro AUC {row.estimate:.4f} "
            f"(95% CI {row.ci_low:.4f}-{row.ci_high:.4f})"
        )


if __name__ == "__main__":
    main()
