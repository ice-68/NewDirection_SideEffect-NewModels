"""Stage 1: build the cached tensors and the LOSO folds.

Usage (from the project root):

    python -m src.build_cache
    python -m src.build_cache --atlas AAL --t 78

Outputs, all under ``cache/``:

    <atlas>_T<t>.npz          X (N, T, R) float32 + FC (N, R, R) float32
                              + subject/site/label arrays
    <atlas>_T<t>_folds.json   LOSO folds for every seed
    <atlas>_T<t>_subjects.csv the cleaned table that produced the cache
    stage1_report.md          human-readable build report

Two tensors are cached:

* ``X``  - the sequence tensor, truncated (and zero-padded if needed) to T
  timepoints. This is the temporal-model input.
* ``FC`` - the static Pearson functional-connectivity matrix, computed from the
  **full-length** series (decision 2026-10-14). FC collapses the time axis, so
  its (R, R) shape is identical for every series length; truncating first would
  only discard data. This is the graph-model input, and it is used to build the
  top-k / thresholded adjacency in ``graph.py``.

Per-ROI standardisation is NOT baked in: it must be fitted on training folds only
(rule R2), see ``preprocess.fit_zscore``. FC is computed per subject from that
subject's own series, so it is fold-independent and cannot leak.
"""
import argparse
import json
import time

import numpy as np

from . import config
from .io_timeseries import read_1d
from .manifest import build_manifest
from .preprocess import cut_or_pad, pearson_fc
from .splits import check_folds, loso_folds


def build(atlas, t=None, verbose=True):
    t = t or config.T
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df = build_manifest(atlas, t=t, verbose=verbose)
    good = df[df["valid"]].sort_values(["site", "subject_id"]).reset_index(drop=True)

    n = len(good)
    n_rois = config.N_ROIS[atlas]
    X = np.zeros((n, t, n_rois), dtype=np.float32)
    FC = np.zeros((n, n_rois, n_rois), dtype=np.float32)
    raw_lengths = np.zeros(n, dtype=np.int32)

    t0 = time.time()
    for i, row in enumerate(good.itertuples(index=False)):
        arr = read_1d(row.path)
        raw_lengths[i] = arr.shape[0]
        X[i] = cut_or_pad(arr, t)
        # FC uses the ORIGINAL full-length series, not the T-truncated X.
        FC[i] = pearson_fc(arr)
    if verbose:
        print(f"[cache] loaded {n} series in {time.time() - t0:.1f}s")

    sites = sorted(good["site"].unique())
    site_to_idx = {s: i for i, s in enumerate(sites)}
    site_idx = good["site"].map(site_to_idx).to_numpy(dtype=np.int64)
    label = good["label"].to_numpy(dtype=np.int64)
    subject_id = good["subject_id"].to_numpy(dtype=np.int64)

    tag = f"{atlas.lower()}_T{t}"
    npz_path = config.CACHE_DIR / f"{tag}.npz"
    np.savez_compressed(
        npz_path,
        X=X,
        FC=FC,
        subject_id=subject_id,
        site_idx=site_idx,
        label=label,
        raw_lengths=raw_lengths,
        sites_json=np.array(json.dumps(sites)),
    )

    folds = {}
    for seed in config.SEEDS:
        f = loso_folds(site_idx, label, len(sites), config.INNER_VAL_FRACTION, seed)
        check_folds(f, n, len(sites))
        folds[str(seed)] = f
    folds_path = config.CACHE_DIR / f"{tag}_folds.json"
    folds_path.write_text(
        json.dumps(
            {
                "atlas": atlas,
                "T": t,
                "sites": sites,
                "n_samples": n,
                "inner_val_fraction": config.INNER_VAL_FRACTION,
                "folds": folds,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    subj_path = config.CACHE_DIR / f"{tag}_subjects.csv"
    good.assign(site_idx=site_idx, raw_lengths=raw_lengths).to_csv(
        subj_path, index=False, encoding="utf-8"
    )

    _write_report(atlas, t, X, FC, good, sites, site_idx, label, raw_lengths, folds)
    print(f"[cache] npz      : {npz_path}")
    print(f"[cache] folds    : {folds_path}")
    print(f"[cache] subjects : {subj_path}")
    print(f"[cache] X={tuple(X.shape)} sites={len(sites)} seeds={len(config.SEEDS)}")
    print(f"[cache] FC={tuple(FC.shape)} (full-length series)")
    return X, good, sites, site_idx, label, folds


def _write_report(atlas, t, X, FC, good, sites, site_idx, label, raw_lengths, folds):
    lines = [
        "# Stage 1 build report",
        "",
        f"- atlas: **{atlas}**  (T = {t}, R = {config.N_ROIS[atlas]})",
        f"- samples: **{len(good)}**",
        f"- cache tensor `X`  : {tuple(X.shape)} float32  (temporal input, truncated to T)",
        f"- cache tensor `FC` : {tuple(FC.shape)} float32  (graph input, FULL-length series)",
        "- standardisation is deliberately NOT baked in (fitted per training fold)",
        "",
        "## Site x class",
        "",
        "| site | n | ASD | control | raw length min/median/max |",
        "|---|---|---|---|---|",
    ]
    for i, s in enumerate(sites):
        m = site_idx == i
        lab = label[m]
        rl = raw_lengths[m]
        lines.append(
            f"| {s} | {int(m.sum())} | {int((lab == 1).sum())} | {int((lab == 0).sum())} "
            f"| {int(rl.min())}/{int(np.median(rl))}/{int(rl.max())} |"
        )
    lines += [
        "",
        "## Totals",
        "",
        f"- ASD: {int((label == 1).sum())}  control: {int((label == 0).sum())}",
        f"- sites: {len(sites)}  folds per seed: {len(sites)}  seeds: {len(config.SEEDS)}",
        f"- test evaluations: {len(folds)} seeds x {len(sites)} folds "
        f"= {len(folds) * len(sites)} site-folds",
        "",
        "Each fold: train = all other sites, val = label-stratified slice of those",
        "training sites, test = the held-out site (never used for model selection).",
    ]
    (config.CACHE_DIR / "stage1_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Stage-1 cache builder")
    ap.add_argument("--atlas", default=config.ATLAS, choices=list(config.ATLAS_DIRS))
    ap.add_argument("--t", type=int, default=config.T,
                    help="fixed number of timepoints to keep (from the start)")
    args = ap.parse_args()
    build(args.atlas, t=args.t)


if __name__ == "__main__":
    main()
