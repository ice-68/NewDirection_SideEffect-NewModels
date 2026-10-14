"""Integrity and leakage checks for the stage-1 pipeline.

Run from the project root:

    python tests/test_pipeline.py

Exits with code 1 if any check fails. These checks are the executable form of
rules R1 (test site never used for model selection) and R2 (no leakage).
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.graph import edge_index, graph_stats, topk_adjacency  # noqa: E402
from src.io_timeseries import read_1d  # noqa: E402
from src.preprocess import apply_zscore, cut_or_pad, fit_zscore, pearson_fc  # noqa: E402
from src.splits import check_folds  # noqa: E402

TAG = f"{config.ATLAS.lower()}_T{config.T}"
NPZ = config.CACHE_DIR / f"{TAG}.npz"
FOLDS = config.CACHE_DIR / f"{TAG}_folds.json"
SUBJECTS = config.CACHE_DIR / f"{TAG}_subjects.csv"

_CACHE = {}


def cache():
    if "npz" not in _CACHE:
        _CACHE["npz"] = np.load(NPZ)
        _CACHE["folds"] = json.loads(FOLDS.read_text(encoding="utf-8"))
    return _CACHE["npz"], _CACHE["folds"]


# ---------------------------------------------------------------- checks ----
def check_cache_exists():
    assert NPZ.exists(), f"missing {NPZ}"
    assert FOLDS.exists(), f"missing {FOLDS}"
    assert SUBJECTS.exists(), f"missing {SUBJECTS}"


def check_cache_shape():
    d, j = cache()
    x = d["X"]
    assert x.ndim == 3, x.shape
    n, t, r = x.shape
    assert t == config.T, f"T mismatch: {t} != {config.T}"
    assert r == config.N_ROIS[config.ATLAS], f"n_rois mismatch: {r}"
    assert n == j["n_samples"], f"n_samples mismatch: {n} != {j['n_samples']}"
    assert np.isfinite(x).all(), "non-finite values in X"
    assert list(np.unique(d["label"])) == [0, 1], "labels must be 0/1"


def check_all_sites_present():
    d, j = cache()
    sites = np.asarray(j["sites"])
    assert len(sites) == 20, f"expected 20 sites, got {len(sites)}"
    seen = np.unique(d["site_idx"])
    assert len(seen) == len(sites), "some site has no samples"
    # every site must contain both classes, otherwise AUC is undefined
    for i, s in enumerate(sites):
        lab = d["label"][d["site_idx"] == i]
        assert set(np.unique(lab)) == {0, 1}, f"site {s} is single-class: {np.unique(lab)}"


def check_no_padding_needed():
    d, _ = cache()
    rl = d["raw_lengths"]
    assert (rl >= config.T).all(), f"padding required for {int((rl < config.T).sum())} subjects"


def check_cache_matches_source():
    d, j = cache()
    order = json.loads(str(d["sites_json"]))
    n = len(d["X"])
    sample = list(range(0, n, max(1, n // 6)))[:6]
    checked = 0
    for i in sample:
        arr = read_1d(_path_for(i, d, order))
        expect = cut_or_pad(arr, config.T)
        assert np.array_equal(d["X"][i], expect), f"cache mismatch at row {i}"
        checked += 1
    assert checked >= 4


def _path_for(i, d, order):
    """Reconstruct the .1D path of cache row `i` from the manifest table."""
    import pandas as pd

    if "subj" not in _CACHE:
        _CACHE["subj"] = pd.read_csv(SUBJECTS).sort_values(
            ["site", "subject_id"]).reset_index(drop=True)
    row = _CACHE["subj"].iloc[i]
    return row["path"]


def check_fold_integrity():
    d, j = cache()
    n = j["n_samples"]
    for seed, folds in j["folds"].items():
        check_folds(folds, n, len(j["sites"]))
    assert len(j["folds"]) == len(config.SEEDS)


def check_test_site_never_in_train():
    """R1: for every fold, no train/val sample may come from the test site."""
    d, j = cache()
    site_idx = d["site_idx"]
    for seed, folds in j["folds"].items():
        for f in folds:
            t = f["test_site_idx"]
            for part in ("train_idx", "val_idx"):
                bad = [i for i in f[part] if site_idx[i] == t]
                assert not bad, f"seed {seed} fold {f['fold']}: {part} contains test-site samples"


def check_zscore_fit_ignores_test():
    """R2: train statistics must not depend on the test fold at all."""
    d, j = cache()
    x = d["X"]
    f = j["folds"][str(config.SEEDS[0])][0]
    mu1, sd1 = fit_zscore(x[f["train_idx"]])
    x2 = x.copy()
    x2[f["test_idx"]] = x2[f["test_idx"]] + 1000.0   # corrupt the held-out site
    x2[f["val_idx"]] = x2[f["val_idx"]] - 500.0
    mu2, sd2 = fit_zscore(x2[f["train_idx"]])
    assert np.array_equal(mu1, mu2), "train mean changed when test fold changed"
    assert np.array_equal(sd1, sd2), "train std changed when test fold changed"


def check_zscore_applies_cleanly():
    d, j = cache()
    x = d["X"]
    f = j["folds"][str(config.SEEDS[0])][0]
    mu, sd = fit_zscore(x[f["train_idx"]])
    z = apply_zscore(x[f["train_idx"]], mu, sd)
    assert abs(float(z.reshape(-1, z.shape[-1]).mean(0).max())) < 1e-4, "train mean not ~0"
    assert z.shape == x[f["train_idx"]].shape


def check_fc_is_deterministic_and_local():
    """FC must be deterministic, symmetric, NaN-free, and handle flat ROIs."""
    d, _ = cache()
    x = d["X"]
    a = pearson_fc(x[0])
    b = pearson_fc(x[0])
    assert np.array_equal(a, b), "pearson_fc is not deterministic"
    assert np.isfinite(a).all(), "FC must not contain NaN/Inf"
    assert np.allclose(a, a.T, atol=1e-4), "FC must be symmetric"
    varying = x[0].std(0) > 0
    diag = np.diag(a)
    assert np.allclose(diag[varying], 1.0, atol=1e-3), "varying ROI must self-correlate ~1"
    assert np.allclose(diag[~varying], 0.0), "flat ROI must be 0, not NaN"
    assert np.allclose(a[~varying, :], 0.0), "flat ROI must be 0 against every ROI"


def check_graph_properties():
    d, _ = cache()
    x = d["X"]
    fc = pearson_fc(x[0])
    k = 10
    adj = topk_adjacency(fc, k)
    st = graph_stats(adj)
    assert st["symmetric"], "adjacency is not symmetric (undirected graph broken)"
    assert st["zero_diagonal"], "adjacency diagonal must be zero"
    # union of two directed kNN graphs: at most k*R, at least k edges
    assert st["n_edges"] <= 2 * k * adj.shape[0], st
    assert st["n_edges"] >= k, st
    ei, ew = edge_index(adj)
    assert ei.shape[1] == st["n_edges"] == ew.shape[0]
    assert ei.max() < adj.shape[0]


def check_fc_cache_shape():
    """The graph input FC must be cached, square, finite and symmetric."""
    d, j = cache()
    assert "FC" in d, "cache is missing the full-length FC tensor"
    fc = d["FC"]
    r = config.N_ROIS[config.ATLAS]
    assert fc.shape == (j["n_samples"], r, r), fc.shape
    assert fc.dtype == np.float32, fc.dtype
    assert np.isfinite(fc).all(), "FC cache contains NaN/Inf"
    n = fc.shape[0]
    for i in range(0, n, max(1, n // 8)):
        assert np.allclose(fc[i], fc[i].T, atol=1e-4), f"FC row {i} not symmetric"


def check_fc_cache_matches_full_source():
    """FC must equal pearson_fc of the ORIGINAL series, read back from the .1D."""
    d, j = cache()
    fc = d["FC"]
    order = json.loads(str(d["sites_json"]))
    n = len(d["X"])
    sample = list(range(0, n, max(1, n // 6)))[:6]
    checked = 0
    for i in sample:
        arr = read_1d(_path_for(i, d, order))
        assert arr.shape[0] == d["raw_lengths"][i], f"raw length mismatch at row {i}"
        assert np.allclose(fc[i], pearson_fc(arr), atol=1e-4), (
            f"cached FC row {i} does not match the full-length series")
        checked += 1
    assert checked >= 4


def check_fc_uses_full_length():
    """At least one subject must show FC(full series) != FC(T-truncated series)."""
    d, _ = cache()
    fc, x, rl = d["FC"], d["X"], d["raw_lengths"]
    longer = [i for i in range(len(rl)) if rl[i] > config.T]
    assert longer, "no subject has a series longer than T"
    differs = any(
        not np.allclose(fc[i], pearson_fc(x[i]), atol=1e-3) for i in longer[:50]
    )
    assert differs, "full-length FC identical to truncated FC - truncation still applied"


CHECKS = [
    ("cache files exist", check_cache_exists),
    ("cache shape / dtype / finiteness", check_cache_shape),
    ("20 sites, each with both classes", check_all_sites_present),
    ("no zero padding needed", check_no_padding_needed),
    ("cached tensor matches source .1D", check_cache_matches_source),
    ("LOSO folds disjoint and complete", check_fold_integrity),
    ("R1: test site never in train/val", check_test_site_never_in_train),
    ("R2: z-score fit ignores test fold", check_zscore_fit_ignores_test),
    ("z-score applies cleanly", check_zscore_applies_cleanly),
    ("FC deterministic / symmetric", check_fc_is_deterministic_and_local),
    ("top-k graph is valid undirected", check_graph_properties),
    ("FC cache shape / symmetry", check_fc_cache_shape),
    ("FC cache matches full-length source", check_fc_cache_matches_full_source),
    ("FC uses full length, not truncated", check_fc_uses_full_length),
]


def main():
    print(f"stage-1 checks  (atlas={config.ATLAS}, T={config.T}, cache={NPZ.name})")
    print("-" * 62)
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}")
            print("        " + f"{type(exc).__name__}: {exc}")
            if "--trace" in sys.argv:
                traceback.print_exc()
    print("-" * 62)
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

