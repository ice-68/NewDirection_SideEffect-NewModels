"""Unit and smoke checks for the stage-2 B3 Mamba baseline."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mambapy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.b3 import (  # noqa: E402
    MambaClassifier,
    _load_checkpoint,
    _save_checkpoint,
    run_loso,
    set_seed,
    summarize,
)
from src.splits import loso_folds  # noqa: E402


def test_mamba_forward():
    set_seed(1)
    model = MambaClassifier(
        n_rois=6,
        hidden=8,
        layers=1,
        dropout=0.0,
        d_state=4,
        expand_factor=2,
    )
    logits = model(torch.randn(4, 10, 6))
    assert logits.shape == (4,)


def test_b3_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "fold.json"
    config = {"hidden": 8, "layers": 1, "backend": "mambapy"}
    payload = {
        "config": config,
        "predictions": [{"score": 0.5}],
        "selection": {"best_epoch": 1},
        "history": [{"epoch": 1}],
    }
    _save_checkpoint(path, payload)
    loaded = _load_checkpoint(path, config)
    assert loaded == payload
    assert _load_checkpoint(path, {"hidden": 9}) is None


def test_b3_loso_smoke_and_summary():
    rng = np.random.default_rng(2)
    n_sites, per_site, t, n_rois = 3, 8, 10, 6
    site_idx = np.repeat(np.arange(n_sites), per_site)
    labels = np.tile(np.array([0, 1, 0, 1, 0, 1, 0, 1]), n_sites)
    subject_ids = np.arange(100, 100 + len(labels))
    x = rng.normal(size=(len(labels), t, n_rois)).astype(np.float32)
    x[:, :, 0] += (labels * 2 - 1)[:, None] * 0.8
    folds = {
        str(seed): loso_folds(site_idx, labels, n_sites, 0.25, seed)
        for seed in (0,)
    }

    predictions, selections, histories = run_loso(
        x=x,
        labels=labels,
        subject_ids=subject_ids,
        site_idx=site_idx,
        sites=["A", "B", "C"],
        folds=folds,
        seeds=(0,),
        hidden=8,
        layers=1,
        dropout=0.0,
        d_state=4,
        expand_factor=2,
        batch_size=8,
        max_epochs=1,
        patience=1,
        device="cpu",
        verbose=False,
    )
    assert len(selections) == 3
    assert len(predictions) == len(labels)
    assert len(histories) == 3
    assert (predictions["score"] >= 0).all()
    assert (predictions["score"] <= 1).all()

    site_metrics, seed_summary, site_summary, summary = summarize(
        predictions, selections
    )
    assert len(site_metrics) == 3
    assert len(seed_summary) == 1
    assert len(site_summary) == 3
    assert np.isfinite(summary["macro_auc_mean"])
    assert np.isfinite(summary["macro_kappa_mean"])
