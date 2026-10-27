"""Unit and smoke checks for the GCA-style stage-2 B1 baseline."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.b1 import (  # noqa: E402
    GCA,
    INFERENCE_MODE,
    _load_checkpoint,
    _save_checkpoint,
    build_fit_adjacency,
    build_population_adjacency,
    fc_upper_features,
    normalize_adjacency,
    run_loso,
    score_test_nodes_inductive,
    select_ridge_features,
    set_seed,
    summarize,
)
from src.splits import loso_folds  # noqa: E402


def test_fc_upper_features():
    fc = np.zeros((2, 3, 3), dtype=np.float32)
    fc[0, 0, 1] = 0.5
    fc[1, 0, 2] = 0.8
    features = fc_upper_features(fc)
    assert features.shape == (2, 3)
    assert np.isclose(features[0, 0], np.arctanh(0.5))
    assert np.isclose(features[1, 1], np.arctanh(0.8))


def test_select_ridge_features():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(40, 20))
    labels = np.tile(np.array([0, 1]), 20)
    features[:, 0] += labels * 1.5
    scaler, selected = select_ridge_features(features, labels, n_features=8, seed=1)
    assert selected.shape == (8,)
    assert 0 in selected
    assert np.isfinite(scaler.transform(features)).all()


def test_population_adjacency():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(6, 4)).astype(np.float32)
    gender = np.array([1, 1, 2, 2, 1, 2])
    age = np.array([10, 11, 20, 21, 30, 31], dtype=np.float32)
    adjacency = build_population_adjacency(features, gender, age, sigma=0.5, age_tau=5)
    normalized = normalize_adjacency(adjacency)
    assert adjacency.shape == (6, 6)
    assert np.allclose(adjacency, adjacency.T)
    assert np.allclose(np.diag(adjacency), 0.0)
    assert np.allclose(normalized, normalized.T)
    assert (normalized >= 0).all()


def test_fit_adjacency_excludes_test_nodes():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(8, 4)).astype(np.float32)
    gender = np.tile(np.array([1, 2]), 4)
    age = np.arange(8, dtype=np.float32)
    fit_idx = np.array([0, 1, 2, 3, 4, 5])
    adjacency = build_fit_adjacency(features, gender, age, fit_idx, sigma=0.5, age_tau=5)
    assert adjacency.shape == (len(fit_idx), len(fit_idx))


def test_inductive_test_scoring_uses_one_test_node_at_a_time():
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shapes = []

        def forward(self, x, adjacency):
            self.shapes.append((x.shape[0], adjacency.shape[0]))
            return torch.zeros(x.shape[0])

    features = np.arange(30, dtype=np.float32).reshape(6, 5)
    gender = np.array([1, 1, 2, 2, 1, 2])
    age = np.arange(6, dtype=np.float32)
    model = RecordingModel()
    scores = score_test_nodes_inductive(
        model=model,
        node_features=features,
        gender=gender,
        age=age,
        fit_idx=np.array([0, 1, 2, 3]),
        test_idx=np.array([4, 5]),
        sigma=0.5,
        age_tau=5.0,
        device="cpu",
    )
    assert scores.shape == (2,)
    assert model.shapes == [(5, 5), (5, 5)]


def test_b1_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "fold.json"
    config = {"hidden": 8, "fold_signature": "abc"}
    payload = {
        "config": config,
        "predictions": [{"score": 0.5}],
        "selection": {"best_epoch": 1},
        "history": [{"epoch": 1}],
    }
    _save_checkpoint(path, payload)
    assert _load_checkpoint(path, config) == payload
    assert _load_checkpoint(path, {"hidden": 9}) is None


def test_gca_forward():
    set_seed(3)
    model = GCA(in_dim=8, hidden=8, layers=3, alpha=0.1, beta=0.3, dropout=0.0)
    x = torch.randn(5, 8)
    adjacency = torch.eye(5) * 0.7 + 0.05
    logits = model(x, adjacency)
    assert logits.shape == (5,)


def test_b1_loso_smoke_and_summary():
    rng = np.random.default_rng(4)
    n_sites, per_site, n_rois = 3, 8, 6
    site_idx = np.repeat(np.arange(n_sites), per_site)
    labels = np.tile(np.array([0, 1, 0, 1, 0, 1, 0, 1]), n_sites)
    subject_ids = np.arange(100, 100 + len(labels))
    gender = np.tile(np.array([1, 1, 1, 2, 2, 2, 1, 2]), n_sites)
    age = np.tile(np.array([10, 11, 20, 21, 30, 31, 40, 41], dtype=np.float32), n_sites)

    fc = rng.normal(scale=0.3, size=(len(labels), n_rois, n_rois))
    fc = np.triu(fc, k=1)
    fc = fc + fc.transpose(0, 2, 1)
    fc += (labels * 2 - 1)[:, None, None] * 0.4
    np.clip(fc, -1.0, 1.0, out=fc)
    features = fc_upper_features(fc)
    folds = {
        str(seed): loso_folds(site_idx, labels, n_sites, 0.25, seed)
        for seed in (0,)
    }

    predictions, selections, histories = run_loso(
        features=features,
        labels=labels,
        subject_ids=subject_ids,
        site_idx=site_idx,
        gender=gender,
        age=age,
        sites=["A", "B", "C"],
        folds=folds,
        seeds=(0,),
        n_features=8,
        hidden=8,
        layers=2,
        max_epochs=2,
        patience=2,
        device="cpu",
        verbose=False,
    )
    assert len(selections) == 3
    assert len(predictions) == len(labels)
    assert len(histories) == 6
    assert (selections["inference_mode"] == INFERENCE_MODE).all()
    assert (selections["n_graph_train_nodes"] == selections["n_fit"]).all()
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
