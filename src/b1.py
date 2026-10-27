"""Stage-2 B1: GCA-style population graph baseline.

Replication of the ABIDE part of "Classification of Developmental and Brain
Disorders via Graph Convolutional Aggregation" (GCA / AN-GCN), adapted to this
project's leakage-safe LOSO protocol.

Each subject is a node. Imaging features are Fisher-transformed upper-triangle
FC values, reduced with a ridge-based feature ranking fitted on the inner
training split only. Edges combine imaging-feature similarity with age and
gender. Acquisition site is deliberately NOT used as an edge feature, because
our research question is about generalisation to unseen sites.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from . import config
from .provenance import fold_signature, run_fingerprint
from .stage2_summary import summarize_predictions


DEFAULT_N_FEATURES = 128
DEFAULT_HIDDEN = 128
DEFAULT_LAYERS = 10
DEFAULT_ALPHA = 0.1
DEFAULT_BETA = 0.3
DEFAULT_SIGMA = 0.25
DEFAULT_AGE_TAU = 5.0
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MAX_EPOCHS = 150
DEFAULT_PATIENCE = 10
INFERENCE_MODE = "inductive_star"
PROTOCOL_VERSION = "b1_inductive_star_v1"


def set_seed(seed):
    """Seed Python, NumPy and PyTorch for one fold."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_path(checkpoint_dir, seed, fold):
    return Path(checkpoint_dir) / f"seed{int(seed)}_fold{int(fold):02d}.json"


def _save_checkpoint(path, payload):
    """Atomically write one fold so an interrupted run can resume safely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _load_checkpoint(path, expected_config):
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("config") != expected_config:
        return None
    required = {"predictions", "selection", "history"}
    if not required <= set(payload):
        return None
    return payload


def fisher_transform(fc):
    """Convert [-1, 1] Pearson correlations to approximately normal values."""
    fc = np.asarray(fc, dtype=np.float64)
    return np.arctanh(np.clip(fc, -1.0 + 1e-7, 1.0 - 1e-7)).astype(np.float32)


def fc_upper_features(fc):
    """(N, R, R) FC -> (N, R*(R-1)/2) Fisher-transformed upper triangle."""
    fc = np.asarray(fc, dtype=np.float32)
    if fc.ndim != 3 or fc.shape[1] != fc.shape[2]:
        raise ValueError(f"expected (N, R, R) FC tensor, got {fc.shape}")
    iu = np.triu_indices(fc.shape[1], k=1)
    return fisher_transform(fc[:, iu[0], iu[1]])


def load_phenotype(path, subject_ids):
    """Return aligned age and gender arrays for the cached subject order."""
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["SUB_ID"]).copy()
    table = df.set_index(df["SUB_ID"].astype(int))

    if not table.index.is_unique:
        raise ValueError("phenotype file contains duplicate SUB_ID values")
    missing = sorted(set(subject_ids.tolist()) - set(table.index.tolist()))
    if missing:
        raise ValueError(f"phenotype is missing {len(missing)} cached subjects")

    age = np.empty(len(subject_ids), dtype=np.float32)
    gender = np.empty(len(subject_ids), dtype=np.int8)
    for i, sid in enumerate(subject_ids):
        row = table.loc[sid]
        if pd.isna(row["AGE_AT_SCAN"]) or pd.isna(row["SEX"]):
            raise ValueError(f"missing age or sex for subject {int(sid)}")
        sex = int(row["SEX"])
        if sex not in (1, 2):
            raise ValueError(f"unexpected SEX code {sex} for subject {int(sid)}")
        age[i] = float(row["AGE_AT_SCAN"])
        gender[i] = sex
    if not np.isfinite(age).all():
        raise ValueError("age contains non-finite values")
    return age, gender


def select_ridge_features(features, labels, n_features, seed):
    """Select ridge-discriminative features on the training split only.

    GCA uses recursive feature elimination with a ridge classifier. A full RFE
    sweep from ~20k FC edges is impractically slow inside every LOSO fold, so
    we keep the same inductive bias by ranking absolute ridge coefficients.
    """
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    n_features = int(n_features)
    if n_features <= 0 or n_features > features.shape[1]:
        raise ValueError(f"n_features must be in [1, {features.shape[1]}]")

    scaler = StandardScaler().fit(features)
    z = scaler.transform(features)
    ridge = RidgeClassifier(
        alpha=1.0,
        random_state=int(seed),
        solver="sparse_cg",
        max_iter=2000,
    ).fit(z, labels)
    importance = np.abs(ridge.coef_.ravel())
    if len(importance) == 0:
        importance = np.ones(features.shape[1], dtype=np.float64)
    selected = np.argsort(importance)[::-1][:n_features]
    return scaler, np.sort(selected).astype(np.int64)


def build_population_adjacency(node_features, gender, age, sigma, age_tau):
    """Build the GCA population adjacency from imaging and non-imaging data."""
    node_features = np.asarray(node_features, dtype=np.float32)
    n = node_features.shape[0]
    gender = np.asarray(gender)
    age = np.asarray(age, dtype=np.float32)
    if gender.shape[0] != n or age.shape[0] != n:
        raise ValueError("gender and age must align with node_features")

    centered = node_features - node_features.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    denom = np.maximum(np.outer(norms, norms), 1e-12)
    cosine = (centered @ centered.T) / denom
    rho = np.clip(1.0 - cosine, 0.0, 2.0)
    kernel = np.exp(-(rho**2) / (2.0 * float(sigma) ** 2))

    same_gender = (gender[:, None] == gender[None, :]).astype(np.float32)
    close_age = (
        np.abs(age[:, None] - age[None, :]) < float(age_tau)
    ).astype(np.float32)
    adjacency = kernel * (same_gender + close_age)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency.astype(np.float32)


def normalize_adjacency(adjacency):
    """Symmetric normalisation with self-loops, as in GCA/GCN."""
    adjacency = np.asarray(adjacency, dtype=np.float32)
    np.fill_diagonal(adjacency, 0.0)
    with_loops = adjacency + np.eye(adjacency.shape[0], dtype=np.float32)
    degree = with_loops.sum(axis=1)
    inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    return (inv_sqrt[:, None] * with_loops * inv_sqrt[None, :]).astype(np.float32)


def build_fit_adjacency(node_features, gender, age, fit_idx, sigma, age_tau):
    """Build the training-side graph without any held-out-site nodes."""
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    if len(np.unique(fit_idx)) != len(fit_idx):
        raise ValueError("fit_idx must not contain duplicate rows")
    return build_population_adjacency(
        node_features[fit_idx],
        gender[fit_idx],
        age[fit_idx],
        sigma=sigma,
        age_tau=age_tau,
    )


def score_test_nodes_inductive(
    model,
    node_features,
    gender,
    age,
    fit_idx,
    test_idx,
    sigma,
    age_tau,
    device,
):
    """Score each test node in its own graph attached to the fixed fit graph."""
    fit_idx = np.asarray(fit_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    if set(fit_idx.tolist()) & set(test_idx.tolist()):
        raise ValueError("fit_idx and test_idx must not overlap")

    model.eval()
    scores = []
    with torch.no_grad():
        for test_node in test_idx:
            graph_idx = np.concatenate([fit_idx, np.asarray([test_node])])
            adjacency = build_population_adjacency(
                node_features[graph_idx],
                gender[graph_idx],
                age[graph_idx],
                sigma=sigma,
                age_tau=age_tau,
            )
            normalized = torch.from_numpy(normalize_adjacency(adjacency)).to(device)
            node_tensor = torch.from_numpy(node_features[graph_idx]).to(device)
            logits = model(node_tensor, normalized)
            scores.append(float(torch.sigmoid(logits[-1]).cpu()))
    return np.asarray(scores, dtype=np.float64)


class GCA(nn.Module):
    """Aggregator-normalisation GCN with skip connections and identity maps."""

    def __init__(
        self,
        in_dim,
        hidden=DEFAULT_HIDDEN,
        layers=DEFAULT_LAYERS,
        alpha=DEFAULT_ALPHA,
        beta=DEFAULT_BETA,
        dropout=0.0,
    ):
        super().__init__()
        if in_dim != hidden:
            raise ValueError("GCA currently requires in_dim == hidden")
        self.layers = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(int(layers))]
        )
        self.head = nn.Linear(hidden, 1)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dropout = float(dropout)
        for linear in self.layers:
            nn.init.normal_(linear.weight, std=0.01)
            with torch.no_grad():
                linear.weight.add_(torch.eye(hidden))

    def forward(self, x, normalized_adjacency):
        initial = x
        hidden = x
        for linear in self.layers:
            diffused = normalized_adjacency @ hidden
            hidden = (
                (1.0 - self.alpha) * diffused
                + self.beta * linear(diffused)
                + self.alpha * initial
                + self.beta * linear(initial)
            )
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.head(hidden).view(-1)


def train_one_fold(
    features_all,
    labels,
    gender,
    age,
    train_idx,
    val_idx,
    test_idx,
    n_features=DEFAULT_N_FEATURES,
    hidden=DEFAULT_HIDDEN,
    layers=DEFAULT_LAYERS,
    alpha=DEFAULT_ALPHA,
    beta=DEFAULT_BETA,
    sigma=DEFAULT_SIGMA,
    age_tau=DEFAULT_AGE_TAU,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    max_epochs=DEFAULT_MAX_EPOCHS,
    patience=DEFAULT_PATIENCE,
    device="cpu",
    seed=0,
    verbose=False,
):
    """Train one transductive population-graph fold and score the test nodes."""
    set_seed(seed)
    device = torch.device(device)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float32)

    scaler, selected = select_ridge_features(
        features_all[train_idx],
        labels[train_idx],
        n_features=n_features,
        seed=seed,
    )
    z_all = scaler.transform(features_all)
    node_features = z_all[:, selected].astype(np.float32)
    fit_idx = np.concatenate([train_idx, val_idx])
    if len(np.unique(fit_idx)) != len(fit_idx):
        raise ValueError("train_idx and val_idx must be disjoint")
    if set(fit_idx.tolist()) & set(test_idx.tolist()):
        raise ValueError("test_idx must not overlap train_idx or val_idx")

    adjacency = build_fit_adjacency(
        node_features,
        gender,
        age,
        fit_idx,
        sigma=sigma,
        age_tau=age_tau,
    )
    normalized = torch.from_numpy(normalize_adjacency(adjacency)).to(device)
    node_tensor = torch.from_numpy(node_features[fit_idx]).to(device)
    fit_labels = torch.from_numpy(labels[fit_idx]).to(device)
    fit_position = np.full(len(labels), -1, dtype=np.int64)
    fit_position[fit_idx] = np.arange(len(fit_idx), dtype=np.int64)
    train_pos = fit_position[train_idx]
    val_pos = fit_position[val_idx]
    if (train_pos < 0).any() or (val_pos < 0).any():
        raise RuntimeError("failed to map train/validation rows into fit graph")

    model = GCA(
        in_dim=n_features,
        hidden=hidden,
        layers=layers,
        alpha=alpha,
        beta=beta,
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = float("-inf")
    best_epoch = 0
    best_state = None
    stale_epochs = 0
    history = []
    started = time.time()

    for epoch in range(1, int(max_epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(node_tensor, normalized)
        loss = criterion(logits[train_pos], fit_labels[train_pos])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        val_score = torch.sigmoid(logits.detach()[val_pos]).cpu().numpy()
        val_label = labels[val_idx]
        if len(np.unique(val_label)) == 2:
            val_auc = float(roc_auc_score(val_label, val_score))
        else:
            val_auc = float("nan")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.detach().cpu()),
                "val_auc": val_auc,
            }
        )

        improved = np.isfinite(val_auc) and val_auc > best_auc + 1e-6
        if improved:
            best_auc = float(val_auc)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if verbose:
            print(
                f"    epoch={epoch:03d} loss={float(loss):.4f} "
                f"val_auc={val_auc:.4f} best={best_auc:.4f}",
                flush=True,
            )
        if stale_epochs >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_scores = score_test_nodes_inductive(
        model=model,
        node_features=node_features,
        gender=gender,
        age=age,
        fit_idx=fit_idx,
        test_idx=test_idx,
        sigma=sigma,
        age_tau=age_tau,
        device=device,
    )
    details = {
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_auc),
        "epochs_run": int(len(history)),
        "elapsed_seconds": float(time.time() - started),
        "inference_mode": INFERENCE_MODE,
        "n_features": int(len(selected)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_fit": int(len(fit_idx)),
        "n_graph_train_nodes": int(adjacency.shape[0]),
        "n_test": int(len(test_idx)),
        "edge_density": float((adjacency > 0).sum()) / max(
            adjacency.shape[0] * (adjacency.shape[0] - 1), 1
        ),
    }
    return test_scores, details, history


def run_loso(
    features,
    labels,
    subject_ids,
    site_idx,
    gender,
    age,
    sites,
    folds,
    seeds,
    n_features=DEFAULT_N_FEATURES,
    hidden=DEFAULT_HIDDEN,
    layers=DEFAULT_LAYERS,
    alpha=DEFAULT_ALPHA,
    beta=DEFAULT_BETA,
    sigma=DEFAULT_SIGMA,
    age_tau=DEFAULT_AGE_TAU,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    max_epochs=DEFAULT_MAX_EPOCHS,
    patience=DEFAULT_PATIENCE,
    threshold=0.5,
    device=None,
    max_folds=None,
    checkpoint_dir=None,
    run_metadata=None,
    verbose=True,
):
    """Run strict inductive GCA classification over leakage-safe LOSO folds."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    subject_ids = np.asarray(subject_ids, dtype=np.int64)
    site_idx = np.asarray(site_idx, dtype=np.int64)
    sites = list(sites)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)

    prediction_rows = []
    selection_rows = []
    history_rows = []
    started = time.time()
    total = min(
        int(max_folds) if max_folds is not None else 10**9,
        sum(len(folds[str(seed)]) for seed in seeds),
    )
    completed = 0
    base_checkpoint_config = {
        "protocol_version": PROTOCOL_VERSION,
        "inference_mode": INFERENCE_MODE,
        "n_features": int(n_features),
        "hidden": int(hidden),
        "layers": int(layers),
        "alpha": float(alpha),
        "beta": float(beta),
        "sigma": float(sigma),
        "age_tau": float(age_tau),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "threshold": float(threshold),
        "run_metadata": run_metadata,
    }

    for seed in seeds:
        for fold in folds[str(seed)]:
            if completed >= total:
                break
            fold_started = time.time()
            test_site_idx = int(fold["test_site_idx"])
            test_site = sites[test_site_idx]
            test_idx = np.asarray(fold["test_idx"], dtype=np.int64)
            expected_checkpoint_config = {
                **base_checkpoint_config,
                "fold_signature": fold_signature(fold),
                "test_site": test_site,
            }
            checkpoint = None
            if checkpoint_dir is not None:
                checkpoint = _load_checkpoint(
                    _checkpoint_path(checkpoint_dir, seed, fold["fold"]),
                    expected_checkpoint_config,
                )

            if checkpoint is not None:
                fold_predictions = checkpoint["predictions"]
                fold_selection = checkpoint["selection"]
                fold_history = checkpoint["history"]
                resumed = True
            else:
                scores, details, history = train_one_fold(
                    features_all=features,
                    labels=labels,
                    gender=gender,
                    age=age,
                    train_idx=fold["train_idx"],
                    val_idx=fold["val_idx"],
                    test_idx=test_idx,
                    n_features=n_features,
                    hidden=hidden,
                    layers=layers,
                    alpha=alpha,
                    beta=beta,
                    sigma=sigma,
                    age_tau=age_tau,
                    lr=lr,
                    weight_decay=weight_decay,
                    max_epochs=max_epochs,
                    patience=patience,
                    device=device,
                    seed=int(seed) * 1000 + int(fold["fold"]),
                )

                fold_predictions = []
                for idx, score in zip(test_idx, scores):
                    fold_predictions.append(
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
                fold_selection = {
                    "seed": int(seed),
                    "fold": int(fold["fold"]),
                    "test_site": test_site,
                    **details,
                    "wall_seconds": float(time.time() - fold_started),
                }
                fold_history = [
                    {
                        "seed": int(seed),
                        "fold": int(fold["fold"]),
                        "test_site": test_site,
                        **row,
                    }
                    for row in history
                ]
                if checkpoint_dir is not None:
                    _save_checkpoint(
                        _checkpoint_path(checkpoint_dir, seed, fold["fold"]),
                        {
                            "config": expected_checkpoint_config,
                            "predictions": fold_predictions,
                            "selection": fold_selection,
                            "history": fold_history,
                        },
                    )
                resumed = False

            prediction_rows.extend(fold_predictions)
            selection_rows.append(fold_selection)
            history_rows.extend(fold_history)

            completed += 1
            if verbose and (completed == total or completed % 5 == 0):
                elapsed = time.time() - started
                print(
                    f"[B1] {completed}/{total} folds | seed={seed} "
                    f"site={test_site} epoch={fold_selection['best_epoch']} "
                    f"val_auc={fold_selection['best_val_auc']:.4f} "
                    f"{'resumed' if resumed else 'trained'} | {elapsed:.1f}s",
                    flush=True,
                )

    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    histories = pd.DataFrame(history_rows)
    return predictions, selections, histories


def summarize(predictions, selections, threshold=0.5):
    """Build seed/site summaries with the common stage-2 metric set."""
    site_metrics, seed_summary, site_summary, summary = summarize_predictions(
        predictions, selections, threshold=threshold
    )
    summary["mean_edge_density"] = float(selections["edge_density"].mean())
    return site_metrics, seed_summary, site_summary, summary


def _write_report(
    path,
    summary,
    site_summary,
    seed_summary,
    n_features,
    hidden,
    layers,
    alpha,
    beta,
):
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
        "# Stage 2 B1 report",
        "",
        "Model: GCA / AN-GCN replication on subject-level population graphs.",
        "",
        "Graph: subjects as nodes; edges combine imaging-feature similarity with",
        "age and gender. Acquisition site is intentionally excluded from edges.",
        f"Features: Fisher-transformed FC upper triangle, ridge-ranked to "
        f"{int(n_features)}. hidden={int(hidden)}, layers={int(layers)}, "
        f"alpha={float(alpha):g}, beta={float(beta):g}.",
        "",
        "Protocol: strict inductive evaluation. The held-out site is absent",
        "from the training and validation graph. Each test subject is scored",
        "in a separate graph with no edges to other held-out subjects.",
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
        f"- folds: {summary['n_folds']} "
        f"({summary['n_seeds']} seeds x {summary['n_sites']} sites)",
        f"- mean best epoch: {summary['mean_best_epoch']:.1f}",
        f"- mean edge density: {summary['mean_edge_density']:.4f}",
        "",
        "## Per-seed",
        "",
        "| seed | ACC | AUC | PR-AUC | SEN | SPE | Balanced ACC | F1 | Kappa | worst-site AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in seed_summary.itertuples(index=False):
        lines.append(
            f"| {row.seed} | {row.macro_accuracy:.4f} | {row.macro_auc:.4f} | "
            f"{row.macro_pr_auc:.4f} | "
            f"{row.macro_sensitivity:.4f} | {row.macro_specificity:.4f} | "
            f"{row.macro_balanced_accuracy:.4f} | {row.macro_f1:.4f} | "
            f"{row.macro_kappa:.4f} | {row.worst_site_auc:.4f} |"
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
        "fold-level estimates are less stable than the other sites.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_from_cache(
    cache_path,
    folds_path,
    pheno_path,
    output_dir,
    seeds,
    n_features=DEFAULT_N_FEATURES,
    hidden=DEFAULT_HIDDEN,
    layers=DEFAULT_LAYERS,
    alpha=DEFAULT_ALPHA,
    beta=DEFAULT_BETA,
    sigma=DEFAULT_SIGMA,
    age_tau=DEFAULT_AGE_TAU,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    max_epochs=DEFAULT_MAX_EPOCHS,
    patience=DEFAULT_PATIENCE,
    device=None,
    max_folds=None,
):
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

    run_metadata = run_fingerprint(cache_path, folds_path)
    features = fc_upper_features(data["FC"])
    age, gender = load_phenotype(pheno_path, data["subject_id"])
    predictions, selections, histories = run_loso(
        features=features,
        labels=data["label"],
        subject_ids=data["subject_id"],
        site_idx=data["site_idx"],
        gender=gender,
        age=age,
        sites=sites,
        folds=fold_payload["folds"],
        seeds=seeds,
        n_features=n_features,
        hidden=hidden,
        layers=layers,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        age_tau=age_tau,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        max_folds=max_folds,
        checkpoint_dir=output_dir / "checkpoints",
        run_metadata=run_metadata,
    )
    site_metrics, seed_summary, site_summary, summary = summarize(
        predictions, selections
    )
    summary.update(
        {
            "model": "B1_gca_population_graph",
            "protocol_version": PROTOCOL_VERSION,
            "inference_mode": INFERENCE_MODE,
            "cache": str(cache_path),
            "folds": str(folds_path),
            "pheno": str(pheno_path),
            "seeds": [int(s) for s in seeds],
            "n_features": int(n_features),
            "hidden": int(hidden),
            "layers": int(layers),
            "alpha": float(alpha),
            "beta": float(beta),
            "sigma": float(sigma),
            "age_tau": float(age_tau),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "device": str(device or ("cuda" if torch.cuda.is_available() else "cpu")),
            "max_folds": None if max_folds is None else int(max_folds),
            "binary_threshold": 0.5,
            "provenance": run_metadata,
        }
    )

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    selections.to_csv(output_dir / "fold_selection.csv", index=False)
    histories.to_csv(output_dir / "training_history.csv", index=False)
    site_metrics.to_csv(output_dir / "site_seed_metrics.csv", index=False)
    seed_summary.to_csv(output_dir / "seed_metrics.csv", index=False)
    site_summary.to_csv(output_dir / "site_summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_report(
        output_dir / "report.md",
        summary,
        site_summary,
        seed_summary,
        n_features=n_features,
        hidden=hidden,
        layers=layers,
        alpha=alpha,
        beta=beta,
    )
    print(
        f"[B1] site-macro AUC {summary['macro_auc_mean']:.4f} "
        f"± {summary['macro_auc_std']:.4f}; "
        f"ACC {summary['macro_accuracy_mean']:.4f}; "
        f"kappa {summary['macro_kappa_mean']:.4f}",
        flush=True,
    )
    print(f"[B1] results: {output_dir}", flush=True)
    return summary


def main():
    tag = f"{config.ATLAS.lower()}_T{config.T}"
    ap = argparse.ArgumentParser(description="Stage-2 B1 GCA population-graph baseline")
    ap.add_argument("--cache", default=str(config.CACHE_DIR / f"{tag}.npz"))
    ap.add_argument("--folds", default=str(config.CACHE_DIR / f"{tag}_folds.json"))
    ap.add_argument("--pheno", default=str(config.PHENO_CSV))
    ap.add_argument(
        "--out",
        default=str(config.PROJECT_ROOT / "outputs" / "stage2_b1_inductive"),
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    ap.add_argument("--n-features", type=int, default=DEFAULT_N_FEATURES)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    ap.add_argument("--age-tau", type=float, default=DEFAULT_AGE_TAU)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = ap.parse_args()

    run_from_cache(
        cache_path=args.cache,
        folds_path=args.folds,
        pheno_path=args.pheno,
        output_dir=args.out,
        seeds=tuple(args.seeds),
        n_features=args.n_features,
        hidden=args.hidden,
        layers=args.layers,
        alpha=args.alpha,
        beta=args.beta,
        sigma=args.sigma,
        age_tau=args.age_tau,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        max_folds=args.max_folds,
    )


if __name__ == "__main__":
    main()
