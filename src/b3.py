"""Stage-2 B3: pure temporal Mamba baseline.

The model receives only the cached ROI time series. The implementation uses
the pure PyTorch ``mambapy`` backend because the official CUDA extension is not
available for this Windows/Python environment. No connectivity matrix, site
metadata, or phenotype is used. Per-ROI standardisation is fitted on each
fold's training sites only.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from . import config
from .b2 import (
    _checkpoint_path,
    _load_checkpoint,
    _make_dataset,
    _make_loader,
    _predict,
    _save_checkpoint,
    set_seed,
)
from .preprocess import apply_zscore, fit_zscore
from .provenance import fold_signature, run_fingerprint
from .stage2_summary import summarize_predictions

try:
    from mambapy.mamba import Mamba, MambaConfig
except ImportError as exc:  # pragma: no cover - exercised via test skip
    raise ImportError(
        "B3 requires mambapy==1.2.0. Install it with "
        "`python -m pip install --no-deps mambapy==1.2.0`."
    ) from exc


DEFAULT_HIDDEN = 64
DEFAULT_LAYERS = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_D_STATE = 16
DEFAULT_EXPAND_FACTOR = 2
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_EPOCHS = 80
DEFAULT_PATIENCE = 10
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
PROTOCOL_VERSION = "b3_mamba_v1"
MAMBAPY_VERSION = version("mambapy")


def configure_runtime():
    """Use stable CUDA settings for the pure-PyTorch Mamba scan."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False


class MambaClassifier(nn.Module):
    """Unidirectional Mamba encoder with mean-over-time pooling."""

    def __init__(
        self,
        n_rois,
        hidden=DEFAULT_HIDDEN,
        layers=DEFAULT_LAYERS,
        dropout=DEFAULT_DROPOUT,
        d_state=DEFAULT_D_STATE,
        expand_factor=DEFAULT_EXPAND_FACTOR,
        pscan=True,
    ):
        super().__init__()
        if hidden <= 0 or layers <= 0:
            raise ValueError("hidden and layers must be positive")
        if d_state <= 0 or expand_factor <= 0:
            raise ValueError("d_state and expand_factor must be positive")
        mamba_config = MambaConfig(
            d_model=int(hidden),
            n_layers=int(layers),
            d_state=int(d_state),
            expand_factor=int(expand_factor),
            pscan=bool(pscan),
            use_cuda=False,
        )
        self.input_proj = nn.Linear(int(n_rois), int(hidden))
        self.encoder = Mamba(mamba_config)
        self.norm = nn.LayerNorm(int(hidden))
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(hidden), 1)

    def forward(self, x):
        x = self.input_proj(x)
        sequence = self.encoder(x)
        pooled = sequence.mean(dim=1)
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        return self.head(pooled).view(-1)


def train_one_fold(
    x,
    labels,
    train_idx,
    val_idx,
    test_idx,
    hidden=DEFAULT_HIDDEN,
    layers=DEFAULT_LAYERS,
    dropout=DEFAULT_DROPOUT,
    d_state=DEFAULT_D_STATE,
    expand_factor=DEFAULT_EXPAND_FACTOR,
    pscan=True,
    batch_size=DEFAULT_BATCH_SIZE,
    max_epochs=DEFAULT_MAX_EPOCHS,
    patience=DEFAULT_PATIENCE,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    threshold=0.5,
    device="cpu",
    seed=0,
    verbose=False,
):
    """Train one leakage-safe temporal fold and return test scores."""
    set_seed(seed)
    device = torch.device(device)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.float32)

    mu, sd = fit_zscore(np.asarray(x[train_idx], dtype=np.float32))
    z = apply_zscore(np.asarray(x, dtype=np.float32), mu, sd)
    dataset = _make_dataset(z, labels)
    train_loader = _make_loader(dataset, train_idx, batch_size, True, seed)
    val_loader = _make_loader(dataset, val_idx, batch_size, False, seed)
    test_loader = _make_loader(dataset, test_idx, batch_size, False, seed)

    model = MambaClassifier(
        n_rois=z.shape[-1],
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        d_state=d_state,
        expand_factor=expand_factor,
        pscan=pscan,
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
        losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_scores, val_labels = _predict(model, val_loader, device)
        if len(np.unique(val_labels)) == 2:
            val_auc = float(roc_auc_score(val_labels, val_scores))
        else:
            val_auc = float("nan")
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(
            {"epoch": epoch, "train_loss": mean_loss, "val_auc": val_auc}
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
                f"    epoch={epoch:03d} loss={mean_loss:.4f} "
                f"val_auc={val_auc:.4f} best={best_auc:.4f}",
                flush=True,
            )
        if stale_epochs >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    scores, _ = _predict(model, test_loader, device)
    details = {
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best_auc),
        "epochs_run": int(len(history)),
        "elapsed_seconds": float(time.time() - started),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
    }
    return scores, details, history


def run_loso(
    x,
    labels,
    subject_ids,
    site_idx,
    sites,
    folds,
    seeds,
    hidden=DEFAULT_HIDDEN,
    layers=DEFAULT_LAYERS,
    dropout=DEFAULT_DROPOUT,
    d_state=DEFAULT_D_STATE,
    expand_factor=DEFAULT_EXPAND_FACTOR,
    pscan=True,
    batch_size=DEFAULT_BATCH_SIZE,
    max_epochs=DEFAULT_MAX_EPOCHS,
    patience=DEFAULT_PATIENCE,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    threshold=0.5,
    device=None,
    max_folds=None,
    checkpoint_dir=None,
    run_metadata=None,
    verbose=True,
):
    """Run B3 over cached time series and leakage-safe LOSO folds."""
    x = np.asarray(x, dtype=np.float32)
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
    checkpoint_config = {
        "protocol_version": PROTOCOL_VERSION,
        "backend": "mambapy",
        "backend_version": MAMBAPY_VERSION,
        "hidden": int(hidden),
        "layers": int(layers),
        "dropout": float(dropout),
        "d_state": int(d_state),
        "expand_factor": int(expand_factor),
        "pscan": bool(pscan),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
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
                **checkpoint_config,
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
                    x=x,
                    labels=labels,
                    train_idx=fold["train_idx"],
                    val_idx=fold["val_idx"],
                    test_idx=test_idx,
                    hidden=hidden,
                    layers=layers,
                    dropout=dropout,
                    d_state=d_state,
                    expand_factor=expand_factor,
                    pscan=pscan,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    lr=lr,
                    weight_decay=weight_decay,
                    threshold=threshold,
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
                    f"[B3] {completed}/{total} folds | seed={seed} "
                    f"site={test_site} epoch={fold_selection['best_epoch']} "
                    f"val_auc={fold_selection['best_val_auc']:.4f} "
                    f"{'resumed' if resumed else 'trained'} | {elapsed:.1f}s",
                    flush=True,
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    histories = pd.DataFrame(history_rows)
    return predictions, selections, histories


def summarize(predictions, selections, threshold=0.5):
    """Delegate to the shared stage-2 metric aggregation."""
    return summarize_predictions(predictions, selections, threshold=threshold)


def _write_report(
    path,
    summary,
    site_summary,
    seed_summary,
    hidden,
    layers,
    d_state,
    expand_factor,
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
        "# Stage 2 B3 report",
        "",
        "Model: unidirectional pure-temporal Mamba over CC200 ROI time series.",
        "",
        f"Backend: mambapy {MAMBAPY_VERSION} (pure PyTorch parallel scan).",
        f"Input: cached X (T=78, R=200). hidden={int(hidden)}, "
        f"layers={int(layers)}, d_state={int(d_state)}, "
        f"expand_factor={int(expand_factor)}.",
        "Only temporal information is used; site and phenotype are excluded.",
        "",
        "Protocol: per-fold ROI standardisation is fitted on training sites only;",
        "the inner validation split comes from training sites only; the held-out",
        "site is used for final prediction only. The encoder is causal before",
        "mean-over-time pooling.",
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
    output_dir,
    seeds,
    hidden=DEFAULT_HIDDEN,
    layers=DEFAULT_LAYERS,
    dropout=DEFAULT_DROPOUT,
    d_state=DEFAULT_D_STATE,
    expand_factor=DEFAULT_EXPAND_FACTOR,
    pscan=True,
    batch_size=DEFAULT_BATCH_SIZE,
    max_epochs=DEFAULT_MAX_EPOCHS,
    patience=DEFAULT_PATIENCE,
    lr=DEFAULT_LR,
    weight_decay=DEFAULT_WEIGHT_DECAY,
    device=None,
    max_folds=None,
):
    configure_runtime()
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
    predictions, selections, histories = run_loso(
        x=data["X"],
        labels=data["label"],
        subject_ids=data["subject_id"],
        site_idx=data["site_idx"],
        sites=sites,
        folds=fold_payload["folds"],
        seeds=seeds,
        hidden=hidden,
        layers=layers,
        dropout=dropout,
        d_state=d_state,
        expand_factor=expand_factor,
        pscan=pscan,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
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
            "model": "B3_unidirectional_mamba",
            "protocol_version": PROTOCOL_VERSION,
            "cache": str(cache_path),
            "folds": str(folds_path),
            "seeds": [int(s) for s in seeds],
            "hidden": int(hidden),
            "layers": int(layers),
            "dropout": float(dropout),
            "d_state": int(d_state),
            "expand_factor": int(expand_factor),
            "pscan": bool(pscan),
            "backend": "mambapy",
            "backend_version": MAMBAPY_VERSION,
            "direction": "unidirectional",
            "batch_size": int(batch_size),
            "max_epochs": int(max_epochs),
            "patience": int(patience),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
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
        hidden=hidden,
        layers=layers,
        d_state=d_state,
        expand_factor=expand_factor,
    )
    print(
        f"[B3] site-macro AUC {summary['macro_auc_mean']:.4f} "
        f"± {summary['macro_auc_std']:.4f}; "
        f"ACC {summary['macro_accuracy_mean']:.4f}; "
        f"kappa {summary['macro_kappa_mean']:.4f}",
        flush=True,
    )
    print(f"[B3] results: {output_dir}", flush=True)
    return summary


def main():
    tag = f"{config.ATLAS.lower()}_T{config.T}"
    ap = argparse.ArgumentParser(description="Stage-2 B3 pure temporal Mamba")
    ap.add_argument("--cache", default=str(config.CACHE_DIR / f"{tag}.npz"))
    ap.add_argument("--folds", default=str(config.CACHE_DIR / f"{tag}_folds.json"))
    ap.add_argument(
        "--out",
        default=str(config.PROJECT_ROOT / "outputs" / "stage2_b3_v1"),
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    ap.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    ap.add_argument("--d-state", type=int, default=DEFAULT_D_STATE)
    ap.add_argument("--expand-factor", type=int, default=DEFAULT_EXPAND_FACTOR)
    ap.add_argument("--no-pscan", action="store_true")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = ap.parse_args()

    run_from_cache(
        cache_path=args.cache,
        folds_path=args.folds,
        output_dir=args.out,
        seeds=tuple(args.seeds),
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        d_state=args.d_state,
        expand_factor=args.expand_factor,
        pscan=not args.no_pscan,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        max_folds=args.max_folds,
    )


if __name__ == "__main__":
    main()
