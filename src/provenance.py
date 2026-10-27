"""Reproducibility metadata for Stage 2 experiments."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from . import config


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root=None):
    """Return the current Git commit, or None when Git is unavailable."""
    root = Path(project_root or config.PROJECT_ROOT)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def fold_signature(fold):
    """Hash the indices that define one outer LOSO fold."""
    payload = {
        "fold": int(fold["fold"]),
        "test_site_idx": int(fold["test_site_idx"]),
        "train_idx": [int(i) for i in fold["train_idx"]],
        "val_idx": [int(i) for i in fold["val_idx"]],
        "test_idx": [int(i) for i in fold["test_idx"]],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def environment_info():
    """Record the interpreter and numeric packages used by a run."""
    try:
        import torch

        torch_version = str(torch.__version__)
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = None if torch.version.cuda is None else str(torch.version.cuda)
    except ImportError:
        torch_version = None
        cuda_available = False
        cuda_version = None
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
    }


def run_fingerprint(cache_path, folds_path):
    """Return data and environment metadata shared by all Stage 2 models."""
    return {
        "cache_sha256": sha256_file(cache_path),
        "folds_sha256": sha256_file(folds_path),
        "git_commit": git_commit(),
        "environment": environment_info(),
    }
