"""Checks for Stage 2 reproducibility metadata."""
import hashlib

from src.provenance import fold_signature, sha256_file


def test_sha256_file(tmp_path):
    path = tmp_path / "payload.txt"
    path.write_bytes(b"stage2")
    expected = hashlib.sha256(b"stage2").hexdigest()
    assert sha256_file(path) == expected


def test_fold_signature_tracks_all_partitions():
    fold = {
        "fold": 0,
        "test_site_idx": 2,
        "train_idx": [0, 1],
        "val_idx": [2],
        "test_idx": [3, 4],
    }
    changed = {**fold, "test_idx": [3]}
    assert fold_signature(fold) == fold_signature(dict(fold))
    assert fold_signature(fold) != fold_signature(changed)
