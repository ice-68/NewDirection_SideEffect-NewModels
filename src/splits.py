"""Leakage-safe LOSO splits.

Rule R1 from the reference document: the test site never takes part in model
selection. The inner validation set is therefore drawn from the training sites
only, stratified by label.
"""
import numpy as np


def _stratified_split(idx, label, fraction, rng):
    """Split `idx` into (train, val), holding back `fraction` per class."""
    val = []
    for cls in np.unique(label[idx]):
        members = idx[label[idx] == cls]
        members = rng.permutation(members)
        k = max(1, int(round(len(members) * fraction)))
        k = min(k, len(members) - 1)  # always leave at least one for training
        val.extend(int(i) for i in members[:k])
    val = np.array(sorted(val), dtype=int)
    train = np.setdiff1d(idx, val)
    return train, val


def loso_folds(site_idx, label, n_sites, val_fraction, seed):
    """One fold per site; returns a list of dicts with train/val/test indices."""
    site_idx = np.asarray(site_idx)
    label = np.asarray(label)
    rng = np.random.default_rng(seed)
    folds = []
    for s in range(n_sites):
        test_idx = np.where(site_idx == s)[0]
        pool = np.where(site_idx != s)[0]
        train_idx, val_idx = _stratified_split(pool, label, val_fraction, rng)
        folds.append(
            {
                "fold": s,
                "test_site_idx": int(s),
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
            }
        )
    return folds


def check_folds(folds, n_samples, n_sites):
    """Assert fold sanity: disjoint parts, full coverage, every site tested once."""
    seen_test = set()
    for f in folds:
        tr, va, te = set(f["train_idx"]), set(f["val_idx"]), set(f["test_idx"])
        assert tr & va == set(), f"fold {f['fold']}: train/val overlap"
        assert tr & te == set(), f"fold {f['fold']}: train/test overlap"
        assert va & te == set(), f"fold {f['fold']}: val/test overlap"
        assert len(tr | va | te) == n_samples, f"fold {f['fold']}: incomplete cover"
        seen_test.add(f["test_site_idx"])
    assert len(seen_test) == n_sites, "not every site is used as a test site"
    return True
