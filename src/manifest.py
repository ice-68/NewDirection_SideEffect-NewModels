"""Build the subject manifest: id, site, label, per-atlas file and dimensions."""
import re

import pandas as pd

from . import config
from .io_timeseries import scan_1d

SUBJECT_RE = re.compile(r"_(\d+)_rois_")


def subject_id_from_name(name):
    m = SUBJECT_RE.search(name)
    return int(m.group(1)) if m else None


def load_labels():
    """SUB_ID -> label (1 = ASD, 0 = control)."""
    df = pd.read_csv(config.LABELS_CSV)
    df = df.dropna(subset=["SUB_ID", "label"])
    return dict(zip(df["SUB_ID"].astype(int), df["label"].astype(int)))


def load_sites():
    """SUB_ID -> SITE_ID from the ABIDE phenotype table."""
    df = pd.read_csv(config.PHENO_CSV, low_memory=False)
    df = df.dropna(subset=["SUB_ID", "SITE_ID"])
    return dict(zip(df["SUB_ID"].astype(int), df["SITE_ID"].astype(str)))


def scan_atlas(atlas):
    """Return {subject_id: (path, n_timepoints, n_rois)} for one atlas."""
    out = {}
    for f in sorted(config.ATLAS_DIRS[atlas].glob("*.1D")):
        sid = subject_id_from_name(f.name)
        if sid is None:
            continue
        if f.stat().st_size == 0:
            out[sid] = (f, 0, 0)
            continue
        rows, cols = scan_1d(f)
        out[sid] = (f, rows, cols)
    return out


def build_manifest(atlas=None, t=None, verbose=True):
    """One row per subject found in `atlas`, with `valid` and a `reason` column.

    A subject is invalid when it is on the audit drop list, has no label or
    site, is empty, has the wrong ROI count, has fewer than `t` timepoints, or
    (by default) is missing from any atlas.
    """
    atlas = atlas or config.ATLAS
    t = t or config.T
    labels = load_labels()
    sites = load_sites()
    per_atlas = {a: scan_atlas(a) for a in config.ATLAS_DIRS}
    primary = per_atlas[atlas]

    rows = []
    for sid, (path, n_t, n_r) in sorted(primary.items()):
        label = labels.get(sid)
        site = sites.get(sid)
        missing = [a for a, d in per_atlas.items() if not (sid in d and d[sid][1] > 0)]

        reason = ""
        if sid in config.DROP_SUBJECTS:
            reason = "audit drop list (broken/missing file)"
        elif label is None:
            reason = "no label"
        elif site is None:
            reason = "no site"
        elif n_t == 0:
            reason = "empty file"
        elif n_r != config.N_ROIS[atlas]:
            reason = f"n_rois={n_r} != {config.N_ROIS[atlas]}"
        elif missing and config.REQUIRE_ALL_ATLASES:
            reason = "missing in " + "/".join(missing)
        elif n_t < t:
            reason = f"{n_t} timepoints < T={t}"

        rec = {
            "subject_id": sid,
            "site": site,
            "label": label,
            "n_timepoints": n_t,
            "n_rois": n_r,
            "path": str(path),
            "valid": reason == "",
            "reason": reason,
        }
        for a, d in per_atlas.items():
            rec[f"has_{a}"] = (sid in d) and (d[sid][1] > 0)
        rows.append(rec)
    df = pd.DataFrame(rows)

    if verbose:
        n_ok = int(df["valid"].sum())
        print(f"[manifest] atlas={atlas} T={t} subjects_found={len(df)} valid={n_ok}")
        if len(df) - n_ok:
            bad = df[~df["valid"]]
            print("[manifest] drop reasons:")
            for reason, grp in bad.groupby("reason"):
                ids = sorted(grp["subject_id"].tolist())
                print(f"    {reason}: {len(ids)} -> {ids[:8]}"
                      f"{' ...' if len(ids) > 8 else ''}")
    return df
