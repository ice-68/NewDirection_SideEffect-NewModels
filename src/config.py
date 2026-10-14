"""Project-wide configuration for the site-effect / site-shortcut study.

Every value here mirrors a recorded decision in <<数据审计报告.md>> and
<<实验TODOLIST.md>>; change them there first, then here.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"

# Primary atlas (decision 2026-10-13). AAL / CC400 are sensitivity analyses.
ATLAS = "CC200"
ATLAS_DIRS = {
    "AAL": DATA_DIR / "AAL" / "rois_aal",
    "CC200": DATA_DIR / "CC200" / "rois_cc200",
    "CC400": DATA_DIR / "CC400" / "rois_cc400",
}
N_ROIS = {"AAL": 116, "CC200": 200, "CC400": 392}

LABELS_CSV = DATA_DIR / "labels.csv"
PHENO_CSV = DATA_DIR / "Phenotypic_V1_0b_preprocessed1.csv"

# Fixed sequence length: keep the first T timepoints. This removes the
# "sequence length == site" shortcut found in the audit.
# T = 78 is the shortest clean length in the dataset (all of OHSU), so every one
# of the 20 sites stays in the study (decision 2026-10-13).
T = 78

# Subjects removed by the data audit (broken / missing files):
#   50012 -> CC200 empty file, 50160 -> CC200 truncated to 9 rows,
#   50778 / 50341 / 50355 -> CC400 truncated, 50166 / 50161 -> no CC200 file.
DROP_SUBJECTS = {50012, 50160, 50778, 50341, 50355, 50166, 50161}

# Keep only subjects that exist and are non-empty in every atlas (877-subject set).
REQUIRE_ALL_ATLASES = True

SEEDS = (0, 1, 2, 3, 4)
INNER_VAL_FRACTION = 0.2

