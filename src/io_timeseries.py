"""Reading CPAC ROI `.1D` time-series files."""
import numpy as np


def read_1d(path):
    """Read one `.1D` file.

    Layout: a leading '#' header row (e.g. `#1  #2  #3`), then one row per
    volume and one column per ROI. Returns a (n_timepoints, n_rois) float32 array.
    """
    arr = np.loadtxt(path, comments="#", dtype=np.float32, ndmin=2)
    return arr


def read_1d_safe(path):
    """Read a `.1D` file, returning (array, None) or (None, error string)."""
    try:
        return read_1d(path), None
    except Exception as exc:  # noqa: BLE001 - keep the raw reason for the audit
        return None, f"{type(exc).__name__}: {exc}"


def scan_1d(path):
    """Cheap dimension scan -> (n_timepoints, n_rois) without a full parse."""
    n_rows = 0
    n_cols = None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            n_rows += 1
            if n_cols is None:
                n_cols = len(s.split())
    return n_rows, (n_cols or 0)
