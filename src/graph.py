"""Static functional connectivity (FC) and graph construction.

Everything here is computed from a single subject's own series, so it is fold
independent and cannot leak information across the train/test boundary (rule R2).

FC collapses the time axis, so the graph input is built from the **full-length**
series (decision 2026-10-14), not the T-truncated sequence tensor. The pipeline
therefore caches the FC matrices directly (``build_cache.py`` stores an (N, R, R)
``FC`` array computed from the original series); ``fc_matrix`` below is the
standalone helper that reproduces one such matrix from a single (T, R) series.
"""
import numpy as np

from .preprocess import pearson_fc


def fc_matrix(x):
    """(T, R) BOLD series -> (R, R) Pearson correlation matrix."""
    return pearson_fc(x)


def upper_triangle(fc):
    """Flatten the strictly-upper triangle of an (R, R) matrix."""
    iu = np.triu_indices(fc.shape[0], k=1)
    return fc[iu]


def topk_adjacency(fc, k, signed=False):
    """Symmetric k-nearest-neighbour graph over ROI correlations.

    For every node keep the `k` strongest edges (by |corr| when `signed` is
    False), take the union of the directed neighbourhoods so the graph is
    undirected, and force a zero diagonal. Returns a dense (R, R) float32 matrix
    holding the original correlation values on the kept edges.
    """
    r = fc.shape[0]
    k = int(min(k, r - 1))
    score = np.abs(fc) if not signed else fc.copy()
    score = score.astype(np.float32, copy=True)
    np.fill_diagonal(score, -np.inf)

    keep = np.zeros((r, r), dtype=bool)
    for i in range(r):
        idx = np.argpartition(score[i], -k)[-k:]
        keep[i, idx] = True

    sym = keep | keep.T
    np.fill_diagonal(sym, False)
    adj = np.where(sym, fc, 0.0).astype(np.float32)
    return adj


def threshold_adjacency(fc, threshold):
    """Keep edges with |corr| >= `threshold`; zero elsewhere (zero diagonal)."""
    adj = np.where(np.abs(fc) >= threshold, fc, 0.0).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def edge_index(adj):
    """(R, R) adjacency -> (edge_index (2, E) int64, edge_weight (E,) float32)."""
    rows, cols = np.nonzero(adj)
    ei = np.vstack([rows, cols]).astype(np.int64)
    ew = adj[rows, cols].astype(np.float32)
    return ei, ew


def graph_stats(adj):
    """Quick sanity numbers for a dense adjacency matrix."""
    r = adj.shape[0]
    n_edges = int((adj != 0).sum())
    return {
        "n_nodes": r,
        "n_edges": n_edges,
        "density": n_edges / (r * (r - 1)),
        "symmetric": bool(np.allclose(adj, adj.T)),
        "zero_diagonal": bool(np.all(np.diag(adj) == 0)),
    }
