# src/LCTB_v2.py
import time
from typing import Optional
import numpy as np
from src.LCT import lct_edge_stat

"""
LCT-B (v2) – Day-11 optimized implementation

Features:
- Cached upper-tri indices
- Optional coarse grid (interior quantiles of |T|) to speed threshold scan
- Streaming bootstrap tail counts (no B×M matrix in RAM)
- float32 storage for boot vectors
- winsorize passthrough to lct_edge_stat
- Diagnostics returned in `info`
"""

# cache: iu, ju by p
_TRI_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _upper_tri_indices(p: int):
    t = _TRI_CACHE.get(p)
    if t is None:
        t = np.triu_indices(p, 1)
        _TRI_CACHE[p] = t
    return t


def _absT_from_XY(X: np.ndarray, Y: np.ndarray, var_method: str, winsorize=None) -> np.ndarray:
    """Compute |T| for upper-tri edges as a 1D vector."""
    T, _, _ = lct_edge_stat(X, Y, var_method=var_method, winsorize=winsorize)
    p = T.shape[0]
    iu, ju = _upper_tri_indices(p)
    return np.abs(T[iu, ju])


def _boot_index_pairs(N: int, n1: int, n2: int, rng: np.random.Generator, replace: bool):
    if replace:
        idx1 = rng.integers(0, N, size=n1)
        idx2 = rng.integers(0, N, size=n2)
    else:
        perm = rng.permutation(N)
        idx1 = perm[:n1]
        idx2 = perm[n1:n1 + n2]
    return idx1, idx2


def lct_threshold_bootstrap_v2(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 0.05,
    B: int = 200,
    var_method: str = "cai_liu",     # "cai_liu" | "gaussian" | "jackknife"
    replace: bool = False,           # False: permutation split; True: with replacement
    n_jobs: int = -1,                # kept for API compatibility (not used in v2)
    rng=None,
    *,
    winsorize: Optional[float] = None,
    coarse_grid: Optional[int] = None,   # e.g., 200 -> scan 200 interior quantiles of |T|
    dtype: str = "float32",              # boot vector dtype
):
    """
    Bootstrap LCT threshold (LCT-B) by resampling under H0 from pooled rows.

    Returns:
      t_hat        : float, chosen threshold
      reject_mask  : 1D bool over upper-tri edges (size M = p*(p-1)/2)
      info         : dict with {t_grid, q_hat, fdr_hat, R_t, M, B, timings, memory, settings}
    """
    t_wall0 = time.perf_counter()
    rng = np.random.default_rng(rng)

    n1, p = X.shape
    n2 = Y.shape[0]
    pooled = np.vstack([X, Y])
    N = pooled.shape[0]

    iu, ju = _upper_tri_indices(p)
    M = iu.size

    # 1) Original |T| and threshold grid
    t0 = time.perf_counter()
    absT = _absT_from_XY(X, Y, var_method=var_method, winsorize=winsorize)
    if coarse_grid is None:
        t_grid = np.unique(np.sort(absT))
    else:
        K = max(10, int(coarse_grid))
        qs = np.linspace(0.0, 1.0, K + 2, endpoint=True)[1:-1]  # drop 0 and 1
        t_grid = np.unique(np.quantile(absT, qs))
    t1 = time.perf_counter()

    # 2) Bootstrap index pairs under H0
    idxs = [_boot_index_pairs(N, n1, n2, rng, replace) for _ in range(B)]

    # 3) Streaming tail counts (memory-light; sort each boot vector once)
    t2 = time.perf_counter()
    counts_exceed = np.zeros(t_grid.size, dtype=np.int64)
    for (idx1, idx2) in idxs:
        Xb = pooled[idx1]
        Yb = pooled[idx2]
        v = _absT_from_XY(Xb, Yb, var_method=var_method, winsorize=winsorize).astype(dtype, copy=False)
        v.sort()  # ascending
        left_idx = np.searchsorted(v, t_grid, side="left")  # number of values < t
        counts_exceed += (M - left_idx).astype(np.int64, copy=False)
    q_hat = counts_exceed / float(B * M)
    t3 = time.perf_counter()

    # 4) FDP estimate and threshold selection
    R_t = np.array([(absT >= t).sum() for t in t_grid], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        fdr_hat = (M * q_hat) / np.maximum(R_t, 1.0)

    t_hat = None
    alpha = float(alpha)
    for t, fdr_val in zip(t_grid[::-1], fdr_hat[::-1]):  # descending
        R = (absT >= t).sum()
        if R == 0:
            continue
        if fdr_val <= alpha:
            t_hat = float(t)
            break

    if t_hat is None:
        t_hat = float(t_grid.max() + 1e-9)
        reject_mask = np.zeros_like(absT, dtype=bool)
    else:
        reject_mask = (absT >= t_hat)

    t_wall1 = time.perf_counter()

    info = {
        "t_grid": t_grid,
        "grid_len": int(t_grid.size),
        "q_hat": q_hat,
        "fdr_hat": fdr_hat,
        "R_t": R_t,
        "M": int(M),
        "B": int(B),
        "alpha": alpha,
        "var_method": var_method,
        "winsorize": winsorize,
        "coarse_grid": coarse_grid,
        "dtype": dtype,
        # timings
        "t_build_absT_s": t1 - t0,
        "t_boot_count_s": t3 - t2,
        "t_total_s": t_wall1 - t_wall0,
        # rough memory estimate for streaming path
        "memory_bytes_est": int(M * np.dtype(dtype).itemsize + absT.nbytes + t_grid.nbytes),
        # kept for API compatibility
        "replace": replace,
        "n_jobs": n_jobs,
    }
    return float(t_hat), reject_mask, info


# Backward-compatible alias so drivers can do:
#   from src.LCTB_v2 import lct_threshold_bootstrap
lct_threshold_bootstrap = lct_threshold_bootstrap_v2