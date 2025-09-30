import numpy as np
from scipy.stats import norm

def lct_edge_stat(X, Y):
    """
    Day-1 placeholder to validate shapes.
    Replace with the Cai–Liu variance-stabilized statistic on Day 5.
    
    """
    R1 = np.corrcoef(X, rowvar=False)
    R2 = np.corrcoef(Y, rowvar=False)
    Z = 0.5 * np.log((1 + R1) / (1 - R1)) - 0.5 * np.log((1 + R2) / (1 - R2))
    np.fill_diagonal(Z, 0.0)
    return Z

def lct_threshold_normal(T, alpha=0.05):
    iu, ju = np.triu_indices(T.shape[0], k=1)
    absT = np.abs(T)[iu, ju]
    t_grid = np.unique(np.sort(absT))
    q = lambda t: 2 * (1 - norm.cdf(t))
    M = absT.size
    t_hat, mask = None, None
    for t in t_grid[::-1]:
        R = (absT >= t).sum()
        if R == 0:
            continue
        est_fdr = M * q(t) / R
        if est_fdr <= alpha:
            t_hat = t
            mask = (absT >= t)
    if t_hat is None:
        return np.max(t_grid) + 1, np.zeros_like(absT, dtype=bool)
    return t_hat, mask
