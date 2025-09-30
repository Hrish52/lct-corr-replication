import numpy as np
from scipy.stats import norm

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)  # Avoid division by zero
    return 0.5 * np.log((1 + r) / (1 - r))

def two_group_z_stat(R1, R2, n1, n2):
    """
    Fisher z two-sample test for equality of correlations.
    Z1 = atanh(r1), Z2 = atanh(r2).
    Standard error: sqrt(1/(n1-3) + 1/(n2-3)).
    Returns a (p,p) matrix with zeros on the diagonal.
    """
    Z1, Z2 = fisher_z(R1), fisher_z(R2)
    se = np.sqrt(1.0 / (n1 - 3.0) + 1.0 / (n2 - 3.0))
    Z = (Z1 - Z2) / se
    np.fill_diagonal(Z, 0.0)
    return Z

def upper_triangular_indices(p):
    return np.triu_indices(p, k=1)

def vectorize_upper(M):
    iu, ju = upper_triangular_indices(M.shape[0])
    return M[iu, ju]

def pvals_from_Z(Z):
    return 2 * (1 - norm.cdf(np.abs(Z)))

def bh_threshold(pvals, alpha=0.05):
    m = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = (np.arange(1, m + 1) / m) * alpha
    k = np.where(ranked <= thresh)[0]
    k = k[-1] if k.size > 0 else 0
    crit = ranked[k] if k > 0 else 0
    return pvals <= crit

def by_threshold(pvals, alpha=0.05):
    m = pvals.size
    c_m = np.sum(1 / np.arange(1, m + 1))
    return bh_threshold(pvals, alpha / c_m)