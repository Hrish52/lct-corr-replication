import numpy as np
from scipy.stats import laplace, norm

def upper_tri_pairs(p: int):
    return np.triu_indices(p, 1)

def truth_mask_block(p: int, block: int) -> np.ndarray:
    iu, ju = upper_tri_pairs(p)
    return (iu < block) & (ju < block)

def make_block_cov(p: int, rho: float = 0.3, block_size: int = 20) -> np.ndarray:
    Sigma = np.eye(p, dtype=float)
    Sigma[:block_size, :block_size] = rho
    np.fill_diagonal(Sigma, 1.0)
    return 0.99 * Sigma + 0.01 * np.eye(p)

def sample_gaussian(n: int, Sigma: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = Sigma.shape[0]
    return rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)

# ---------- Non-Gaussian helpers & samplers ----------

def _chol_or_eig(Sigma: np.ndarray) -> np.ndarray:
    """
    Cholesky if possible; otherwise eigen fallback (with small eigenvalue floor).
    Returns a factor L such that L @ L.T ≈ Sigma.
    """
    try:
        return np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Sigma)
        w = np.clip(w, 1e-8, None)
        return V @ np.diag(np.sqrt(w))

def _gaussian_copula(n: int, Sigma: np.ndarray, rng=None) -> np.ndarray:
    """
    Draw Z ~ N(0, Sigma); standardize each column to N(0,1); map to U=Phi(Z) in (0,1).
    Produces Uniform(0,1) marginals with Gaussian-copula dependence Sigma.
    """
    rng = np.random.default_rng(rng)
    p = Sigma.shape[0]
    L = _chol_or_eig(Sigma)
    Z = rng.normal(size=(n, p)) @ L.T
    # standardize columns to protect against numeric drift
    Z = (Z - Z.mean(axis=0, keepdims=True)) / Z.std(axis=0, ddof=1, keepdims=True)
    U = norm.cdf(Z)
    # numerical safety near 0/1
    return np.clip(U, 1e-12, 1 - 1e-12)

def sample_t(n: int, df: float, Sigma: np.ndarray, rng=None) -> np.ndarray:
    """
    Multivariate t with df>2, correlation Sigma, scaled so Var≈1 per feature.
    Construction: Z~N(0,Sigma), s~Chi2(df)/df, T=Z/sqrt(s), then scale by sqrt((df-2)/df).
    Final z-score per column to clean MC drift.
    """
    if df <= 2:
        raise ValueError("df must be > 2 for finite variance")
    rng = np.random.default_rng(rng)
    p = Sigma.shape[0]
    L = _chol_or_eig(Sigma)
    Z = rng.normal(size=(n, p)) @ L.T
    s = rng.chisquare(df, size=(n, 1)) / df
    T = Z / np.sqrt(s)
    T *= np.sqrt((df - 2.0) / df)
    # light recenter/scale
    T = (T - T.mean(axis=0, keepdims=True)) / T.std(axis=0, ddof=1, keepdims=True)
    return T

def sample_laplace(n: int, b: float, Sigma: np.ndarray, rng=None) -> np.ndarray:
    """
    Laplace marginals with Gaussian-copula dependence Sigma via inverse-CDF.
    For unit variance Laplace, use b = 1/sqrt(2). Columns are re-standardized to Var≈1.
    """
    U = _gaussian_copula(n, Sigma, rng=rng)
    X = laplace.ppf(U, loc=0.0, scale=b)
    # recentre & z-score to protect against MC noise
    X = X - X.mean(axis=0, keepdims=True)
    X = X / X.std(axis=0, ddof=1, keepdims=True)
    return X

def sample_exp(n: int, rate: float, Sigma: np.ndarray, rng=None, zscore: bool = True) -> np.ndarray:
    """
    Exponential(rate) (>0) marginals with Gaussian-copula dependence Sigma via inverse-CDF.
    Mean-center and (optionally) z-score columns (recommended True for comparability).
    """
    U = _gaussian_copula(n, Sigma, rng=rng)
    # F^{-1}(u) for Exp(rate): -ln(1-u)/rate
    X = -np.log1p(-U) / float(rate)
    # center; strong skew remains by design
    X = X - X.mean(axis=0, keepdims=True)
    if zscore:
        sd = X.std(axis=0, ddof=1, keepdims=True)
        sd = np.where(sd == 0, 1.0, sd)
        X = X / sd
    return X

def make_block_ar1_cov(p: int, rho: float, block_size: int = 20) -> np.ndarray:
    """
    Σ = I_p except top-left block (k×k) is AR(1) Toeplitz with entries rho^{|i-j|}.
    """
    k = int(block_size)
    Sigma = np.eye(p, dtype=float)
    if k <= 1 or abs(rho) < 1e-15:
        return Sigma
    idx = np.arange(k)
    toe = rho ** np.abs(idx[:, None] - idx[None, :])
    np.fill_diagonal(toe, 1.0)  # ensure diag=1
    Sigma[:k, :k] = toe
    return Sigma

def make_block_decay_cov(p: int, rho: float, block_size: int = 20, decay: float = 0.6) -> np.ndarray:
    """
    Σ = I_p except top-left block has decaying correlations: rho * decay^{|i-j|}, with diag=1.
    """
    k = int(block_size)
    Sigma = np.eye(p, dtype=float)
    if k <= 1 or abs(rho) < 1e-15:
        return Sigma
    idx = np.arange(k)
    blk = (rho * (decay ** np.abs(idx[:, None] - idx[None, :])))
    np.fill_diagonal(blk, 1.0)
    Sigma[:k, :k] = blk
    return Sigma