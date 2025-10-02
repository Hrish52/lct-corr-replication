import numpy as np

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
