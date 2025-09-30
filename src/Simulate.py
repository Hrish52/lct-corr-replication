import numpy as np

def make_block_cov(p, rho=0.3, block_size=20, seed=0):
    rng = np.random.default_rng(seed)
    Sigma = np.eye(p)
    for k in range(0, p, block_size):
        b = slice(k, min(k+block_size, p))
        Sigma[b, b] = rho
        np.fill_diagonal(Sigma[b, b], 1.0)
    Sigma = 0.99*Sigma + 0.01*np.eye(p)
    return Sigma

def sample_gaussian(n, Sigma, seed=0):
    rng = np.random.default_rng(seed)
    p = Sigma.shape[0]
    X = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
    return X