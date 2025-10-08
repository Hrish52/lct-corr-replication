import numpy as np
from src.Simulate import make_block_cov, sample_t, sample_laplace, sample_exp

def _basic_checks(X):
    assert X.ndim == 2
    assert np.all(np.isfinite(X))
    s = X.std(axis=0, ddof=1)
    assert np.all(s > 0)
    assert abs(s.mean() - 1.0) < 0.15  # allow slack

def test_sample_t_shapes_and_variance():
    p, n = 80, 120
    Sigma = make_block_cov(p, rho=0.3, block_size=12)
    X = sample_t(n, df=6, Sigma=Sigma, rng=0)
    assert X.shape == (n, p)
    _basic_checks(X)

def test_sample_laplace_shapes_and_variance():
    p, n = 80, 120
    Sigma = make_block_cov(p, rho=0.3, block_size=12)
    X = sample_laplace(n, b=1/np.sqrt(2), Sigma=Sigma, rng=1)
    assert X.shape == (n, p)
    _basic_checks(X)

def test_sample_exp_shapes_and_variance():
    p, n = 80, 120
    Sigma = make_block_cov(p, rho=0.25, block_size=10)
    X = sample_exp(n, rate=1.0, Sigma=Sigma, rng=2, zscore=True)
    assert X.shape == (n, p)
    _basic_checks(X)
