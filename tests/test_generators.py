import numpy as np
from src.Simulate import make_block_cov, sample_t, sample_laplace, sample_exp
from src.Simulate import sample_t_cl, sample_exp_cl, sample_normal_mixture
from src.LCT import _kappa_hat, _zscore_columns

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

def test_cl_generators_shapes():
    p, n = 60, 200
    S = make_block_cov(p, rho=0.5, block_size=10)
    for X in (sample_t_cl(n, 6, S, rng=0),
              sample_exp_cl(n, 1.0, S, rng=1),
              sample_normal_mixture(n, S, rng=2)):
        assert X.shape == (n, p)
        assert np.all(np.isfinite(X))

def test_normal_mixture_kappa_exceeds_one():
    """Cai-Liu's normal mixture is elliptical with kappa = 9/5, not 1.
    Fisher-z assumes kappa = 1; this is what breaks it."""
    p, n = 80, 4000
    X = sample_normal_mixture(n, np.eye(p), rng=0)
    k = _kappa_hat(_zscore_columns(X))
    assert 1.4 < k < 2.4, f"kappa {k:.3f}, expected ~1.8"