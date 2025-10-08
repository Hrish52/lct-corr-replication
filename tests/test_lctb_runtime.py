import numpy as np
from src.Simulate import make_block_cov, sample_t
from src.LCTB import lct_threshold_bootstrap

def test_lctb_smoke_runtime_small():
    p, n = 60, 80
    Sigma = make_block_cov(p, rho=0.3, block_size=10)
    X = np.random.default_rng(0).normal(size=(n, p))
    Y = sample_t(n, df=6, Sigma=Sigma, rng=1)
    t, mask, info = lct_threshold_bootstrap(
        X, Y, alpha=0.10, B=30, var_method="cai_liu", rng=0
    )
    assert mask.size == p*(p-1)//2
