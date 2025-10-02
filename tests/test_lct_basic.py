import numpy as np
from src.LCT import lct_edge_stat, lct_threshold_normal
from src.Simulate import make_block_cov, sample_gaussian, upper_tri_pairs, truth_mask_block

def test_lct_runs_and_rejects_some_edges_gaussian():
    rng = np.random.default_rng(0)
    p, n1, n2, rho, block = 120, 80, 80, 0.35, 12
    X = rng.normal(size=(n1, p))
    Sigma = make_block_cov(p, rho=rho, block_size=block)
    Y = sample_gaussian(n2, Sigma, seed=1)

    T, _, _ = lct_edge_stat(X, Y, var_method="gaussian")
    t_hat, mask = lct_threshold_normal(T, alpha=0.10)

    # basic sanity
    assert mask.size == p*(p-1)//2
    assert mask.sum() >= 1  # at least some rejections at alpha=0.10
