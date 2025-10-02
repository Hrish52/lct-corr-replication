import numpy as np
from src.Simulate import make_block_cov, sample_gaussian, upper_tri_pairs, truth_mask_block
from src.FisherBaselines import two_group_z_stat, pvals_from_Z, bh_threshold, by_threshold

def test_pipeline_runs_and_makes_sense():
    rng = np.random.default_rng(42)
    p, n1, n2, rho, block = 60, 60, 60, 0.4, 10

    X = rng.normal(size=(n1, p))
    Sigma = make_block_cov(p, rho=rho, block_size=block)
    Y = sample_gaussian(n2, Sigma, seed=123)

    Z  = two_group_z_stat(
        np.corrcoef(X, rowvar=False),
        np.corrcoef(Y, rowvar=False),
        n1, n2
    )
    iu, ju = upper_tri_pairs(p)
    pvals = pvals_from_Z(Z)[iu, ju]
    truth = truth_mask_block(p, block)

    bh = bh_threshold(pvals, 0.10)
    by = by_threshold(pvals, 0.10)

    # Sanity: BY should be no more liberal than BH
    assert by.sum() <= bh.sum()
    # There should be at least some true edges in the block
    assert truth.sum() == block * (block - 1) // 2
