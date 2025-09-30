import numpy as np
from src.FisherBaselines import two_group_z_stat, pvals_from_Z, bh_threshold, by_threshold

def test_no_signal_gives_few_rejections():
    rng = np.random.default_rng(0)
    p, n1, n2 = 20, 60, 60
    X = rng.normal(size=(n1, p))
    Y = rng.normal(size=(n2, p))
    R1 = np.corrcoef(X, rowvar=False)
    R2 = np.corrcoef(Y, rowvar=False)
    Z = two_group_z_stat(R1, R2, n1, n2)
    iu, ju = np.triu_indices(p, 1)
    pvals = pvals_from_Z(Z)[iu, ju]
    # Expect very few at alpha=0.05; exact count may be 0–2 due to randomness
    assert bh_threshold(pvals, 0.05).sum() <= 3
    assert by_threshold(pvals, 0.05).sum() <= 3

def test_block_signal_increases_rejections():
    rng = np.random.default_rng(1)
    p, n1, n2 = 30, 80, 80  # slightly larger n for stability

    # Group 1: independent standard normal
    X = rng.normal(size=(n1, p))

    # Group 2: MVN with a 6x6 correlated block (rho=0.6) in the top-left
    rho = 0.6
    Sigma = np.eye(p)
    Sigma[:6, :6] = rho
    np.fill_diagonal(Sigma, 1.0)
    Sigma = 0.99 * Sigma + 0.01 * np.eye(p)  # PSD shrink
    Y = rng.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n2)

    R1 = np.corrcoef(X, rowvar=False)
    R2 = np.corrcoef(Y, rowvar=False)
    Z = two_group_z_stat(R1, R2, n1, n2)

    iu, ju = np.triu_indices(p, 1)
    pvals = pvals_from_Z(Z)[iu, ju]

    # With 15 truly different edges in the 6x6 block, BH at 0.10 should flag some
    assert bh_threshold(pvals, 0.10).sum() >= 1

