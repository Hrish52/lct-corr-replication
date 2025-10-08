import numpy as np
from src.LCTB import lct_threshold_bootstrap
from src.Simulate import make_block_cov, sample_gaussian

def test_lctb_runs_and_controls_under_null():
    rng = np.random.default_rng(0)
    p, n1, n2 = 60, 80, 80
    X = rng.normal(size=(n1, p))
    Y = rng.normal(size=(n2, p))
    # under H0 we expect very few or zero rejections at alpha=0.05 with modest B
    t, mask, info = lct_threshold_bootstrap(X, Y, alpha=0.05, B=50, var_method="cai_liu", rng=0)
    assert mask.size == p*(p-1)//2
    # allow a few random hits, but not many:
    assert mask.sum() <= 5
