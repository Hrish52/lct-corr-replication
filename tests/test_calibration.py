import numpy as np
from src.LCT import lct_edge_stat

def test_lct_null_calibration_cai_liu_small_p():
    rng = np.random.default_rng(0)
    p, n1, n2 = 60, 120, 120
    X = rng.normal(size=(n1, p))
    Y = rng.normal(size=(n2, p))
    T, _, _ = lct_edge_stat(X, Y, var_method="cai_liu")
    iu, ju = np.triu_indices(p, 1)
    t = T[iu, ju]
    # mean near 0, sd near 1 under H0 (tolerances generous for CI over finite m)
    assert abs(t.mean()) < 0.08
    assert 0.85 < t.std(ddof=1) < 1.15
