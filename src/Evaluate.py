import numpy as np

def empirical_corr(X):
    return np.corrcoef(X, rowvar=False)

def fdr_power(truth_mask, reject_mask):
    R = reject_mask.sum()
    V = np.logical_and(~truth_mask, reject_mask).sum()
    S = np.logical_and(truth_mask, reject_mask).sum()
    m1 = truth_mask.sum()
    fdr = V / max(R, 1)
    power = S / max(m1, 1)
    return fdr, power, {"R": int(R), "V": int(V), "S": int(S), "m1": int(m1)}