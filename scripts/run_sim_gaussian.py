import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import csv, time
from pathlib import Path
import numpy as np

from src.FisherBaselines import two_group_z_stat, pvals_from_Z, bh_threshold, by_threshold
from src.Simulate import make_block_cov, sample_gaussian, upper_tri_pairs, truth_mask_block

OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

def run_once(p=250, n1=80, n2=80, rho=0.3, block=20, seed=0):
    t_start = time.perf_counter()
    
    # Group 1: independent ~ I_p
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n1, p))

    # Group 2: MVN with a top-left block of correlation rho
    Sigma = make_block_cov(p, rho=rho, block_size=block)
    Y = sample_gaussian(n2, Sigma, seed=seed)

    # Correlation matrices
    R1 = np.corrcoef(X, rowvar=False)
    R2 = np.corrcoef(Y, rowvar=False)

    # Two-sample Fisher z test
    Z = two_group_z_stat(R1, R2, n1, n2)

    # Vectorize upper-tri
    iu, ju = upper_tri_pairs(p)
    pvals = pvals_from_Z(Z)[iu, ju]

    # Ground truth: edges in the block differ
    truth = truth_mask_block(p, block)

    row = {"p": p, "n1": n1, "n2": n2, "rho": rho, "block": block, "seed": seed}
    for alpha in (0.05, 0.10):
        bh = bh_threshold(pvals, alpha)
        by = by_threshold(pvals, alpha)

        # metrics
        R_bh, R_by = int(bh.sum()), int(by.sum())
        V_bh, V_by = int((~truth & bh).sum()), int((~truth & by).sum())
        S_bh, S_by = int((truth & bh).sum()), int((truth & by).sum())
        m1 = int(truth.sum())

        fdr_bh = V_bh / max(R_bh, 1)
        fdr_by = V_by / max(R_by, 1)
        pow_bh = S_bh / max(m1, 1)
        pow_by = S_by / max(m1, 1)

        row.update({
            f"R_bh_{alpha}": R_bh, f"V_bh_{alpha}": V_bh, f"S_bh_{alpha}": S_bh,
            f"R_by_{alpha}": R_by, f"V_by_{alpha}": V_by, f"S_by_{alpha}": S_by,
            f"fdr_bh_{alpha}": fdr_bh, f"power_bh_{alpha}": pow_bh,
            f"fdr_by_{alpha}": fdr_by, f"power_by_{alpha}": pow_by
        })
    
    row["wall_time_s"] = round(time.perf_counter() - t_start, 6)
    return row

def run_grid():
    grids = [
        {"p": 250, "n1": 80, "n2": 80, "rho": 0.3,  "block": 20, "reps": 50},
        {"p": 500, "n1": 80, "n2": 80, "rho": 0.25, "block": 20, "reps": 50},
    ]
    for g in grids:
        rows = []
        t0 = time.time()
        for r in range(g["reps"]):
            rows.append(run_once(g["p"], g["n1"], g["n2"], g["rho"], g["block"], seed=r))
            if (r + 1) % 10 == 0:
                print(f'  [{g["p"]},{g["rho"]}] finished {r+1}/{g["reps"]}')
        out = OUT / f'gaussian_p{g["p"]}_n{g["n1"]}_{g["n2"]}_rho{g["rho"]}_b{g["block"]}_R{g["reps"]}.csv'
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f'Wrote {out} in {time.time()-t0:.1f}s')

def main():
    run_grid()

if __name__ == "__main__":
    main()