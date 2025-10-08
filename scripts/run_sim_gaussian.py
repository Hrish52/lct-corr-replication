# scripts/run_sim_gaussian.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import csv, time
from pathlib import Path
import numpy as np
import argparse

from src.FisherBaselines import two_group_z_stat, pvals_from_Z, bh_threshold, by_threshold
from src.Simulate import make_block_cov, sample_gaussian, upper_tri_pairs, truth_mask_block
from src.LCT import lct_edge_stat, lct_threshold_normal
from src.LCTB import lct_threshold_bootstrap

OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

def run_once(p=250, n1=80, n2=80, rho=0.3, block=20, seed=0):
    t_start = time.perf_counter()

    # --- data ---
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n1, p))  # group 1: ~ I_p
    Sigma = make_block_cov(p, rho=rho, block_size=block)
    Y = sample_gaussian(n2, Sigma, seed=seed)  # group 2: block-correlated

    # --- correlations & Fisher baselines ---
    R1 = np.corrcoef(X, rowvar=False)
    R2 = np.corrcoef(Y, rowvar=False)
    Z  = two_group_z_stat(R1, R2, n1, n2)

    iu, ju = upper_tri_pairs(p)
    pvals  = pvals_from_Z(Z)[iu, ju]
    truth  = truth_mask_block(p, block)

    row = {"p": p, "n1": n1, "n2": n2, "rho": rho, "block": block, "seed": seed}

    for alpha in (0.05, 0.10):
        # BH/BY
        bh = bh_threshold(pvals, alpha)
        by = by_threshold(pvals, alpha)

        R_bh, R_by = int(bh.sum()), int(by.sum())
        V_bh, V_by = int((~truth & bh).sum()), int((~truth & by).sum())
        S_bh, S_by = int((truth & bh).sum()),  int((truth & by).sum())
        m1 = int(truth.sum())

        row.update({
            f"R_bh_{alpha}": R_bh, f"V_bh_{alpha}": V_bh, f"S_bh_{alpha}": S_bh,
            f"R_by_{alpha}": R_by, f"V_by_{alpha}": V_by, f"S_by_{alpha}": S_by,
            f"fdr_bh_{alpha}": V_bh / max(R_bh, 1),  f"power_bh_{alpha}": S_bh / max(m1, 1),
            f"fdr_by_{alpha}": V_by / max(R_by, 1),  f"power_by_{alpha}": S_by / max(m1, 1),
        })

    # --- LCT-N (Day 5: Cai–Liu variance; use "gaussian" for Day 4) ---
    T, _, _ = lct_edge_stat(X, Y, var_method="cai_liu")
    for alpha in (0.05, 0.10):
        t_hat, mask_lct = lct_threshold_normal(T, alpha=alpha)
        R_lct = int(mask_lct.sum())
        V_lct = int((~truth & mask_lct).sum())
        S_lct = int((truth & mask_lct).sum())
        m1 = int(truth.sum())
        row.update({
            f"t_lct_{alpha}": float(t_hat),
            f"R_lct_{alpha}": R_lct, f"V_lct_{alpha}": V_lct, f"S_lct_{alpha}": S_lct,
            f"fdr_lct_{alpha}": V_lct / max(R_lct, 1),
            f"power_lct_{alpha}": S_lct / max(m1, 1),
        })
    
    # --- LCT-B (bootstrap threshold), start modest B then scale
    t_b, mask_b, info_b = lct_threshold_bootstrap(
        X, Y, alpha=0.05, B=100, var_method="cai_liu", n_jobs=-1, rng=seed
    )
    R_b = int(mask_b.sum())
    V_b = int((~truth & mask_b).sum())
    S_b = int((truth & mask_b).sum())
    m1 = int(truth.sum())
    row.update({
        "t_lctb_0.05": float(t_b),
        "R_lctb_0.05": R_b, "V_lctb_0.05": V_b, "S_lctb_0.05": S_b,
        "fdr_lctb_0.05": V_b / max(R_b, 1),
        "power_lctb_0.05": S_b / max(m1, 1),
    })

    row["wall_time_s"] = round(time.perf_counter() - t_start, 6)
    return row

def run_grid():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p-only", type=int, default=None, help="Run only this p (e.g., 500)")
    parser.add_argument("--reps", type=int, default=None, help="Override reps")
    args = parser.parse_args()

    base_grids = [
        {"p": 250, "n1": 80, "n2": 80, "rho": 0.3,  "block": 20, "reps": 50},
        {"p": 500, "n1": 80, "n2": 80, "rho": 0.25, "block": 20, "reps": 50},
    ]

    grids = [g for g in base_grids if (args.p_only is None or g["p"] == args.p_only)]
    if args.reps is not None:
        for g in grids:
            g["reps"] = args.reps

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
            w.writeheader()
            w.writerows(rows)
        print(f'Wrote {out} in {time.time() - t0:.1f}s')

def main():
    run_grid()

if __name__ == "__main__":
    main()