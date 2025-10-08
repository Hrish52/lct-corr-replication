# scripts/run_sim_gaussian.py  (now: gaussian + non-gaussian grids)
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import csv, time
from pathlib import Path
import numpy as np
import argparse

from src.FisherBaselines import two_group_z_stat, pvals_from_Z, bh_threshold, by_threshold
from src.Simulate import (
    make_block_cov, sample_gaussian, sample_t, sample_laplace, sample_exp,
    upper_tri_pairs, truth_mask_block
)
from src.LCT import lct_edge_stat, lct_threshold_normal
from src.LCTB import lct_threshold_bootstrap

OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

# cache upper-tri indices once per p
_IU_CACHE = {}
def tri_pairs(p):
    if p not in _IU_CACHE:
        _IU_CACHE[p] = upper_tri_pairs(p)
    return _IU_CACHE[p]


def _dataset(model, n1, n2, p, rho, block, seed, extra):
    """
    Returns (X, Y) where X is null group ~ N(0, I) and Y has dependence per `model`
    using the same top-left block covariance for ground-truth fairness.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n1, p))  # null group ~ I_p
    Sigma = make_block_cov(p, rho=rho, block_size=block)

    if model == "gaussian":
        Y = sample_gaussian(n2, Sigma, seed=seed)
    elif model == "t":
        Y = sample_t(n2, df=extra.get("df", 6), Sigma=Sigma, rng=seed)
    elif model == "laplace":
        b = extra.get("b", 1/np.sqrt(2))  # unit variance Laplace
        Y = sample_laplace(n2, b=b, Sigma=Sigma, rng=seed)
    elif model == "exp":
        rate = extra.get("rate", 1.0)
        Y = sample_exp(n2, rate=rate, Sigma=Sigma, rng=seed, zscore=True)
    else:
        raise ValueError(f"Unknown model: {model}")

    return X, Y


def run_once(model="gaussian", p=250, n1=80, n2=80, rho=0.3, block=20, seed=0, extra=None):
    extra = extra or {}
    t_start = time.perf_counter()

    # --- data ---
    X, Y = _dataset(model, n1, n2, p, rho, block, seed, extra)

    # --- correlations & Fisher baselines ---
    R1 = np.corrcoef(X, rowvar=False)
    R2 = np.corrcoef(Y, rowvar=False)
    Z  = two_group_z_stat(R1, R2, n1, n2)

    iu, ju = tri_pairs(p)
    pvals  = pvals_from_Z(Z)[iu, ju]
    truth  = truth_mask_block(p, block)

    row = {
        "model": model, "p": p, "n1": n1, "n2": n2, "rho": rho, "block": block, "seed": seed,
        **{f"k_{k}": v for k, v in (extra or {}).items()}
    }

    for alpha in (0.05, 0.10):
        # BH/BY
        sel_bh = bh_threshold(pvals, alpha)
        sel_by = by_threshold(pvals, alpha)

        R_bh, R_by = int(sel_bh.sum()), int(sel_by.sum())
        V_bh, V_by = int((~truth & sel_bh).sum()), int((~truth & sel_by).sum())
        S_bh, S_by = int((truth & sel_bh).sum()),  int((truth & sel_by).sum())
        m1 = int(truth.sum())

        row.update({
            f"R_bh_{alpha}": R_bh, f"V_bh_{alpha}": V_bh, f"S_bh_{alpha}": S_bh,
            f"R_by_{alpha}": R_by, f"V_by_{alpha}": V_by, f"S_by_{alpha}": S_by,
            f"fdr_bh_{alpha}": V_bh / max(R_bh, 1),  f"power_bh_{alpha}": S_bh / max(m1, 1),
            f"fdr_by_{alpha}": V_by / max(R_by, 1),  f"power_by_{alpha}": S_by / max(m1, 1),
        })

    # --- LCT-N (Cai–Liu variance) ---
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

    # --- LCT-B (bootstrap thresholds) ---
    B_list = [100]
    if p == 250:
        B_list = [100, 200, 500]
    for B in B_list:
        for alpha in (0.05, 0.10):
            t_b, mask_b, info_b = lct_threshold_bootstrap(
                X, Y, alpha=alpha, B=B, var_method="cai_liu",
                n_jobs=-1, rng=seed
            )
            Rb = int(mask_b.sum())
            Vb = int((~truth & mask_b).sum())
            Sb = int((truth & mask_b).sum())
            m1 = int(truth.sum())
            row.update({
                f"t_lctb_{alpha}_B{B}": float(t_b),
                f"R_lctb_{alpha}_B{B}": Rb, f"V_lctb_{alpha}_B{B}": Vb, f"S_lctb_{alpha}_B{B}": Sb,
                f"fdr_lctb_{alpha}_B{B}": Vb / max(Rb, 1),
                f"power_lctb_{alpha}_B{B}": Sb / max(m1, 1),
            })

    row["wall_time_s"] = round(time.perf_counter() - t_start, 6)
    return row


def run_grid():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="One of {gaussian, t, laplace, exp}. If omitted, runs all.")
    parser.add_argument("--p-only", type=int, default=None, help="Run only this p (e.g., 500)")
    parser.add_argument("--reps", type=int, default=None, help="Override reps (default 50)")
    args = parser.parse_args()

    grids = []

    # Gaussian (existing)
    grids += [
        dict(model="gaussian", p=250, n1=80, n2=80, rho=0.30, block=20, reps=50, extra={}),
        dict(model="gaussian", p=500, n1=80, n2=80, rho=0.25, block=20, reps=50, extra={}),
    ]

    # Student-t with df=6 (heavy tails)
    grids += [
        dict(model="t", p=250, n1=80, n2=80, rho=0.30, block=20, reps=50, extra={"df": 6}),
        dict(model="t", p=500, n1=80, n2=80, rho=0.25, block=20, reps=50, extra={"df": 6}),
    ]

    # Laplace unit variance (b = 1/sqrt(2))
    grids += [
        dict(model="laplace", p=250, n1=80, n2=80, rho=0.30, block=20, reps=50, extra={"b": 1/np.sqrt(2)}),
        dict(model="laplace", p=500, n1=80, n2=80, rho=0.25, block=20, reps=50, extra={"b": 1/np.sqrt(2)}),
    ]

    # Exponential(1), centered & z-scored
    grids += [
        dict(model="exp", p=250, n1=80, n2=80, rho=0.30, block=20, reps=50, extra={"rate": 1.0}),
        dict(model="exp", p=500, n1=80, n2=80, rho=0.25, block=20, reps=50, extra={"rate": 1.0}),
    ]

    # filter by flags
    if args.model:
        grids = [g for g in grids if g["model"] == args.model]
    if args.p_only is not None:
        grids = [g for g in grids if g["p"] == args.p_only]
    if args.reps is not None:
        for g in grids:
            g["reps"] = args.reps

    # run
    for g in grids:
        rows = []
        t0 = time.time()
        for r in range(g["reps"]):
            rows.append(run_once(
                model=g["model"], p=g["p"], n1=g["n1"], n2=g["n2"],
                rho=g["rho"], block=g["block"], seed=r, extra=g["extra"]
            ))
            if (r + 1) % 10 == 0:
                print(f'  [{g["model"]}, p={g["p"]}, rho={g["rho"]}] finished {r+1}/{g["reps"]}')
        tag = f'{g["model"]}_p{g["p"]}_n{g["n1"]}_{g["n2"]}_rho{g["rho"]}_b{g["block"]}_R{g["reps"]}'
        if g["extra"]:
            extra_tag = "_".join([f'{k}{v}' for k, v in g["extra"].items()])
            tag = f'{tag}_{extra_tag}'
        out = OUT / f'{tag}.csv'
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'Wrote {out} in {time.time() - t0:.1f}s')


def main():
    run_grid()


if __name__ == "__main__":
    main()