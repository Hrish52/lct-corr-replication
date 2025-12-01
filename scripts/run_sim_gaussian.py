# scripts/run_sim_gaussian.py
# Gaussian + non-Gaussian grids with BH/BY, LCT-N (Cai–Liu), and LCT-B.
# Supports fast flags: --skip-lctb, --B-list, --n-jobs, and progress per rep.

import sys, pathlib, os, argparse, csv, time
from pathlib import Path
import numpy as np
import argparse

# repo root on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# repo root on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.FisherBaselines import two_group_z_stat, pvals_from_Z, bh_threshold, by_threshold
from src.Simulate import (
    make_block_cov, sample_gaussian, sample_t, sample_laplace, sample_exp,
    upper_tri_pairs, truth_mask_block
)
from src.LCT import lct_edge_stat, lct_threshold_normal
try:
    from src.LCTB_v2 import lct_threshold_bootstrap as lctb  # Day-11 fast path
except ImportError:
    from src.LCTB import lct_threshold_bootstrap as lctb

from src.defaults import get_defaults_for  # Day-12 defaults resolver

OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

# cache upper-tri indices once per p
_IU_CACHE = {}
def tri_pairs(p: int):
    if p not in _IU_CACHE:
        _IU_CACHE[p] = upper_tri_pairs(p)
    return _IU_CACHE[p]

# globals for speed flags (set by argparse in run_grid)
_SKIP_LCTB = False
_B_LIST = None
_N_JOBS = -1
# Day-12: defaults toggles
_USE_DEFAULTS = False
_DEFAULTS_FILE = "results/defaults.json"

def _dataset(model: str, n1: int, n2: int, p: int, rho: float, block: int, seed: int, extra: dict):
    """
    Return X (null group ~ N(0, I)) and Y with dependence per `model`,
    using same top-left block Sigma for fair truth.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n1, p))
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

def run_once(model="gaussian", p=250, n1=80, n2=80, rho=0.30, block=20, seed=0, extra=None):
    extra = extra or {}
    t_start = time.perf_counter()

    # data
    X, Y = _dataset(model, n1, n2, p, rho, block, seed, extra)

    # Fisher baselines
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

    # BH / BY
    for alpha in (0.05, 0.10):
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

    # LCT-N (Cai–Liu variance)
    T, _, _ = lct_edge_stat(X, Y, var_method="cai_liu")
    for alpha in (0.05, 0.10):
        t_hat, mask = lct_threshold_normal(T, alpha=alpha)
        R = int(mask.sum()); V = int((~truth & mask).sum()); S = int((truth & mask).sum())
        m1 = int(truth.sum())
        row.update({
            f"t_lct_{alpha}": float(t_hat),
            f"R_lct_{alpha}": R, f"V_lct_{alpha}": V, f"S_lct_{alpha}": S,
            f"fdr_lct_{alpha}": V / max(R, 1),
            f"power_lct_{alpha}": S / max(m1, 1),
        })

    # LCT-B (bootstrap thresholds)
    if _SKIP_LCTB:
        B_list = []
    else:
        if _B_LIST is not None:
            B_list = _B_LIST
        else:
            B_list = [100, 200, 500] if p == 250 else [100]
    # Day-12: if using defaults and no explicit B-list, sentinel to resolve per-α
    if _USE_DEFAULTS and _B_LIST is None:
        B_list = [-1]

    for B in B_list:
        for alpha in (0.05, 0.10):
            # Resolve defaults (only if requested and using sentinel)
            var_method = "cai_liu"
            wins = None
            kwargs_extra = {}
            B_eff = B
            if _USE_DEFAULTS and _B_LIST is None:
                d = get_defaults_for(p, alpha, path=_DEFAULTS_FILE) or {}
                if "B" in d and d["B"] is not None:
                    B_eff = int(d["B"])
                if d.get("coarse_grid") is not None:
                    kwargs_extra["coarse_grid"] = int(d["coarse_grid"])
                if d.get("winsorize") is not None:
                    wins = float(d["winsorize"])
                if d.get("var_method"):
                    var_method = str(d["var_method"])

            # Call LCT-B
            t_b, mask_b, info_b = lctb(
                X, Y, alpha=alpha, B=B_eff, var_method=var_method,
                winsorize=wins, n_jobs=_N_JOBS, rng=seed, **kwargs_extra
            )

            # Tally results
            Rb = int(mask_b.sum())
            Vb = int((~truth & mask_b).sum())
            Sb = int((truth & mask_b).sum())
            m1 = int(truth.sum())
            row.update({
                f"t_lctb_{alpha}_B{B_eff}": float(t_b),
                f"R_lctb_{alpha}_B{B_eff}": Rb, f"V_lctb_{alpha}_B{B_eff}": Vb, f"S_lctb_{alpha}_B{B_eff}": Sb,
                f"fdr_lctb_{alpha}_B{B_eff}": Vb / max(Rb, 1),
                f"power_lctb_{alpha}_B{B_eff}": Sb / max(m1, 1),
            })

    row["wall_time_s"] = round(time.perf_counter() - t_start, 6)
    return row


def run_grid():
    # --- CLI ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="gaussian|t|laplace|exp")
    parser.add_argument("--p-only", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    # speed knobs
    parser.add_argument("--skip-lctb", action="store_true", help="skip LCT-B for speed")
    parser.add_argument("--B-list", type=str, default=None,
                        help="comma list for LCT-B, e.g. '50,100'. Default: p=250 -> 100,200,500; else 100.")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="workers for LCT-B; on Windows prefer 1 to avoid spawn overhead.")
    # Day-12: defaults toggles
    parser.add_argument("--use-defaults", action="store_true",
                        help="Use results/defaults.json to auto-set (B, coarse_grid, winsorize, var_method) per (p, α).")
    parser.add_argument("--defaults-file", type=str, default="results/defaults.json",
                        help="Path to defaults.json (from scripts/make_defaults.py).")
    args = parser.parse_args()

    # set globals for run_once
    global _SKIP_LCTB, _B_LIST, _N_JOBS, _USE_DEFAULTS, _DEFAULTS_FILE
    _SKIP_LCTB = args.skip_lctb
    _B_LIST = None if args.B_list is None else [int(x) for x in args.B_list.split(",") if x.strip()]
    win = (os.name == "nt")
    _N_JOBS = 1 if (win and args.n_jobs is None) else (args.n_jobs if args.n_jobs is not None else -1)
    _USE_DEFAULTS = bool(args.use_defaults)
    _DEFAULTS_FILE = args.defaults_file

    # --- grids ---
    grids = []
    # Gaussian
    grids += [
        dict(model="gaussian", p=250, n1=80, n2=80, rho=0.30, block=20, reps=50, extra={}),
        dict(model="gaussian", p=500, n1=80, n2=80, rho=0.25, block=20, reps=50, extra={}),
    ]
    # t(df=6)
    grids += [
        dict(model="t", p=250, n1=80, n2=80, rho=0.30, block=20, reps=50, extra={"df": 6}),
        dict(model="t", p=500, n1=80, n2=80, rho=0.25, block=20, reps=50, extra={"df": 6}),
    ]
    # Laplace (b=1/sqrt(2))
    grids += [
        dict(model="laplace", p=250, n1=80, n2=80, rho=0.30, block=20, reps=50, extra={"b": 1/np.sqrt(2)}),
        dict(model="laplace", p=500, n1=80, n2=80, rho=0.25, block=20, reps=50, extra={"b": 1/np.sqrt(2)}),
    ]
    # Exponential(1), centered & z-scored
    grids += [
        dict(model="exp", p=250, n1=80, n2=80, rho=0.30, block=20, reps=50, extra={"rate": 1.0}),
        dict(model="exp", p=500, n1=80, n2=80, rho=0.25, block=20, reps=50, extra={"rate": 1.0}),
    ]

    # filter by CLI
    if args.model:
        grids = [g for g in grids if g["model"] == args.model]
    if args.p_only is not None:
        grids = [g for g in grids if g["p"] == args.p_only]
    if args.reps is not None:
        for g in grids:
            g["reps"] = args.reps

    # --- run ---
    for g in grids:
        rows = []
        t0 = time.time()
        for r in range(g["reps"]):
            rows.append(run_once(
                model=g["model"], p=g["p"], n1=g["n1"], n2=g["n2"],
                rho=g["rho"], block=g["block"], seed=r, extra=g["extra"]
            ))
            # progress per rep
            print(f'  [{g["model"]}, p={g["p"]}] rep {r+1}/{g["reps"]} '
                  f'done; last wall_time_s={rows[-1]["wall_time_s"]:.2f}')
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
