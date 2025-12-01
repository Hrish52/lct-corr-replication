# scripts/run_robustness.py
import sys, pathlib, os, argparse, csv, time
from pathlib import Path
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.FisherBaselines import two_group_z_stat, pvals_from_Z, bh_threshold, by_threshold
from src.Simulate import (
    sample_gaussian, sample_t, sample_laplace, sample_exp,
    make_block_cov, make_block_ar1_cov, make_block_decay_cov,
    truth_mask_block, upper_tri_pairs
)
from src.LCT import lct_edge_stat, lct_threshold_normal
try:
    from src.LCTB_v2 import lct_threshold_bootstrap   # faster impl
except ImportError:
    from src.LCTB import lct_threshold_bootstrap      # fallback to original

from src.defaults import get_defaults_for  # Day-12: resolver for results/defaults.json

OUT = Path("results/tables"); OUT.mkdir(parents=True, exist_ok=True)
_IU = {}
def tri_pairs(p):
    if p not in _IU: _IU[p] = upper_tri_pairs(p)
    return _IU[p]

def _Sigma(kind: str, p: int, rho: float, block: int, decay: float):
    if kind == "block":       return make_block_cov(p, rho=rho, block_size=block)
    if kind == "block_ar1":   return make_block_ar1_cov(p, rho=rho, block_size=block)
    if kind == "block_decay": return make_block_decay_cov(p, rho=rho, block_size=block, decay=decay)
    raise ValueError(f"Unknown cov-kind: {kind}")

def _dataset(model: str, n: int, p: int, Sigma: np.ndarray, seed: int, extra: dict):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))  # null group
    if model == "gaussian":
        Y = sample_gaussian(n, Sigma, seed=seed)
    elif model == "t":
        Y = sample_t(n, df=extra.get("df", 6), Sigma=Sigma, rng=seed)
    elif model == "laplace":
        Y = sample_laplace(n, b=extra.get("b", 1/np.sqrt(2)), Sigma=Sigma, rng=seed)
    elif model == "exp":
        Y = sample_exp(n, rate=extra.get("rate", 1.0), Sigma=Sigma, rng=seed, zscore=True)
    else:
        raise ValueError(f"Unknown model: {model}")
    return X, Y

def run_once(model, p, n1, n2, rho, block, cov_kind, var_methods, B_list, seed,
             decay, winsorize, n_jobs, extra, use_defaults=False, defaults_file="results/defaults.json"):
    t0 = time.perf_counter()
    Sigma = _Sigma(cov_kind, p, rho=rho, block=block, decay=decay)

    # Group 1 (null): size n1; Group 2 (signal): size n2 with Sigma
    X1, _ = _dataset("gaussian", n1, p, np.eye(p), seed, {})     # null group ~ N(0, I)
    _,  Y = _dataset(model,     n2, p, Sigma,      seed+12345, extra or {})

    iu, ju = tri_pairs(p)
    truth = truth_mask_block(p, block)
    row = {"model": model, "p": p, "n1": n1, "n2": n2, "rho": rho, "block": block,
           "cov_kind": cov_kind, "decay": decay, "seed": seed}

    # Fisher baselines
    R1 = np.corrcoef(X1, rowvar=False)
    R2 = np.corrcoef(Y,  rowvar=False)
    Z  = two_group_z_stat(R1, R2, n1, n2)
    pvals = pvals_from_Z(Z)[iu, ju]
    for a in (0.05, 0.10):
        sel_bh = bh_threshold(pvals, a)
        sel_by = by_threshold(pvals, a)
        Rb, RY = int(sel_bh.sum()), int(sel_by.sum())
        Vb, VY = int((~truth & sel_bh).sum()), int((~truth & sel_by).sum())
        Sb, SY = int((truth & sel_bh).sum()),  int((truth & sel_by).sum())
        m1 = int(truth.sum())
        row.update({
            f"R_bh_{a}": Rb, f"V_bh_{a}": Vb, f"S_bh_{a}": Sb,
            f"R_by_{a}": RY, f"V_by_{a}": VY, f"S_by_{a}": SY,
            f"fdr_bh_{a}": Vb / max(Rb, 1),  f"power_bh_{a}": Sb / max(m1, 1),
            f"fdr_by_{a}": VY / max(RY, 1),  f"power_by_{a}": SY / max(m1, 1),
        })

    # LCT-N ablation (var_method ∈ {cai_liu, gaussian, jackknife})
    for vm in var_methods:
        T, _, _ = lct_edge_stat(X1, Y, var_method=vm, winsorize=winsorize)
        for a in (0.05, 0.10):
            t_hat, mask = lct_threshold_normal(T, alpha=a)
            R = int(mask.sum()); V = int((~truth & mask).sum()); S = int((truth & mask).sum())
            m1 = int(truth.sum())
            row.update({
                f"t_lct_{vm}_{a}": float(t_hat),
                f"R_lct_{vm}_{a}": R, f"V_lct_{vm}_{a}": V, f"S_lct_{vm}_{a}": S,
                f"fdr_lct_{vm}_{a}": V / max(R, 1), f"power_lct_{vm}_{a}": S / max(m1, 1),
            })

    # LCT-B (use first var_method unless defaults override)
    eff_B_list = (B_list or [])
    if use_defaults and not eff_B_list:
        eff_B_list = [-1]  # sentinel -> resolve per α from defaults.json

    for B in eff_B_list:
        for a in (0.05, 0.10):
            vm = var_methods[0] if var_methods else "cai_liu"
            wins = winsorize
            kwargs_extra = {}
            B_eff = B
            if use_defaults and B == -1:
                d = (get_defaults_for(p, a, path=defaults_file) or {})
                if "B" in d and d["B"] is not None:
                    B_eff = int(d["B"])
                if d.get("coarse_grid") is not None:
                    kwargs_extra["coarse_grid"] = int(d["coarse_grid"])
                if d.get("winsorize") is not None:
                    wins = float(d["winsorize"])
                if d.get("var_method"):
                    vm = str(d["var_method"])

            t_b, mask_b, _ = lct_threshold_bootstrap(
                X1, Y, alpha=a, B=B_eff, var_method=vm,
                winsorize=wins, n_jobs=n_jobs, rng=seed, **kwargs_extra
            )
            Rb = int(mask_b.sum()); Vb = int((~truth & mask_b).sum()); Sb = int((truth & mask_b).sum())
            m1 = int(truth.sum())
            row.update({
                f"t_lctb_{a}_B{B_eff}": float(t_b),
                f"R_lctb_{a}_B{B_eff}": Rb, f"V_lctb_{a}_B{B_eff}": Vb, f"S_lctb_{a}_B{B_eff}": Sb,
                f"fdr_lctb_{a}_B{B_eff}": Vb / max(Rb, 1), f"power_lctb_{a}_B{B_eff}": Sb / max(m1, 1),
            })

    row["wall_time_s"] = round(time.perf_counter() - t0, 6)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gaussian", help="gaussian|t|laplace|exp")
    ap.add_argument("--p", type=int, default=250)
    ap.add_argument("--rho-list", type=str, default="0.25,0.30")
    ap.add_argument("--n1-list", type=str, default="60,80,120")
    ap.add_argument("--n2-list", type=str, default="60,80,120")
    ap.add_argument("--cov-kind", type=str, default="block_ar1", help="block|block_ar1|block_decay")
    ap.add_argument("--decay", type=float, default=0.6, help="only for block_decay")
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--var-methods", type=str, default="cai_liu,gaussian,jackknife")
    ap.add_argument("--B-list", type=str, default="")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--winsorize", type=float, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    # Day-12: defaults toggles
    ap.add_argument("--use-defaults", action="store_true",
                    help="Use results/defaults.json to auto-set (B, coarse_grid, winsorize, var_method) per (p, α).")
    ap.add_argument("--defaults-file", type=str, default="results/defaults.json",
                    help="Path to defaults.json (from scripts/make_defaults.py).")
    args = ap.parse_args()

    rhos = [float(x) for x in args.rho_list.split(",")]
    n1s  = [int(x)   for x in args.n1_list.split(",")]
    n2s  = [int(x)   for x in args.n2_list.split(",")]
    var_methods = [s.strip() for s in args.var_methods.split(",") if s.strip()]

    if args.B_list:
        B_list = [int(x) for x in args.B_list.split(",")]
    else:
        B_list = [100, 200] if args.p == 250 else [50, 100]
    # If using defaults and no explicit B_list, let run_once resolve per α
    if args.use_defaults and not args.B_list:
        B_list = []

    win = (os.name == "nt")
    n_jobs = 1 if (win and args.n_jobs is None) else (args.n_jobs if args.n_jobs is not None else -1)

    extra = {}
    if args.model == "t":       extra = {"df": 6}
    if args.model == "laplace": extra = {"b": 1/np.sqrt(2)}
    if args.model == "exp":     extra = {"rate": 1.0}

    for n1 in n1s:
        for n2 in n2s:
            for rho in rhos:
                rows = []
                for seed in range(args.reps):
                    rows.append(run_once(
                        model=args.model, p=args.p, n1=n1, n2=n2, rho=rho, block=args.block,
                        cov_kind=args.cov_kind, var_methods=var_methods, B_list=B_list,
                        seed=seed, decay=args.decay, winsorize=args.winsorize,
                        n_jobs=n_jobs, extra=extra,
                        use_defaults=bool(args.use_defaults), defaults_file=args.defaults_file
                    ))
                    if (seed+1) % 5 == 0 or seed == 0:
                        print(f"[{args.model} p={args.p} cov={args.cov_kind} n1={n1} n2={n2} rho={rho}] rep {seed+1}/{args.reps}")

                tag = f"robust_{args.model}_p{args.p}_cov{args.cov_kind}_n{n1}_{n2}_rho{rho}_b{args.block}_R{args.reps}"
                out = OUT / f"{tag}.csv"
                with out.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)
                print("Wrote", out)

if __name__ == "__main__":
    main()