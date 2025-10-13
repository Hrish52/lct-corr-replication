# scripts/run_null_calibration.py
import sys, pathlib, csv, time, argparse, os
from pathlib import Path
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.Simulate import sample_gaussian, sample_t, sample_laplace, sample_exp, truth_mask_block
from src.LCT import lct_edge_stat, lct_threshold_normal
from src.LCTB import lct_threshold_bootstrap

OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

def make_null(model, n1, n2, p, seed, extra):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n1, p))  # null group ~ N(0, I)
    I = np.eye(p)
    if model == "gaussian":
        Y = sample_gaussian(n2, I, seed=seed)
    elif model == "t":
        Y = sample_t(n2, df=extra.get("df", 6), Sigma=I, rng=seed)
    elif model == "laplace":
        Y = sample_laplace(n2, b=extra.get("b", 1/np.sqrt(2)), Sigma=I, rng=seed)
    elif model == "exp":
        Y = sample_exp(n2, rate=extra.get("rate", 1.0), Sigma=I, rng=seed, zscore=True)
    else:
        raise ValueError("Unknown model")
    return X, Y

def run_once(model, p, n1, n2, seed, alpha_list, B_list, var_method="cai_liu", winsorize=None, n_jobs=-1, extra=None):
    extra = extra or {}
    t0 = time.perf_counter()
    X, Y = make_null(model, n1, n2, p, seed, extra)

    # truth under null: no edges (all False)
    truth_mask = truth_mask_block(p, block=0)
    iu, ju = np.triu_indices(p, 1)
    M = iu.size

    row = {"model": model, "p": p, "n1": n1, "n2": n2, "seed": seed}

    # LCT-N
    T, _, _ = lct_edge_stat(X, Y, var_method=var_method, winsorize=winsorize)
    for alpha_s in alpha_list:
        a = float(alpha_s)  # cast to float for computation
        t_n, mask_n = lct_threshold_normal(T, alpha=a)
        Rn = int(mask_n.sum())
        row.update({
            f"t_lct_{alpha_s}": float(t_n),
            f"R_lct_{alpha_s}": Rn,
            f"fdp_lct_{alpha_s}": (1.0 if Rn > 0 else 0.0),
            f"any_reject_lct_{alpha_s}": int(Rn > 0),
            f"R_over_M_lct_{alpha_s}": Rn / M,
        })

    # LCT-B
    for B in B_list:
        tB = time.perf_counter()
        for alpha_s in alpha_list:
            a = float(alpha_s)  # cast to float for computation
            t_b, mask_b, _ = lct_threshold_bootstrap(
                X, Y, alpha=a, B=B, var_method=var_method, n_jobs=n_jobs, rng=seed
            )
            Rb = int(mask_b.sum())
            row.update({
                f"t_lctb_{alpha_s}_B{B}": float(t_b),
                f"R_lctb_{alpha_s}_B{B}": Rb,
                f"fdp_lctb_{alpha_s}_B{B}": (1.0 if Rb > 0 else 0.0),
                f"any_reject_lctb_{alpha_s}_B{B}": int(Rb > 0),
                f"R_over_M_lctb_{alpha_s}_B{B}": Rb / M,
            })
        row[f"lctb_B{B}_wall_time_s"] = time.perf_counter() - tB

    row["wall_time_s"] = time.perf_counter() - t0
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="gaussian|t|laplace|exp (default: all)")
    ap.add_argument("--p", type=int, default=250)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--n1", type=int, default=80)
    ap.add_argument("--n2", type=int, default=80)
    ap.add_argument("--alpha", type=str, default="0.05,0.10")
    ap.add_argument("--B", type=str, default="50,100,200,500")
    ap.add_argument("--winsorize", type=float, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()

    models = ["gaussian", "t", "laplace", "exp"] if args.model is None else [args.model]
    alphas = [a.strip() for a in args.alpha.split(",")]
    Bs = [int(b) for b in args.B.split(",")]

    # Windows-friendly default
    win = (os.name == "nt")
    n_jobs = 1 if (win and args.n_jobs is None) else (args.n_jobs if args.n_jobs is not None else -1)

    for model in models:
        rows = []
        for seed in range(args.reps):
            extra = {}
            if model == "t": extra = {"df": 6}
            if model == "laplace": extra = {"b": 1/np.sqrt(2)}
            if model == "exp": extra = {"rate": 1.0}
            rows.append(run_once(
                model=model, p=args.p, n1=args.n1, n2=args.n2, seed=seed,
                alpha_list=alphas, B_list=Bs, var_method="cai_liu",
                winsorize=args.winsorize, n_jobs=n_jobs, extra=extra
            ))

        tag = f"nullcal_{model}_p{args.p}_n{args.n1}_{args.n2}_R{args.reps}"
        out = OUT / f"{tag}.csv"
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out} (rows={len(rows)})")

if __name__ == "__main__":
    main()
