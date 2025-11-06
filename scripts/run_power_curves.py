# scripts/run_power_curves.py
import sys, pathlib, os, argparse, csv, time
from pathlib import Path
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.FisherBaselines import two_group_z_stat, pvals_from_Z, bh_threshold, by_threshold
from src.Simulate import make_block_cov, sample_gaussian, sample_t, sample_laplace, sample_exp, upper_tri_pairs, truth_mask_block
from src.LCT import lct_edge_stat, lct_threshold_normal
try:
    from src.LCTB_v2 import lct_threshold_bootstrap   # faster impl
except ImportError:
    from src.LCTB import lct_threshold_bootstrap      # fallback to original

OUT = Path("results/tables")
OUT.mkdir(parents=True, exist_ok=True)

_IU_CACHE = {}
def tri_pairs(p: int):
    if p not in _IU_CACHE:
        _IU_CACHE[p] = upper_tri_pairs(p)
    return _IU_CACHE[p]

def _dataset(model: str, n: int, p: int, rho: float, block: int, seed: int, extra: dict):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    Sigma = make_block_cov(p, rho=rho, block_size=block)
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

def run_once(model="gaussian", p=250, n=80, rho=0.30, block=20, seed=0, extra=None, winsorize=None, B_list=None, n_jobs=-1):
    t0 = time.perf_counter()
    X, Y = _dataset(model, n, p, rho, block, seed, extra or {})
    truth = truth_mask_block(p, block)
    iu, ju = tri_pairs(p)

    # Fisher baselines
    R1 = np.corrcoef(X, rowvar=False)
    R2 = np.corrcoef(Y, rowvar=False)
    Z  = two_group_z_stat(R1, R2, n, n)
    pvals = pvals_from_Z(Z)[iu, ju]

    row = {"model": model, "p": p, "n": n, "rho": rho, "block": block, "seed": seed}
    for alpha in (0.05, 0.10):
        sel_bh = bh_threshold(pvals, alpha)
        sel_by = by_threshold(pvals, alpha)
        Rb, RY = int(sel_bh.sum()), int(sel_by.sum())
        Vb, VY = int((~truth & sel_bh).sum()), int((~truth & sel_by).sum())
        Sb, SY = int((truth & sel_bh).sum()),  int((truth & sel_by).sum())
        m1 = int(truth.sum())
        row.update({
            f"R_bh_{alpha}": Rb, f"V_bh_{alpha}": Vb, f"S_bh_{alpha}": Sb,
            f"R_by_{alpha}": RY, f"V_by_{alpha}": VY, f"S_by_{alpha}": SY,
            f"fdr_bh_{alpha}": Vb / max(Rb, 1),  f"power_bh_{alpha}": Sb / max(m1, 1),
            f"fdr_by_{alpha}": VY / max(RY, 1),  f"power_by_{alpha}": SY / max(m1, 1),
        })

    # LCT-N (Cai–Liu)
    T, _, _ = lct_edge_stat(X, Y, var_method="cai_liu", winsorize=winsorize)
    absT = np.abs(T[iu, ju])
    for alpha in (0.05, 0.10):
        t_hat, mask = lct_threshold_normal(T, alpha=alpha)
        R = int(mask.sum()); V = int((~truth & mask).sum()); S = int((truth & mask).sum())
        m1 = int(truth.sum())
        row.update({
            f"t_lct_{alpha}": float(t_hat),
            f"R_lct_{alpha}": R, f"V_lct_{alpha}": V, f"S_lct_{alpha}": S,
            f"fdr_lct_{alpha}": V / max(R, 1), f"power_lct_{alpha}": S / max(m1, 1),
        })

    # LCT-B
    for B in (B_list or []):
        for alpha in (0.05, 0.10):
            t_b, mask_b, info_b = lct_threshold_bootstrap(
                X, Y, alpha=alpha, B=B, var_method="cai_liu", n_jobs=n_jobs, rng=seed, winsorize=winsorize
            )
            Rb = int(mask_b.sum()); Vb = int((~truth & mask_b).sum()); Sb = int((truth & mask_b).sum())
            m1 = int(truth.sum())
            row.update({
                f"t_lctb_{alpha}_B{B}": float(t_b),
                f"R_lctb_{alpha}_B{B}": Rb, f"V_lctb_{alpha}_B{B}": Vb, f"S_lctb_{alpha}_B{B}": Sb,
                f"fdr_lctb_{alpha}_B{B}": Vb / max(Rb, 1), f"power_lctb_{alpha}_B{B}": Sb / max(m1, 1),
            })

    row["wall_time_s"] = round(time.perf_counter() - t0, 6)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default="gaussian,t,laplace,exp",
                    help="comma list: gaussian,t,laplace,exp")
    ap.add_argument("--p", type=int, default=250)
    ap.add_argument("--n-list", type=str, default="60,80,120")
    ap.add_argument("--rho-list", type=str, default="0.20,0.25,0.30,0.35")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--B-list", type=str, default="")
    ap.add_argument("--winsorize", type=float, default=None)
    ap.add_argument("--n-jobs", type=int, default=None)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    n_list = [int(x) for x in args.n_list.split(",")]
    rho_list = [float(x) for x in args.rho_list.split(",")]

    # Default B per p if not given (Windows-friendly)
    if args.B_list:
        B_list = [int(x) for x in args.B_list.split(",")]
    else:
        B_list = [100, 200] if args.p == 250 else [50, 100]

    win = (os.name == "nt")
    n_jobs = 1 if (win and args.n_jobs is None) else (args.n_jobs if args.n_jobs is not None else -1)

    for model in models:
        extra = {}
        if model == "t": extra = {"df": 6}
        if model == "laplace": extra = {"b": 1/np.sqrt(2)}
        if model == "exp": extra = {"rate": 1.0}

        for n in n_list:
            for rho in rho_list:
                rows = []
                for r in range(args.reps):
                    rows.append(run_once(model=model, p=args.p, n=n, rho=rho, block=20, seed=r,
                                         extra=extra, winsorize=args.winsorize, B_list=B_list, n_jobs=n_jobs))
                    if (r+1) % 5 == 0 or r == 0:
                        print(f"[{model} p={args.p} n={n} rho={rho:.2f}] rep {r+1}/{args.reps} done")

                tag = f"power_{model}_p{args.p}_n{n}_rho{rho}_b20_R{args.reps}"
                out = OUT / f"{tag}.csv"
                with out.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)
                print(f"Wrote {out}")

if __name__ == "__main__":
    main()