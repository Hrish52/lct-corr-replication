# scripts/make_defaults.py
import argparse, json, re
from pathlib import Path
import pandas as pd
import numpy as np

TABLES = Path("results") / "tables"
OUTDIR = Path("results")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "defaults.json"

ALPHA_RE = re.compile(r"_(0\.05|0\.10)")
B_RE = re.compile(r"_B(\d+)")

def _collect_frames(patterns):
    files = []
    for pat in patterns:
        files.extend(sorted(TABLES.glob(pat)))
    if not files:
        raise SystemExit(f"No CSVs found under {TABLES} for patterns: {patterns}")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f).assign(file=f.name))
        except Exception as e:
            print(f"[warn] skipping {f}: {e}")
    if not dfs:
        raise SystemExit("No readable CSVs.")
    return pd.concat(dfs, ignore_index=True)

def _melt_fdp(df):
    fcols = [c for c in df.columns if isinstance(c, str) and c.startswith("fdp_lctb_")]
    if not fcols:
        raise SystemExit("No fdp_lctb_* columns found. Run run_null_calibration first.")
    long = df.melt(id_vars=[c for c in df.columns if c not in fcols],
                   value_vars=fcols, var_name="metric", value_name="fdp")
    long["alpha"] = long["metric"].apply(lambda s: (ALPHA_RE.search(s) or [None, None])[1])
    long["B"] = long["metric"].apply(lambda s: int(B_RE.search(s).group(1)) if B_RE.search(s) else np.nan)
    long = long.dropna(subset=["alpha","B"])
    long["B"] = long["B"].astype(int)
    return long

def _melt_runtime(df):
    rcols = [c for c in df.columns if isinstance(c, str)
             and c.startswith("lctb_B") and c.endswith("_wall_time_s")]
    if not rcols:
        return pd.DataFrame(columns=["B","sec"])
    rt = df.melt(id_vars=[c for c in df.columns if c not in rcols],
                 value_vars=rcols, var_name="metric", value_name="sec")
    rt["B"] = rt["metric"].str.extract(r'B(\d+)', expand=False).astype(int)
    return rt[["B","sec"]]

def choose_B_for_alpha(sub_fdp, sub_rt, alpha, fdr_tol, prefer_low_runtime=True):
    """Return (B_star, agg_fdp DataFrame)."""
    agg = (sub_fdp.groupby("B")["fdp"]
           .agg(mean="mean", se=lambda x: x.std(ddof=1)/np.sqrt(len(x)))
           .reset_index())
    thresh = float(alpha) + float(fdr_tol)
    qualified = agg[agg["mean"] <= thresh].sort_values(["B"])
    if not qualified.empty:
        if prefer_low_runtime and not sub_rt.empty:
            rt = sub_rt.groupby("B")["sec"].mean().reset_index().rename(columns={"sec":"rt"})
            q = qualified.merge(rt, on="B", how="left").sort_values(["rt","B"])
            return int(q.iloc[0]["B"]), agg
        return int(qualified.iloc[0]["B"]), agg
    # fallback: smallest mean FDP (then smallest B)
    agg = agg.sort_values(["mean","B"])
    return int(agg.iloc[0]["B"]), agg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", type=str,
        default="nullcal_*_p250_*.csv,nullcal_*_p500_*.csv,nullcal_*_p1000_*.csv",
        help="Comma-separated glob patterns under results/tables/")
    ap.add_argument("--alphas", type=str, default="0.05,0.10")
    ap.add_argument("--fdr-tol", type=float, default=0.00, help="Allowed excess over alpha.")
    ap.add_argument("--winsorize", type=float, default=None, help="Recorded default (e.g., 5).")
    ap.add_argument("--var-method", type=str, default="cai_liu",
        choices=["cai_liu","gaussian","jackknife"])
    ap.add_argument("--coarse-grid-when-p-ge", type=int, default=500)
    ap.add_argument("--coarse-grid-K", type=int, default=200)
    ap.add_argument("--outfile", type=str, default=str(OUTFILE))
    args = ap.parse_args()

    patterns = [s.strip() for s in args.patterns.split(",") if s.strip()]
    df = _collect_frames(patterns)

    # infer p if not present
    if "p" not in df.columns:
        df["p"] = df["file"].str.extract(r'_p(\d+)_', expand=False).astype(int)

    long_fdp = _melt_fdp(df)
    rt_long = _melt_runtime(df)
    alphas = [a.strip() for a in args.alphas.split(",")]

    result = {}
    for p_val in sorted(df["p"].dropna().astype(int).unique()):
        result[str(p_val)] = {}
        for alpha in alphas:
            sub = long_fdp[(long_fdp["p"] == p_val) & (long_fdp["alpha"] == alpha)]
            if sub.empty:
                print(f"[warn] no FDP rows for p={p_val}, alpha={alpha}")
                continue
            B_star, _ = choose_B_for_alpha(sub, rt_long, alpha, args.fdr_tol)
            coarse = None
            if args.coarse_grid_when_p_ge and p_val >= args.coarse_grid_when_p_ge:
                coarse = int(args.coarse_grid_K)
            result[str(p_val)][alpha] = {
                "B": int(B_star),
                "coarse_grid": coarse,
                "winsorize": args.winsorize,
                "var_method": args.var_method,
            }
            print(f"[defaults] p={p_val} α={alpha}: B={B_star} "
                  f"coarse_grid={coarse} winsorize={args.winsorize} var={args.var_method}")

    with open(args.outfile, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.outfile}")

if __name__ == "__main__":
    main()