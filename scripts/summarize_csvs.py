import glob, pandas as pd, re

def summarize(path_pattern="results/tables/*_p*.csv"):
    rows = []
    for p in sorted(glob.glob(path_pattern)):
        df = pd.read_csv(p)
        mean = df.mean(numeric_only=True)
        row = {"file": p, "Avg_wall_time_s": mean.get("wall_time_s", float("nan"))}
        for c in df.columns:
            if c.startswith(("fdr_","power_")):
                row[f"mean_{c}"] = mean.get(c, float("nan"))
        m = re.search(r'/(gaussian|t|laplace|exp)_p(\d+)_n(\d+)_(\d+)_rho([0-9.]+)_b(\d+)_R(\d+)', p)
        if m:
            row.update({
                "model": m.group(1), "p": int(m.group(2)), "n1": int(m.group(3)),
                "n2": int(m.group(4)), "rho": float(m.group(5)), "block": int(m.group(6)),
                "reps": int(m.group(7)),
            })
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    summary = summarize()
    print(summary.to_string(index=False))
