import glob, pandas as pd

def summarize(path_pattern):
    rows = []
    for p in sorted(glob.glob(path_pattern)):
        df = pd.read_csv(p)
        mean = df.mean(numeric_only=True)
        rows.append({
            "file": p,
            "BH_FDR@0.05": mean["fdr_bh_0.05"],
            "BH_Power@0.05": mean["power_bh_0.05"],
            "BY_FDR@0.05": mean["fdr_by_0.05"],
            "BY_Power@0.05": mean["power_by_0.05"],
            "Avg_wall_time_s": mean["wall_time_s"],
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    summary = summarize("results/tables/gaussian_*.csv")
    print(summary.to_string(index=False))
