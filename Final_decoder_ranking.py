import os, glob, argparse
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

# ---------------- IO ----------------
def load_results(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "test/crossday_results_*.pkl")))
    if not files:
        raise FileNotFoundError("No crossday_results_*.pkl files in --results_dir")
    df = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)

    # normalize columns we need
    if "day_int" in df: df["day_int"] = pd.to_numeric(df["day_int"], errors="coerce")
    elif "day" in df:   df["day_int"] = pd.to_numeric(df["day"], errors="coerce")
    else: raise ValueError("Need 'day_int' or 'day' in PKLs")

    if "decoder" not in df.columns: raise ValueError("Need 'decoder'")
    if "dim_red" not in df.columns: raise ValueError("Need 'dim_red'")
    if "vaf" not in df.columns: raise ValueError("Need 'vaf'")
    if "fold" not in df.columns: df["fold"] = 0
    if "seed" not in df.columns: df["seed"] = 0
    if "emg_channel" not in df.columns: df["emg_channel"] = -1

    col = "align" if "align" in df.columns else ("alignment" if "alignment" in df.columns else None)
    if col is None: raise ValueError("Need 'align'")
    v = df[col].astype(str).str.lower()
    df["align_norm"] = np.where(v.str.contains("align"), "aligned",
                         np.where(v.str.contains("direct"), "direct",
                                  np.where(v.str.contains("cross"), "crossval", v)))
    return df

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def holm(pvals):
    p = np.array(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, float); prev = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * p[idx]
        val = max(val, prev)
        adj[idx] = min(val, 1.0)
        prev = adj[idx]
    return adj

# ---------------- core ----------------
def mean_over_muscles(df):
    """Mean VAF over EMG channels per (decoder, dim_red, align, day, fold, seed)."""
    keys = ["decoder","dim_red","align_norm","day_int","fold","seed"]
    out = df.groupby(keys, dropna=False)["vaf"].mean().reset_index()
    out = out.rename(columns={"vaf":"vaf_mean_musc"})
    return out

def day_level_median(df_avg, dim_red="PCA", align="aligned"):
    """Median across pairs (fold×seed) per day for each decoder."""
    g = df_avg[(df_avg["dim_red"]==dim_red) & (df_avg["align_norm"]==align)].copy()
    g["pair_id"] = g["fold"].astype(str) + "_" + g["seed"].astype(str)
    # collapse to day-level medians per decoder
    tbl = g.groupby(["day_int","decoder"], dropna=False)["vaf_mean_musc"].median().reset_index()
    tbl = tbl.rename(columns={"day_int":"day","vaf_mean_musc":"day_median"})
    return tbl

def make_complete_matrix(tbl):
    """Return matrix (days×decoders) with complete cases only, plus the list of decoders & days used."""
    decoders = sorted(tbl["decoder"].unique().tolist())
    # matrix with possibly missing
    M = tbl.pivot(index="day", columns="decoder", values="day_median")
    # keep only days where all decoders are present
    M = M.dropna(axis=0, how="any")
    used_days = M.index.values.astype(int).tolist()
    return M, decoders, used_days

def friedman_and_pairwise(M):
    """Friedman across decoders; then pairwise Wilcoxon across days (rows)."""
    # Friedman
    series = [M[c].values for c in M.columns]
    stat, p = friedmanchisquare(*series)

    # average ranks (higher VAF = better rank 1)
    # rank per row (day): argsort descending
    ranks = np.vstack([(-M.values[i]).argsort().argsort()+1 for i in range(M.shape[0])])
    # Convert to average rank per decoder keeping column order
    avg_ranks = ranks.mean(axis=0)

    # pairwise Wilcoxon (blocked by day)
    rows = []
    C = list(M.columns)
    for i in range(len(C)):
        for j in range(i+1, len(C)):
            a = M[C[i]].values
            b = M[C[j]].values
            # paired across days
            diffs = a - b
            # need variability
            if np.allclose(diffs, diffs[0]):
                W, p_ab = (0.0, 1.0)  # degenerate
            else:
                mode = "exact" if len(diffs) <= 25 else "approx"
                W, p_ab = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", mode=mode)
            rows.append({
                "decoder_A": C[i],
                "decoder_B": C[j],
                "test": "Wilcoxon (days)",
                "W": float(W),
                "p": float(p_ab),
                "n_days": int(M.shape[0]),
                "median_A": float(np.median(a)),
                "median_B": float(np.median(b)),
                "median_diff_AminusB": float(np.median(a-b))
            })
    pairwise = pd.DataFrame(rows)
    if not pairwise.empty:
        pairwise["p_holm"] = holm(pairwise["p"].values)
    return stat, p, avg_ranks, pairwise

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Final ranking of decoders across days (non-parametric).")
    ap.add_argument("--results_dir", type=str, default=".")
    ap.add_argument("--out_dir", type=str, default="stats_rank")
    ap.add_argument("--dim_red", type=str, default="PCA", choices=["PCA","UMAP"])
    ap.add_argument("--align", type=str, default="aligned", choices=["aligned","direct"])
    args = ap.parse_args()

    df = load_results(args.results_dir)
    df_avg = mean_over_muscles(df)
    day_tbl = day_level_median(df_avg, dim_red=args.dim_red, align=args.align)
    M, decoders, used_days = make_complete_matrix(day_tbl)

    outdir = ensure_dir(args.out_dir)
    if M.shape[0] < 2:
        raise RuntimeError("Not enough complete days to compare decoders.")

    stat, p, avg_ranks, pairwise = friedman_and_pairwise(M)

    # Save matrix and results
    M.to_csv(os.path.join(outdir, f"day_medians_{args.dim_red}_{args.align}.csv"))
    summary = pd.DataFrame({"decoder": M.columns, "avg_rank": avg_ranks}).sort_values("avg_rank")
    summary.to_csv(os.path.join(outdir, f"ranking_{args.dim_red}_{args.align}.csv"), index=False)
    pairwise.to_csv(os.path.join(outdir, f"pairwise_{args.dim_red}_{args.align}.csv"), index=False)



    print("Used days (complete across decoders):", used_days)
    print("Saved results in", outdir)

if __name__ == "__main__":
    main()
