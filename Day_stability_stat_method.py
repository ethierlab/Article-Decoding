#!/usr/bin/env python3
# Day0_validation_stats.py
# Day-0 held-out CV validation:
# - build Day-0 units (VAF averaged over muscles; pairing on fold×seed)
# - violin per decoder  (saved as ...__violin.png)
# - boxplot per decoder (saved as ...__box.png)
# - paired Wilcoxon across decoders (two-sided), Holm step-down correction

import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import wilcoxon

# Consistent color scheme
DECODER_COLORS = {"GRU":"#d62728", "LSTM":"#1f77b4", "LiGRU":"#ff7f0e", "Linear":"#2ca02c"}
DECODER_ORDER = ["GRU", "LSTM", "LiGRU", "Linear"]

# -------------------------- helpers --------------------------
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def norm_align_name(x: str) -> str:
    """Map various spellings to {'aligned','direct','crossval'}."""
    s = str(x).lower()
    if "align" in s:  return "aligned"
    if "direct" in s: return "direct"
    if "cross" in s:  return "crossval"
    return s

def canon_decoder(x: str) -> str:
    s = str(x).lower()
    if s == "ligru":  return "LiGRU"
    if s == "lstm":   return "LSTM"
    if s == "gru":    return "GRU"
    if s == "linear": return "Linear"
    return x

def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (returns array aligned to input order)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)               # ascending
    adj = np.empty(m, float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        k = m - rank                    # (m, m-1, ..., 1)
        val = k * p[idx]
        val = max(val, running_max)     # enforce monotonicity
        adj[idx] = min(val, 1.0)
        running_max = adj[idx]
    return adj

# -------------------------- IO / loading --------------------------
def load_all_pkls(results_dir: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, "crossday_results_*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No PKL files match {pattern}")
    dfs = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            dfs.append(df)
            print(f"[load] {f}  shape={df.shape}")
        except Exception as e:
            print(f"[warn] Could not read {f}: {e}")
    if not dfs:
        raise RuntimeError("No results could be loaded.")
    df = pd.concat(dfs, ignore_index=True)

    # Required columns (with a few flexible names)
    if "decoder" not in df.columns:
        raise ValueError("Missing 'decoder' column.")
    if "dim_red" not in df.columns:
        raise ValueError("Missing 'dim_red' column.")
    align_col = "align" if "align" in df.columns else ("alignment" if "alignment" in df.columns else None)
    if align_col is None:
        raise ValueError("Need 'align' (or 'alignment') column.")
    need_cols = ["fold", "emg_channel", "vaf"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in results.")
    # day index
    if "day_int" not in df.columns:
        if "day" in df.columns:
            df["day_int"] = pd.to_numeric(df["day"], errors="coerce")
        else:
            raise ValueError("Need 'day_int' (or 'day') to identify Day-0.")

    # Standardize basics
    df["day_int"] = pd.to_numeric(df["day_int"], errors="coerce")
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    df["emg_channel"] = pd.to_numeric(df["emg_channel"], errors="coerce")

    # Normalize align names and decoder labels
    df["align_norm"] = df[align_col].apply(norm_align_name)
    df["decoder_canon"] = df["decoder"].apply(canon_decoder)

    # Seed/replicate column (optional)
    rep_col = None
    for cand in ["replicate_id", "replicate", "seed", "run_id", "session"]:
        if cand in df.columns:
            rep_col = cand
            break
    if rep_col is None:
        rep_col = "replicate_id"
        df[rep_col] = 0
    df["rep_col_name"] = rep_col
    df[rep_col] = pd.to_numeric(df[rep_col], errors="coerce").fillna(0).astype(int)

    return df

# --------------------- build Day-0 units -----------------
def average_over_muscles(df: pd.DataFrame, exclude_channels=None) -> pd.DataFrame:
    """Return one row per (decoder, dim_red, align_norm, day_int, fold, seed/rep) with mean VAF across muscles."""
    rep_col = df["rep_col_name"].iloc[0]
    g = df.copy()
    if exclude_channels:
        g = g[~g["emg_channel"].isin(exclude_channels)]
    keys = ["decoder_canon", "dim_red", "align_norm", "day_int", "fold", rep_col]
    out = (
        g.groupby(keys, dropna=False)["vaf"]
         .mean()
         .reset_index()
         .rename(columns={"vaf": "vaf_mean_musc", rep_col: "seed"})
    )
    # Pairing key = fold × seed
    out["unit_id"] = out["fold"].astype(str) + "_" + out["seed"].astype(str)
    return out

def day0_units(df_avg: pd.DataFrame, dim_red="PCA", align_norm="crossval") -> pd.DataFrame:
    """Filter to Day-0 for the chosen condition and keep the pairing columns."""
    sub = df_avg[
        (df_avg["dim_red"] == dim_red) &
        (df_avg["align_norm"] == align_norm) &
        (df_avg["day_int"] == 0)
    ].copy()
    if sub.empty:
        raise ValueError("No rows after filtering to day 0 + dim_red + align_norm.")
    units = sub.rename(columns={
        "decoder_canon": "decoder",
        "vaf_mean_musc": "VAF_unit"
    })[["decoder", "dim_red", "align_norm", "day_int", "unit_id", "fold", "seed", "VAF_unit"]]
    return units

# --------------------- plotting (separate figs) -------------------
def _prepare_plot_arrays(units: pd.DataFrame):
    # order decoders consistently
    decoders_present = [d for d in DECODER_ORDER if d in units["decoder"].unique().tolist()]
    data = [units.loc[units["decoder"] == d, "VAF_unit"].values for d in decoders_present]
    meds = [np.median(x) if len(x) else np.nan for x in data]
    return decoders_present, data, meds

def plot_violin_day0(units: pd.DataFrame, out_png: str):
    decoders, data, meds = _prepare_plot_arrays(units)
    plt.figure(figsize=(8, 6))
    parts = plt.violinplot(data, showextrema=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_alpha(0.35)
        b.set_facecolor(DECODER_COLORS.get(decoders[i], "gray"))
    # jitter points
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0: continue
        x = np.full_like(vals, i, dtype=float) + rng.uniform(-0.07, 0.07, size=len(vals))
        plt.scatter(x, vals, s=18, alpha=0.9, c=DECODER_COLORS.get(decoders[i-1], "gray"))
    # medians
    plt.scatter(np.arange(1, len(meds)+1), meds, s=30, c="black", zorder=10, label="Median")
    plt.xticks(np.arange(1, len(decoders)+1), decoders)
    plt.ylabel("VAF (Day 0, mean over muscles)")
    # plt.title("Day-0 Validation • Violin (cross-validated)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"[save] {out_png}")
    plt.close()

def plot_box_day0(units: pd.DataFrame, out_png: str):
    decoders, data, _ = _prepare_plot_arrays(units)
    plt.figure(figsize=(8, 6))
    bp = plt.boxplot(data, patch_artist=True, showmeans=False)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(DECODER_COLORS.get(decoders[i], "gray"))
        patch.set_alpha(0.35)
    for whisker in bp["whiskers"]:
        whisker.set(color="black", alpha=0.6)
    for cap in bp["caps"]:
        cap.set(color="black", alpha=0.6)
    for median in bp["medians"]:
        median.set(color="black", linewidth=2)
    plt.xticks(np.arange(1, len(decoders)+1), decoders)
    plt.ylabel("VAF (Day 0, mean over muscles)")
    # plt.title("Day-0 Validation • Boxplot (cross-validated)")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"[save] {out_png}")
    plt.close()

# --------------------- stats (Wilcoxon + Holm) -------------------
def wilcoxon_paired_table(units: pd.DataFrame,
                          decoders=None,
                          alternative: str = "two-sided",
                          holm_correction: bool = True) -> pd.DataFrame:
    """
    Paired Wilcoxon across decoders (pairing by unit_id = fold×seed).
    Day-0 comparisons are two-sided (per your methods).
    We report median_diff_AminusB with a consistent sign.
    """
    if decoders is None:
        decoders = sorted(units["decoder"].dropna().unique())
    rows = []
    for A, B in combinations(decoders, 2):
        Ua = units[units.decoder == A].set_index("unit_id")["VAF_unit"]
        Ub = units[units.decoder == B].set_index("unit_id")["VAF_unit"]
        common = Ua.index.intersection(Ub.index)
        if len(common) < 2:
            rows.append([A, B, "wilcoxon_paired", np.nan, np.nan, len(common),
                         np.nan, np.nan, np.nan])
            continue
        x = Ua.loc[common].values  # A
        y = Ub.loc[common].values  # B
        diffs = x - y              # A minus B
        method = "exact" if len(diffs) <= 25 else "approx"
        stat, p = wilcoxon(diffs, zero_method="wilcox",
                           alternative=alternative, method=method)
        rows.append([
            A, B, "wilcoxon_paired",
            float(stat), float(p), int(len(common)),
            float(np.median(x)), float(np.median(y)),
            float(np.median(diffs))
        ])
    res = pd.DataFrame(rows, columns=[
        "decoder_A","decoder_B","test","W","p_value","n_pairs",
        "median_A","median_B","median_diff_AminusB"
    ])
    if holm_correction and not res["p_value"].isna().all():
        res["p_holm"] = holm_bonferroni(res["p_value"].values)
    return res

# --------------------- main ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Day-0 validation stats (CV) + separate violin & boxplot")
    ap.add_argument("--results_dir", type=str, default=".", help="Folder with crossday_results_*.pkl")
    ap.add_argument("--out_dir",     type=str, default="figs_day0", help="Where to save outputs")
    ap.add_argument("--dim_red",     type=str, default="PCA", choices=["PCA","UMAP"], help="Dimensionality reduction")
    ap.add_argument("--align",       type=str, default="crossval", help="Alignment condition for Day-0 (e.g., crossval)")
    ap.add_argument("--exclude_channels", nargs="*", type=int, default=None,
                    help="EMG channels to exclude BEFORE averaging (e.g., 0 5 6)")
    ap.add_argument("--decoders", nargs="*", type=str, default=None,
                    help="Which decoders to keep (default: all present)")
    ap.add_argument("--out_prefix", type=str, default="day0_validation",
                    help="Prefix for output files")
    args = ap.parse_args()

    df = load_all_pkls(args.results_dir)
    df_avg = average_over_muscles(df, exclude_channels=args.exclude_channels)
    align_norm = norm_align_name(args.align)
    units = day0_units(df_avg, dim_red=args.dim_red, align_norm=align_norm)

    if args.decoders:
        keep = {canon_decoder(d) for d in args.decoders}
        units = units[units["decoder"].isin(keep)].copy()

    # quick sanity: how many points per decoder?
    counts = units.groupby("decoder")["VAF_unit"].count().rename("n_points")
    print("\n[summary] points per decoder (≈ #folds × #seeds at Day-0):")
    print(counts.to_string())

    # outputs
    ensure_dir(args.out_dir)
    suf_excl = (f"__excl_{'-'.join(map(str, args.exclude_channels))}"
                if args.exclude_channels else "")
    base = f"{args.out_prefix}__{args.dim_red}__{align_norm}{suf_excl}"

    # separate figures
    out_png_violin = os.path.join(args.out_dir, base + "__violin.png")
    out_png_box    = os.path.join(args.out_dir, base + "__box.png")
    plot_violin_day0(units, out_png_violin)
    plot_box_day0(units, out_png_box)

    # stats
    decs = sorted(units["decoder"].unique())
    res = wilcoxon_paired_table(units, decoders=decs, alternative="two-sided", holm_correction=True)
    out_csv = os.path.join(args.out_dir, base + "__stats.csv")
    res.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}\n")
    print(res.to_string(index=False))

if __name__ == "__main__":
    main()
