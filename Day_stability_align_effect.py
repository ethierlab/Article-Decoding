# Day_stability_align_effect.py
# Figure: mean-over-muscles per (fold,seed), all decoders, two alignment methods per decoder on the same plot
# Stats: paired Wilcoxon per day, per decoder (aligned vs direct/naive), Holm correction across all tests

import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ---------- fixed colors ----------
DECODER_ORDER = ["GRU", "LSTM", "LiGRU", "Linear"]
DECODER_COLORS = {"GRU": "tab:red", "LSTM": "tab:blue", "LiGRU": "tab:orange", "Linear": "tab:green"}
COND_COLORS    = {"aligned": "tab:blue", "direct": "tab:orange", "naive":"tab:orange", "crossval":"tab:purple"}

# ---------- IO ----------
def load_results(results_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, "crossday_results_*.pkl")))
    if not files:
        raise FileNotFoundError(f"No PKL files in {results_dir} matching crossday_results_*.pkl")
    dfs = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] could not read {f}: {e}")
    if not dfs:
        raise RuntimeError("No results could be loaded.")
    df = pd.concat(dfs, ignore_index=True)

    # normalize expected columns
    if "day_int" not in df.columns:
        # fall back to 'day' if present (already int), else try date → relative days
        if "day" in df.columns:
            df["day_int"] = pd.to_numeric(df["day"], errors="coerce")
        elif "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce")
            df["day_int"] = (d - d.min()).dt.days
        else:
            raise ValueError("No 'day_int', 'day', or 'date' column found.")
    df["day_int"] = pd.to_numeric(df["day_int"], errors="coerce")

    for col, default in [("decoder", "UNK"), ("dim_red", "PCA"), ("align", "aligned"),
                         ("fold", 0), ("seed", 0), ("emg_channel", -1)]:
        if col not in df.columns:
            df[col] = default
    if "vaf" not in df.columns:
        raise ValueError("Expected a 'vaf' column in PKL files.")

    return df

# ---------- helpers ----------
def norm_align_name(x: str) -> str:
    """Map a few aliases to stable names."""
    x = str(x).lower()
    if "align" in x:    return "aligned"
    if "direct" in x:   return "direct"
    if "naive" in x:    return "direct"
    if "cross" in x:    return "crossval"
    return x

def average_over_muscles(df: pd.DataFrame, exclude_channels=None) -> pd.DataFrame:
    """
    Return one row per (decoder, dim_red, align_norm, day, fold, seed)
    with mean VAF across muscles. Keeps 'align_norm' even if the raw
    column was named 'align'.
    """
    sub = df.copy()
    if exclude_channels:
        sub = sub[~sub["emg_channel"].isin(exclude_channels)]

    # prefer normalized alignment if present
    align_col = "align_norm" if "align_norm" in sub.columns else "align"

    keys = ["decoder", "dim_red", align_col, "day_int", "fold", "seed"]
    out = (
        sub.groupby(keys, dropna=False)["vaf"]
           .mean()
           .reset_index(name="vaf_mean_musc")
    )

    # ensure we always have an 'align_norm' column downstream
    if "align_norm" not in out.columns:
        out["align_norm"] = out[align_col].map(norm_align_name)

    return out


def holm_bonferroni(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment (returns adjusted p-values)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for k, i in enumerate(order):
        rank = m - k
        val = pvals[i] * rank
        val = max(val, prev)  # ensure monotone non-decreasing once sorted back
        adj[i] = min(val, 1.0)
        prev = adj[i]
    return adj

# ---------- plotting ----------
def overlay_pair_same_x_per_day(df, decoders, condA="aligned", condB="direct",
                                title="", save=None, ylim=None, cond_eps=0.0):
    """
    Day on x-axis. For each day, and for each decoder in `decoders`,
    draw *two* boxplots (condA & condB) on the SAME x (overlaid).
    Requires columns: day_int, decoder, align_norm, vaf_mean_musc.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # make sure we have the columns
    need = {"day_int","decoder","align_norm","vaf_mean_musc"}
    missing = need - set(df.columns)
    if missing:
        print("[plot] missing columns:", missing); return

    days = sorted(df["day_int"].dropna().unique().tolist())
    if not days or not decoders:
        print("[plot] nothing to plot"); return

    # spacing: keep each decoder at a fixed offset inside the day cluster
    day_gap  = 1.6                                 # gap between day clusters
    base_pos = np.arange(len(days)) * day_gap
    dec_gap  = 0.9                                 # span used by decoders within a day
    dec_offs = np.linspace(-dec_gap/2, dec_gap/2, len(decoders))

    # both conditions share same x; optionally a tiny epsilon to avoid perfect overlap
    cond_offs = np.array([-cond_eps/2, +cond_eps/2])  # set cond_eps=0 for exact same x
    boxw = 0.24

    positions, data, colors, hatches = [], [], [], []
    for di, dec in enumerate(decoders):
        for ci, cond in enumerate([condA, condB]):
            for k, day in enumerate(days):
                x = base_pos[k] + dec_offs[di] + cond_offs[ci]
                vals = df[(df["day_int"]==day) &
                          (df["decoder"]==dec) &
                          (df["align_norm"]==cond)]["vaf_mean_musc"].values
                data.append(vals)
                positions.append(x)
                colors.append("C0" if cond==condA else "C1")
                hatches.append(None if cond==condA else "//")  # optional visual cue

    plt.figure(figsize=(16,6))
    bp = plt.boxplot(data, positions=positions, widths=boxw,
                     showfliers=False, patch_artist=True)

    for patch, c, ht in zip(bp["boxes"], colors, hatches):
        patch.set_facecolor(c); patch.set_alpha(0.40)
        patch.set_edgecolor(c); patch.set_linewidth(1.2)
        if ht is not None: patch.set_hatch(ht)  # makes the overlay obvious

    # tidy lines
    for elem in ["medians","whiskers","caps"]:
        for artist in bp[elem]:
            artist.set_linewidth(1.0)

    # x ticks = real day values
    plt.xticks(base_pos, [str(int(d)) for d in days])
    if ylim is not None: plt.ylim(ylim)
    plt.grid(True, axis="y", alpha=0.25)
    plt.xlabel("Day")
    plt.ylabel("VAF (mean over muscles)")
    plt.title(title)

    legend = [mpatches.Patch(facecolor="C0", alpha=0.40, label=condA),
              mpatches.Patch(facecolor="C1", alpha=0.40, label=condB, hatch="//")]
    plt.legend(handles=legend, title="Alignment", frameon=False, loc="upper right")

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight"); print("saved:", save)
    plt.show()

def grouped_boxplot_by_day(df, decoders, condA, condB, title, save, ylim=None):
    """
    Boxplots par JOUR, avec pour chaque jour:
        [ (decoder1,condA), (decoder1,condB), (decoder2,condA), (decoder2,condB), ... ]
    df doit contenir: day_int, decoder, align_norm, vaf_mean_musc
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    days = sorted(df["day_int"].dropna().unique().tolist())
    if not days:
        print("[plot] no days"); return

    decs  = list(decoders)
    conds = [condA, condB]

    # --- positions: base (jour) + offset (décodeur) + offset (condition) ---
    day_gap = 1.8                         # espace entre blocs de jours
    base = np.arange(len(days)) * day_gap # positions centrales des jours

    span_within_day = 1.2                  # largeur totale allouée à tous les décodeurs dans un jour
    dec_offsets  = np.linspace(-span_within_day/2, span_within_day/2, len(decs))
    cond_offsets = np.array([-0.12, +0.12])   # petit décalage A vs B
    boxw = min(0.10, (span_within_day/len(decs)) * 0.35)

    # palette pour les conditions (fixe → légende claire)
    cond_color = {condA: "C0", condB: "C1"}

    data, positions, colors = [], [], []

    for di, dec in enumerate(decs):
        for ci, cond in enumerate(conds):
            g = df[(df["decoder"] == dec) & (df["align_norm"] == cond)]
            for i, day in enumerate(days):
                vals = g.loc[g["day_int"] == day, "vaf_mean_musc"].values
                data.append(vals if len(vals) else np.array([]))
                pos = base[i] + dec_offsets[di] + cond_offsets[ci]
                positions.append(pos)
                colors.append(cond_color.get(cond, "gray"))

    # --- plot ---
    plt.figure(figsize=(18, 7))
    bp = plt.boxplot(data, positions=positions, widths=boxw,
                     showfliers=False, patch_artist=True)

    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for elem in ["medians", "whiskers", "caps"]:
        for artist in bp[elem]:
            artist.set_linewidth(1.0)

    # ticks = jours
    plt.xticks(base, [str(int(d)) for d in days])
    plt.xlim(base[0] - day_gap*0.6, base[-1] + day_gap*0.6)
    if ylim is not None:
        plt.ylim(ylim)

    # grille et légende (conditions)
    for x in base: plt.axvline(x, color="k", alpha=0.06, linewidth=1)
    plt.grid(True, axis="y", alpha=0.25)
    plt.xlabel("Day"); plt.ylabel("VAF (mean over muscles)")
    plt.title(title)

    handles = [mpatches.Patch(facecolor=cond_color[c], alpha=0.6, label=c) for c in conds]
    plt.legend(handles=handles, title="Alignment", loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight"); print("saved:", save)
    plt.show()


# ---------- stats ----------
def paired_wilcoxon_daywise(df_avg, decoders, condA, condB, out_csv):
    """
    For each day and decoder: paired Wilcoxon between condA vs condB,
    pairing on (fold, seed). Returns the stats DataFrame and writes CSV.
    """
    rows = []
    for dec in decoders:
        sub_dec = df_avg[df_avg["decoder"]==dec]
        days = sorted(sub_dec["day_int"].dropna().unique())
        for d in days:
            ga = sub_dec[(sub_dec["align_norm"]==condA) & (sub_dec["day_int"]==d)]
            gb = sub_dec[(sub_dec["align_norm"]==condB) & (sub_dec["day_int"]==d)]
            if ga.empty or gb.empty:
                continue
            # inner-join on (fold,seed)
            key = ["fold","seed"]
            merged = pd.merge(ga[key+["vaf_mean_musc"]], gb[key+["vaf_mean_musc"]],
                              on=key, how="inner", suffixes=("_A","_B"))
            if len(merged) < 3:
                continue
            stat, p = wilcoxon(merged["vaf_mean_musc_A"], merged["vaf_mean_musc_B"], zero_method="wilcox", alternative="two-sided", method="exact")
            rows.append({
                "decoder": dec,
                "day": int(d),
                "test": "wilcoxon_paired",
                "W": float(stat),
                "p": float(p),
                "n_pairs": int(len(merged)),
                "median_A": float(np.median(merged["vaf_mean_musc_A"])),
                "median_B": float(np.median(merged["vaf_mean_musc_B"])),
                "median_diff_BminusA": float(np.median(merged["vaf_mean_musc_B"] - merged["vaf_mean_musc_A"]))
            })
    if not rows:
        print("[stats] no comparable day/decoder pairs found.")
        return pd.DataFrame()

    stats = pd.DataFrame(rows).sort_values(["decoder","day"]).reset_index(drop=True)

    # Holm across ALL tests together
    stats["p_holm"] = holm_bonferroni(stats["p"].values)

    if out_csv:
        stats.to_csv(out_csv, index=False)
        print("saved:", out_csv)
    return stats

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Alignment effect across decoders (mean over muscles).")
    ap.add_argument("--results_dir", type=str, default=".", help="Where the PKLs live")
    ap.add_argument("--out_dir", type=str, default="figs_align_effect", help="Output folder")
    ap.add_argument("--dim_red", type=str, default="PCA", choices=["PCA","UMAP"], help="Which DR to plot")
    ap.add_argument("--cond_a", type=str, default="aligned", help="Alignment A (e.g., aligned)")
    ap.add_argument("--cond_b", type=str, default="direct",  help="Alignment B (e.g., direct/naive)")
    ap.add_argument("--exclude_channels", nargs="*", type=int, default=None, help="Channels to drop (e.g., 0 5 6)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_results(args.results_dir)
    # normalize alignment labels & keep only requested pair
    df["align_norm"] = df["align"].map(norm_align_name)
    condA = norm_align_name(args.cond_a)
    condB = norm_align_name(args.cond_b)

    df = df[df["dim_red"] == args.dim_red]
    df = df[df["align_norm"].isin([condA, condB])].copy()
    if df.empty:
        raise RuntimeError("No rows match the requested dim_red/alignment pair.")

    # avg across muscles → distribution across fold×seed
    df_avg = average_over_muscles(df, exclude_channels=args.exclude_channels)

    # plot
    title = "All decoders • PCA • mean over muscles • aligned vs direct (overlay per decoder)"
    out   = os.path.join(args.out_dir, "aligned_vs_direct_overlay_per_day.png")

    overlay_pair_same_x_per_day(
        df=df_avg[df_avg["dim_red"]=="PCA"],
        decoders=present_decoders,
        condA="aligned",
        condB="direct",
        title=title,
        save=out,
        ylim=(-0.5, 1.05),
        cond_eps=0.00   # set to 0.06 if you want a tiny side-by-side nudge
    )

    # stats
    csv_path = os.path.join(args.out_dir, f"align_effect_stats_{args.dim_red}_{condA}_vs_{condB}.csv")
    stats = paired_wilcoxon_daywise(df_avg, present_decoders, condA, condB, csv_path)
    if not stats.empty:
        print(stats.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
