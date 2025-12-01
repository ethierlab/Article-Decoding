# avg_muscle_plots.py
import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------- fixed colors per decoder ----------
DECODER_COLORS = {"GRU":"red","LSTM":"tab:blue","Linear":"tab:green","LiGRU":"tab:orange"}
# Fixed plotting order
DECODER_ORDER = ["GRU", "LSTM", "LiGRU", "Linear"]  # note: "Linear" and "LiGRU" capitalization

# ---------- IO ----------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_results(results_dir, recalc_day_from_date=False):
    pattern = os.path.join(results_dir, "crossday_results_*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No PKL files match {pattern}")
    dfs = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            dfs.append(df)
            print(f"Loaded {f}  shape={df.shape}")
        except Exception as e:
            print(f"Could not read {f}: {e}")
    if not dfs:
        raise RuntimeError("No results could be loaded.")
    all_df = pd.concat(dfs, ignore_index=True)

    # Optional: force global timeline from real recording dates
    if recalc_day_from_date:
        date_col = "day" if "day" in all_df.columns else ("date" if "date" in all_df.columns else None)
        if date_col:
            all_df[date_col] = pd.to_datetime(all_df[date_col], errors="coerce")
            base = all_df[date_col].min()
            all_df["day_int"] = (all_df[date_col] - base).dt.days
            print(f"[INFO] Recomputed day_int from {date_col}. Baseline = {base.date()}")
        else:
            print("[WARN] No 'day' or 'date' column found; using stored day_int.")

    if "day_int" in all_df:
        all_df["day_int"] = pd.to_numeric(all_df["day_int"], errors="coerce")

    # Normalize columns we may rely on
    if "emg_channel" not in all_df.columns:
        all_df["emg_channel"] = -1
    if "fold" not in all_df.columns:
        all_df["fold"] = 0

    return all_df

# ---------- helpers ----------
def build_avg_muscles_points(df, exclude_channels=(0,5,6),
                             decoder=None, dim_red=None, align=None,
                             per_day=True):
    """
    Each row = one point averaged across muscles (channels not excluded).
    Keys preserved: decoder, dim_red, align, day_int, fold (and seed if present).
    If per_day=True (default) you get one point per (day, fold[,seed]).
    """
    d = df.copy()
    if decoder: d = d[d["decoder"]==decoder]
    if dim_red: d = d[d["dim_red"]==dim_red]
    if align:   d = d[d["align"]==align]

    d = d[~d["emg_channel"].isin(exclude_channels)]

    keys = ["decoder","dim_red","align","day_int","fold"]
    if "seed" in d.columns:
        keys.append("seed")

    g = (
        d.groupby(keys, dropna=False)["vaf"]
         .mean()
         .reset_index()
         .rename(columns={"vaf":"vaf_avg_muscles"})
    )
    return g

def _auto_ylim_from_series(yvals, pad_frac=0.03):
    y = np.asarray(yvals, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (0.0, 1.0)
    ymin, ymax = float(y.min()), float(y.max())
    pad = (ymax - ymin) * pad_frac
    # in case of very flat data
    if pad == 0:
        pad = 0.01 * max(1.0, abs(ymin))
    return (ymin - pad, ymax + pad)


def violin_avg_muscles_by_day_grouped(df_points, title="", save=None, ylim=None):
    """
    Grouped violin: per day on x-axis; within each day, one violin per decoder.
    df_points must have columns: decoder, day_int, vaf_avg_muscles.
    """
    if df_points.empty:
        print("[violin-grouped] nothing to plot"); return

    days   = sorted(df_points["day_int"].dropna().unique())
    present = df_points["decoder"].dropna().unique().tolist()
    decs = [d for d in DECODER_ORDER if d in present]
    # denser packing
    day_gap = 1.10
    base_pos = np.arange(len(days), dtype=float) * day_gap
    G = len(decs)
    max_span = 0.90
    offsets = np.linspace(-max_span/2, max_span/2, G)
    widths  = min(0.32, (max_span/G) * 0.95)

    data, positions, colors, med_x, med_y = [], [], [], [], []
    for gi, dec in enumerate(decs):
        gdec = df_points[df_points["decoder"]==dec]
        for i, d in enumerate(days):
            vals = gdec[gdec["day_int"]==d]["vaf_avg_muscles"].values
            data.append(vals if len(vals) else np.array([]))
            x = base_pos[i] + offsets[gi]
            positions.append(x)
            colors.append(DECODER_COLORS.get(dec, None))
            if len(vals):
                med_x.append(x)
                med_y.append(np.median(vals))

    plt.figure(figsize=(12,5))
    parts = plt.violinplot(data, positions=positions, widths=widths, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        if c: body.set_facecolor(c)
        body.set_alpha(0.50)

    if med_x:
        plt.scatter(med_x, med_y, s=18, color="k", zorder=3, label="Median")

    plt.xticks(base_pos, [str(int(d)) for d in days])

    # auto y-lims if not provided
    if ylim is None:
        ymin, ymax = _auto_ylim_from_series(df_points["vaf_avg_muscles"])
        plt.ylim(ymin, ymax)
    else:
        plt.ylim(*ylim)

    plt.grid(True, axis="y", alpha=0.25)
    plt.xlabel("Number of days since model trained"); plt.ylabel("VAF (avg across muscles)")
    # plt.title(title)

    # legend ABOVE to avoid right-side blank column
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=DECODER_COLORS.get(dec,"gray"), alpha=0.50, label=dec) for dec in decs]
    handles.append(Patch(color="black", label="Median"))
    plt.legend(handles=handles, title="Decoder", ncol=len(handles),
               loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)

    plt.margins(x=0.01)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight"); print("saved:", save)
    plt.show()


def box_by_day_grouped(df, x="day_int", y="y", group="decoder", title="", save=None, ylim=None):
    """
    Grouped boxplot per day with extra spacing to avoid overlap between days.
    Expects columns: 'day_int', 'decoder', and y='y'.
    """
    if df.empty:
        print("[box] nothing to plot"); return

    days = sorted(df[x].dropna().unique())
    present = df[group].dropna().unique().tolist()
    groups = [d for d in DECODER_ORDER if d in present]

    # denser packing
    day_gap = 1.10
    base_positions = np.arange(len(days), dtype=float) * day_gap
    G = len(groups)
    max_group_span = 0.80
    offsets = np.linspace(-max_group_span/2, max_group_span/2, G)
    boxw = min(0.22, (max_group_span / G) * 0.85)

    data, positions, colors = [], [], []
    for gi, gv in enumerate(groups):
        gdf = df[df[group] == gv]
        for i, d in enumerate(days):
            vals = gdf[gdf[x] == d][y].values
            data.append(vals if len(vals) else np.array([]))
            positions.append(base_positions[i] + offsets[gi])
            colors.append(DECODER_COLORS.get(gv, None) if group == "decoder" else None)

    plt.figure(figsize=(12,5))
    bp = plt.boxplot(data, positions=positions, widths=boxw, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        if c is not None:
            patch.set_facecolor(c); patch.set_alpha(0.60)
    for elem in ["medians", "whiskers", "caps"]:
        for artist in bp[elem]:
            artist.set_linewidth(1.0)

    plt.xticks(base_positions, [str(int(d)) for d in days])

    # auto y-lims if not provided
    if ylim is None:
        ymin, ymax = _auto_ylim_from_series(df[y])
        plt.ylim(ymin, ymax)
    else:
        plt.ylim(*ylim)

    plt.grid(True, axis="y", alpha=0.25)
    plt.xlabel("Number of days since model trained"); plt.ylabel("VAF") 

    # legend ABOVE (no right padding)
    from matplotlib.patches import Patch
    if group == "decoder":
        handles = [Patch(facecolor=DECODER_COLORS.get(g, "gray"), alpha=0.60, label=str(g))
                   for g in groups]
        plt.legend(handles=handles, title="Decoder", ncol=len(handles),
                   loc="upper center", bbox_to_anchor=(0.5, 1.15), frameon=False)

    plt.margins(x=0.01)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=200, bbox_inches="tight"); print("saved:", save)
    plt.show()


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Avg-across-muscles violin & box plots (PCA • align=aligned • no day 0).")
    ap.add_argument("--results_dir", type=str, required=True, help="Folder with crossday_results_*.pkl")
    ap.add_argument("--out_dir", type=str, default="figs_avg_muscles", help="Where to save figures")
    ap.add_argument("--exclude_channels", nargs="+", type=int, default=[0,5,6], help="Channels to exclude")
    ap.add_argument("--exclude_days", nargs="+", type=int, default=[0], help="Days to exclude (default: 0)")
    ap.add_argument("--align", type=str, default="aligned", choices=["aligned","direct","crossval"], help="Alignment")
    ap.add_argument("--dimred", type=str, default="PCA", choices=["PCA","UMAP"], help="Dimensionality reduction")
    ap.add_argument("--recalc_day_from_date", action="store_true",
                    help="Recompute day_int from real recording dates across all files")
    args = ap.parse_args()

    df = load_results(args.results_dir, recalc_day_from_date=args.recalc_day_from_date)

    # Build avg-across-muscles points with your filters
    pts = build_avg_muscles_points(
        df,
        exclude_channels=tuple(args.exclude_channels),
        decoder=None,          # all decoders on same plots
        dim_red=args.dimred,   # PCA as requested
        align=args.align,      # aligned (not direct)
        per_day=True
    )

    # Drop excluded days (default: day 0)
    if len(args.exclude_days):
        pts = pts[~pts["day_int"].isin(set(args.exclude_days))]

    # Remove NaN days if any
    pts = pts.dropna(subset=["day_int"])

    # ---- VIOLIN ----
    out_violin = ensure_dir(os.path.join(args.out_dir, "violin_avg_muscles_grouped"))
    violin_avg_muscles_by_day_grouped(
        pts,
        title=f"Avg across muscles • {args.dimred} • align={args.align}",
        save=os.path.join(out_violin, "violin_grouped.png"),
        ylim=(0, 1.05)
    )

    # ---- BOX ----
    out_box = ensure_dir(os.path.join(args.out_dir, "box_avg_muscles_grouped"))
    df_box = pts.rename(columns={"vaf_avg_muscles":"y"})
    box_by_day_grouped(
        df_box,
        x="day_int",
        y="y",
        group="decoder",
        title=f"Avg across muscles • {args.dimred} • align={args.align}",
        save=os.path.join(out_box, "box_grouped.png"),
        ylim=(0, 1.05)
    )

if __name__ == "__main__":
    main()
