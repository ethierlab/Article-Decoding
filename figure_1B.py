# avg_muscle_plots_day0.py
import os, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- fixed colors per decoder ----------
DECODER_COLORS = {
    "GRU":   "red",
    "LSTM":  "tab:blue",
    "Linear":"tab:green",
    "LiGRU":"tab:orange"
}

# Fixed plotting order on x-axis
DECODER_ORDER = ["Linear", "LSTM", "LiGRU", "GRU"]


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

    # Optional: recompute day_int from true dates
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

    if "emg_channel" not in all_df.columns:
        all_df["emg_channel"] = -1
    if "fold" not in all_df.columns:
        all_df["fold"] = 0

    return all_df


# ---------- helpers ----------
def _auto_ylim_from_series(yvals, pad_frac=0.03):
    y = np.asarray(yvals, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (0.0, 1.0)
    ymin, ymax = float(y.min()), float(y.max())
    pad = (ymax - ymin) * pad_frac
    if pad == 0:
        pad = 0.01 * max(1.0, abs(ymin))
    return (ymin - pad, ymax + pad)


def build_day0_muscle_points(
    df,
    day_int=0,
    include_channels=None,
    exclude_channels=None,
    dim_red="PCA",
    align="crossval"
):
    """
    Retourne un DataFrame où chaque ligne = un point:
        moyenne VAF sur les muscles sélectionnés pour un (decoder, fold[, seed])
    Filtré sur:
        - day_int == day_int
        - dim_red
        - align
        - include_channels / exclude_channels
    """
    d = df.copy()

    # filtres principaux
    d = d[d["dim_red"] == dim_red]
    d = d[d["align"] == align]

    # jour choisi (par défaut 0)
    if "day_int" not in d.columns:
        raise ValueError("Column 'day_int' not found in results DataFrame.")
    d = d[d["day_int"] == day_int]

    # filtrage EMG channels
    if include_channels is not None and len(include_channels) > 0:
        d = d[d["emg_channel"].isin(include_channels)]
    elif exclude_channels is not None and len(exclude_channels) > 0:
        d = d[~d["emg_channel"].isin(exclude_channels)]

    if d.empty:
        print("[build_day0_muscle_points] No data left after filtering.")
        return pd.DataFrame(columns=["decoder", "fold", "vaf_muscles"])

    # group keys: average across selected muscles
    keys = ["decoder", "fold"]
    if "seed" in d.columns:
        keys.append("seed")

    g = (
        d.groupby(keys, dropna=False)["vaf"]
         .mean()
         .reset_index()
         .rename(columns={"vaf": "vaf_muscles"})
    )
    return g


def box_by_decoder_single_day(df_points, muscle_label="", ylim=None, save=None):
    """
    Boxplot: un box par décodeur (Linear, LSTM, LiGRU, GRU).
    df_points doit contenir colonnes: decoder, vaf_muscles.
    """
    if df_points.empty:
        print("[box_by_decoder_single_day] nothing to plot")
        return

    decoders_present = df_points["decoder"].dropna().unique().tolist()
    decoders = [d for d in DECODER_ORDER if d in decoders_present]
    if not decoders:
        print("[box_by_decoder_single_day] no known decoders present")
        return

    data = [df_points[df_points["decoder"] == dec]["vaf_muscles"].values
            for dec in decoders]
    positions = np.arange(len(decoders))

    plt.figure(figsize=(6, 5))
    bp = plt.boxplot(
        data,
        positions=positions,
        widths=0.6,
        showfliers=False,
        patch_artist=True
    )

    # couleurs
    for patch, dec in zip(bp["boxes"], decoders):
        c = DECODER_COLORS.get(dec, "gray")
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    for elem in ["medians", "whiskers", "caps"]:
        for artist in bp[elem]:
            artist.set_linewidth(1.0)

    plt.xticks(positions, decoders)
    if ylim is None:
        ymin, ymax = _auto_ylim_from_series(df_points["vaf_muscles"])
        plt.ylim(ymin, ymax)
    else:
        plt.ylim(*ylim)

    plt.grid(True, axis="y", alpha=0.25)
    plt.ylabel("VAF")
    if muscle_label:
        plt.title(muscle_label)

    plt.margins(x=0.05)
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
        print("saved:", save)

    plt.show()


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Boxplot jour 0 par décodeur, avec filtrage de muscles."
    )
    ap.add_argument("--results_dir", type=str, required=True,
                    help="Folder with crossday_results_*.pkl")
    ap.add_argument("--out_dir", type=str, default="figs_day0_muscles",
                    help="Where to save figures")
    ap.add_argument("--day_int", type=int, default=0,
                    help="Which day_int to use (default: 0 = jour de training)")
    ap.add_argument("--include_channels", nargs="+", type=int, default=None,
                    help="EMG channel indices to INCLUDE (0-based). "
                         "If set, exclude_channels is ignored.")
    ap.add_argument("--exclude_channels", nargs="+", type=int, default=None,
                    help="EMG channel indices to EXCLUDE (0-based). "
                         "Used only if include_channels is not provided.")
    ap.add_argument("--align", type=str, default="crossval",
                    choices=["aligned", "direct", "crossval"],
                    help="Alignment type (for jour 0, crossval a du sens).")
    ap.add_argument("--dimred", type=str, default="PCA",
                    choices=["PCA", "UMAP"],
                    help="Dimensionality reduction used in results.")
    ap.add_argument("--muscle_label", type=str, default="",
                    help="Label to put in figure title, e.g. 'Extensors'.")
    ap.add_argument("--recalc_day_from_date", action="store_true",
                    help="Recompute day_int from recording dates.")
    args = ap.parse_args()

    df = load_results(args.results_dir, recalc_day_from_date=args.recalc_day_from_date)

    pts = build_day0_muscle_points(
        df,
        day_int=args.day_int,
        include_channels=args.include_channels,
        exclude_channels=args.exclude_channels,
        dim_red=args.dimred,
        align=args.align
    )

    if pts.empty:
        print("No points to plot after filtering; check your options.")
        return

    out_dir = ensure_dir(args.out_dir)

    out_file = os.path.join(
        out_dir,
        f"day{args.day_int}_box_{args.dimred}_{args.align}.png"
        if not args.muscle_label
        else f"day{args.day_int}_box_{args.dimred}_{args.align}_{args.muscle_label.replace(' ', '_')}.png"
    )

    box_by_decoder_single_day(
        pts,
        muscle_label=args.muscle_label,
        ylim=(0, 1.05),
        save=out_file
    )


if __name__ == "__main__":
    main()
