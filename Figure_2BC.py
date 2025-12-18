#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figures B et C : stabilité du VAF au fil des jours.

B: un décodeur (par défaut GRU), aligned vs direct.
C: tous les décodeurs, aligned vs direct.

Exemples:

python figure_BC_day_stability.py \
  --results_dir crossday_results \
  --out_dir figs_day_stab \
  --which B \
  --decoder_B GRU \
  --exclude_channels 0 5 6

python figure_BC_day_stability.py \
  --results_dir crossday_results \
  --out_dir figs_day_stab \
  --which C \
  --exclude_channels 0 5 6
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DECODER_ORDER = ["GRU", "LSTM", "LiGRU", "Linear"]
DECODER_COLORS = {
    "GRU": "tab:red",
    "LSTM": "tab:blue",
    "LiGRU": "tab:orange",
    "Linear": "tab:green",
}


# ---------------------------------------------------------------------
# IO + helpers
# ---------------------------------------------------------------------
def norm_align_name(x: str) -> str:
    x = str(x).lower()
    if "align" in x:
        return "aligned"
    if "direct" in x:
        return "direct"
    if "naive" in x:
        return "direct"
    if "cross" in x:
        return "crossval"
    return x


def load_results(results_dir: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, "crossday_results_*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No PKL files in {results_dir} matching crossday_results_*.pkl")
    dfs = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            dfs.append(df)
            print(f"[INFO] Loaded {f} shape={df.shape}")
        except Exception as e:
            print(f"[WARN] could not read {f}: {e}")
    if not dfs:
        raise RuntimeError("No results could be loaded.")
    df = pd.concat(dfs, ignore_index=True)

    # day_int
    if "day_int" not in df.columns:
        if "day" in df.columns:
            df["day_int"] = pd.to_numeric(df["day"], errors="coerce")
        elif "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce")
            df["day_int"] = (d - d.min()).dt.days
        else:
            raise ValueError("No 'day_int', 'day', or 'date' column found.")
    df["day_int"] = pd.to_numeric(df["day_int"], errors="coerce")

    # colonnes minimales
    for col, default in [
        ("decoder", "UNK"),
        ("dim_red", "PCA"),
        ("align", "aligned"),
        ("fold", 0),
        ("seed", 0),
        ("emg_channel", -1),
    ]:
        if col not in df.columns:
            df[col] = default

    if "vaf" not in df.columns:
        raise ValueError("Expected a 'vaf' column in PKL files.")

    return df


def average_over_muscles(df: pd.DataFrame, exclude_channels=None) -> pd.DataFrame:
    sub = df.copy()
    if exclude_channels:
        sub = sub[~sub["emg_channel"].isin(exclude_channels)]

    align_col = "align_norm" if "align_norm" in sub.columns else "align"
    keys = ["decoder", "dim_red", align_col, "day_int", "fold", "seed"]

    out = (
        sub.groupby(keys, dropna=False)["vaf"]
        .mean()
        .reset_index(name="vaf_mean_musc")
    )

    if "align_norm" not in out.columns:
        out["align_norm"] = out[align_col].map(norm_align_name)
    return out


# ---------------------------------------------------------------------
# Figure B : un décodeur, aligned vs direct
# ---------------------------------------------------------------------
def plot_panel_B(df_results: pd.DataFrame,
                 decoder: str,
                 dim_red: str,
                 condA: str,
                 condB: str,
                 exclude_channels=None,
                 out_path: str = None):
    df = df_results.copy()
    df["align_norm"] = df["align"].map(norm_align_name)
    condA = norm_align_name(condA)
    condB = norm_align_name(condB)

    df = df[df["dim_red"] == dim_red]
    df = df[df["align_norm"].isin([condA, condB])]
    df = df[df["decoder"] == decoder]
    if df.empty:
        raise RuntimeError("No rows after filtering for panel B.")

    df_avg = average_over_muscles(df, exclude_channels=exclude_channels)

    days = sorted(df_avg["day_int"].dropna().unique())
    colorA = "tab:blue"   # aligned
    colorB = "tab:orange"  # direct
    jitter = 0.06
    rng = np.random.default_rng(0)

    plt.figure(figsize=(6, 3))
    ax = plt.gca()

    for day in days:
        for cond, col in [(condA, colorA), (condB, colorB)]:
            vals = df_avg[
                (df_avg["day_int"] == day)
                & (df_avg["align_norm"] == cond)
            ]["vaf_mean_musc"].values
            if vals.size == 0:
                continue
            xs = day + rng.uniform(-jitter, jitter, size=len(vals))
            ax.scatter(xs, vals, s=10, alpha=0.4, color=col)
            ax.errorbar(
                day,
                np.mean(vals),
                yerr=np.std(vals),
                fmt="o",
                color=col,
                capsize=3,
                markersize=4,
                linewidth=1,
            )

    ax.set_xlabel("Day")
    ax.set_ylabel("VAF (mean over muscles)")
    ax.set_title(f"{decoder} • {dim_red} • {condA} vs {condB}")
    ax.set_xlim(min(days) - 0.8, max(days) + 0.8)
    ax.set_ylim(-0.5, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend([condA, condB], loc="lower left", frameon=False)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved figure B to {out_path}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------
# Figure C : tous les décodeurs, aligned vs direct
# ---------------------------------------------------------------------
def plot_panel_C(df_results: pd.DataFrame,
                 dim_red: str,
                 condA: str,
                 condB: str,
                 exclude_channels=None,
                 out_path: str = None):
    df = df_results.copy()
    df["align_norm"] = df["align"].map(norm_align_name)
    condA = norm_align_name(condA)
    condB = norm_align_name(condB)

    df = df[df["dim_red"] == dim_red]
    df = df[df["align_norm"].isin([condA, condB])]
    if df.empty:
        raise RuntimeError("No rows after filtering for panel C.")

    df_avg = average_over_muscles(df, exclude_channels=exclude_channels)

    present_decoders = [d for d in DECODER_ORDER if d in df_avg["decoder"].unique()]
    if not present_decoders:
        raise RuntimeError("No decoders found for panel C.")

    days = sorted(df_avg["day_int"].dropna().unique())

    plt.figure(figsize=(7, 3.2))
    ax = plt.gca()

    rng = np.random.default_rng(0)
    jitter = 0.05
    markerA = "o"
    markerB = "s"

    for dec in present_decoders:
        col = DECODER_COLORS.get(dec, "black")
        sub = df_avg[df_avg["decoder"] == dec]

        for cond, marker in [(condA, markerA), (condB, markerB)]:
            means, stds, xs = [], [], []
            for day in days:
                vals = sub[
                    (sub["day_int"] == day)
                    & (sub["align_norm"] == cond)
                ]["vaf_mean_musc"].values
                if vals.size == 0:
                    continue
                xs.append(day)
                means.append(np.mean(vals))
                stds.append(np.std(vals))

                xx = day + rng.uniform(-jitter, jitter, size=len(vals))
                ax.scatter(xx, vals, s=6, alpha=0.25, color=col)

            if xs:
                ax.errorbar(
                    xs,
                    means,
                    yerr=stds,
                    fmt=marker + "-",
                    color=col,
                    markersize=4,
                    linewidth=1,
                    capsize=3,
                    label=f"{dec} ({cond})",
                )

    ax.set_xlabel("Day")
    ax.set_ylabel("VAF (mean over muscles)")
    ax.set_title(f"All decoders • {dim_red} • {condA} vs {condB}")
    ax.set_xlim(min(days) - 0.8, max(days) + 0.8)
    ax.set_ylim(-0.5, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Saved figure C to {out_path}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Figures B et C : day stability.")
    ap.add_argument("--results_dir", type=str, default=".")
    ap.add_argument("--out_dir", type=str, default="figs_day_stab")
    ap.add_argument("--dim_red", type=str, default="PCA", choices=["PCA", "UMAP"])
    ap.add_argument(
        "--cond_a",
        type=str,
        default="aligned",
        help="condition A (e.g. aligned)",
    )
    ap.add_argument(
        "--cond_b",
        type=str,
        default="direct",
        help="condition B (e.g. direct/naive)",
    )
    ap.add_argument(
        "--exclude_channels",
        nargs="*",
        type=int,
        default=None,
        help="EMG channels to exclude (e.g. 0 5 6)",
    )
    ap.add_argument(
        "--decoder_B",
        type=str,
        default="GRU",
        help="decoder utilisé pour la figure B",
    )
    ap.add_argument(
        "--which",
        type=str,
        default="both",
        choices=["B", "C", "both"],
        help="quelle(s) figure(s) générer",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_results(args.results_dir)

    if args.which in ("B", "both"):
        out_B = os.path.join(
            args.out_dir,
            f"figB_{args.decoder_B}_{args.dim_red}_{args.cond_a}_vs_{args.cond_b}.png",
        )
        plot_panel_B(
            df_results=df,
            decoder=args.decoder_B,
            dim_red=args.dim_red,
            condA=args.cond_a,
            condB=args.cond_b,
            exclude_channels=args.exclude_channels,
            out_path=out_B,
        )

    if args.which in ("C", "both"):
        out_C = os.path.join(
            args.out_dir,
            f"figC_allDecoders_{args.dim_red}_{args.cond_a}_vs_{args.cond_b}.png",
        )
        plot_panel_C(
            df_results=df,
            dim_red=args.dim_red,
            condA=args.cond_a,
            condB=args.cond_b,
            exclude_channels=args.exclude_channels,
            out_path=out_C,
        )


if __name__ == "__main__":
    main()
