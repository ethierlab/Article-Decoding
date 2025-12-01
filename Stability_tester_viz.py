#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------- style ----------
plt.rcParams.update({
    "figure.figsize": (14, 4.5),
    "axes.grid": True,
    "grid.alpha": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "semibold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# Palette per decoder (soft)
DECODER_COLORS = {
    "GRU":   "#f28e8c",   # soft red
    "LSTM":  "#8bb8e8",   # soft blue
    "Linear":"#98c77b",   # soft green
    "LiGRU": "#f2b278",   # soft orange
}

def _fmt_x(v):
    # pretty xtick labels
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:g}"

def _gather_long(df, x_col, y_col, hue_col):
    x_vals = sorted(df[x_col].dropna().unique().tolist())
    groups = sorted(df[hue_col].dropna().unique().tolist())
    data = defaultdict(list)   # (x, group) -> list[y]
    for _, r in df[[x_col, hue_col, y_col]].dropna().iterrows():
        data[(r[x_col], r[hue_col])].append(float(r[y_col]))
    return x_vals, groups, data

def _positions(n_x, n_g, cluster_width=0.82):
    base = np.arange(n_x, dtype=float)
    if n_g > 1:
        offs = np.linspace(-cluster_width/2, cluster_width/2, n_g)
    else:
        offs = np.array([0.0])
    return base, offs, cluster_width

def plot_median_only(ax, df, x_col, y_col, hue_col, xlabel, title, max_xticks=30):
    """Standalone figure: per-decoder medians across x (dots+line)."""
    if df is None or df.empty:
        ax.set_title(title + " (no data)")
        return

    groups = sorted(df[hue_col].dropna().unique().tolist())
    x_vals = sorted(df[x_col].dropna().unique().tolist())
    if not groups or not x_vals:
        ax.set_title(title + " (no data)")
        return

    for gi, g in enumerate(groups):
        color = DECODER_COLORS.get(g, f"C{gi}")
        meds = []
        for xv in x_vals:
            s = df[(df[x_col] == xv) & (df[hue_col] == g)][y_col].dropna()
            meds.append(float(np.median(s)) if len(s) else np.nan)
        ax.plot(x_vals, meds, marker="o", markersize=6, lw=1.6,
                color=color, markeredgecolor="black", label=g)

    # ticks / labels
    if len(x_vals) > max_xticks:
        step = int(np.ceil(len(x_vals) / max_xticks))
        x_show = x_vals[::step]
    else:
        x_show = x_vals
    ax.set_xticks(x_show)
    ax.set_xticklabels([f"{x:g}" if isinstance(x, float) else str(x) for x in x_show])

    ax.set_xlim(min(x_vals)-0.1, max(x_vals)+0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("VAF (median across muscles/folds)")
    # ax.set_title(title)
    ax.legend(title="Decoder", ncols=len(groups), frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.18))



def plot_grouped_violins(ax, df, x_col, y_col, hue_col, xlabel, title, max_xticks=30):
    x_vals, groups, data = _gather_long(df, x_col, y_col, hue_col)
    base, offs, cwidth = _positions(len(x_vals), len(groups))
    if not x_vals or not groups:
        ax.set_title(title + " (no data)")
        return

    for gi, g in enumerate(groups):
        pos = base + offs[gi]
        width_v = (cwidth / (len(groups) + 0.5))
        color = DECODER_COLORS.get(g, f"C{gi}")

        # per-x arrays
        arrays = [data.get((x, g), []) for x in x_vals]
        # violin
        v = ax.violinplot(arrays, positions=pos, widths=width_v,
                          showmeans=False, showextrema=False, showmedians=False)
        for b in v["bodies"]:
            b.set_facecolor(color)
            b.set_edgecolor("none")
            b.set_alpha(0.45)

        # overlay median as a black dot
        meds = [np.nan if len(a)==0 else float(np.median(a)) for a in arrays]
        ax.scatter(pos, meds, s=18, color="black", zorder=3)

    # axes cosmetics
    # tick density
    if len(x_vals) > max_xticks:
        step = int(np.ceil(len(x_vals)/max_xticks))
        show_idx = list(range(0, len(x_vals), step))
    else:
        show_idx = list(range(len(x_vals)))
    ax.set_xticks(base[show_idx])
    ax.set_xticklabels([_fmt_x(x_vals[i]) for i in show_idx])
    ax.set_xlim(-0.8, len(x_vals)-0.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("VAF (avg across muscles)")
    # ax.set_title(title)

    # legend
    handles = [plt.Line2D([0],[0], lw=10, color=DECODER_COLORS.get(g, f"C{i}")) for i,g in enumerate(groups)]
    ax.legend(handles, groups, title="Decoder", ncols=len(groups), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25))

def plot_grouped_boxes(ax, df, x_col, y_col, hue_col, xlabel, title, max_xticks=30):
    x_vals, groups, data = _gather_long(df, x_col, y_col, hue_col)
    base, offs, cwidth = _positions(len(x_vals), len(groups))
    if not x_vals or not groups:
        ax.set_title(title + " (no data)")
        return

    for gi, g in enumerate(groups):
        pos = base + offs[gi]
        width_b = (cwidth / (len(groups) + 0.5)) * 0.58
        color = DECODER_COLORS.get(g, f"C{gi}")
        arrays = [data.get((x, g), []) for x in x_vals]

        # Matplotlib boxplot per-position
        bp = ax.boxplot(
            arrays, positions=pos, widths=width_b, patch_artist=True, manage_ticks=False
        )
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.55)
            box.set_edgecolor("black")
            box.set_linewidth(0.8)
        for whisk in bp["whiskers"]:
            whisk.set_color("black"); whisk.set_linewidth(0.8)
        for cap in bp["caps"]:
            cap.set_color("black"); cap.set_linewidth(0.8)
        for med in bp["medians"]:
            med.set_color("black"); med.set_linewidth(1.2)
        for fl in bp.get("fliers", []):
            fl.set_marker("o"); fl.set_markersize(2.5); fl.set_alpha(0.25); fl.set_markeredgecolor("black")

        # black median dots on top (like the reference)
        meds = [np.nan if len(a)==0 else float(np.median(a)) for a in arrays]
        ax.scatter(pos, meds, s=18, color="black", zorder=3)

    # axes cosmetics
    if len(x_vals) > max_xticks:
        step = int(np.ceil(len(x_vals)/max_xticks))
        show_idx = list(range(0, len(x_vals), step))
    else:
        show_idx = list(range(len(x_vals)))
    ax.set_xticks(base[show_idx])
    ax.set_xticklabels([_fmt_x(x_vals[i]) for i in show_idx])
    ax.set_xlim(-0.8, len(x_vals)-0.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("VAF (avg across muscles)")
    # ax.set_title(title)

    handles = [plt.Line2D([0],[0], lw=10, color=DECODER_COLORS.get(g, f"C{i}")) for i,g in enumerate(groups)]
    ax.legend(handles, groups, title="Decoder", ncols=len(groups), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25))

def main():
    parser = argparse.ArgumentParser(description="Pretty plots for robustness experiments")
    parser.add_argument('--neuron_pkl', type=str, default='neuron_robustness_day0.pkl')
    parser.add_argument('--noise_pkl',  type=str, default='noise_robustness_day0.pkl')
    parser.add_argument('--outdir',     type=str, default='plots_pretty')
    parser.add_argument('--dpi',        type=int, default=300)
    parser.add_argument('--max_xticks', type=int, default=30)
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load
    df_neuron = pd.read_pickle(args.neuron_pkl) if os.path.exists(args.neuron_pkl) else None
    df_noise  = pd.read_pickle(args.noise_pkl)  if os.path.exists(args.noise_pkl)  else None

    # ---------- Neuron-loss ----------
    if df_neuron is not None and not df_neuron.empty and \
       {'decoder','removed','vaf'}.issubset(df_neuron.columns):
        dn = df_neuron[['decoder','removed','vaf']].copy()
        dn = dn.replace([np.inf,-np.inf], np.nan).dropna()

        # Violin
        fig, ax = plt.subplots()
        plot_grouped_violins(
            ax, dn, x_col='removed', y_col='vaf', hue_col='decoder',
            xlabel='# Neurons removed', title='Robustness to neuron loss (Day 0, CV Val)',
            max_xticks=args.max_xticks
        )
        path = os.path.join(args.outdir, 'neuron_violin.png')
        fig.tight_layout(); fig.savefig(path, dpi=args.dpi); plt.close(fig)

        # Box
        fig, ax = plt.subplots()
        plot_grouped_boxes(
            ax, dn, x_col='removed', y_col='vaf', hue_col='decoder',
            xlabel='# Neurons removed', title='Robustness to neuron loss (Day 0, CV Val)',
            max_xticks=args.max_xticks
        )
        path = os.path.join(args.outdir, 'neuron_box.png')
        fig.tight_layout(); fig.savefig(path, dpi=args.dpi); plt.close(fig)
        print("[OK] Neuron-loss plots saved.")
    
    else:
        print("[INFO] Neuron-loss PKL missing or empty / wrong columns.")

    # ---------- Noise ----------
    if df_noise is not None and not df_noise.empty and \
       {'decoder','noise_sigma','vaf'}.issubset(df_noise.columns):
        dz = df_noise[['decoder','noise_sigma','vaf']].copy()
        dz['noise_sigma'] = pd.to_numeric(dz['noise_sigma'], errors='coerce')
        dz = dz.replace([np.inf,-np.inf], np.nan).dropna()

        # Violin
        fig, ax = plt.subplots()
        plot_grouped_violins(
            ax, dz, x_col='noise_sigma', y_col='vaf', hue_col='decoder',
            xlabel='Noise σ (spike units)', title='Robustness to noise (Day 0, CV Val)',
            max_xticks=args.max_xticks
        )
        path = os.path.join(args.outdir, 'noise_violin.png')
        fig.tight_layout(); fig.savefig(path, dpi=args.dpi); plt.close(fig)

        # Box
        fig, ax = plt.subplots()
        plot_grouped_boxes(
            ax, dz, x_col='noise_sigma', y_col='vaf', hue_col='decoder',
            xlabel='Noise σ (spike units)', title='Robustness to noise (Day 0, CV Val)',
            max_xticks=args.max_xticks
        )
        path = os.path.join(args.outdir, 'noise_box.png')
        fig.tight_layout(); fig.savefig(path, dpi=args.dpi); plt.close(fig)
        print("[OK] Noise plots saved.")

    else:
        print("[INFO] Noise PKL missing or empty / wrong columns.")

        # ---------- Neuron-loss MEDIAN-ONLY ----------
    if df_neuron is not None and not df_neuron.empty and \
    {'decoder','removed','vaf'}.issubset(df_neuron.columns):
        dn = df_neuron[['decoder','removed','vaf']].copy()
        dn = dn.replace([np.inf,-np.inf], np.nan).dropna()

        fig, ax = plt.subplots(figsize=(14, 4.5))
        plot_median_only(
            ax, dn, x_col='removed', y_col='vaf', hue_col='decoder',
            xlabel='# Neurons removed',
            title='Median VAF vs. # Neurons removed (Day 0, CV Val)',
            max_xticks=args.max_xticks
        )
        path = os.path.join(args.outdir, 'neuron_median.png')
        fig.tight_layout(); fig.savefig(path, dpi=args.dpi); plt.close(fig)
        print(f"[OK] Saved median-only: {path}")

    # ---------- Noise MEDIAN-ONLY ----------
    if df_noise is not None and not df_noise.empty and \
    {'decoder','noise_sigma','vaf'}.issubset(df_noise.columns):
        dz = df_noise[['decoder','noise_sigma','vaf']].copy()
        dz['noise_sigma'] = pd.to_numeric(dz['noise_sigma'], errors='coerce')
        dz = dz.replace([np.inf,-np.inf], np.nan).dropna()

        fig, ax = plt.subplots(figsize=(14, 4.5))
        plot_median_only(
            ax, dz, x_col='noise_sigma', y_col='vaf', hue_col='decoder',
            xlabel='Noise σ (spike units)',
            title='Median VAF vs. Noise σ (Day 0, CV Val)',
            max_xticks=args.max_xticks
        )
        path = os.path.join(args.outdir, 'noise_median.png')
        fig.tight_layout(); fig.savefig(path, dpi=args.dpi); plt.close(fig)
        print(f"[OK] Saved median-only: {path}")


if __name__ == "__main__":
    main()
