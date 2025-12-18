import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

# ========= PARAMS =========
PKL_PATH = "ALL_gridsearch_results_v2.pkl"

with open(PKL_PATH, "rb") as f:
    rows = pickle.load(f)
df_raw = pd.DataFrame(rows)

print("=== Décoders bruts dans le PKL ===")
print(df_raw["decoder"].value_counts())

# Normaliser un peu les noms
df_raw["decoder"] = df_raw["decoder"].astype(str).str.strip().str.lower()

print("\n=== Décoders après normalisation ===")
print(df_raw["decoder"].value_counts())

print("\n=== Quelques lignes Kalman si elles existent ===")
print(df_raw[df_raw["decoder"] == "kalman"].head())

# Ordre de print / plots
DECODER_ORDER = ['ligru', 'lstm', 'gru', 'linear', 'wiener']   

TOPN = 50                                            # for boxplot selection (non utilisé ici mais laissé)
TARGETS = [0.80, 0.90, 0.95]                         # vertical goal lines on median curve
INCLUDE_SEED_IN_CONFIG = True                        # set False to ignore 'seed' when grouping
# ==========================

# === FIXED colors + legend order/labels ===
COLOR_MAP = {
    'ligru' : "#ff7f0e",  # orange
    'lstm'  : "#1f77b4",  # blue
    'gru'   : "#d62728",  # red
    'linear': "#2ca02c",  # green
    'wiener': "#9467bd",  # purple  
    # 'kalman': "#19f0ff",  # blue-cyan   
}

# Ordre d’affichage dans les légendes / violins
HUE_ORDER = ['gru', 'lstm', 'ligru', 'linear', 'wiener']  

LEGEND_LABELS = {
    'gru'   : 'GRU',
    'lstm'  : 'LSTM',
    'ligru' : 'LiGRU',
    'linear': 'Linear',
    'wiener': 'Wiener',   
}

def reorder_and_relabel_legend(ax, title='Decoder'):
    """Put legend in desired order with custom labels."""
    handles, labels = ax.get_legend_handles_labels()
    # labels seront les valeurs de la colonne 'decoder'
    lut = dict(zip(labels, handles))
    ordered = []
    for key in HUE_ORDER:
        if key in lut:
            ordered.append((lut[key], LEGEND_LABELS.get(key, key)))
    if ordered:
        ax.legend(
            [h for h, _ in ordered],
            [lab for _, lab in ordered],
            title=title, bbox_to_anchor=(1.02, 1), loc='upper left'
        )

# ================== LOAD & SANITIZE ==================
with open(PKL_PATH, "rb") as f:
    rows = pickle.load(f)
df = pd.DataFrame(rows)

# Normaliser les noms de décodeurs (important si certains sont en majuscules ou avec espaces)
if "decoder" in df.columns:
    df["decoder"] = df["decoder"].astype(str).str.strip().str.lower()  # <<< NEW

# --- SANITIZE df ---
need = ["decoder", "mean_vaf", "num_params", "fold_vafs"]
for c in need:
    if c not in df.columns:
        df[c] = np.nan

# robust numeric casts
df["mean_vaf"]   = pd.to_numeric(df["mean_vaf"], errors="coerce")
df["num_params"] = pd.to_numeric(df["num_params"], errors="coerce")

# valid runs
df = df.dropna(subset=["decoder", "mean_vaf", "num_params"])
df = df[df["num_params"] > 0]

# ------------- HARDEN VAFS -------------
df["mean_vaf"] = pd.to_numeric(df["mean_vaf"], errors="coerce")

# Normaliser les VAF style pourcentage (ex: 87 -> 0.87)
mask_pct = (df["mean_vaf"] > 1.5) & (df["mean_vaf"] <= 1000)
df.loc[mask_pct, "mean_vaf"] = df.loc[mask_pct, "mean_vaf"] / 100.0

# Remove non-finite and out-of-range values
df = df[np.isfinite(df["mean_vaf"])]
df = df[df["mean_vaf"].between(-1.0, 1.0)]

# ===== Choose config keys that exist (no channel needed) =====
CAND_CONFIG_KEYS = [
    'decoder', 'num_params', 'hidden_dim', 'k_lag', 'n_pca', 'lr', 'num_epochs', 'seed',
    'dropout', 'bidirectional', 'batch_size', 'optimizer', 'activation',
    'k_history', 'window',
    # si tu as d’autres hyperparam pour Wiener/Kalman, tu peux les ajouter ici
    'history_bins', 'alpha', 'lambda', 'ridge', 'lag'  # <<< éventuellement utiles
]
CONFIG_KEYS = [k for k in CAND_CONFIG_KEYS if k in df.columns]
if not INCLUDE_SEED_IN_CONFIG and 'seed' in CONFIG_KEYS:
    CONFIG_KEYS.remove('seed')
# minimum keys to keep plots meaningful:
for must in ['decoder', 'num_params']:
    if must not in CONFIG_KEYS:
        CONFIG_KEYS = [must] + CONFIG_KEYS
# dedupe while preserving order
CONFIG_KEYS = list(dict.fromkeys(CONFIG_KEYS))

# ============ AVERAGE across ALL muscles (rows) ============

# 1) run-level average of mean_vaf across rows sharing same config
df_avg = (
    df.groupby(CONFIG_KEYS, dropna=False)['mean_vaf']
      .mean()
      .reset_index()
      .rename(columns={'mean_vaf': 'mean_vaf_avg_all_muscles'})
)
# count how many rows contributed (approx #muscles × folds × seeds)
counts = (
    df.groupby(CONFIG_KEYS, dropna=False)
      .size()
      .reset_index(name='n_contrib')
)
df_avg = df_avg.merge(counts, on=CONFIG_KEYS, how='left')

# 2) fold-level distribution averaged across rows sharing same config & fold_num
flat = []
for _, row in df.iterrows():
    vafs = row["fold_vafs"] if isinstance(row["fold_vafs"], (list, tuple, np.ndarray)) else []
    for fold_num, v in enumerate(vafs):
        rec = {k: row.get(k, None) for k in CONFIG_KEYS}
        rec.update({"fold_num": fold_num, "vaf": float(v)})
        flat.append(rec)

flatdf = pd.DataFrame(flat)
if not flatdf.empty:
    flatdf["vaf"] = pd.to_numeric(flatdf["vaf"], errors="coerce")
    flatdf = flatdf.dropna(subset=["vaf"])
    grp_keys = CONFIG_KEYS + ["fold_num"]
    flat_avg = (
        flatdf.groupby(grp_keys, dropna=False)["vaf"]
              .mean()
              .reset_index()
              .rename(columns={"vaf": "vaf_avg_all_muscles"})
    )
    counts_fold = (
        flatdf.groupby(grp_keys, dropna=False)
              .size()
              .reset_index(name="n_contrib")
    )
    flat_avg = flat_avg.merge(counts_fold, on=grp_keys, how="left")
else:
    flat_avg = pd.DataFrame(columns=CONFIG_KEYS + ["fold_num", "vaf_avg_all_muscles", "n_contrib"])

# ========= DIAGNOSTICS (optional) =========
print(df_avg[["decoder","mean_vaf_avg_all_muscles"]].dtypes)
print(df_avg[["decoder","mean_vaf_avg_all_muscles","n_contrib"]].head(5))
if not flat_avg.empty:
    print(flat_avg[["decoder","vaf_avg_all_muscles"]].dtypes)

# ======== TOP per decoder (averaged across all muscles) ========
print("\n# ==== TOP per decoder (by mean VAF averaged across all muscles) ====")
for decoder in DECODER_ORDER:
    subdf = df_avg[df_avg.decoder == decoder]
    if subdf.empty:
        continue
    top = subdf.sort_values("mean_vaf_avg_all_muscles", ascending=False).head(20)
    print(f"\n## {decoder.upper()} ##")
    for _, r in top.iterrows():
        # gestion propre des NaN pour afficher '-'
        def fmt_val(x):
            return "-" if pd.isna(x) else x

        print(
            f"VAFµ={r['mean_vaf_avg_all_muscles']:.3f} | "
            f"hid={fmt_val(r.get('hidden_dim', np.nan)):<3} | "
            f"n_pca={fmt_val(r.get('n_pca', np.nan)):<3} | "
            f"k_lag={fmt_val(r.get('k_lag', np.nan)):<3} | "
            f"lr={fmt_val(r.get('lr', np.nan)):<8} | "
            f"epochs={fmt_val(r.get('num_epochs', np.nan)):<4} | "
            f"params={fmt_val(r.get('num_params', np.nan)):<7} | "
            f"seed={fmt_val(r.get('seed', np.nan)):<3} | "
            f"n_contrib={fmt_val(r.get('n_contrib', np.nan))}"
        )
print("\n# ===== END TOP PRINT =====\n")

# ========= SCATTER (avg across all muscles) =========
plt.figure(figsize=(13, 7))
ax = sns.scatterplot(
    data=df_avg,
    x="num_params", y="mean_vaf_avg_all_muscles",
    hue="decoder",
    hue_order=HUE_ORDER,
    palette=COLOR_MAP,
    alpha=0.7, s=60, edgecolor=None
)
ax.set_xscale('log')
ax.set_xlabel("Number of trainable parameters (log)")
ax.set_ylabel("mean VAF (avg across all muscles)")
ax.set_ylim(0, 1.2)
reorder_and_relabel_legend(ax, title='Decoder')
plt.tight_layout()
plt.savefig("optim_scatter_vaf_vs_params_avg_all_muscles.png", dpi=350)
plt.show()

# ========= VIOLIN (folds avg across all muscles) =========
if not flat_avg.empty:
    plt.figure(figsize=(10, 6))
    order_present = [d for d in HUE_ORDER if d in flat_avg['decoder'].unique()]
    palette_sub = {k: COLOR_MAP[k] for k in order_present}  # sous-dict de couleurs

    ax = sns.violinplot(
        data=flat_avg, x="decoder", y="vaf_avg_all_muscles",
        order=order_present,
        cut=0,
        bw_adjust=0.5,
        palette=palette_sub,
        inner="point"
    )
    ax.set_ylim(0, 1.2)
    ax.set_title("Distribution of VAFs (fold-level) by decoder — averaged across all muscles")
    ax.set_ylabel("VAF per fold (avg across all muscles)")
    ax.set_xlabel("Decoder")
    # Pas de légende, l'axe X encode déjà le décodeur
    plt.tight_layout()
    plt.savefig("optim_violin_vaf_folds_by_decoder_avg_all_muscles.png", dpi=300)
    plt.show()
else:
    print("[VIOLIN] Skipped (no fold_vafs present).")

# ========= MEDIAN CURVE: X = median VAF, Y = #params (log), one line per decoder =========
if not flat_avg.empty:
    medsrc = flat_avg.dropna(subset=['decoder', 'num_params', 'vaf_avg_all_muscles']).copy()
    medsrc = medsrc[medsrc['num_params'] > 0]

    medg = (
        medsrc.groupby(['decoder', 'num_params'], dropna=False)['vaf_avg_all_muscles']
              .median()
              .reset_index()
              .rename(columns={'vaf_avg_all_muscles': 'median_vaf'})
    )

    # Calcul des crossing targets AVANT plot
    crossings = {t: {} for t in TARGETS}
    for dec, g in medg.groupby('decoder'):
        g = g.sort_values('median_vaf')
        for t in TARGETS:
            g2 = g[g['median_vaf'] >= t]
            crossings[t][dec] = None if g2.empty else float(g2['num_params'].min())

    plt.figure(figsize=(12, 7))
    handles = {}
    for dec in HUE_ORDER:
        g = medg[medg['decoder'] == dec].sort_values('median_vaf')
        if g.empty:
            continue
        (ln,) = plt.plot(
            g['median_vaf'], g['num_params'],
            marker='o', markersize=4, linewidth=1.6,
            label=LEGEND_LABELS.get(dec, dec),
            color=COLOR_MAP.get(dec, 'gray'), alpha=0.95
        )
        handles[dec] = ln

    # vertical lines for targets
    for t in TARGETS:
        plt.axvline(t, linestyle='--', linewidth=1.0, alpha=0.4)

    # highlight the minimum-params point at each target (if it exists)
    for t in TARGETS:
        for dec, nmin in crossings[t].items():
            if nmin is not None and dec in COLOR_MAP:
                plt.scatter([t], [nmin], s=70, edgecolors='k', linewidths=0.8,
                            color=COLOR_MAP[dec], zorder=5)

    plt.yscale('log')
    plt.xlim(0, 1.0)
    plt.xlabel('Median VAF for a given #parameters (avg across all muscles/folds/seeds)')
    plt.ylabel('# Parameters (log scale)')
    plt.grid(True, which='both', axis='both', alpha=0.3)

    present = [d for d in HUE_ORDER if d in handles]
    plt.legend(
        [handles[d] for d in present],
        [LEGEND_LABELS[d] for d in present],
        title='Decoder', bbox_to_anchor=(1.02, 1), loc='upper left'
    )
    plt.tight_layout()
    plt.savefig('optim_curve_numparams_vs_median_vaf_inverted_axes.png', dpi=350)
    plt.show()

    # Console summary: minimum params to reach targets
    print("\n# Min #params to reach target median VAF (median across all rows sharing num_params):")
    for t in TARGETS:
        print(f"\nTarget VAF ≥ {t:.2f}:")
        for dec in HUE_ORDER:
            nmin = crossings[t].get(dec)
            if nmin is None:
                print(f"  {dec:<10} : not reached")
            else:
                nmin_str = f"{int(round(nmin)):,}"
                print(f"  {dec:<10} : ~{nmin_str} params")
else:
    print("[MEDIAN CURVE] Skipped (flat_avg is empty).")
