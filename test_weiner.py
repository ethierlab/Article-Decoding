#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run-all (no args): compare Wiener-Cascade training methods on Day0 with K-fold no-leak CV.

- Same preprocessing logic as your current pipeline:
  * Downsample spikes+EMG (BIN_FACTOR)
  * Gaussian smoothing on spikes (segment-wise)
  * Low-pass Butterworth on |EMG| (segment-wise, filtfilt)
  * PCA fit on TRAIN only
  * Contiguous time K-fold with LEFT+RIGHT train, middle VAL
  * Cuts used to forbid windows crossing trial boundaries

- Wiener-Cascade training methods:
  (A) "GD"  : PyTorch gradient descent (your current approach)
  (B) "OLS" : Analytical two-stage OLS (teacher method): W by pinv, then per-output quadratic pinv

Outputs:
  - results_wiener_compare.pkl  : summary per config
  - rows_wiener_compare.pkl     : long-format fold×channel×seed
  - viz_out_wiener_compare/*.png : plots

Launch:
  python run_wiener_compare_gd_vs_ols.py

Notes:
  - This script does NOT include Kalman.
  - By default, EMBARGO uses only smoothing-based embargo (NOT max(k_lag, ...)),
    because k_lag is already handled by window construction. You can flip it back if desired.
"""

import os, gc, time, warnings, pickle, itertools
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.")

# ============================ USER KNOBS (EDIT HERE) ============================
COMBINED_PICKLE = "combined.pkl"   # <-- change me
OUTDIR          = Path("viz_out_wiener_compare")
RESULTS_FILE    = Path("results_wiener_compare.pkl")
ROWS_FILE       = Path("rows_wiener_compare.pkl")

FOLDS = 10
SEEDS = [0, 1, 2]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing
BIN_FACTOR = 20          # 1000 Hz -> 50 Hz
BIN_SIZE = 0.001
SMOOTHING_LENGTH = 0.05  # s
SAMPLING_RATE = 1000
GAUSS_TRUNCATE = 4.0

# Training (GD variant only)
BATCH_SIZE = 256
NUM_WORKERS = -1         # -1 => auto from $SLURM_CPUS_PER_TASK
USE_AMP = True
PERF_MODE = True

# Embargo choice:
# - If True: EMB = max(k_lag, smoothing_embargo)  (your earlier conservative behavior)
# - If False: EMB = smoothing_embargo only        (recommended for fairer window counts)
EMBARGO_INCLUDES_KLAG = False

# Grid of configs (same config applies to both training methods)
GRID = dict(
    n_pca=[16, 32, 64],
    k_lag=[10, 20, 25, 40],     # at 50 Hz: 25 ~= 500 ms
    num_epochs=[150],
    lr=[1e-3],
    weight_decay=[0.0],         # optional; used only for GD optimizer
)

# Compare these two training methods
TRAIN_METHODS = ["GD", "OLS"]   # "GD"=torch, "OLS"=teacher pinv cascade

# ============================ UTILS ============================
SEED_BASE = 42

def set_seed(seed=SEED_BASE):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if PERF_MODE:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def auto_num_workers(default=8):
    try:
        n = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        if n > 0:
            return max(2, n - 1)
    except Exception:
        pass
    return default

def get_all_unit_names(combined_df: pd.DataFrame) -> List[str]:
    unit_set = set()
    for _, row in combined_df.iterrows():
        sc = row.get("spike_counts", None)
        if isinstance(sc, pd.DataFrame):
            unit_set.update(sc.columns)
    return sorted(list(unit_set))

def butter_lowpass(data, fs, order=4, cutoff_hz=5.0):
    nyq = 0.5 * fs
    norm = cutoff_hz / nyq
    b, a = butter(order, norm, btype="low", analog=False)
    return filtfilt(b, a, data, axis=0)

def downsample_spike_and_emg(spike_df, emg_data, bin_factor=10):
    if spike_df.empty or spike_df.shape[0] < bin_factor:
        return spike_df, emg_data

    T_old, n_units = spike_df.shape
    T_new = T_old // bin_factor
    spk_arr = spike_df.values[: T_new * bin_factor, :]
    spk_arr = spk_arr.reshape(T_new, bin_factor, n_units).sum(axis=1)
    ds_spike_df = pd.DataFrame(spk_arr, columns=spike_df.columns)

    if isinstance(emg_data, pd.DataFrame):
        e_arr = emg_data.values
        col_names = emg_data.columns
    else:
        e_arr = np.array(emg_data)
        col_names = None

    if e_arr.shape[0] < bin_factor:
        return ds_spike_df, emg_data

    e_arr = e_arr[: T_new * bin_factor, ...]
    if e_arr.ndim == 2:
        e_arr = e_arr.reshape(T_new, bin_factor, e_arr.shape[1]).mean(axis=1)
        ds_emg = pd.DataFrame(e_arr, columns=col_names) if col_names is not None else e_arr
    else:
        ds_emg = emg_data

    return ds_spike_df, ds_emg

def build_continuous_dataset_raw(df, bin_factor, all_units=None):
    spikes_all, emg_all, lengths = [], [], []
    for _, row in df.iterrows():
        spike_df = row["spike_counts"]
        emg_val  = row["EMG"]
        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty:
            continue
        if emg_val is None:
            continue

        if all_units is not None:
            spike_df = spike_df.reindex(columns=all_units, fill_value=0)

        ds_spike_df, ds_emg = downsample_spike_and_emg(spike_df, emg_val, bin_factor)
        if ds_spike_df.shape[0] == 0:
            continue

        Xr = ds_spike_df.values.astype(np.float32)
        Yr = ds_emg.values.astype(np.float32) if isinstance(ds_emg, pd.DataFrame) else np.asarray(ds_emg, dtype=np.float32)

        spikes_all.append(Xr)
        emg_all.append(Yr)
        lengths.append(len(Xr))

    if not spikes_all:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), []

    cuts = np.cumsum(lengths)[:-1].tolist()
    return np.concatenate(spikes_all, axis=0), np.concatenate(emg_all, axis=0), cuts

def smooth_spike_data(x_2d, eff_bin, smoothing_length):
    sigma = (smoothing_length / eff_bin) / 2.0
    return gaussian_filter1d(x_2d.astype(np.float32), sigma=sigma, axis=0)

def preprocess_segment(Xseg, Yseg, bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    eff_fs  = SAMPLING_RATE // bin_factor
    eff_bin = bin_factor * bin_size
    Xs = smooth_spike_data(Xseg, eff_bin, smoothing_length)
    Ys = butter_lowpass(np.abs(Yseg), eff_fs)
    return Xs, Ys

def sigma_bins(bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    eff_bin = bin_factor * bin_size
    return (smoothing_length / eff_bin) / 2.0

def smoothing_embargo_bins(bin_factor=BIN_FACTOR, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH, truncate=GAUSS_TRUNCATE):
    emb = int(np.ceil(truncate * sigma_bins(bin_factor, bin_size, smoothing_length)))
    return emb

def embargo_bins(k_lag, include_klag: bool):
    emb = smoothing_embargo_bins()
    return max(k_lag, emb) if include_klag else emb

def time_kfold_splits(n_time, n_splits) -> List[Tuple[int, int]]:
    block = n_time // n_splits
    splits = []
    for k in range(n_splits):
        v0 = k * block
        v1 = (k + 1) * block if k < n_splits - 1 else n_time
        splits.append((v0, v1))
    return splits

def adjust_cuts_for_segment(start, end, cuts_global, trim_left=0, trim_right=0, seg_len=None):
    local = [c - start for c in cuts_global if start < c < end]
    if seg_len is None:
        seg_len = end - start
    new_start = trim_left
    new_end   = seg_len - trim_right
    return [c - new_start for c in local if new_start < c < new_end]

def valid_window_indices(n_time, k, cuts, start=0, end=None, stride=1):
    end = n_time if end is None else end
    out = []
    for t in range(start + k, end, stride):
        if any(t - k < c < t for c in cuts):
            continue
        out.append(t)
    return out

def build_seq_with_cuts(Z, Y, K_LAG, cuts, stride=1):
    idx = valid_window_indices(Z.shape[0], K_LAG, cuts, stride=stride)
    if not idx:
        return (np.empty((0, K_LAG * Z.shape[1]), dtype=np.float32),
                np.empty((0, Y.shape[1]), dtype=np.float32))
    X = np.stack([Z[t-K_LAG:t, :].reshape(-1) for t in idx], axis=0).astype(np.float32)
    Yb = np.stack([Y[t, :] for t in idx], axis=0).astype(np.float32)
    return X, Yb

# ============================ MODELS ============================
class WienerCascadeTorch(nn.Module):
    def __init__(self, input_dim: int, n_out: int):
        super().__init__()
        self.lin = nn.Linear(input_dim, n_out, bias=True)
        self.c0  = nn.Parameter(torch.zeros(n_out))
        self.c1  = nn.Parameter(torch.ones(n_out))
        self.c2  = nn.Parameter(torch.ones(n_out) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.lin(x)
        return self.c0 + self.c1 * z + self.c2 * (z ** 2)

def solve_wiener_cascade_ols(X_train: np.ndarray, Y_train: np.ndarray):
    """
    Teacher method:
      1) Solve linear stage W by pinv with bias
      2) Solve per-output quadratic c by pinv on [1, z, z^2]
    Returns:
      Y_hat, W_linear, C_poly
    """
    X_bias = np.column_stack([np.ones(X_train.shape[0], dtype=X_train.dtype), X_train])
    W_linear = np.linalg.pinv(X_bias) @ Y_train
    Z = X_bias @ W_linear

    C_poly = np.zeros((Y_train.shape[1], 3), dtype=np.float64)
    Y_hat = np.zeros_like(Y_train, dtype=np.float64)
    for m in range(Y_train.shape[1]):
        z_m = Z[:, m]
        M_poly = np.column_stack([np.ones_like(z_m), z_m, z_m**2])
        c = np.linalg.pinv(M_poly) @ Y_train[:, m]
        C_poly[m] = c
        Y_hat[:, m] = M_poly @ c
    return Y_hat.astype(np.float32), W_linear.astype(np.float32), C_poly.astype(np.float32)

# ============================ TRAIN / METRICS ============================
def train_torch(model, X_train, Y_train, num_epochs, lr, batch_size, num_workers=None, use_amp=True, weight_decay=0.0):
    if num_workers is None:
        num_workers = auto_num_workers()

    x_cpu = torch.as_tensor(X_train, dtype=torch.float32)
    y_cpu = torch.as_tensor(Y_train, dtype=torch.float32)
    dset = TensorDataset(x_cpu, y_cpu)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))

    model.train()
    for ep in range(1, num_epochs + 1):
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = model(xb)
                loss = crit(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.item())
        if ep == 1 or ep % 50 == 0:
            print(f"    epoch {ep}/{num_epochs}  loss={total/len(loader):.4f}")
    return model

def eval_vaf_full_numpy(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
    v = []
    for ch in range(yt.shape[1]):
        vt = np.var(yt[:, ch])
        v.append(np.nan if vt < 1e-12 else 1.0 - np.var(yt[:, ch] - yp[:, ch]) / vt)
    return np.asarray(v, dtype=np.float32)

# ============================ RUN ============================
def run_compare():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED_BASE)

    combined_df = pd.read_pickle(COMBINED_PICKLE)
    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"], errors="coerce")

    unique_days = sorted(combined_df["date"].dropna().unique())
    if not unique_days:
        raise RuntimeError("No days in combined_df")
    day0 = unique_days[0]
    train_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)

    all_units = get_all_unit_names(combined_df)

    # detect EMG channels
    n_emg_channels = 0
    for _, row in combined_df.iterrows():
        emg_val = row.get("EMG", None)
        if emg_val is not None:
            if isinstance(emg_val, pd.DataFrame) and not emg_val.empty:
                n_emg_channels = emg_val.shape[1]
                break
            if isinstance(emg_val, np.ndarray) and emg_val.size > 0:
                n_emg_channels = emg_val.shape[1]
                break
    if n_emg_channels == 0:
        raise RuntimeError("Could not detect EMG channels.")

    # raw day0
    X0_raw, Y0_raw, cuts0 = build_continuous_dataset_raw(train_df, BIN_FACTOR, all_units=all_units)
    if X0_raw.size == 0:
        raise RuntimeError("Empty day0 after downsampling.")

    splits = time_kfold_splits(X0_raw.shape[0], FOLDS)

    results: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    # build combos
    keys, vals = zip(*GRID.items())
    cfgs = [dict(zip(keys, combo)) for combo in itertools.product(*vals)]

    total_runs = len(cfgs) * len(SEEDS) * len(TRAIN_METHODS)
    run_idx = 0

    for cfg in cfgs:
        n_pca = int(cfg["n_pca"])
        k_lag = int(cfg["k_lag"])
        num_epochs = int(cfg["num_epochs"])
        lr = float(cfg["lr"])
        weight_decay = float(cfg.get("weight_decay", 0.0))

        for seed in SEEDS:
            for method in TRAIN_METHODS:
                run_idx += 1
                print(f"\n=== [{run_idx}/{total_runs}] method={method} cfg={cfg} seed={seed} ===")
                set_seed(SEED_BASE + seed)

                EMB = embargo_bins(k_lag, include_klag=EMBARGO_INCLUDES_KLAG)
                WORKERS = auto_num_workers() if NUM_WORKERS == -1 else NUM_WORKERS

                vafs_fold = []
                fold_times = []
                param_count = None

                for i_fold, (val_start, val_end) in enumerate(splits):
                    # raw segments
                    X_left_raw  = X0_raw[:val_start];     Y_left_raw  = Y0_raw[:val_start]
                    X_val_raw   = X0_raw[val_start:val_end]; Y_val_raw = Y0_raw[val_start:val_end]
                    X_right_raw = X0_raw[val_end:];       Y_right_raw = Y0_raw[val_end:]

                    # preprocess independently (no leakage)
                    Xl, Yl = preprocess_segment(X_left_raw, Y_left_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH) if len(X_left_raw) else (np.empty((0,)), np.empty((0,)))
                    if len(Xl) > EMB:
                        Xl = Xl[:len(Xl)-EMB]; Yl = Yl[:len(Yl)-EMB]
                        cuts_left = adjust_cuts_for_segment(0, len(X_left_raw), cuts0, trim_left=0, trim_right=EMB, seg_len=len(X_left_raw))
                    else:
                        Xl = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
                        Yl = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
                        cuts_left = []

                    Xv, Yv = preprocess_segment(X_val_raw, Y_val_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH) if len(X_val_raw) else (np.empty((0,)), np.empty((0,)))
                    if len(Xv) > 2*EMB:
                        Xv = Xv[EMB:len(Xv)-EMB]; Yv = Yv[EMB:len(Yv)-EMB]
                        cuts_val = adjust_cuts_for_segment(val_start, val_end, cuts0, trim_left=EMB, trim_right=EMB, seg_len=len(X_val_raw))
                    else:
                        Xv = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
                        Yv = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
                        cuts_val = []

                    Xr, Yr = preprocess_segment(X_right_raw, Y_right_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH) if len(X_right_raw) else (np.empty((0,)), np.empty((0,)))
                    if len(Xr) > EMB:
                        Xr = Xr[EMB:]; Yr = Yr[EMB:]
                        cuts_right = adjust_cuts_for_segment(val_end, len(X0_raw), cuts0, trim_left=EMB, trim_right=0, seg_len=len(X_right_raw))
                    else:
                        Xr = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
                        Yr = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
                        cuts_right = []

                    # concat train
                    if Xl.size and Xr.size:
                        X_train_time = np.vstack([Xl, Xr])
                        Y_train_time = np.vstack([Yl, Yr])
                        cuts_train = cuts_left + [len(Xl)] + [c + len(Xl) for c in cuts_right]
                    elif Xl.size:
                        X_train_time, Y_train_time, cuts_train = Xl, Yl, cuts_left
                    else:
                        X_train_time, Y_train_time, cuts_train = Xr, Yr, cuts_right

                    if X_train_time.shape[0] <= k_lag or Xv.shape[0] <= k_lag:
                        continue

                    # PCA on TRAIN only
                    pca_model = PCA(n_components=max(n_pca, 2), random_state=SEED_BASE + seed + i_fold)
                    pca_model.fit(X_train_time)
                    Z_tr = pca_model.transform(X_train_time)[:, :n_pca]
                    Z_va = pca_model.transform(Xv)[:, :n_pca]

                    # build windows with cuts
                    X_tr, Y_tr = build_seq_with_cuts(Z_tr, Y_train_time, k_lag, cuts_train, stride=1)
                    X_va, Y_va = build_seq_with_cuts(Z_va, Yv,            k_lag, cuts_val,   stride=1)
                    if X_tr.shape[0] == 0 or X_va.shape[0] == 0:
                        continue

                    t0 = time.perf_counter()

                    if method == "GD":
                        model = WienerCascadeTorch(k_lag * n_pca, Y_tr.shape[1]).to(DEVICE)
                        if param_count is None:
                            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

                        train_torch(
                            model, X_tr, Y_tr,
                            num_epochs=num_epochs,
                            lr=lr,
                            batch_size=BATCH_SIZE,
                            num_workers=WORKERS,
                            use_amp=USE_AMP,
                            weight_decay=weight_decay
                        )
                        with torch.no_grad():
                            Yp = model(torch.from_numpy(X_va).float().to(DEVICE)).cpu().numpy()
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    else:  # OLS
                        Yp, W_lin, C_poly = solve_wiener_cascade_ols(X_tr, Y_tr)
                        if param_count is None:
                            # W_lin includes bias row
                            param_count = int(W_lin.size + C_poly.size)

                    fold_times.append(time.perf_counter() - t0)

                    vaf_ch = eval_vaf_full_numpy(Y_va, Yp)
                    mean_vaf = float(np.nanmean(vaf_ch))
                    vafs_fold.append(mean_vaf if not np.isnan(mean_vaf) else -1.0)

                    for ch_idx, v in enumerate(vaf_ch):
                        long_rows.append(dict(
                            train_method=method,
                            n_pca=int(n_pca),
                            k_lag=int(k_lag),
                            num_epochs=int(num_epochs),
                            lr=float(lr),
                            weight_decay=float(weight_decay),
                            seed=int(seed),
                            fold=int(i_fold),
                            emg_channel=int(ch_idx),
                            vaf=float(v),
                        ))

                    gc.collect()

                if not vafs_fold:
                    print("  [WARN] no valid folds; skipping.")
                    continue

                results.append(dict(
                    train_method=method,
                    seed=int(seed),
                    num_params=int(param_count) if param_count is not None else None,
                    mean_vaf=float(np.mean(vafs_fold)),
                    fold_vafs=[float(x) for x in vafs_fold],
                    fold_times=[float(s) for s in fold_times],
                    mean_time=float(np.mean(fold_times)) if fold_times else float("nan"),
                    n_pca=int(n_pca),
                    k_lag=int(k_lag),
                    num_epochs=int(num_epochs),
                    lr=float(lr),
                    weight_decay=float(weight_decay),
                    embargo=int(EMB),
                    embargo_includes_klag=bool(EMBARGO_INCLUDES_KLAG),
                ))

                pickle.dump(results, open(RESULTS_FILE, "wb"))
                pickle.dump(long_rows, open(ROWS_FILE, "wb"))

    print("\nSaved:")
    print(" ", RESULTS_FILE.resolve())
    print(" ", ROWS_FILE.resolve())
    return results, long_rows

# ============================ PLOTTING ============================
def make_plots(results, long_rows):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df_sum  = pd.DataFrame(results)
    df_long = pd.DataFrame(long_rows)

    if df_sum.empty:
        print("[WARN] Nothing to plot.")
        return

    # 1) Mean VAF per run (sorted)
    df_sum_sorted = df_sum.sort_values("mean_vaf", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(12, 5))
    labels = [
        f'{m}|pca{p}|k{k}|sd{s}'
        for m, p, k, s in zip(df_sum_sorted["train_method"], df_sum_sorted["n_pca"], df_sum_sorted["k_lag"], df_sum_sorted["seed"])
    ]
    x = np.arange(len(df_sum_sorted))
    plt.bar(x, df_sum_sorted["mean_vaf"].values)
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Mean VAF (val)")
    plt.title("Wiener-Cascade: mean VAF per configuration (GD vs OLS)")
    plt.tight_layout()
    plt.savefig(OUTDIR / "01_mean_vaf_per_run_sorted.png", dpi=150)
    plt.close()

    # 2) Boxplot per method
    if "fold_vafs" in df_sum.columns:
        fold_records = []
        for _, r in df_sum.iterrows():
            for i, v in enumerate(r["fold_vafs"]):
                fold_records.append({"train_method": r["train_method"], "fold": i, "vaf": v})
        df_fold = pd.DataFrame(fold_records)
        if not df_fold.empty:
            methods = [m for m in TRAIN_METHODS if m in df_fold["train_method"].unique()]
            data = [df_fold[df_fold["train_method"] == m]["vaf"].values for m in methods]
            plt.figure(figsize=(7, 5))
            plt.boxplot(data, labels=methods, showmeans=True)
            plt.ylabel("Fold mean VAF")
            plt.title("Per-fold VAF by training method")
            plt.tight_layout()
            plt.savefig(OUTDIR / "02_vaf_by_method_boxplot.png", dpi=150)
            plt.close()

    # 3) Histogram of per-channel VAF (all)
    if not df_long.empty and "vaf" in df_long.columns:
        v = df_long["vaf"].replace([np.inf, -np.inf], np.nan).dropna().values
        if v.size > 0:
            plt.figure(figsize=(7, 5))
            plt.hist(v, bins=30)
            plt.xlabel("Per-channel VAF")
            plt.ylabel("Count")
            plt.title("Distribution of per-channel VAF (all runs)")
            plt.tight_layout()
            plt.savefig(OUTDIR / "03_vaf_channel_hist.png", dpi=150)
            plt.close()

    # 4) Aggregated table by method
    print("\n=== SUMMARY ===")
    cols = ["train_method", "n_pca", "k_lag", "num_epochs", "lr", "weight_decay", "seed", "mean_vaf", "mean_time", "num_params", "embargo", "embargo_includes_klag"]
    show_cols = [c for c in cols if c in df_sum_sorted.columns]
    print(df_sum_sorted[show_cols].head(12).to_string(index=False))
    print("\nPer-method stats:")
    print(df_sum.groupby("train_method")["mean_vaf"].agg(["count", "mean", "std", "max", "min"]).to_string())
    print("\nOutputs in:")
    print(" ", OUTDIR.resolve())

# ============================ MAIN ============================
if __name__ == "__main__":
    res, rows = run_compare()
    make_plots(res, rows)
