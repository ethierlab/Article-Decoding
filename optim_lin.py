#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Day-0 no-leak cross-validation grid-search for Wiener (FIR+quad) and Kalman decoders.

- Architectures:
    * WienerCascadeEthier (FIR + quad, linéaire au sens "décodage low-complexity")
    * KalmanDecoder (état=EMG, obs=PCA spikes)

- Preprocessing:
    * Downsample spikes+EMG (BIN_FACTOR)
    * Gaussian smoothing sur spikes (segment-wise)
    * Low-pass Butterworth sur |EMG| (segment-wise)
    * PCA sur TRAIN uniquement
    * Embargo basé sur la portée du lissage (no-leak train/val)

- CV:
    * K-fold temporel contigu (pas de shuffle)
    * train = LEFT + RIGHT, val = segment central avec trims EMB

- MIG / HPC friendly:
    * AMP (mixed precision) activé par défaut si GPU dispo (Wiener)
    * TF32 activé en PERF_MODE
    * auto_num_workers plafonné (évite d’exploser une instance MIG)
    * stride = 1 pour un baseline propre
    * libération explicite mémoire GPU entre configs

Inputs:
    --combined_pickle : pickle avec un DataFrame "combined_df"
                        colonnes au moins: "date", "spike_counts", "EMG"

Outputs:
    - gridsearch_results_<jobid>.pkl  (résumé par config)
    - gridsearch_rows_<jobid>.pkl     (long-format fold × canal × seed)
"""

import os, gc, time, argparse, warnings, pickle, itertools
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

warnings.filterwarnings(
    "ignore",
    message="n_jobs value 1 overridden to 1 by setting random_state."
)

# ============================ GLOBAL CONFIG ============================
SEED_BASE        = 42
BIN_FACTOR       = 20      # 1 kHz -> /20 -> 50 Hz
BIN_SIZE         = 0.001   # s (1 ms d'origine)
SMOOTHING_LENGTH = 0.05    # s (~50 ms)
SAMPLING_RATE    = 1000    # Hz
GAUSS_TRUNCATE   = 4.0     # embargo couvre ~truncate*sigma

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERF_MODE = True           # True = perf (TF32+AMP), False = repro stricte

# ============================ SEED / PERF ============================
def set_seed(seed: int = SEED_BASE):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if PERF_MODE:
        # Mode perf MIG/HPC: TF32 + benchmark, non déterministe
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
    else:
        # Mode repro stricte
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def auto_num_workers(default: int = 4) -> int:
    """
    MIG-friendly: lit SLURM_CPUS_PER_TASK et plafonne à 4 workers.
    """
    try:
        n = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
        if n > 0:
            return max(2, min(4, n - 1))
    except Exception:
        pass
    return default

def _seed_worker(worker_id: int):
    worker_seed = SEED_BASE + worker_id
    np.random.seed(worker_seed)

def _dl_generator():
    g = torch.Generator()
    g.manual_seed(SEED_BASE)
    return g

# ============================ DATA HELPERS ============================
def get_all_unit_names(combined_df: pd.DataFrame) -> List[str]:
    unit_set = set()
    for _, row in combined_df.iterrows():
        sc = row.get("spike_counts", None)
        if isinstance(sc, pd.DataFrame):
            unit_set.update(sc.columns)
    return sorted(list(unit_set))

def get_emg_labels_from_df(df: pd.DataFrame, fallback_n: int) -> List[str]:
    for emg in df["EMG"]:
        if isinstance(emg, pd.DataFrame) and not emg.empty:
            return [str(c) for c in emg.columns]
    return [f"ch{c}" for c in range(fallback_n)]

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
    """
    Concat trials (downsampled) sans lissage/filtrage; retourne X_raw, Y_raw, cuts.
    """
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
        if isinstance(ds_emg, pd.DataFrame):
            Yr = ds_emg.values.astype(np.float32)
        else:
            Yr = np.asarray(ds_emg, dtype=np.float32)

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

def preprocess_segment(Xseg, Yseg, bin_factor,
                       bin_size=BIN_SIZE,
                       smoothing_length=SMOOTHING_LENGTH):
    """
    Lissage/filtrage PAR SEGMENT (pas global) -> pas de fuite.
    """
    eff_fs  = SAMPLING_RATE // bin_factor
    eff_bin = bin_factor * bin_size
    Xs = smooth_spike_data(Xseg, eff_bin, smoothing_length)
    Ys = butter_lowpass(np.abs(Yseg), eff_fs)
    return Xs, Ys

def sigma_bins(bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    eff_bin = bin_factor * bin_size
    return (smoothing_length / eff_bin) / 2.0

def embargo_bins(k_lag, bin_factor, bin_size=BIN_SIZE,
                 smoothing_length=SMOOTHING_LENGTH,
                 truncate=GAUSS_TRUNCATE):
    emb = int(np.ceil(truncate * sigma_bins(bin_factor, bin_size, smoothing_length)))
    return max(k_lag, emb)

def time_kfold_splits(n_time: int, n_splits: int) -> List[Tuple[int, int]]:
    block = n_time // n_splits
    splits = []
    for k in range(n_splits):
        v0 = k * block
        v1 = (k + 1) * block if k < n_splits - 1 else n_time
        splits.append((v0, v1))
    return splits

def adjust_cuts_for_segment(start, end, cuts_global,
                            trim_left=0, trim_right=0, seg_len=None):
    local = [c - start for c in cuts_global if start < c < end]
    if seg_len is None:
        seg_len = end - start
    new_start = trim_left
    new_end   = seg_len - trim_right
    return [c - new_start for c in local if new_start < c < new_end]

def valid_window_indices(n_time, k, cuts,
                         start=0, end=None, stride=1):
    end = n_time if end is None else end
    out = []
    for t in range(start + k, end, stride):
        if any(t - k < c < t for c in cuts):
            # interdit de traverser une frontière d'essai
            continue
        out.append(t)
    return out

def build_seq_with_cuts(Z, Y, K_LAG, cuts, stride):
    idx = valid_window_indices(Z.shape[0], K_LAG, cuts, stride=stride)
    if not idx:
        return (np.empty((0, K_LAG * Z.shape[1]), dtype=np.float32),
                np.empty((0, Y.shape[1]), dtype=np.float32))
    X = np.stack([Z[t-K_LAG:t, :].reshape(-1) for t in idx],
                 axis=0).astype(np.float32)
    Yb = np.stack([Y[t, :] for t in idx],
                  axis=0).astype(np.float32)
    return X, Yb

def contiguous_blocks(n_time: int, cuts: List[int]) -> List[Tuple[int, int]]:
    if n_time <= 0:
        return []
    points = [0] + sorted([c for c in cuts if 0 < c < n_time]) + [n_time]
    return [(points[i], points[i+1]) for i in range(len(points)-1)]

# ============================ MODELS (Wiener + Kalman) ============================
class WienerCascadeEthier(nn.Module):
    """ FIR multi-entrée -> non-linéarité quadratique (par canal EMG). """
    def __init__(self, input_dim: int, n_out: int, poly_order: int = 2, bias: bool = True):
        super().__init__()
        assert poly_order in (1, 2)
        self.poly_order = poly_order
        self.lin = nn.Linear(input_dim, n_out, bias=bias)
        self.c0  = nn.Parameter(torch.zeros(n_out))
        self.c1  = nn.Parameter(torch.ones(n_out))
        if self.poly_order == 2:
            self.c2 = nn.Parameter(0.1 * torch.ones(n_out))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        z = self.lin(x)
        y = self.c0 + self.c1 * z
        if self.poly_order == 2:
            y = y + self.c2 * (z * z)
        return y

class KFParams:
    def __init__(self, A, H, Q, R, x0, P0):
        self.A  = A
        self.H  = H
        self.Q  = Q
        self.R  = R
        self.x0 = x0
        self.P0 = P0

class KalmanDecoder:
    """ KF état=EMG, obs=features neuraux (PCA). """
    def __init__(self):
        self.params: Optional[KFParams] = None

    @staticmethod
    def _fit_A(X_blocks: List[np.ndarray]) -> np.ndarray:
        Xt_list, Xt1_list = [], []
        for X in X_blocks:
            if len(X) < 2:
                continue
            Xt_list.append(X[:-1])
            Xt1_list.append(X[1:])
        if not Xt_list:
            n_out = X_blocks[0].shape[1]
            return np.eye(n_out)
        Xt  = np.vstack(Xt_list)
        Xt1 = np.vstack(Xt1_list)
        A = (np.linalg.pinv(Xt) @ Xt1).T
        return A

    @staticmethod
    def _fit_H(Z: np.ndarray, X: np.ndarray) -> np.ndarray:
        Xp = np.linalg.pinv(X)
        H  = Z.T @ Xp.T
        return H

    def fit(self, Z_train: np.ndarray, Y_train: np.ndarray, cuts_train: List[int]):
        blocks = contiguous_blocks(len(Y_train), cuts_train)
        X_blocks = [Y_train[s:e] for (s, e) in blocks]

        A = self._fit_A(X_blocks)
        H = self._fit_H(Z_train, Y_train)

        # Process noise Q
        W_list = []
        for X in X_blocks:
            if len(X) < 2:
                continue
            X_pred = X[:-1] @ A.T
            W_list.append(X[1:] - X_pred)
        if W_list:
            W = np.vstack(W_list)
            Q = np.cov(W.T) + 1e-6 * np.eye(W.shape[1])
        else:
            Q = 1e-3 * np.eye(Y_train.shape[1])

        # Obs noise R
        Z_pred = Y_train @ H.T
        V = Z_train - Z_pred
        R = np.cov(V.T) + 1e-6 * np.eye(Z_train.shape[1])

        x0 = Y_train[0].copy()
        P0 = np.cov(Y_train.T) + 1e-3 * np.eye(Y_train.shape[1])

        self.params = KFParams(A, H, Q, R, x0, P0)

    def filter_block(self, Z: np.ndarray, x0: np.ndarray, P0: np.ndarray):
        A = self.params.A; H = self.params.H
        Q = self.params.Q; R = self.params.R

        x = x0.copy()
        P = P0.copy()
        nT = len(Z)
        n_out = x.shape[0]
        X_hat = np.zeros((nT, n_out), dtype=float)

        AT, HT = A.T, H.T
        I = np.eye(n_out)

        for t in range(nT):
            # Predict
            x = A @ x
            P = A @ P @ AT + Q
            # Update
            z = Z[t]
            S = H @ P @ HT + R
            K = P @ HT @ np.linalg.inv(S)
            y = z - (H @ x)
            x = x + K @ y
            P = (I - K @ H) @ P
            X_hat[t] = x

        return X_hat, x, P

    def filter(self, Z: np.ndarray, cuts_val: List[int]) -> np.ndarray:
        assert self.params is not None
        blocks = contiguous_blocks(len(Z), cuts_val)
        x, P = self.params.x0, self.params.P0
        out = np.zeros((len(Z), len(x)), dtype=float)
        for s, e in blocks:
            Xb, x, P = self.filter_block(Z[s:e], x, P)
            out[s:e] = Xb
        return out

# ============================ PCA ============================
def fit_pca(X, n_components, seed):
    model = PCA(n_components=max(n_components, 2), random_state=seed)
    model.fit(X)
    return model

def pca_transform(model, X):
    return model.transform(X)

# ============================ METRICS / TRAIN ============================
def eval_vaf_full_numpy(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
    v = []
    for ch in range(yt.shape[1]):
        vt = np.var(yt[:, ch])
        if vt < 1e-12:
            v.append(np.nan)
        else:
            v.append(1.0 - np.var(yt[:, ch] - yp[:, ch]) / vt)
    return np.asarray(v, dtype=np.float32)

def train_wiener_model(model: nn.Module,
                       X_train: np.ndarray,
                       Y_train: np.ndarray,
                       num_epochs: int,
                       lr: float,
                       batch_size: int,
                       num_workers: Optional[int],
                       use_amp: bool = True):
    if num_workers is None:
        num_workers = auto_num_workers()

    x_cpu = torch.as_tensor(X_train, dtype=torch.float32)
    y_cpu = torch.as_tensor(Y_train, dtype=torch.float32)
    dset  = TensorDataset(x_cpu, y_cpu)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        generator=None if PERF_MODE else _dl_generator(),
        worker_init_fn=None if PERF_MODE else _seed_worker
    )

    opt  = optim.Adam(model.parameters(), lr=lr)
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
                yp = model(xb)
                loss = crit(yp, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        if ep == 1 or ep % 50 == 0:
            print(f"  epoch {ep}/{num_epochs}  loss={total/len(loader):.4f}")
    return model

# ============================ GRID (LINEAIRE ONLY) ============================
GRID: Dict[str, Dict[str, List[Any]]] = {
    "wiener": dict(
        n_pca=[8, 16, 24, 32, 48, 64, 80, 96],
        k_lag=[5, 10, 15, 20, 25, 40, 80, 120, 160],
        num_epochs=[100, 150, 200, 300],
        lr=[1e-3, 3e-3, 1e-2],
    )
 }

DECODER_DISPLAY = {
    "wiener": "Wiener",
    "kalman": "Kalman",
}

def cartesian_product(param_dict: Dict[str, List[Any]]):
    keys, vals = zip(*param_dict.items())
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

# ============================ MAIN ============================
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--decoders", nargs="+",
                    default=["wiener", "kalman"],
                    help="Choix: wiener, kalman")
    ap.add_argument("--seeds", type=int, default=3,
                    help="Nombre de seeds (0..seeds-1)")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--progress", type=int, default=50,
                    help="Print heartbeat every N runs")
    ap.add_argument("--combined_pickle", type=str,
                    default="combined.pkl")
    # HPC / MIG partition
    ap.add_argument("--start_idx", type=int, default=0,
                    help="Index de départ dans la liste de configs")
    ap.add_argument("--num_per_job", type=int, default=50,
                    help="Nombre de configs traitées dans ce job")
    # Perf
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=-1,
                    help="-1 => auto (plafonné via auto_num_workers)")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--perf_mode", action="store_true")
    args = ap.parse_args()

    global PERF_MODE
    if args.perf_mode:
        PERF_MODE = True

    # ---------- Load data ----------
    combined_df = pd.read_pickle(args.combined_pickle)
    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"], errors="coerce")

    unique_days = sorted(combined_df["date"].dropna().unique())
    if not unique_days:
        raise RuntimeError("No days in combined_df")

    day0 = unique_days[0]
    train_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)

    ALL_UNITS = get_all_unit_names(combined_df)

    # Detect EMG channels
    n_emg_channels = 0
    for _, row in combined_df.iterrows():
        emg_val = row.get("EMG", None)
        if emg_val is not None:
            if isinstance(emg_val, pd.DataFrame) and not emg_val.empty:
                n_emg_channels = emg_val.shape[1]
                break
            elif isinstance(emg_val, np.ndarray) and emg_val.size > 0:
                n_emg_channels = emg_val.shape[1]
                break
    if n_emg_channels == 0:
        raise RuntimeError("Could not detect EMG channels.")
    EMG_LABELS = get_emg_labels_from_df(combined_df, n_emg_channels)

    # raw day0 (no smoothing yet)
    X0_raw, Y0_raw, cuts0 = build_continuous_dataset_raw(
        train_df, BIN_FACTOR, all_units=ALL_UNITS
    )
    if X0_raw.size == 0:
        raise RuntimeError("Empty day0 after downsampling.")

    splits = time_kfold_splits(X0_raw.shape[0], args.folds)

    # Perf knobs
    set_seed(SEED_BASE)
    BATCH   = args.batch_size
    WORKERS = auto_num_workers() if args.num_workers == -1 else args.num_workers
    USE_AMP = (not args.no_amp)

    # Build combos
    combos = []
    for dec in args.decoders:
        for cfg in cartesian_product(GRID[dec]):
            for seed in range(args.seeds):
                combos.append((dec, cfg, seed))

    start = args.start_idx
    end   = min(len(combos), start + args.num_per_job)
    my_combos = combos[start:end]
    print(f"This job handles combos [{start}..{end-1}] of {len(combos)} total.")

    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")
    out_summary = Path(f"gridsearch_results_{job_id}.pkl")
    out_rows    = Path(f"gridsearch_rows_{job_id}.pkl")

    # Load existing partial results (resume)
    if out_summary.exists():
        results: List[Dict[str, Any]] = pickle.load(open(out_summary, "rb"))
    else:
        results = []
    if out_rows.exists():
        long_rows: List[Dict[str, Any]] = pickle.load(open(out_rows, "rb"))
    else:
        long_rows = []

    done_keys = {
        (r["decoder"], r["n_pca"], r["k_lag"],
         r["num_epochs"], r["lr"], r["seed"])
        for r in results
    }

    total = 0
    for decoder, cfg, seed in my_combos:
        key = (decoder, cfg["n_pca"], cfg["k_lag"],
               cfg["num_epochs"], cfg["lr"], seed)
        if key in done_keys:
            continue

        total += 1
        if total % args.progress == 0:
            print(f"  …{total} runs done in this job")

        try:
            set_seed(SEED_BASE + seed)

            n_pca     = int(cfg["n_pca"])
            k_lag     = int(cfg["k_lag"])
            num_ep    = int(cfg["num_epochs"])
            lr        = float(cfg["lr"])
            stride    = 1  # pas de stride >1 sinon tu dilues le problème

            # EMB: pour Kalman on met k_lag_effectif=1 (comme dans le script de base)
            k_for_embargo = k_lag if decoder == "wiener" else 1
            EMB = embargo_bins(k_for_embargo, BIN_FACTOR, BIN_SIZE,
                               SMOOTHING_LENGTH, GAUSS_TRUNCATE)

            vafs_fold  = []
            fold_times = []
            param_count = None

            for i_fold, (val_start, val_end) in enumerate(splits):
                # Segments bruts
                X_left_raw  = X0_raw[:val_start];   Y_left_raw  = Y0_raw[:val_start]
                X_val_raw   = X0_raw[val_start:val_end]; Y_val_raw = Y0_raw[val_start:val_end]
                X_right_raw = X0_raw[val_end:];     Y_right_raw = Y0_raw[val_end:]

                # LEFT (trim right = EMB)
                Xl, Yl = (preprocess_segment(X_left_raw, Y_left_raw,
                                             BIN_FACTOR, BIN_SIZE,
                                             SMOOTHING_LENGTH)
                          if len(X_left_raw) else (np.empty((0,)), np.empty((0,))))
                if len(Xl) > EMB:
                    Xl = Xl[:len(Xl) - EMB]
                    Yl = Yl[:len(Yl) - EMB]
                    cuts_left = adjust_cuts_for_segment(
                        0, len(X_left_raw), cuts0,
                        trim_left=0, trim_right=EMB, seg_len=len(X_left_raw)
                    )
                else:
                    Xl = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
                    Yl = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
                    cuts_left = []

                # VAL (trim both sides = EMB)
                Xv, Yv = (preprocess_segment(X_val_raw, Y_val_raw,
                                             BIN_FACTOR, BIN_SIZE,
                                             SMOOTHING_LENGTH)
                          if len(X_val_raw) else (np.empty((0,)), np.empty((0,))))
                if len(Xv) > 2 * EMB:
                    Xv = Xv[EMB:len(Xv) - EMB]
                    Yv = Yv[EMB:len(Yv) - EMB]
                    cuts_val = adjust_cuts_for_segment(
                        val_start, val_end, cuts0,
                        trim_left=EMB, trim_right=EMB, seg_len=len(X_val_raw)
                    )
                else:
                    Xv = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
                    Yv = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
                    cuts_val = []

                # RIGHT (trim left = EMB)
                Xr, Yr = (preprocess_segment(X_right_raw, Y_right_raw,
                                             BIN_FACTOR, BIN_SIZE,
                                             SMOOTHING_LENGTH)
                          if len(X_right_raw) else (np.empty((0,)), np.empty((0,))))
                if len(Xr) > EMB:
                    Xr = Xr[EMB:]
                    Yr = Yr[EMB:]
                    cuts_right = adjust_cuts_for_segment(
                        val_end, len(X0_raw), cuts0,
                        trim_left=EMB, trim_right=0, seg_len=len(X_right_raw)
                    )
                else:
                    Xr = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
                    Yr = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
                    cuts_right = []

                # concat train (LEFT + RIGHT)
                if Xl.size and Xr.size:
                    X_train_time = np.vstack([Xl, Xr])
                    Y_train_time = np.vstack([Yl, Yr])
                    cuts_train = cuts_left + [len(Xl)] + [c + len(Xl) for c in cuts_right]
                elif Xl.size:
                    X_train_time, Y_train_time, cuts_train = Xl, Yl, cuts_left
                else:
                    X_train_time, Y_train_time, cuts_train = Xr, Yr, cuts_right

                if X_train_time.shape[0] == 0 or Xv.shape[0] == 0:
                    continue

                # PCA sur TRAIN seulement
                pca_model = fit_pca(X_train_time, n_components=n_pca,
                                    seed=SEED_BASE + seed + i_fold)
                Z_tr = pca_transform(pca_model, X_train_time)[:, :n_pca]
                Z_va = pca_transform(pca_model, Xv)[:, :n_pca]

                if decoder == "wiener":
                    # Fenêtres (lag K) sur PCA
                    if X_train_time.shape[0] <= k_lag or Xv.shape[0] <= k_lag:
                        continue
                    X_tr, Y_tr = build_seq_with_cuts(
                        Z_tr, Y_train_time, k_lag, cuts_train, stride
                    )
                    X_va, Y_va = build_seq_with_cuts(
                        Z_va, Yv,           k_lag, cuts_val,   stride
                    )
                    if X_tr.shape[0] == 0 or X_va.shape[0] == 0:
                        continue

                    model = WienerCascadeEthier(k_lag * n_pca, Y_tr.shape[1]).to(DEVICE)
                    if param_count is None:
                        param_count = sum(p.numel() for p in model.parameters()
                                          if p.requires_grad)

                    t0 = time.perf_counter()
                    train_wiener_model(
                        model, X_tr, Y_tr,
                        num_epochs=num_ep, lr=lr,
                        batch_size=BATCH,
                        num_workers=WORKERS,
                        use_amp=USE_AMP
                    )
                    fold_times.append(time.perf_counter() - t0)

                    with torch.no_grad():
                        Yp = model(torch.from_numpy(X_va).float().to(DEVICE)).cpu().numpy()
                    vaf_ch = eval_vaf_full_numpy(Y_va, Yp)
                    mean_vaf = float(np.nanmean(vaf_ch))

                else:  # kalman
                    # Pas de fenêtrage: KF travaille à l'échelle du temps
                    kf = KalmanDecoder()
                    t0 = time.perf_counter()
                    kf.fit(Z_tr, Y_train_time, cuts_train)
                    Y_hat = kf.filter(Z_va, cuts_val)
                    fold_times.append(time.perf_counter() - t0)
                    vaf_ch = eval_vaf_full_numpy(Yv, Y_hat)
                    mean_vaf = float(np.nanmean(vaf_ch))

                    if param_count is None:
                        n_out  = Y_train_time.shape[1]
                        n_feat = Z_tr.shape[1]
                        param_count = (
                            n_out * n_out +             # A
                            n_feat * n_out +            # H
                            n_out * (n_out+1)//2 +      # Q (sym)
                            n_feat * (n_feat+1)//2 +    # R (sym)
                            n_out +                     # x0
                            n_out * (n_out+1)//2        # P0 (sym)
                        )

                vafs_fold.append(mean_vaf if not np.isnan(mean_vaf) else -1.0)

                dec_name = DECODER_DISPLAY.get(decoder, decoder)
                for ch_idx, v in enumerate(vaf_ch):
                    long_rows.append(dict(
                        decoder=dec_name,
                        dim_red="PCA",
                        align="crossval",
                        day_int=0,
                        fold=i_fold,
                        emg_channel=int(ch_idx),
                        emg_label=str(EMG_LABELS[ch_idx]) if ch_idx < len(EMG_LABELS) else f"ch{ch_idx}",
                        vaf=float(v),
                        seed=int(seed),
                        n_pca=int(n_pca),
                        k_lag=int(k_lag),
                        hidden_dim=0,
                        num_epochs=int(num_ep),
                        lr=float(lr),
                    ))

                if decoder == "wiener":
                    del model
                    torch.cuda.empty_cache()
                gc.collect()

            if not vafs_fold:
                raise RuntimeError("No valid folds (after embargo/windowing) for this config.")

            results.append(dict(
                decoder=decoder,
                seed=int(seed),
                num_params=int(param_count) if param_count is not None else None,
                mean_vaf=float(np.mean(vafs_fold)),
                fold_vafs=[float(x) for x in vafs_fold],
                fold_times=[float(s) for s in fold_times],
                mean_time=float(np.mean(fold_times)) if fold_times else float("nan"),
                n_pca=int(n_pca),
                k_lag=int(k_lag),
                hidden_dim=0,
                num_epochs=int(num_ep),
                lr=float(lr),
            ))
            done_keys.add(key)

            # checkpoints fréquents pour HPC
            pickle.dump(results, open(out_summary, "wb"))
            pickle.dump(long_rows, open(out_rows, "wb"))

        except RuntimeError as e:
            warnings.warn(f"{key} failed: {e}")
            continue

    print("\nJob done. Saved:")
    print("  ", out_summary.resolve())
    print("  ", out_rows.resolve())

if __name__ == "__main__":
    main()
