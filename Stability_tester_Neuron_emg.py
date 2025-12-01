#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import warnings
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import pinv
from sklearn.decomposition import PCA

# Optional UMAP
try:
    import umap
except Exception:
    umap = None

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.")

###############################################################################
# CONFIG (defaults; overridable by CLI)
###############################################################################
SEED = 42
BIN_FACTOR = 20           # 1 kHz -> 50 Hz when original BIN_SIZE=1ms
BIN_SIZE = 0.001          # seconds (1 ms base); effective bin = BIN_FACTOR * BIN_SIZE
SMOOTHING_LENGTH = 0.05   # seconds (50 ms gaussian)
SAMPLING_RATE = 1000      # Hz
GAUSS_TRUNCATE = 4.0      # embargo coverage ~ truncate*sigma

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERF_MODE = True  # True=fast, non-strict determinism

###############################################################################
# RUNTIME / SEED
###############################################################################
def set_seed(seed=SEED):
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

def _seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

###############################################################################
# DATA HELPERS (borrowed/adapted from script 2)
###############################################################################
def get_all_unit_names(combined_df):
    unit_set = set()
    for _, row in combined_df.iterrows():
        sc = row.get("spike_counts", None)
        if isinstance(sc, pd.DataFrame):
            unit_set.update(sc.columns)
    return sorted(list(unit_set))

def butter_lowpass(data, fs, order=4, cutoff_hz=5.0):
    nyq = 0.5 * fs
    norm = cutoff_hz / nyq
    b, a = butter(order, norm, btype='low', analog=False)
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

def smooth_spike_data(x_2d, bin_size=0.001, smoothing_length=0.05):
    sigma = (smoothing_length / bin_size) / 2
    return gaussian_filter1d(x_2d.astype(np.float32), sigma=sigma, axis=0)

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
        if isinstance(ds_emg, pd.DataFrame):
            Yr = ds_emg.values.astype(np.float32)
        else:
            Yr = np.asarray(ds_emg, dtype=np.float32)

        spikes_all.append(Xr)
        emg_all.append(Yr)
        lengths.append(len(Xr))

    if len(spikes_all) == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), []

    cuts = np.cumsum(lengths)[:-1].tolist()
    return np.concatenate(spikes_all, axis=0), np.concatenate(emg_all, axis=0), cuts

def preprocess_segment(Xseg, Yseg, bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    eff_fs = SAMPLING_RATE // bin_factor
    Xs = smooth_spike_data(Xseg, bin_size * bin_factor, smoothing_length)
    Ys = butter_lowpass(np.abs(Yseg), eff_fs)
    return Xs, Ys

def sigma_bins(bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    return (smoothing_length / (bin_size * bin_factor)) / 2.0

def embargo_bins(K_LAG, bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH, truncate=GAUSS_TRUNCATE):
    emb = int(np.ceil(truncate * sigma_bins(bin_factor, bin_size, smoothing_length)))
    return max(K_LAG, emb)

def time_kfold_splits(n_time, n_splits):
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
    new_end = seg_len - trim_right
    return [c - new_start for c in local if new_start < c < new_end]

def valid_window_indices(n_time, k, cuts, stride=1, start=0, end=None):
    end = n_time if end is None else end
    out = []
    for t in range(start + k, end, stride):
        if any(t - k < c < t for c in cuts):
            continue
        out.append(t)
    return out

def build_seq_with_cuts(Z, Y, K_LAG, cuts, stride, is_linear):
    idx = valid_window_indices(Z.shape[0], K_LAG, cuts, stride=stride)
    if not idx:
        if is_linear:
            return np.empty((0, K_LAG * Z.shape[1]), dtype=np.float32), np.empty((0, Y.shape[1]), dtype=np.float32)
        else:
            return np.empty((0, K_LAG, Z.shape[1]), dtype=np.float32), np.empty((0, Y.shape[1]), dtype=np.float32)
    if is_linear:
        X = np.stack([Z[t-K_LAG:t, :].reshape(-1) for t in idx], axis=0).astype(np.float32)
    else:
        X = np.stack([Z[t-K_LAG:t, :] for t in idx], axis=0).astype(np.float32)
    Yb = np.stack([Y[t, :] for t in idx], axis=0).astype(np.float32)
    return X, Yb

###############################################################################
# MODELS (same shapes as your first script, multi-output = EMG channels)
###############################################################################
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class LinearLagDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act  = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, output_size)
    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        return self.lin2(x)

class LiGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.x2z = nn.Linear(input_size, hidden_size)
        self.h2z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=False)
    def forward(self, x, h):
        z = torch.sigmoid(self.x2z(x) + self.h2z(h))
        h_candidate = torch.relu(self.x2h(x) + self.h2h(h))
        return (1 - z) * h + z * h_candidate

class LiGRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LiGRUCell(input_size, hidden_size)
        self.fc   = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        b, T, _ = x.size()
        h = torch.zeros(b, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)

###############################################################################
# DIM REDUCTION
###############################################################################
def get_dimred_model(data, method, n_components, seed):
    if method.upper() == "PCA":
        model = PCA(n_components=max(n_components, 2), random_state=seed)
        model.fit(data)
        return model
    elif method.upper() == "UMAP":
        if umap is None:
            raise RuntimeError("umap-learn not installed. Run 'pip install umap-learn'")
        model = umap.UMAP(n_components=n_components, random_state=seed)
        model.fit(data)
        return model
    else:
        raise ValueError(f"Unknown dim. reduction: {method}")

def transform_dimred(model, data, method):
    if method.upper() in ("PCA", "UMAP"):
        return model.transform(data)
    else:
        raise ValueError(f"Unknown dim. reduction: {method}")

###############################################################################
# METRICS + TRAIN/EVAL
###############################################################################
def compute_vaf_1d(y_true, y_pred):
    var_resid = np.var(y_true - y_pred)
    var_true  = np.var(y_true)
    if var_true < 1e-12:
        return np.nan
    return 1.0 - (var_resid / var_true)

def compute_multichannel_vaf(y_true, y_pred):
    if y_true.shape[0] == 0:
        return np.array([])
    n_ch = y_true.shape[1]
    return np.array([compute_vaf_1d(y_true[:, ch], y_pred[:, ch]) for ch in range(n_ch)])

def train_model(model, X_train, Y_train, num_epochs=200, lr=0.001,
                batch_size=256, num_workers=None, use_amp=True):
    if num_workers is None:
        num_workers = auto_num_workers()

    x_cpu = torch.as_tensor(X_train, dtype=torch.float32)
    y_cpu = torch.as_tensor(Y_train, dtype=torch.float32)
    dset  = TensorDataset(x_cpu, y_cpu)
    loader = DataLoader(
        dset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4,
        worker_init_fn=None if PERF_MODE else _seed_worker
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))

    model.train()
    for ep in range(1, num_epochs+1):
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        if ep % 10 == 0:
            print(f"  Epoch {ep}/{num_epochs} - loss={total/len(loader):.4f}")
    return model

def evaluate_model(model, X, Y, batch_size=256, use_amp=True):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.as_tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(use_amp and torch.cuda.is_available())):
                out = model(bx)
            preds.append(out.cpu().numpy())
    if preds:
        preds = np.concatenate(preds, axis=0)
        vafs = compute_multichannel_vaf(Y, preds)
        return float(np.nanmean(vafs)), vafs
    else:
        return np.nan, np.full((Y.shape[1],), np.nan)

###############################################################################
# HYPERPARAMS (from your second script)
###############################################################################
ARCH_HYPERPARAMS = {
    "GRU":    dict(N_PCA=32, K_LAG=25, HIDDEN=96,   NUM_EPOCHS=200, LR=0.003),
    "LSTM":   dict(N_PCA=24, K_LAG=25, HIDDEN=128,  NUM_EPOCHS=300, LR=0.003),
    "Linear": dict(N_PCA=32, K_LAG=16, HIDDEN=64,   NUM_EPOCHS=100, LR=0.003),
    "LiGRU":  dict(N_PCA=32, K_LAG=16, HIDDEN=5,    NUM_EPOCHS=200, LR=0.001),
}

###############################################################################
# NEURON REMOVAL (keep original semantics: zeroing)
###############################################################################
def zero_out_units_in_matrix(X, unit_indices_to_zero):
    """Return a COPY of X where selected unit columns are set to zero."""
    if len(unit_indices_to_zero) == 0:
        return X.copy()
    Xz = X.copy()
    Xz[:, unit_indices_to_zero] = 0.0
    return Xz

###############################################################################
# MAIN
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Day-0 neuron-loss robustness using preprocessing & CV from script 2")
    parser.add_argument('--combined_pickle', type=str, default='combined.pkl')
    parser.add_argument('--decoders', type=str, nargs='+', default=['GRU','LSTM','Linear','LiGRU'],
                        choices=list(ARCH_HYPERPARAMS.keys()))
    parser.add_argument('--dimred', type=str, default='PCA', choices=['PCA','UMAP'])
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--perf_mode', action='store_true')  # optional override (default True)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--stride_mul', type=float, default=1.0)
    parser.add_argument('--max_removed', type=int, default=-1, help='cap on #removed units; -1 = all possible')
    parser.add_argument('--removal_mode', type=str, default='systematic', choices=['systematic','random'])
    parser.add_argument('--random_repeats', type=int, default=3, help='only used if removal_mode=random')

    args = parser.parse_args()

    global PERF_MODE
    if args.perf_mode:
        PERF_MODE = True
    set_seed(args.seed)

    WORKERS = auto_num_workers() if args.num_workers == -1 else args.num_workers
    USE_AMP = (not args.no_amp)

    # Load combined DF
    combined_df = pd.read_pickle(args.combined_pickle)
    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"])

    # Units and EMG channels
    ALL_UNITS = get_all_unit_names(combined_df)
    if len(ALL_UNITS) == 0:
        print("[ERROR] No units found in combined_df.")
        return

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
        print("[ERROR] Could not detect EMG channels from DataFrame.")
        return

    # Day 0 only
    unique_days = sorted(combined_df["date"].unique())
    if len(unique_days) == 0:
        print("[ERROR] No days found.")
        return
    day0 = unique_days[0]
    day0_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)

    # Raw day0 (no global smoothing)
    X0_raw, Y0_raw, cuts0 = build_continuous_dataset_raw(day0_df, BIN_FACTOR, all_units=ALL_UNITS)
    if X0_raw.size == 0:
        print("[ERROR] empty day0")
        return

    # CV splits
    splits = time_kfold_splits(X0_raw.shape[0], args.n_folds)

    # Results container
    rows = []

    # === Loop over decoders ===
    for dec_name in args.decoders:
        hp = ARCH_HYPERPARAMS[dec_name]
        N_PCA, K_LAG, HIDDEN, NUM_EPOCHS, LR = (
            hp["N_PCA"], hp["K_LAG"], hp["HIDDEN"], hp["NUM_EPOCHS"], hp["LR"]
        )
        STRIDE = max(1, int(args.stride_mul * K_LAG))
        EMB = embargo_bins(K_LAG, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, GAUSS_TRUNCATE)
        is_linear = (dec_name == "Linear")

        print(f"\n===== Decoder: {dec_name} | N_PCA={N_PCA}, K_LAG={K_LAG}, H={HIDDEN}, E={NUM_EPOCHS}, LR={LR} =====")

        # Per-fold training (no removal) and test with progressive removal
        for fold_idx, (val_start, val_end) in enumerate(splits):
            print(f"\n[Fold {fold_idx+1}/{args.n_folds}] val=[{val_start}:{val_end})")

            # Raw segments
            X_left_raw  = X0_raw[:val_start];   Y_left_raw  = Y0_raw[:val_start]
            X_val_raw   = X0_raw[val_start:val_end]; Y_val_raw = Y0_raw[val_start:val_end]
            X_right_raw = X0_raw[val_end:];     Y_right_raw = Y0_raw[val_end:]

            # Preprocess + embargo trims per segment (to avoid leakage)
            # LEFT: trim right by EMB
            X_left_p, Y_left_p = preprocess_segment(X_left_raw, Y_left_raw, BIN_FACTOR) if len(X_left_raw) else (np.empty((0,)), np.empty((0,)))
            if len(X_left_p) > EMB:
                X_left_p = X_left_p[:len(X_left_p)-EMB]; Y_left_p = Y_left_p[:len(Y_left_p)-EMB]
                cuts_left = adjust_cuts_for_segment(0, len(X_left_raw), cuts0, trim_left=0, trim_right=EMB, seg_len=len(X_left_raw))
            else:
                X_left_p = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
                Y_left_p = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
                cuts_left = []

            # VAL: trim both sides by EMB
            X_val_p, Y_val_p = preprocess_segment(X_val_raw, Y_val_raw, BIN_FACTOR) if len(X_val_raw) else (np.empty((0,)), np.empty((0,)))
            if len(X_val_p) > 2*EMB:
                X_val_p = X_val_p[EMB:len(X_val_p)-EMB]; Y_val_p = Y_val_p[EMB:len(Y_val_p)-EMB]
                cuts_val = adjust_cuts_for_segment(val_start, val_end, cuts0, trim_left=EMB, trim_right=EMB, seg_len=len(X_val_raw))
            else:
                X_val_p = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
                Y_val_p = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
                cuts_val = []

            # RIGHT: trim left by EMB
            X_right_p, Y_right_p = preprocess_segment(X_right_raw, Y_right_raw, BIN_FACTOR) if len(X_right_raw) else (np.empty((0,)), np.empty((0,)))
            if len(X_right_p) > EMB:
                X_right_p = X_right_p[EMB:]; Y_right_p = Y_right_p[EMB:]
                cuts_right = adjust_cuts_for_segment(val_end, len(X0_raw), cuts0, trim_left=EMB, trim_right=0, seg_len=len(X_right_raw))
            else:
                X_right_p = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
                Y_right_p = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
                cuts_right = []

            # Train set = LEFT + RIGHT
            if X_left_p.size and X_right_p.size:
                X_tr_time = np.vstack([X_left_p, X_right_p])
                Y_tr_time = np.vstack([Y_left_p, Y_right_p])
                cuts_train = cuts_left + [c + len(X_left_p) for c in cuts_right] + [len(X_left_p)]
            elif X_left_p.size:
                X_tr_time, Y_tr_time, cuts_train = X_left_p, Y_left_p, cuts_left
            else:
                X_tr_time, Y_tr_time, cuts_train = X_right_p, Y_right_p, cuts_right

            if X_tr_time.shape[0] <= K_LAG or X_val_p.shape[0] <= K_LAG:
                print("  [WARN] not enough samples after embargo; skipping fold")
                continue

            # Dim-red fit on TRAIN only
            dimred_model = get_dimred_model(X_tr_time, args.dimred, N_PCA, args.seed + fold_idx)
            Z_tr = transform_dimred(dimred_model, X_tr_time, args.dimred)[:, :N_PCA]
            Z_va = transform_dimred(dimred_model, X_val_p,   args.dimred)[:, :N_PCA]

            # Windowing
            X_tr, Y_tr = build_seq_with_cuts(Z_tr, Y_tr_time, K_LAG, cuts_train, STRIDE, is_linear)
            X_te_base, Y_te = build_seq_with_cuts(Z_va, Y_val_p, K_LAG, cuts_val, STRIDE, is_linear)
            if X_tr.shape[0] == 0 or X_te_base.shape[0] == 0:
                print("  [WARN] empty after windowing; skipping fold")
                continue

            # Build model
            if dec_name == "GRU":
                model = GRUDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
            elif dec_name == "LSTM":
                model = LSTMDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
            elif dec_name == "Linear":
                model = LinearLagDecoder(K_LAG * N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
            else:
                model = LiGRUDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)

            # Train (no neuron removal)
            print("  Training (no removal)...")
            model = train_model(model, X_tr, Y_tr, num_epochs=NUM_EPOCHS, lr=LR,
                                batch_size=args.batch_size, num_workers=WORKERS, use_amp=USE_AMP)

            # --- Baseline (0 removed) ---
            vaf_mean, vaf_ch = evaluate_model(model, X_te_base, Y_te, batch_size=args.batch_size, use_amp=USE_AMP)
            rows.append(dict(
                day=day0, fold=fold_idx, decoder=dec_name, removed=0,
                vaf_mean=vaf_mean, vaf_ch=vaf_ch, mode='baseline'
            ))
            print(f"  [baseline] mean VAF = {vaf_mean:.3f}")

            # --- Progressive removal on VALIDATION ONLY (zeroing units) ---
            n_units = X_val_p.shape[1]
            max_remove = n_units if args.max_removed < 0 else min(n_units, args.max_removed)

            if args.removal_mode == 'systematic':
                # fixed order: ALL_UNITS order matches columns order
                removal_order = list(range(n_units))
                repeats = 1
            else:
                repeats = max(1, args.random_repeats)

            for rep in range(repeats):
                if args.removal_mode == 'random':
                    removal_order = list(range(n_units))
                    random.Random(args.seed + 1000*fold_idx + 17*rep).shuffle(removal_order)
                removed_set = set()

                print(f"  Evaluating robustness ({args.removal_mode}, rep={rep+1}/{repeats})...")
                # step = number removed so far
                for step in range(1, max_remove + 1):
                    removed_set.add(removal_order[step-1])

                    # Zero-out on the preprocessed validation spike matrix (before projection)
                    X_val_removed = zero_out_units_in_matrix(X_val_p, sorted(list(removed_set)))

                    # Recompute projection via the SAME dimred model (fit on train)
                    Z_va_removed = transform_dimred(dimred_model, X_val_removed, args.dimred)[:, :N_PCA]

                    # Re-window with same cuts/stride
                    X_te, Y_te_same = build_seq_with_cuts(Z_va_removed, Y_val_p, K_LAG, cuts_val, STRIDE, is_linear)
                    if X_te.shape[0] == 0:
                        vaf_mean = np.nan
                        vaf_ch   = np.full((n_emg_channels,), np.nan)
                    else:
                        vaf_mean, vaf_ch = evaluate_model(model, X_te, Y_te_same,
                                                          batch_size=args.batch_size, use_amp=USE_AMP)

                    rows.append(dict(
                        day=day0, fold=fold_idx, decoder=dec_name, removed=step,
                        vaf_mean=vaf_mean, vaf_ch=vaf_ch,
                        mode=('random' if args.removal_mode=='random' else 'systematic'),
                        rep=(rep if args.removal_mode=='random' else 0)
                    ))
                    if step % max(1, max_remove//5) == 0 or step in (1, max_remove):
                        print(f"    removed={step}/{max_remove} -> mean VAF={vaf_mean:.3f}")

    # Save results as tidy rows (explode per-channel VAFs too)
    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    # Expand channel VAFs for convenience
    expanded = []
    for _, r in df.iterrows():
        vch = r['vaf_ch']
        for ch_i, v in enumerate(vch):
            rr = r.copy()
            rr['emg_channel'] = ch_i
            rr['vaf'] = float(v)
            del rr['vaf_ch']
            expanded.append(rr)
    out_df = pd.DataFrame(expanded)
    save_path = os.path.join(args.save_dir, "neuron_robustness_day0.pkl")
    pd.to_pickle(out_df, save_path)
    print(f"\n[INFO] Saved results to {save_path}")

    # Also write a small CSV summary (mean over folds/channels/rep per (decoder, removed))
    summary = (out_df
               .groupby(['decoder','removed'], as_index=False)['vaf']
               .mean()
               .rename(columns={'vaf':'mean_vaf'}))
    csv_path = os.path.join(args.save_dir, "neuron_robustness_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"[INFO] Saved summary curve to {csv_path}")

if __name__ == "__main__":
    main()
