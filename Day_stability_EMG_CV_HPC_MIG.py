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

# UMAP (optionnel)
try:
    import umap
except Exception:
    umap = None

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.")

###############################################################################
# CONFIG (par défaut; override via CLI)
###############################################################################
SEED = 42
BIN_FACTOR = 20           # regroupe 20 échantillons (1 kHz -> 50 Hz)
BIN_SIZE = 0.001          # 1 ms à l'origine (sera multiplié par BIN_FACTOR)
SMOOTHING_LENGTH = 0.05   # 50 ms
SAMPLING_RATE = 1000      # Hz
GAUSS_TRUNCATE = 4.0      # portée effective ~ truncate*sigma pour l'embargo

# Device + modes perf/repro
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERF_MODE = True  # True = vitesse (recommandé HPC/MIG); False = repro stricte

###############################################################################
# SEED & RUNTIME SETTINGS (optimisés GPU/MIG)
###############################################################################
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if PERF_MODE:
        # Vitesse (non déterministe possible)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
    else:
        # Repro stricte (plus lent)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def auto_num_workers(default=8):
    # Essaie d'utiliser la quote CPU Slurm si dispo
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

def _dl_generator():
    g = torch.Generator()
    g.manual_seed(SEED)
    return g

###############################################################################
# DATA HELPERS
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
    # vectorisé (plus de boucle python canal par canal)
    sigma = (smoothing_length / bin_size) / 2
    return gaussian_filter1d(x_2d.astype(np.float32), sigma=sigma, axis=0)

def build_continuous_dataset_raw(df, bin_factor, all_units=None):
    """Concatène les essais sans lissage/filtrage. Retourne X_raw, Y_raw, cuts."""
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
    """Lissage/filtrage PAR SEGMENT (pas global) -> pas de fuite train↔val."""
    eff_fs = SAMPLING_RATE // bin_factor
    Xs = smooth_spike_data(Xseg, bin_size * bin_factor, smoothing_length)
    Ys = butter_lowpass(np.abs(Yseg), eff_fs)
    return Xs, Ys

def preprocess_within_cuts(X_raw, Y_raw, cuts, bin_factor):
    """Applique le prétraitement indépendamment dans chaque bloc délimité par `cuts`."""
    if not cuts:
        return preprocess_segment(X_raw, Y_raw, bin_factor)
    pieces_X, pieces_Y = [], []
    start = 0
    for c in cuts + [len(X_raw)]:
        Xs, Ys = preprocess_segment(X_raw[start:c], Y_raw[start:c], bin_factor)
        pieces_X.append(Xs); pieces_Y.append(Ys)
        start = c
    return np.concatenate(pieces_X, axis=0), np.concatenate(pieces_Y, axis=0)

def sigma_bins(bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    return (smoothing_length / (bin_size * bin_factor)) / 2.0

def embargo_bins(K_LAG, bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH, truncate=GAUSS_TRUNCATE):
    emb = int(np.ceil(truncate * sigma_bins(bin_factor, bin_size, smoothing_length)))
    return max(K_LAG, emb)

def time_kfold_splits(n_time, n_splits):
    """Folds contigus sur l'axe temps (sans shuffle)."""
    block = n_time // n_splits
    splits = []
    for k in range(n_splits):
        v0 = k * block
        v1 = (k + 1) * block if k < n_splits - 1 else n_time
        splits.append((v0, v1))
    return splits

def adjust_cuts_for_segment(start, end, cuts_global, trim_left=0, trim_right=0, seg_len=None):
    """Décale les cuts dans [start,end) -> indices locaux après trims."""
    local = [c - start for c in cuts_global if start < c < end]
    if seg_len is None:
        seg_len = end - start
    new_start = trim_left
    new_end = seg_len - trim_right
    return [c - new_start for c in local if new_start < c < new_end]

def valid_window_indices(n_time, k, cuts, stride=1, start=0, end=None):
    """Fenêtres [t-k, t) pour t in [start+k, end), respectant cuts et stride."""
    end = n_time if end is None else end
    out = []
    for t in range(start + k, end, stride):
        if any(t - k < c < t for c in cuts):  # interdit de traverser une frontière d'essai
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
# MODELS
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
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)

###############################################################################
# DIM REDUCTION (PCA / UMAP)
###############################################################################
def get_dimred_model(data, method, n_components, seed):
    if method.upper() == "PCA":
        model = PCA(n_components=n_components, random_state=seed)
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
# METRICS + TRAIN/EVAL (optimisés GPU/AMP)
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
        generator=None if PERF_MODE else _dl_generator(),
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
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        if ep % 10 == 0:
            print(f"Epoch {ep}/{num_epochs} - loss={total/len(loader):.4f}")
    return model

def evaluate_model(model, X, Y, batch_size=256, use_amp=True):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.as_tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and torch.cuda.is_available())):
                out = model(bx)
            preds.append(out.cpu().numpy())
    if preds:
        preds = np.concatenate(preds, axis=0)
        vafs = compute_multichannel_vaf(Y, preds)
        return float(np.nanmean(vafs)), vafs
    else:
        return np.nan, np.full((Y.shape[1],), np.nan)

###############################################################################
# ALIGNMENT (Moore–Penrose)
###############################################################################
def align_linear_pinv(Zx: np.ndarray, Z0: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    """UMAP: résout Zx A ≈ Z0 (centré) -> A = (X^T X + lam I)^(-1) X^T Y; retourne Zx @ A (décalé)."""
    if Zx.shape != Z0.shape:
        raise ValueError(f"pinv align requires same shape, got {Zx.shape} vs {Z0.shape}")
    X = Zx - Zx.mean(axis=0, keepdims=True)
    Y = Z0 - Z0.mean(axis=0, keepdims=True)
    d = X.shape[1]
    A = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)  # Moore–Penrose régularisé
    return (Zx - Zx.mean(axis=0, keepdims=True)) @ A + Z0.mean(axis=0, keepdims=True)

###############################################################################
# HYPERPARAMS
###############################################################################
ARCH_HYPERPARAMS = {
    "GRU":    dict(N_PCA=32, K_LAG=25, HIDDEN=96,   NUM_EPOCHS=200, LR=0.003),
    "LSTM":   dict(N_PCA=24, K_LAG=25, HIDDEN=128,  NUM_EPOCHS=300, LR=0.003),
    "Linear": dict(N_PCA=32, K_LAG=16, HIDDEN=64,   NUM_EPOCHS=100, LR=0.003),
    "LiGRU":  dict(N_PCA=32, K_LAG=16, HIDDEN=5,    NUM_EPOCHS=200, LR=0.001),
}

###############################################################################
# MAIN (CV temporel corrigé + compute MIG)
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', type=str, required=True, choices=list(ARCH_HYPERPARAMS.keys()))
    parser.add_argument('--dimred', type=str, default="PCA", choices=["PCA", "UMAP"])
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default=".")
    parser.add_argument('--combined_pickle', type=str, default="combined.pkl")

    # runtime perf (hérité MIG)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=-1, help='-1 = auto à partir de $SLURM_CPUS_PER_TASK')
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--perf_mode', action='store_true')  # force PERF (par défaut déjà True)
    parser.add_argument('--stride_mul', type=float, default=1.0, help='stride = max(1, int(stride_mul*K_LAG))')

    args = parser.parse_args()

    # Mode perf/repro
    global PERF_MODE
    if args.perf_mode:
        PERF_MODE = True
    set_seed(args.seed)

    # Hyperparams décodeur
    hp = ARCH_HYPERPARAMS[args.decoder]
    N_PCA, K_LAG, HIDDEN, NUM_EPOCHS, LR = (
        hp["N_PCA"], hp["K_LAG"], hp["HIDDEN"], hp["NUM_EPOCHS"], hp["LR"]
    )
    BATCH = args.batch_size
    WORKERS = auto_num_workers() if args.num_workers == -1 else args.num_workers
    USE_AMP = (not args.no_amp)
    STRIDE = max(1, int(args.stride_mul * K_LAG))  # stride large (évite duplications de fenêtres quasi identiques)

    # Data
    combined_df = pd.read_pickle(args.combined_pickle)
    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"])

    ALL_UNITS = get_all_unit_names(combined_df)
    unique_days = sorted(combined_df["date"].unique())
    if len(unique_days) == 0:
        print("[ERROR] No days found in combined_df!")
        return

    # Détection #EMG
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

    # Jour 0
    day0 = unique_days[0]
    train_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)

    # Raw day0 (sans lissage global)
    X0_raw, Y0_raw, cuts0 = build_continuous_dataset_raw(train_df, BIN_FACTOR, all_units=ALL_UNITS)
    if X0_raw.size == 0:
        print("[ERROR] empty day0")
        return

    # Splits temporels contigus
    splits = time_kfold_splits(X0_raw.shape[0], args.n_folds)

    results = []
    for fold_idx, (val_start, val_end) in enumerate(splits):
        print(f"\n[Fold {fold_idx+1}/{args.n_folds}] val=[{val_start}:{val_end})")

        # Segments bruts
        X_left_raw  = X0_raw[:val_start];   Y_left_raw  = Y0_raw[:val_start]
        X_val_raw   = X0_raw[val_start:val_end]; Y_val_raw = Y0_raw[val_start:val_end]
        X_right_raw = X0_raw[val_end:];     Y_right_raw = Y0_raw[val_end:]

        # Embargo (>= K_LAG et couvrant la portée du lissage)
        EMB = embargo_bins(K_LAG, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, GAUSS_TRUNCATE)

        # Prétraitement + ajustement des cuts dans chaque segment avec trims
        # LEFT
        trimL = 0; trimR = EMB
        X_left_p, Y_left_p = preprocess_segment(X_left_raw, Y_left_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH) if len(X_left_raw) else (np.empty((0,)), np.empty((0,)))
        cuts_left = adjust_cuts_for_segment(0, len(X_left_raw), cuts0, trim_left=trimL, trim_right=trimR, seg_len=len(X_left_raw))
        if len(X_left_p) > trimR:
            X_left_p = X_left_p[:len(X_left_p)-trimR]; Y_left_p = Y_left_p[:len(Y_left_p)-trimR]
        else:
            X_left_p = np.empty((0, X0_raw.shape[1] if X0_raw.size else 0), dtype=np.float32)
            Y_left_p = np.empty((0, Y0_raw.shape[1] if Y0_raw.size else n_emg_channels), dtype=np.float32)
            cuts_left = []

        # VAL
        trimL = EMB; trimR = EMB
        X_val_p, Y_val_p = preprocess_segment(X_val_raw, Y_val_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH) if len(X_val_raw) else (np.empty((0,)), np.empty((0,)))
        cuts_val = adjust_cuts_for_segment(val_start, val_end, cuts0, trim_left=trimL, trim_right=trimR, seg_len=len(X_val_raw))
        if len(X_val_p) > (trimL + trimR):
            X_val_p = X_val_p[trimL:len(X_val_p)-trimR]; Y_val_p = Y_val_p[trimL:len(Y_val_p)-trimR]
        else:
            X_val_p = np.empty((0, X0_raw.shape[1] if X0_raw.size else 0), dtype=np.float32)
            Y_val_p = np.empty((0, Y0_raw.shape[1] if Y0_raw.size else n_emg_channels), dtype=np.float32)
            cuts_val = []

        # RIGHT
        trimL = EMB; trimR = 0
        X_right_p, Y_right_p = preprocess_segment(X_right_raw, Y_right_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH) if len(X_right_raw) else (np.empty((0,)), np.empty((0,)))
        cuts_right = adjust_cuts_for_segment(val_end, len(X0_raw), cuts0, trim_left=trimL, trim_right=trimR, seg_len=len(X_right_raw))
        if len(X_right_p) > trimL:
            X_right_p = X_right_p[trimL:]; Y_right_p = Y_right_p[trimL:]
        else:
            X_right_p = np.empty((0, X0_raw.shape[1] if X0_raw.size else 0), dtype=np.float32)
            Y_right_p = np.empty((0, Y0_raw.shape[1] if Y0_raw.size else n_emg_channels), dtype=np.float32)
            cuts_right = []

        # Concat train (left+right)
        if X_left_p.size and X_right_p.size:
            X_train_time = np.vstack([X_left_p, X_right_p]); Y_train_time = np.vstack([Y_left_p, Y_right_p])
            cuts_train = cuts_left + [c + len(X_left_p) for c in cuts_right] + [len(X_left_p)]  # fusion: on garde la frontière left/right
        elif X_left_p.size:
            X_train_time, Y_train_time, cuts_train = X_left_p, Y_left_p, cuts_left
        else:
            X_train_time, Y_train_time, cuts_train = X_right_p, Y_right_p, cuts_right

        if X_train_time.shape[0] <= K_LAG or X_val_p.shape[0] <= K_LAG:
            print("  [WARN] not enough samples after embargo; skipping fold")
            continue

        # Fit manifold sur TRAIN uniquement
        dimred_model_day0 = get_dimred_model(X_train_time, args.dimred, max(N_PCA, 2), args.seed + fold_idx)
        Z_tr = transform_dimred(dimred_model_day0, X_train_time, args.dimred)[:, :N_PCA]
        Z_va = transform_dimred(dimred_model_day0, X_val_p,   args.dimred)[:, :N_PCA]

        # Fenêtrage avec cuts et stride large
        is_linear = (args.decoder == "Linear")
        X_tr, Y_tr = build_seq_with_cuts(Z_tr, Y_train_time, K_LAG, cuts_train, STRIDE, is_linear)
        X_te, Y_te = build_seq_with_cuts(Z_va, Y_val_p,       K_LAG, cuts_val,   STRIDE, is_linear)
        if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
            print("  [WARN] empty after windowing; skipping fold")
            continue

        # Modèle
        if args.decoder == "GRU":
            model = GRUDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
        elif args.decoder == "LSTM":
            model = LSTMDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
        elif args.decoder == "Linear":
            model = LinearLagDecoder(K_LAG * N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
        else:  # LiGRU
            model = LiGRUDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)

        # Train/Eval day0 (AMP, DataLoader optimisés)
        model = train_model(model, X_tr, Y_tr, num_epochs=NUM_EPOCHS, lr=LR,
                            batch_size=BATCH, num_workers=WORKERS, use_amp=USE_AMP)
        vaf_te, vaf_ch_te = evaluate_model(model, X_te, Y_te, batch_size=BATCH, use_amp=USE_AMP)
        for ch_idx, vaf_single in enumerate(vaf_ch_te):
            results.append({
                "day": day0,
                "day_int": 0,
                "align": "crossval",
                "decoder": args.decoder,
                "dim_red": args.dimred,
                "fold": fold_idx,
                "emg_channel": ch_idx,
                "vaf": vaf_single
            })

        # ---------- Cross-days (direct + aligned) avec ce modèle/fold ----------
        for d_val in unique_days:
            if pd.to_datetime(d_val) == pd.to_datetime(day0): 
                continue
            day_df = combined_df[combined_df["date"] == d_val].reset_index(drop=True)
            X_raw, Y_raw, cuts_test = build_continuous_dataset_raw(day_df, BIN_FACTOR, all_units=ALL_UNITS)
            if X_raw.shape[0] == 0:
                continue

            # Prétraitement sans fuite entre essais: par bloc "cuts"
            X_proc, Y_proc = preprocess_within_cuts(X_raw, Y_raw, cuts_test, BIN_FACTOR)

            # Projection "direct" via modèle day0 (fit sur TRAIN du fold)
            zx_direct = transform_dimred(dimred_model_day0, X_proc, args.dimred)[:, :N_PCA]

            # Modèle propre au jour X
            dimred_model_dayX = get_dimred_model(X_proc, args.dimred, N_PCA, args.seed + fold_idx)
            zx_dayX = transform_dimred(dimred_model_dayX, X_proc, args.dimred)[:, :N_PCA]

            for align_mode in ["direct", "aligned"]:
                if align_mode == "direct":
                    zx_test = zx_direct
                else:
                    if args.dimred.upper() == "PCA":
                        # MP sur les bases PCA
                        V0 = dimred_model_day0.components_[:N_PCA, :].T
                        Vx = dimred_model_dayX.components_[:N_PCA, :].T
                        try:
                            R = pinv(Vx) @ V0
                            zx_test = zx_dayX @ R
                        except Exception:
                            zx_test = zx_dayX
                    else:
                        # UMAP: Moore–Penrose linéaire régularisé
                        try:
                            zx_test = align_linear_pinv(zx_dayX, zx_direct, lam=1e-6)
                        except Exception:
                            zx_test = zx_dayX

                # Fenêtrage test (respect des cuts; pas d'embargo nécessaire)
                X_seq, Y_seq = build_seq_with_cuts(zx_test, Y_proc, K_LAG, cuts_test, STRIDE, is_linear)
                if X_seq.shape[0] == 0:
                    continue

                vaf, vaf_ch = evaluate_model(model, X_seq, Y_seq, batch_size=BATCH, use_amp=USE_AMP)
                for ch_idx, vaf_single in enumerate(vaf_ch):
                    results.append({
                        "day": d_val,
                        "day_int": (pd.to_datetime(d_val) - pd.to_datetime(day0)).days,
                        "align": align_mode,
                        "decoder": args.decoder,
                        "dim_red": args.dimred,
                        "fold": fold_idx,
                        "emg_channel": ch_idx,
                        "vaf": vaf_single,
                        "mean_vaf": vaf
                    })

        print(f"[fold={fold_idx+1}] done.")

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"crossday_results_{args.decoder}_{args.dimred}.pkl")
    pd.to_pickle(pd.DataFrame(results), save_path)
    print(f"\n[INFO] Saved all results to {save_path}")

if __name__ == "__main__":
    main()
