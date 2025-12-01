#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stride-sweep CV (no-leak) pour un seul décodeur.
- Hyperparams FIXES par décodeur; SEUL 'stride' varie (peut être négatif).
- Même logique: PCA fit/train-only, embargo anti-fuite, GPU AMP/TF32.

Sorties:
  - stride_sweep_results_{decoder}_{jobid}.pkl  (résumé par stride)
  - stride_sweep_rows_{decoder}_{jobid}.pkl     (par fold×canal×seed×stride)
  - stride_sweep_summary_{decoder}_{jobid}.csv  (une ligne par stride)
"""

import os, gc, time, argparse, warnings, pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.")

# ============================ CONSTANTES ============================
SEED = 42
BIN_FACTOR = 20           # 1 kHz -> /20 -> 50 Hz
BIN_SIZE = 0.001          # s, bin d'origine (1 ms). Bin effectif = BIN_FACTOR * BIN_SIZE
SMOOTHING_LENGTH = 0.05   # s, longueur fenêtre gaussienne
SAMPLING_RATE = 1000      # Hz (origine)
GAUSS_TRUNCATE = 4.0      # embargo ~ truncate * sigma

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERF_MODE = True          # True = perf (AMP+TF32), False = repro stricte

# Strides par défaut (50 Hz effectif ⇒ 20 ms/step), couvrent avant & arrière
DEFAULT_STRIDES = [-200, -100, -50, -25, -20, -10, -5, -2, -1, 0 , 1, 2, 5, 10, 20, 25, 50, 100, 200]
# DEFAULT_STRIDES = [0 , 1, 2, 3, 4, 5, 6, 7, 8, 9,10]

# Hyperparams FIXES par décodeur (overridable via CLI)
FIXED: Dict[str, Dict[str, Any]] = {
    "gru":    dict(n_pca=32, k_lag=25, hidden_dim=96, num_epochs=200, lr=0.003),
    "lstm":   dict(n_pca=64, k_lag=25, hidden_dim=256, num_epochs=200, lr=3e-3),
    "ligru":  dict(n_pca=64, k_lag=25, hidden_dim=256, num_epochs=200, lr=3e-3),
    "linear": dict(n_pca=64, k_lag=25, hidden_dim=512, num_epochs=100, lr=1e-3),
}
DECODER_DISPLAY = {"gru":"GRU", "lstm":"LSTM", "ligru":"LiGRU", "linear":"Linear"}

# ============================ PERF / SEED ============================
def set_seed(seed=SEED):
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
        if n > 0: return max(2, n - 1)
    except Exception:
        pass
    return default

def _seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)

def _dl_generator():
    g = torch.Generator()
    g.manual_seed(SEED)
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
            return list(map(str, emg.columns))
    return [f"ch{c}" for c in range(fallback_n)]

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

def build_continuous_dataset_raw(df, bin_factor, all_units=None):
    """Concat trials (downsampled) WITHOUT smoothing/filtering; return X_raw, Y_raw, cuts."""
    spikes_all, emg_all, lengths = [], [], []
    for _, row in df.iterrows():
        spike_df = row["spike_counts"]
        emg_val  = row["EMG"]
        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty: continue
        if emg_val is None: continue
        if all_units is not None:
            spike_df = spike_df.reindex(columns=all_units, fill_value=0)
        ds_spike_df, ds_emg = downsample_spike_and_emg(spike_df, emg_val, bin_factor)
        if ds_spike_df.shape[0] == 0: continue

        Xr = ds_spike_df.values.astype(np.float32)
        if isinstance(ds_emg, pd.DataFrame):
            Yr = ds_emg.values.astype(np.float32)
        else:
            Yr = np.asarray(ds_emg, dtype=np.float32)
        spikes_all.append(Xr); emg_all.append(Yr); lengths.append(len(Xr))

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

def embargo_bins(k_lag, bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH, truncate=GAUSS_TRUNCATE):
    emb = int(np.ceil(truncate * sigma_bins(bin_factor, bin_size, smoothing_length)))
    return max(k_lag, emb)

def time_kfold_splits(n_time, n_splits) -> List[Tuple[int,int]]:
    block = n_time // n_splits
    splits = []
    for k in range(n_splits):
        v0 = k * block
        v1 = (k + 1) * block if k < n_splits - 1 else n_time
        splits.append((v0, v1))
    return splits

def adjust_cuts_for_segment(start, end, cuts_global, trim_left=0, trim_right=0, seg_len=None):
    local = [c - start for c in cuts_global if start < c < end]
    if seg_len is None: seg_len = end - start
    new_start = trim_left
    new_end   = seg_len - trim_right
    return [c - new_start for c in local if new_start < c < new_end]

# =================== INDEXATION PHASE-AWARE + STRIDE NEGATIF ===================
def valid_window_indices(n_time, k, cuts, start=0, end=None, stride=1, phase=0):
    """
    Renvoie des indices t (fin de fenêtre [t-k, t)) valides:
    - stride > 0 : ancré au début (start), t croît.
    - stride < 0 : ancré à la fin  (end),   t décroit lors du calcul (puis trié).
    - phase toujours modulo |stride|.
    - on évite les fenêtres qui traversent un 'cut'.
    """
    end = n_time if end is None else end
    s = abs(int(stride)) if int(stride) != 0 else 1
    phase = int(phase) % s
    t_min = start + k
    t_max = end  # exclusif pour range upper bound

    idx = []
    if stride > 0:
        first = t_min + phase
        for t in range(first, t_max, s):
            if any(t - k < c < t for c in cuts):
                continue
            idx.append(t)
    else:
        # on part de la fin: dernier index possible = t_max - 1
        last_possible = t_max - 1
        if last_possible < t_min:
            return []
        # choisir t0 <= last_possible tel que (t0 - t_min - phase) % s == 0
        offset = (last_possible - t_min - phase) % s
        t0 = last_possible - offset
        for t in range(t0, t_min - 1, -s):
            if any(t - k < c < t for c in cuts):
                continue
            idx.append(t)
        idx.sort()  # on renvoie toujours croissant pour cohérence
    return idx

def build_seq_with_cuts(Z, Y, K_LAG, cuts, stride, phase, is_linear):
    idx = valid_window_indices(Z.shape[0], K_LAG, cuts, stride=stride, phase=phase)
    if not idx:
        if is_linear:
            return (np.empty((0, K_LAG * Z.shape[1]), dtype=np.float32),
                    np.empty((0, Y.shape[1]), dtype=np.float32))
        else:
            return (np.empty((0, K_LAG, Z.shape[1]), dtype=np.float32),
                    np.empty((0, Y.shape[1]), dtype=np.float32))
    if is_linear:
        X = np.stack([Z[t-K_LAG:t, :].reshape(-1) for t in idx], axis=0).astype(np.float32)
    else:
        X = np.stack([Z[t-K_LAG:t, :]          for t in idx], axis=0).astype(np.float32)
    Yb = np.stack([Y[t, :]                     for t in idx], axis=0).astype(np.float32)
    return X, Yb

# ============================ MODÈLES ============================
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x); out = out[:, -1, :]
        return self.fc(out)

class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x); out = out[:, -1, :]
        return self.fc(out)

class LinearLagDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act  = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, output_size)
    def forward(self, x):
        x = self.lin1(x); x = self.act(x)
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

# ============================ DIM-RED ============================
def fit_pca(X, n_components, seed):
    model = PCA(n_components=max(n_components, 2), random_state=seed)
    model.fit(X)
    return model

def pca_transform(model, X):
    return model.transform(X)

# ============================ METRICS + TRAIN/EVAL ============================
def eval_vaf_full(model, X_np, Y_np, batch_size, use_amp=True):
    if X_np.shape[0] == 0:
        return float("nan"), np.full((Y_np.shape[1],), np.nan, dtype=np.float32)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.from_numpy(X_np[i:i+batch_size]).to(DEVICE, non_blocking=True).float()
            with torch.cuda.amp.autocast(enabled=(use_amp and DEVICE.type=="cuda")):
                out = model(xb).cpu().numpy()
            preds.append(out)
    Yp = np.concatenate(preds, axis=0)
    vafs = []
    for ch in range(Y_np.shape[1]):
        yt, yp = Y_np[:, ch], Yp[:, ch]
        vt = np.var(yt)
        if vt < 1e-12: vafs.append(np.nan)
        else:          vafs.append(1.0 - np.var(yt - yp)/vt)
    vafs = np.asarray(vafs, dtype=np.float32)
    return float(np.nanmean(vafs)), vafs

def train_model(model, X_train, Y_train, num_epochs, lr,
                batch_size, num_workers=None, use_amp=True):
    if num_workers is None:
        num_workers = auto_num_workers()

    x_cpu = torch.as_tensor(X_train, dtype=torch.float32)
    y_cpu = torch.as_tensor(Y_train, dtype=torch.float32)
    dset  = TensorDataset(x_cpu, y_cpu)

    loader = DataLoader(
        dset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4,
        generator=None if PERF_MODE else _dl_generator(),
        worker_init_fn=None if PERF_MODE else _seed_worker
    )

    opt  = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch.cuda.is_available()))

    model.train()
    for ep in range(1, num_epochs+1):
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
            total += loss.item()
        if ep % 50 == 0 or ep == 1:
            print(f"  epoch {ep}/{num_epochs}  loss={total/len(loader):.4f}")
    return model

# ============================ MAIN (Stride sweep) ============================
def get_model(decoder: str, n_pca: int, k_lag: int, hidden_dim: int, n_out: int):
    if decoder == "gru":
        return GRUDecoder(n_pca, hidden_dim, n_out).to(DEVICE)
    elif decoder == "lstm":
        return LSTMDecoder(n_pca, hidden_dim, n_out).to(DEVICE)
    elif decoder == "ligru":
        return LiGRUDecoder(n_pca, hidden_dim, n_out).to(DEVICE)
    elif decoder == "linear":
        return LinearLagDecoder(k_lag * n_pca, hidden_dim, n_out).to(DEVICE)
    else:
        raise ValueError(f"Unknown decoder '{decoder}'")

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--decoder", choices=list(FIXED.keys()), required=True,
                    help="Décodeur à tester (hyperparams fixes)")
    ap.add_argument("--strides", nargs="+", type=int, default=DEFAULT_STRIDES,
                    help="Valeurs de stride à balayer (peuvent être négatives)")
    ap.add_argument("--seeds", type=int, default=3, help="nombre de seeds (0..seeds-1)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--combined_pickle", type=str, default="combined.pkl")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=-1, help="-1 => auto depuis $SLURM_CPUS_PER_TASK")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--strict_repro", action="store_true", help="désactive TF32 & benchmark pour repro")
    # Overrides optionnels des hyperparams FIXED
    ap.add_argument("--n_pca", type=int, default=None)
    ap.add_argument("--k_lag", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--num_epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    global PERF_MODE
    PERF_MODE = not args.strict_repro

    # Hyperparams fixes du décodeur (overrides CLI possibles)
    cfg = FIXED[args.decoder].copy()
    for k in ["n_pca","k_lag","hidden_dim","num_epochs","lr"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    # ---------- Données ----------
    combined_df = pd.read_pickle(args.combined_pickle)
    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"], errors="coerce")
    unique_days = sorted(combined_df["date"].dropna().unique())
    if not unique_days:
        raise RuntimeError("No days in combined_df")
    day0 = unique_days[0]
    train_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)

    ALL_UNITS = get_all_unit_names(combined_df)

    # détecter nb canaux EMG
    n_emg_channels = 0
    for _, row in combined_df.iterrows():
        emg_val = row.get("EMG", None)
        if emg_val is not None:
            if isinstance(emg_val, pd.DataFrame) and not emg_val.empty:
                n_emg_channels = emg_val.shape[1]; break
            elif isinstance(emg_val, np.ndarray) and emg_val.size > 0:
                n_emg_channels = emg_val.shape[1]; break
    if n_emg_channels == 0:
        raise RuntimeError("Could not detect EMG channels.")

    EMG_LABELS = get_emg_labels_from_df(combined_df, n_emg_channels)

    # day0 brut (sans lissage)
    X0_raw, Y0_raw, cuts0 = build_continuous_dataset_raw(train_df, BIN_FACTOR, all_units=ALL_UNITS)
    if X0_raw.size == 0:
        raise RuntimeError("Empty day0 after downsampling.")

    # splits K-fold contigus
    splits = time_kfold_splits(X0_raw.shape[0], args.folds)

    # perf
    set_seed(SEED)
    BATCH = args.batch_size
    WORKERS = auto_num_workers() if args.num_workers == -1 else args.num_workers
    USE_AMP = (not args.no_amp)

    # sorties
    job_id = os.environ.get("SLURM_ARRAY_TASK_ID", "local")
    out_summary = Path(f"stride_sweep_results_{args.decoder}_{job_id}.pkl")
    out_rows    = Path(f"stride_sweep_rows_{args.decoder}_{job_id}.pkl")
    out_csv     = Path(f"stride_sweep_summary_{args.decoder}_{job_id}.csv")

    results: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []

    # constantes
    decoder = args.decoder
    n_pca = int(cfg["n_pca"])
    k_lag = int(cfg["k_lag"])
    hidden_dim = int(cfg["hidden_dim"])
    num_epochs = int(cfg["num_epochs"])
    lr = float(cfg["lr"])
    is_linear = (decoder == "linear")
    phase_tr = 0
    phase_va = 0

    EMB = embargo_bins(k_lag, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, GAUSS_TRUNCATE)

    # préproc par fold (dépend de EMB)
    def fold_segments(val_start, val_end):
        X_left_raw  = X0_raw[:val_start];   Y_left_raw  = Y0_raw[:val_start]
        X_val_raw   = X0_raw[val_start:val_end]; Y_val_raw = Y0_raw[val_start:val_end]
        X_right_raw = X0_raw[val_end:];     Y_right_raw = Y0_raw[val_end:]

        # LEFT
        Xl, Yl = (preprocess_segment(X_left_raw, Y_left_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
                  if len(X_left_raw) else (np.empty((0,)), np.empty((0,))))
        if len(Xl) > EMB:
            Xl = Xl[:len(Xl)-EMB]; Yl = Yl[:len(Yl)-EMB]
            cuts_left = adjust_cuts_for_segment(0, len(X_left_raw), cuts0, trim_left=0, trim_right=EMB, seg_len=len(X_left_raw))
        else:
            Xl = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
            Yl = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
            cuts_left = []

        # VAL
        Xv, Yv = (preprocess_segment(X_val_raw, Y_val_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
                  if len(X_val_raw) else (np.empty((0,)), np.empty((0,))))
        if len(Xv) > 2*EMB:
            Xv = Xv[EMB:len(Xv)-EMB]; Yv = Yv[EMB:len(Yv)-EMB]
            cuts_val = adjust_cuts_for_segment(val_start, val_end, cuts0, trim_left=EMB, trim_right=EMB, seg_len=len(X_val_raw))
        else:
            Xv = np.empty((0, X0_raw.shape[1]), dtype=np.float32)
            Yv = np.empty((0, Y0_raw.shape[1]), dtype=np.float32)
            cuts_val = []

        # RIGHT
        Xr, Yr = (preprocess_segment(X_right_raw, Y_right_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
                  if len(X_right_raw) else (np.empty((0,)), np.empty((0,))))
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

        return X_train_time, Y_train_time, cuts_train, Xv, Yv, cuts_val

    # balayage des strides
    for stride in [int(s) if int(s) != 0 else 1 for s in args.strides]:
        print(f"\n=== Decoder={decoder}  stride={stride}  (fixes: n_pca={n_pca}, k_lag={k_lag}, "
              f"hidden={hidden_dim}, epochs={num_epochs}, lr={lr}) ===")
        set_seed(SEED)

        vafs_fold_all: List[float] = []
        fold_times_all: List[float] = []
        param_count: int = None

        for seed in range(args.seeds):
            set_seed(SEED + seed)
            for i_fold, (val_start, val_end) in enumerate(splits):
                Xtr_t, Ytr_t, cuts_tr, Xva_t, Yva_t, cuts_va = fold_segments(val_start, val_end)

                if Xtr_t.shape[0] <= k_lag or Xva_t.shape[0] <= k_lag:
                    continue

                # PCA sur TRAIN uniquement
                pca_model = fit_pca(Xtr_t, n_components=n_pca, seed=SEED + seed + i_fold)
                Z_tr = pca_transform(pca_model, Xtr_t)[:, :n_pca]
                Z_va = pca_transform(pca_model, Xva_t)[:, :n_pca]

                # séquences (phase = 0)
                if decoder == "linear":
                    X_tr, Y_tr = build_seq_with_cuts(Z_tr, Ytr_t, k_lag, cuts_tr, stride, 0, True)
                    X_va, Y_va = build_seq_with_cuts(Z_va, Yva_t, k_lag, cuts_va, stride, 0, True)
                else:
                    X_tr, Y_tr = build_seq_with_cuts(Z_tr, Ytr_t, k_lag, cuts_tr, stride, 0, False)
                    X_va, Y_va = build_seq_with_cuts(Z_va, Yva_t, k_lag, cuts_va, stride, 0, False)

                if X_tr.shape[0] == 0 or X_va.shape[0] == 0:
                    continue

                # modèle
                model = get_model(decoder, n_pca, k_lag, hidden_dim, n_out=n_emg_channels)
                if param_count is None:
                    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

                # train + eval
                t0 = time.perf_counter()
                train_model(model, X_tr, Y_tr, num_epochs=num_epochs, lr=lr,
                            batch_size=BATCH, num_workers=WORKERS, use_amp=USE_AMP)
                fold_time = time.perf_counter() - t0
                fold_times_all.append(float(fold_time))

                mean_vaf, vaf_ch = eval_vaf_full(model, X_va, Y_va, batch_size=BATCH, use_amp=USE_AMP)
                vafs_fold_all.append(float(mean_vaf if not np.isnan(mean_vaf) else -1.0))

                # long-format par canal
                for ch_idx, v in enumerate(vaf_ch):
                    long_rows.append(dict(
                        decoder=DECODER_DISPLAY.get(decoder, decoder),
                        stride=int(stride),
                        seed=int(seed),
                        fold=int(i_fold),
                        emg_channel=int(ch_idx),
                        emg_label=str(EMG_LABELS[ch_idx]) if ch_idx < len(EMG_LABELS) else f"ch{ch_idx}",
                        vaf=float(v),
                        n_pca=int(n_pca),
                        k_lag=int(k_lag),
                        hidden_dim=int(hidden_dim),
                        num_epochs=int(num_epochs),
                        lr=float(lr),
                    ))

                # cleanup
                del model; torch.cuda.empty_cache(); gc.collect()

        if not vafs_fold_all:
            warnings.warn(f"No valid folds for stride={stride}. Skipping summary.")
            continue

        res = dict(
            decoder=decoder,
            stride=int(stride),
            num_params=int(param_count) if param_count is not None else None,
            mean_vaf=float(np.mean(vafs_fold_all)),
            std_vaf=float(np.std(vafs_fold_all)),
            n_scores=len(vafs_fold_all),
            mean_time=float(np.mean(fold_times_all)) if fold_times_all else np.nan,
            n_pca=int(n_pca),
            k_lag=int(k_lag),
            hidden_dim=int(hidden_dim),
            num_epochs=int(num_epochs),
            lr=float(lr),
        )
        results.append(res)
        print(f" -> mean VAF={res['mean_vaf']:.4f} (±{res['std_vaf']:.4f}, n={res['n_scores']})  "
              f"mean fold time={res['mean_time']:.2f}s")

        # checkpoints fréquents
        pickle.dump(results, open(out_summary, "wb"))
        pickle.dump(long_rows, open(out_rows, "wb"))

    # CSV final
    if results:
        df = pd.DataFrame(results).sort_values(["stride"])
        df.to_csv(out_csv, index=False)
        print("\nSaved:")
        print("  ", out_summary.resolve())
        print("  ", out_rows.resolve())
        print("  ", out_csv.resolve())
        print("\nSummary:")
        print(df.to_string(index=False))
    else:
        print("\nNo results produced (check data / settings).")

if __name__ == "__main__":
    main()
