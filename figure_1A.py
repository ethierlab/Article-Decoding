#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from numpy.linalg import pinv

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.")

###############################################################################
# GLOBAL CONFIG (must match your training script)
###############################################################################
SEED = 42
BIN_FACTOR = 20           # 1 kHz -> 50 Hz
BIN_SIZE = 0.001          # 1 ms
SMOOTHING_LENGTH = 0.05   # 50 ms
SAMPLING_RATE = 1000      # Hz

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# SEED
###############################################################################
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

###############################################################################
# DATA HELPERS (copied/simplified from your final script)
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

def preprocess_segment(Xseg, Yseg, bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    eff_fs = SAMPLING_RATE // bin_factor
    Xs = smooth_spike_data(Xseg, bin_size * bin_factor, smoothing_length)
    Ys = butter_lowpass(np.abs(Yseg), eff_fs)
    return Xs, Ys

def preprocess_within_cuts(X_raw, Y_raw, cuts, bin_factor):
    if not cuts:
        return preprocess_segment(X_raw, Y_raw, bin_factor)
    pieces_X, pieces_Y = [], []
    start = 0
    for c in cuts + [len(X_raw)]:
        Xs, Ys = preprocess_segment(X_raw[start:c], Y_raw[start:c], bin_factor)
        pieces_X.append(Xs); pieces_Y.append(Ys)
        start = c
    return np.concatenate(pieces_X, axis=0), np.concatenate(pieces_Y, axis=0)

def valid_window_indices(n_time, k, cuts, stride=1, start=0, end=None):
    end = n_time if end is None else end
    out = []
    for t in range(start + k, end, stride):
        if any(t - k < c < t for c in cuts):
            continue
        out.append(t)
    return out

def build_seq_with_cuts_and_indices(Z, Y, K_LAG, cuts, stride, start, end, is_linear):
    idx = valid_window_indices(Z.shape[0], K_LAG, cuts, stride=stride, start=start, end=end)
    if not idx:
        if is_linear:
            X = np.empty((0, K_LAG * Z.shape[1]), dtype=np.float32)
        else:
            X = np.empty((0, K_LAG, Z.shape[1]), dtype=np.float32)
        Yb = np.empty((0, Y.shape[1]), dtype=np.float32)
        return X, Yb, np.array(idx, dtype=int)

    if is_linear:
        X = np.stack([Z[t-K_LAG:t, :].reshape(-1) for t in idx], axis=0).astype(np.float32)
    else:
        X = np.stack([Z[t-K_LAG:t, :] for t in idx], axis=0).astype(np.float32)
    Yb = np.stack([Y[t, :] for t in idx], axis=0).astype(np.float32)
    return X, Yb, np.array(idx, dtype=int)

###############################################################################
# CONTINUOUS DATA + TRIAL CENTERS
###############################################################################
def build_continuous_dataset_raw_with_trials(df, bin_factor, all_units=None):
    """
    Like build_continuous_dataset_raw, but also returns:
      - cuts
      - trial_centers: list of global indices of trial-starts (downsampled)
      - trial_info: list of (row_idx, local_trial_idx, trial_start_time)
    """
    spikes_all, emg_all, lengths = [], [], []
    trial_centers = []
    trial_info = []

    global_offset = 0

    for row_idx, row in df.iterrows():
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

        # continuous arrays
        Xr = ds_spike_df.values.astype(np.float32)
        if isinstance(ds_emg, pd.DataFrame):
            Yr = ds_emg.values.astype(np.float32)
            emg_names = list(ds_emg.columns)
        else:
            Yr = np.asarray(ds_emg, dtype=np.float32)
            emg_names = None

        T_row = Xr.shape[0]

        # downsampled time frame (for locating trial starts)
        time_frame = np.asarray(row["time_frame"]).flatten()
        T_old = len(time_frame)
        T_new = T_old // bin_factor
        if T_new <= 0:
            continue
        tf_ds = time_frame[:T_new * bin_factor].reshape(T_new, bin_factor).mean(axis=1)

        # trial starts
        trial_starts = np.asarray(row["trial_start_time"]).flatten()
        for loc_idx, ts in enumerate(trial_starts):
            if len(tf_ds) == 0:
                continue
            local_center = int(np.argmin(np.abs(tf_ds - ts)))
            if local_center < T_row:  # safety
                global_center = global_offset + local_center
                trial_centers.append(global_center)
                trial_info.append((row_idx, loc_idx, float(ts)))

        spikes_all.append(Xr)
        emg_all.append(Yr)
        lengths.append(T_row)
        global_offset += T_row

    if len(spikes_all) == 0:
        return (np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                [], [], [], [])

    cuts = np.cumsum(lengths)[:-1].tolist()
    X_raw = np.concatenate(spikes_all, axis=0)
    Y_raw = np.concatenate(emg_all, axis=0)
    return X_raw, Y_raw, cuts, trial_centers, trial_info, emg_names

###############################################################################
# MODELS (same as training script)
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
# DIM REDUCTION (PCA only here; extend to UMAP if needed)
###############################################################################
def get_dimred_model(data, method, n_components, seed):
    if method.upper() == "PCA":
        model = PCA(n_components=n_components, random_state=seed)
        model.fit(data)
        return model
    else:
        raise ValueError("Only PCA supported in this plotting script for now.")

def transform_dimred(model, data, method):
    if method.upper() == "PCA":
        return model.transform(data)
    else:
        raise ValueError("Only PCA supported in this plotting script for now.")

###############################################################################
# METRICS + TRAIN/EVAL
###############################################################################
def compute_vaf_1d(y_true, y_pred):
    var_resid = np.var(y_true - y_pred)
    var_true  = np.var(y_true)
    if var_true < 1e-12:
        return np.nan
    return 1.0 - (var_resid / var_true)

def train_model(model, X_train, Y_train, num_epochs=200, lr=0.001,
                batch_size=256):
    x_cpu = torch.as_tensor(X_train, dtype=torch.float32)
    y_cpu = torch.as_tensor(Y_train, dtype=torch.float32)
    dset  = TensorDataset(x_cpu, y_cpu)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(1, num_epochs+1):
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if ep % 10 == 0:
            print(f"Epoch {ep}/{num_epochs} - loss={total/len(loader):.4f}")
    return model

def evaluate_window(model, X_win, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_win), batch_size):
            bx = torch.as_tensor(X_win[i:i+batch_size], dtype=torch.float32).to(DEVICE)
            out = model(bx)
            preds.append(out.cpu().numpy())
    if preds:
        return np.concatenate(preds, axis=0)
    else:
        return np.empty((0,))

###############################################################################
# HYPERPARAMS (copy from your final script)
###############################################################################
ARCH_HYPERPARAMS = {
    "GRU":    dict(N_PCA=32, K_LAG=25, HIDDEN=96,   NUM_EPOCHS=200, LR=0.003),
    "LSTM":   dict(N_PCA=24, K_LAG=25, HIDDEN=128,  NUM_EPOCHS=300, LR=0.003),
    "Linear": dict(N_PCA=32, K_LAG=16, HIDDEN=64,   NUM_EPOCHS=100, LR=0.003),
    "LiGRU":  dict(N_PCA=32, K_LAG=16, HIDDEN=5,    NUM_EPOCHS=200, LR=0.001),
}

###############################################################################
# CHECKPOINT HANDLING
###############################################################################
def save_checkpoint(path, decoder_name, dimred_method, dimred_model,
                    model, hp, emg_names, all_units):
    ckpt = {
        "decoder": decoder_name,
        "dimred": dimred_method,
        "N_PCA": hp["N_PCA"],
        "K_LAG": hp["K_LAG"],
        "HIDDEN": hp["HIDDEN"],
        "NUM_EPOCHS": hp["NUM_EPOCHS"],
        "LR": hp["LR"],
        "state_dict": model.state_dict(),
        "dimred_model": dimred_model,
        "emg_names": emg_names,
        "all_units": all_units,
        "bin_factor": BIN_FACTOR,
        "bin_size": BIN_SIZE,
        "smoothing_length": SMOOTHING_LENGTH,
        "sampling_rate": SAMPLING_RATE,
    }
    torch.save(ckpt, path)
    print(f"[INFO] Saved checkpoint to {path}")

def load_checkpoint(path):
    ckpt = torch.load(path, map_location=DEVICE)
    return ckpt

###############################################################################
# PLOTTING
###############################################################################
def plot_single_muscle(time_rel, y_true, y_pred, muscle_name, decoder_name,
                       vaf, out_path):
    plt.figure(figsize=(6, 3))
    plt.plot(time_rel, y_true, label="Actual", color="black")
    plt.plot(time_rel, y_pred, label=f"{decoder_name} (VAF={vaf:.2f})",
             linestyle="--")
    plt.xlabel("Time (s, aligned to trial start)")
    plt.ylabel(f"EMG {muscle_name}")
    plt.title(f"{decoder_name} on {muscle_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()
    print(f"[INFO] Saved plot to {out_path}")

###############################################################################
# MAIN
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Plot single trial window (continuous day0) for one muscle and one decoder."
    )
    parser.add_argument("--combined_pickle", type=str, default="combined.pkl")
    parser.add_argument("--decoder", type=str, required=True,
                        choices=list(ARCH_HYPERPARAMS.keys()))
    parser.add_argument("--dimred", type=str, default="PCA",
                        choices=["PCA"])
    parser.add_argument("--muscle", type=str, required=True,
                        help="EMG column name, e.g. ECR")
    parser.add_argument("--trial_idx", type=int, required=True,
                        help="Index into the list of trial centers (0-based)")
    parser.add_argument("--checkpoint", type=str, default="decoder_day0.ckpt")
    parser.add_argument("--retrain", action="store_true",
                        help="Force retrain even if checkpoint exists")
    parser.add_argument("--pre_s", type=float, default=1.0,
                        help="Seconds before trial center")
    parser.add_argument("--post_s", type=float, default=4.0,
                        help="Seconds after trial center")
    parser.add_argument("--stride_mul", type=float, default=1.0,
                        help="stride = max(1, int(stride_mul*K_LAG))")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None,
                        help="Output SVG filename")

    args = parser.parse_args()
    set_seed(args.seed)

    # Hyperparams for chosen decoder
    hp = ARCH_HYPERPARAMS[args.decoder]
    N_PCA, K_LAG, HIDDEN, NUM_EPOCHS, LR = (
        hp["N_PCA"], hp["K_LAG"], hp["HIDDEN"], hp["NUM_EPOCHS"], hp["LR"]
    )
    STRIDE = max(1, int(args.stride_mul * K_LAG))

    # Load combined.pkl
    combined_df = pd.read_pickle(args.combined_pickle)
    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"])

    ALL_UNITS = get_all_unit_names(combined_df)
    unique_days = sorted(combined_df["date"].unique())
    if len(unique_days) == 0:
        raise RuntimeError("No days found in combined_df.")

    # Day 0
    day0 = unique_days[0]
    day0_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)

    # Build continuous day0 + trial centers
    X_raw, Y_raw, cuts, trial_centers, trial_info, emg_names = \
        build_continuous_dataset_raw_with_trials(day0_df, BIN_FACTOR, all_units=ALL_UNITS)

    if X_raw.size == 0:
        raise RuntimeError("Empty day0 after downsampling.")
    if not trial_centers:
        raise RuntimeError("No trial centers found in day0.")

    print(f"[INFO] Found {len(trial_centers)} trial centers in day0.")
    if args.trial_idx < 0 or args.trial_idx >= len(trial_centers):
        raise IndexError(f"trial_idx {args.trial_idx} out of range (0..{len(trial_centers)-1}).")

    if emg_names is None:
        raise RuntimeError("Could not infer EMG channel names.")
    if args.muscle not in emg_names:
        raise ValueError(f"Muscle '{args.muscle}' not in EMG names: {emg_names}")
    muscle_idx = emg_names.index(args.muscle)

    # Preprocess (smoothing + low-pass) per block
    X_proc, Y_proc = preprocess_within_cuts(X_raw, Y_raw, cuts, BIN_FACTOR)

    # Train or load decoder + dimred
    need_train = args.retrain or (not os.path.exists(args.checkpoint))
    if need_train:
        print("[INFO] Training new decoder on day0 ...")
        # Fit dimred on whole day0 preprocessed
        dimred_model = get_dimred_model(X_proc, args.dimred, max(N_PCA, 2), args.seed)
        Z0 = transform_dimred(dimred_model, X_proc, args.dimred)[:, :N_PCA]

        is_linear = (args.decoder == "Linear")
        X_tr, Y_tr, _ = build_seq_with_cuts_and_indices(
            Z0, Y_proc, K_LAG, cuts, stride=STRIDE,
            start=0, end=Z0.shape[0], is_linear=is_linear
        )
        if X_tr.shape[0] == 0:
            raise RuntimeError("No training windows after build_seq_with_cuts_and_indices.")

        if args.decoder == "GRU":
            model = GRUDecoder(N_PCA, HIDDEN, Y_proc.shape[1]).to(DEVICE)
        elif args.decoder == "LSTM":
            model = LSTMDecoder(N_PCA, HIDDEN, Y_proc.shape[1]).to(DEVICE)
        elif args.decoder == "Linear":
            model = LinearLagDecoder(K_LAG * N_PCA, HIDDEN, Y_proc.shape[1]).to(DEVICE)
        else:
            model = LiGRUDecoder(N_PCA, HIDDEN, Y_proc.shape[1]).to(DEVICE)

        model = train_model(model, X_tr, Y_tr, num_epochs=NUM_EPOCHS,
                            lr=LR, batch_size=args.batch_size)
        save_checkpoint(args.checkpoint, args.decoder, args.dimred,
                        dimred_model, model, hp, emg_names, ALL_UNITS)
    else:
        print(f"[INFO] Loading decoder from {args.checkpoint}")
        ckpt = load_checkpoint(args.checkpoint)
        if ckpt["decoder"] != args.decoder:
            raise ValueError(f"Checkpoint decoder {ckpt['decoder']} != requested {args.decoder}")
        if ckpt["dimred"].upper() != args.dimred.upper():
            raise ValueError(f"Checkpoint dimred {ckpt['dimred']} != requested {args.dimred}")
        dimred_model = ckpt["dimred_model"]
        emg_names_ckpt = ckpt.get("emg_names", emg_names)
        if emg_names_ckpt != emg_names:
            print("[WARN] EMG names in checkpoint differ from current; using checkpoint names.")
            emg_names = emg_names_ckpt
            if args.muscle not in emg_names:
                raise ValueError(f"Muscle '{args.muscle}' not in checkpoint EMG names: {emg_names}")
            muscle_idx = emg_names.index(args.muscle)

        N_PCA = ckpt["N_PCA"]
        K_LAG = ckpt["K_LAG"]
        HIDDEN = ckpt["HIDDEN"]

        if args.decoder == "GRU":
            model = GRUDecoder(N_PCA, HIDDEN, Y_proc.shape[1]).to(DEVICE)
        elif args.decoder == "LSTM":
            model = LSTMDecoder(N_PCA, HIDDEN, Y_proc.shape[1]).to(DEVICE)
        elif args.decoder == "Linear":
            model = LinearLagDecoder(K_LAG * N_PCA, HIDDEN, Y_proc.shape[1]).to(DEVICE)
        else:
            model = LiGRUDecoder(N_PCA, HIDDEN, Y_proc.shape[1]).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])

    # Recompute Z0 from current preprocessed data
    Z0_full = transform_dimred(dimred_model, X_proc, args.dimred)[:, :N_PCA]

    # Window around chosen trial center (global index in continuous downsampled time)
    center_idx = trial_centers[args.trial_idx]
    fs_ds = SAMPLING_RATE // BIN_FACTOR
    pre_bins = int(round(args.pre_s * fs_ds))
    post_bins = int(round(args.post_s * fs_ds))

    start_idx = max(0, center_idx - pre_bins)
    end_idx = min(Z0_full.shape[0], center_idx + post_bins)

    is_linear = (args.decoder == "Linear")
    X_win, Y_win, idx_used = build_seq_with_cuts_and_indices(
        Z0_full, Y_proc, K_LAG, cuts, stride=1,
        start=start_idx, end=end_idx, is_linear=is_linear
    )
    if X_win.shape[0] == 0:
        raise RuntimeError("No valid windows in selected trial window (maybe crosses a cut or too close to start).")

    preds = evaluate_window(model, X_win, batch_size=args.batch_size)
    # idx_used are the center times t corresponding to each prediction row
    y_true_muscle = Y_win[:, muscle_idx]
    y_pred_muscle = preds[:, muscle_idx]

    vaf = compute_vaf_1d(y_true_muscle, y_pred_muscle)

    # Time axis relative to center_idx
    dt = BIN_FACTOR * BIN_SIZE  # seconds per downsampled bin
    time_rel = (idx_used - center_idx) * dt

    # Output filename
    if args.out is None:
        base = f"day0_trial{args.trial_idx}_{args.decoder}_{args.muscle}.svg"
        out_path = base
    else:
        out_path = args.out

    plot_single_muscle(time_rel, y_true_muscle, y_pred_muscle,
                       args.muscle, args.decoder, vaf, out_path)


if __name__ == "__main__":
    main()
    

# python figure_1A.py \
#   --decoder GRU \
#   --muscle ECR \
#   --trial_idx 5 \
#   --combined_pickle combined.pkl \
#   --checkpoint gru_day0.ckpt \
#   --retrain \
#   --out day0_trial5_GRU_ECR.svg
