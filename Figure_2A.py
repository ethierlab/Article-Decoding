#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure A : trace EMG d'un seul muscle autour d'un essai,
avec deux décoders (par ex. aligned vs direct) chargés depuis des checkpoints.

Exemple d'appel :

python figure_A_single_muscle_trace.py \
  --combined_pickle combined.pkl \
  --decoder GRU \
  --dimred PCA \
  --muscle ECR \
  --day_idx 0 \
  --trial_idx 5 \
  --checkpoint_a gru_aligned_day0.ckpt \
  --checkpoint_b gru_direct_day0.ckpt \
  --label_a "aligned" \
  --label_b "direct" \
  --out figA_GRU_ECR_D0_trial5.svg
"""

import os
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings(
    "ignore", message="n_jobs value 1 overridden to 1 by setting random_state."
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Data helpers (version light, sans entraînement)
# ---------------------------------------------------------------------
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


def smooth_spike_data(x_2d, bin_size, smoothing_length):
    sigma = (smoothing_length / bin_size) / 2.0
    return gaussian_filter1d(x_2d.astype(np.float32), sigma=sigma, axis=0)


def preprocess_segment(
    Xseg,
    Yseg,
    bin_factor,
    bin_size,
    smoothing_length,
    sampling_rate,
):
    eff_fs = sampling_rate // bin_factor
    Xs = smooth_spike_data(Xseg, bin_size * bin_factor, smoothing_length)
    Ys = butter_lowpass(np.abs(Yseg), eff_fs)
    return Xs, Ys


def preprocess_within_cuts(
    X_raw,
    Y_raw,
    cuts,
    bin_factor,
    bin_size,
    smoothing_length,
    sampling_rate,
):
    if not cuts:
        return preprocess_segment(
            X_raw,
            Y_raw,
            bin_factor,
            bin_size,
            smoothing_length,
            sampling_rate,
        )

    pieces_X, pieces_Y = [], []
    start = 0
    all_cuts = cuts + [len(X_raw)]
    for c in all_cuts:
        Xs, Ys = preprocess_segment(
            X_raw[start:c],
            Y_raw[start:c],
            bin_factor,
            bin_size,
            smoothing_length,
            sampling_rate,
        )
        pieces_X.append(Xs)
        pieces_Y.append(Ys)
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
        X = np.stack([Z[t - K_LAG : t, :].reshape(-1) for t in idx], axis=0).astype(
            np.float32
        )
    else:
        X = np.stack([Z[t - K_LAG : t, :] for t in idx], axis=0).astype(np.float32)
    Yb = np.stack([Y[t, :] for t in idx], axis=0).astype(np.float32)
    return X, Yb, np.array(idx, dtype=int)


def build_continuous_dataset_raw_with_trials(df, bin_factor, all_units=None):
    """
    Construit X_raw, Y_raw, cuts, trial_centers, trial_info, emg_names
    pour un jour donné (df).
    """
    spikes_all, emg_all, lengths = [], [], []
    trial_centers = []
    trial_info = []
    global_offset = 0
    emg_names = None

    for row_idx, row in df.iterrows():
        spike_df = row["spike_counts"]
        emg_val = row["EMG"]
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
            emg_names = list(ds_emg.columns)
        else:
            Yr = np.asarray(ds_emg, dtype=np.float32)

        T_row = Xr.shape[0]

        time_frame = np.asarray(row["time_frame"]).flatten()
        T_old = len(time_frame)
        T_new = T_old // bin_factor
        if T_new <= 0:
            continue
        tf_ds = time_frame[: T_new * bin_factor].reshape(T_new, bin_factor).mean(axis=1)

        trial_starts = np.asarray(row["trial_start_time"]).flatten()
        for loc_idx, ts in enumerate(trial_starts):
            if len(tf_ds) == 0:
                continue
            local_center = int(np.argmin(np.abs(tf_ds - ts)))
            if local_center < T_row:
                global_center = global_offset + local_center
                trial_centers.append(global_center)
                trial_info.append((row_idx, loc_idx, float(ts)))

        spikes_all.append(Xr)
        emg_all.append(Yr)
        lengths.append(T_row)
        global_offset += T_row

    if len(spikes_all) == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
            [],
            [],
            emg_names,
        )

    cuts = np.cumsum(lengths)[:-1].tolist()
    X_raw = np.concatenate(spikes_all, axis=0)
    Y_raw = np.concatenate(emg_all, axis=0)
    return X_raw, Y_raw, cuts, trial_centers, trial_info, emg_names


# ---------------------------------------------------------------------
# Models (comme dans ton script original, sans entraînement)
# ---------------------------------------------------------------------
class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


class LSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class LinearLagDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
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
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)


def evaluate_window(model, X_win, batch_size=256):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_win), batch_size):
            bx = torch.as_tensor(X_win[i : i + batch_size], dtype=torch.float32).to(DEVICE)
            out = model(bx)
            preds.append(out.cpu().numpy())
    if preds:
        return np.concatenate(preds, axis=0)
    else:
        return np.empty((0,))


def compute_vaf_1d(y_true, y_pred):
    var_resid = np.var(y_true - y_pred)
    var_true = np.var(y_true)
    if var_true < 1e-12:
        return np.nan
    return 1.0 - (var_resid / var_true)


# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------
def plot_single_trial_dual_decoder(
    time_rel,
    y_true,
    y_pred_a,
    y_pred_b,
    label_a,
    label_b,
    muscle_name,
    decoder_name,
    vaf_a,
    vaf_b,
    out_path,
):
    plt.figure(figsize=(6, 3))
    plt.plot(time_rel, y_true, label="actual", color="black", linewidth=1.0)
    plt.plot(
        time_rel,
        y_pred_a,
        label=f"{label_a} (VAF={vaf_a:.2f})",
        color="tab:red",
        linewidth=1.0,
    )
    plt.plot(
        time_rel,
        y_pred_b,
        label=f"{label_b} (VAF={vaf_b:.2f})",
        color="tab:gray",
        linestyle="--",
        linewidth=1.0,
    )
    plt.xlabel("Time (s, aligned to trial start)")
    plt.ylabel(f"EMG {muscle_name}")
    plt.title(f"{decoder_name} • {muscle_name}")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, format=os.path.splitext(out_path)[1][1:])
    plt.close()
    print(f"[INFO] Saved figure A to {out_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Figure A : single muscle, single trial, two checkpoints (e.g. aligned vs direct)."
    )
    parser.add_argument("--combined_pickle", type=str, default="combined.pkl")
    parser.add_argument(
        "--decoder",
        type=str,
        required=True,
        choices=["GRU", "LSTM", "Linear", "LiGRU"],
    )
    parser.add_argument("--dimred", type=str, default="PCA", choices=["PCA"])
    parser.add_argument("--muscle", type=str, required=True)
    parser.add_argument("--day_idx", type=int, default=0, help="index dans la liste des dates triées")
    parser.add_argument("--trial_idx", type=int, required=True)

    parser.add_argument("--checkpoint_a", type=str, required=True, help="checkpoint 1 (e.g. aligned)")
    parser.add_argument("--checkpoint_b", type=str, required=True, help="checkpoint 2 (e.g. direct)")
    parser.add_argument("--label_a", type=str, default="aligned")
    parser.add_argument("--label_b", type=str, default="direct")

    parser.add_argument("--pre_s", type=float, default=1.0)
    parser.add_argument("--post_s", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out", type=str, default="figA_single_muscle.svg")

    args = parser.parse_args()

    # Charger checkpoints
    if not os.path.exists(args.checkpoint_a) or not os.path.exists(args.checkpoint_b):
        raise FileNotFoundError("One of the checkpoints does not exist.")

    ckpt_a = torch.load(args.checkpoint_a, map_location=DEVICE)
    ckpt_b = torch.load(args.checkpoint_b, map_location=DEVICE)

    # Sanity check: même type de decoder / dimred
    for ckpt, label in [(ckpt_a, "A"), (ckpt_b, "B")]:
        if ckpt["decoder"] != args.decoder:
            raise ValueError(
                f"Checkpoint {label} has decoder {ckpt['decoder']} != requested {args.decoder}"
            )
        if ckpt["dimred"].upper() != args.dimred.upper():
            raise ValueError(
                f"Checkpoint {label} has dimred {ckpt['dimred']} != requested {args.dimred}"
            )

    # Utiliser les hyperparamètres / binning du checkpoint A (supposés identiques à B)
    bin_factor = ckpt_a.get("bin_factor", 20)
    bin_size = ckpt_a.get("bin_size", 0.001)
    smoothing_length = ckpt_a.get("smoothing_length", 0.05)
    sampling_rate = ckpt_a.get("sampling_rate", 1000)

    # Charger combined.pkl
    combined_df = pd.read_pickle(args.combined_pickle)
    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"])

    all_units = get_all_unit_names(combined_df)
    unique_days = sorted(combined_df["date"].unique())
    if len(unique_days) == 0:
        raise RuntimeError("No days found in combined_df.")

    if args.day_idx < 0 or args.day_idx >= len(unique_days):
        raise IndexError(
            f"day_idx {args.day_idx} out of range (0..{len(unique_days)-1})."
        )

    day = unique_days[args.day_idx]
    day_df = combined_df[combined_df["date"] == day].reset_index(drop=True)
    print(f"[INFO] Using day index {args.day_idx} -> date {day}")

    X_raw, Y_raw, cuts, trial_centers, trial_info, emg_names = (
        build_continuous_dataset_raw_with_trials(day_df, bin_factor, all_units=all_units)
    )

    if X_raw.size == 0:
        raise RuntimeError("Empty day after downsampling.")
    if not trial_centers:
        raise RuntimeError("No trial centers found in selected day.")
    if emg_names is None:
        raise RuntimeError("Could not infer EMG channel names.")
    if args.muscle not in emg_names:
        raise ValueError(f"Muscle '{args.muscle}' not in EMG names: {emg_names}")

    muscle_idx = emg_names.index(args.muscle)

    # Pré-processing global jour
    X_proc, Y_proc = preprocess_within_cuts(
        X_raw,
        Y_raw,
        cuts,
        bin_factor,
        bin_size,
        smoothing_length,
        sampling_rate,
    )

    # Fenêtre autour du centre d'essai choisi
    center_idx = trial_centers[args.trial_idx]
    fs_ds = sampling_rate // bin_factor
    pre_bins = int(round(args.pre_s * fs_ds))
    post_bins = int(round(args.post_s * fs_ds))

    start_idx = max(0, center_idx - pre_bins)
    end_idx = min(X_proc.shape[0], center_idx + post_bins)

    # Fonction utilitaire pour un checkpoint
    def decode_one_checkpoint(ckpt):
        N_PCA = ckpt["N_PCA"]
        K_LAG = ckpt["K_LAG"]
        HIDDEN = ckpt["HIDDEN"]
        dimred_model = ckpt["dimred_model"]

        emg_names_ckpt = ckpt.get("emg_names", emg_names)
        if emg_names_ckpt != emg_names:
            print("[WARN] EMG names in checkpoint differ from current; using checkpoint names.")
            if args.muscle not in emg_names_ckpt:
                raise ValueError(
                    f"Muscle '{args.muscle}' not in checkpoint EMG names: {emg_names_ckpt}"
                )
            m_idx = emg_names_ckpt.index(args.muscle)
        else:
            m_idx = muscle_idx

        # Dim red sur tout le jour
        Z_full = dimred_model.transform(X_proc)[:, :N_PCA]

        is_linear = (ckpt["decoder"] == "Linear")
        X_win, Y_win, idx_used = build_seq_with_cuts_and_indices(
            Z_full,
            Y_proc,
            K_LAG,
            cuts,
            stride=1,
            start=start_idx,
            end=end_idx,
            is_linear=is_linear,
        )
        if X_win.shape[0] == 0:
            raise RuntimeError(
                "No valid windows in selected trial window (maybe crosses a cut or too close to edges)."
            )

        # Reconstruire le modèle
        if ckpt["decoder"] == "GRU":
            model = GRUDecoder(N_PCA, HIDDEN, Y_proc.shape[1]).to(DEVICE)
        elif ckpt["decoder"] == "LSTM":
            model = LSTMDecoder(N_PCA, HIDDEN, Y_proc.shape[1]).to(DEVICE)
        elif ckpt["decoder"] == "Linear":
            model = LinearLagDecoder(K_LAG * N_PCA, HIDDEN, Y_proc.shape[1]).to(DEVICE)
        else:
            model = LiGRUDecoder(N_PCA, HIDDEN, Y_proc.shape[1]).to(DEVICE)

        model.load_state_dict(ckpt["state_dict"])

        preds = evaluate_window(model, X_win, batch_size=args.batch_size)

        y_true_m = Y_win[:, m_idx]
        y_pred_m = preds[:, m_idx]

        vaf = compute_vaf_1d(y_true_m, y_pred_m)

        dt = bin_factor * bin_size
        time_rel = (idx_used - center_idx) * dt
        return time_rel, y_true_m, y_pred_m, vaf

    # Decoder A et B (on assume même fenêtrage)
    time_rel_a, y_true_a, y_pred_a, vaf_a = decode_one_checkpoint(ckpt_a)
    time_rel_b, y_true_b, y_pred_b, vaf_b = decode_one_checkpoint(ckpt_b)

    # Vérification : même temps / y_true
    if not np.array_equal(time_rel_a, time_rel_b):
        print("[WARN] time axes differ between checkpoints; using A's time.")
    if not np.allclose(y_true_a, y_true_b, atol=1e-6):
        print("[WARN] y_true differs between checkpoints; using A's y_true.")

    plot_single_trial_dual_decoder(
        time_rel=time_rel_a,
        y_true=y_true_a,
        y_pred_a=y_pred_a,
        y_pred_b=y_pred_b,
        label_a=args.label_a,
        label_b=args.label_b,
        muscle_name=args.muscle,
        decoder_name=args.decoder,
        vaf_a=vaf_a,
        vaf_b=vaf_b,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
