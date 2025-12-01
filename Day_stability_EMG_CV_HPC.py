#!/usr/bin/env python3
import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from numpy.linalg import pinv
from scipy.linalg import orthogonal_procrustes  # NEW: for UMAP alignment
from sklearn.decomposition import PCA

# UMAP (optionnel)
try:
    import umap
except Exception:
    umap = None

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state.")

# ============================== CONFIG ==============================
COMBINED_PICKLE_FILE = "combined.pkl"
SAVE_RESULTS_PKL     = "multi_day_align_results_cv.pkl"

SEED              = 42
BIN_FACTOR        = 20
BIN_SIZE          = 0.001        # 1 ms (sera multiplié par BIN_FACTOR)
SMOOTHING_LENGTH  = 0.05         # 50 ms
SAMPLING_RATE     = 1000
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE        = 128
GAUSS_TRUNCATE    = 4.0           # portée ~ truncate * sigma
WINDOW_STRIDE     = None          # None => sera mis à K_LAG plus bas (train & val)

ARCH_HYPERPARAMS = {
    "GRU":    dict(N_PCA=32, K_LAG=25, HIDDEN=96,   NUM_EPOCHS=200, LR=0.003),
    "LSTM":   dict(N_PCA=24, K_LAG=25, HIDDEN=128,  NUM_EPOCHS=300, LR=0.003),
    "Linear": dict(N_PCA=32, K_LAG=16, HIDDEN=64,   NUM_EPOCHS=100, LR=0.003),
    "LiGRU":  dict(N_PCA=32, K_LAG=16, HIDDEN=5,    NUM_EPOCHS=200, LR=0.001),
}

# ============================== HELPERS ==============================
def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

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
        e_arr = emg_data.values; col_names = emg_data.columns
    else:
        e_arr = np.array(emg_data); col_names = None

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
    sigma = (smoothing_length / bin_size) / 2.0
    return gaussian_filter1d(x_2d.astype(float), sigma=sigma, axis=0)

def build_continuous_dataset_raw(df, bin_factor, all_units=None):
    """Concatène les essais d'une journée SANS lissage/filtrage.
       Retourne X_raw, Y_raw, cuts (frontières cumulées entre essais)."""
    spikes_all, emg_all, lengths = [], [], []
    for _, row in df.iterrows():
        spike_df = row["spike_counts"]; emg_val = row["EMG"]
        if not isinstance(spike_df, pd.DataFrame) or spike_df.empty: continue
        if emg_val is None: continue
        if all_units is not None:
            spike_df = spike_df.reindex(columns=all_units, fill_value=0)
        ds_spike_df, ds_emg = downsample_spike_and_emg(spike_df, emg_val, bin_factor)
        if ds_spike_df.shape[0] == 0: continue
        Xr = ds_spike_df.values.astype(float)
        if isinstance(ds_emg, pd.DataFrame):
            Yr = ds_emg.values.astype(float)
        else:
            Yr = np.asarray(ds_emg, dtype=float)
        spikes_all.append(Xr); emg_all.append(Yr); lengths.append(len(Xr))
    if not spikes_all:
        return np.empty((0,)), np.empty((0,)), []
    cuts = np.cumsum(lengths)[:-1].tolist()
    return np.concatenate(spikes_all, axis=0), np.concatenate(emg_all, axis=0), cuts

def preprocess_segment(Xseg, Yseg, bin_factor, bin_size=BIN_SIZE, smoothing_length=SMOOTHING_LENGTH):
    """Lissage/filtrage PAR SEGMENT (pas global) -> pas de fuite train↔val."""
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
    """K folds contigus sur l'axe temps (sans shuffle)."""
    block = n_time // n_splits
    splits = []
    for k in range(n_splits):
        v0 = k * block
        v1 = (k + 1) * block if k < n_splits - 1 else n_time
        splits.append((v0, v1))
    return splits

def valid_window_indices(n_time, k, cuts, stride=1, start=0, end=None):
    """Fenêtres [t-k, t) pour t in [start+k, end), respectant cuts et stride."""
    end = n_time if end is None else end
    out = []
    for t in range(start + k, end, stride):
        # invalide si une cut est dans (t-k, t)
        if any(t - k < c < t for c in cuts):
            continue
        out.append(t)
    return out

# ========================= DIM REDUCTION =========================
def get_dimred_model(data, method, n_components, seed):
    if method.upper() == "PCA":
        model = PCA(n_components=n_components, random_state=seed)
    elif method.upper() == "UMAP":
        if umap is None:
            raise RuntimeError("umap-learn not installed: pip install umap-learn")
        model = umap.UMAP(n_components=n_components, random_state=seed)
    else:
        raise ValueError(f"Unknown dim. reduction: {method}")
    model.fit(data)
    return model

def transform_dimred(model, data, method):
    return model.transform(data)

# =============================== MODELS ===============================
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

# ========================== TRAIN / EVAL ==========================
def train_model(model, X_train, Y_train, num_epochs, lr):
    ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                       torch.tensor(Y_train, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    opt  = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for ep in range(1, num_epochs + 1):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
    return model

def compute_vaf_1d(y_true, y_pred):
    var_true = np.var(y_true)
    if var_true < 1e-12: return np.nan
    var_resid = np.var(y_true - y_pred)
    return 1.0 - (var_resid / var_true)

def compute_multichannel_vaf(y_true, y_pred):
    if y_true.shape[0] == 0: return np.array([])
    return np.array([compute_vaf_1d(y_true[:, ch], y_pred[:, ch]) for ch in range(y_true.shape[1])])

def evaluate_model(model, X, Y):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            bx = torch.tensor(X[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
            out = model(bx)
            preds.append(out.cpu().numpy())
    preds = np.concatenate(preds, axis=0) if preds else np.empty((0,))
    vafs = compute_multichannel_vaf(Y, preds) if preds.size else np.full((Y.shape[1],), np.nan)
    return float(np.nanmean(vafs)), vafs

# ============================ ALIGNMENT ============================
def align_linear_pinv(Zx: np.ndarray, Z0: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    """UMAP/PCA-agnostic : résout Zx A ≈ Z0 (centré) -> A = (X^T X + lam I)^(-1) X^T Y."""
    if Zx.shape != Z0.shape:
        raise ValueError(f"pinv align requires same shape, got {Zx.shape} vs {Z0.shape}")
    X = Zx - Zx.mean(axis=0, keepdims=True)
    Y = Z0 - Z0.mean(axis=0, keepdims=True)
    d = X.shape[1]
    A = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)  # Moore–Penrose régularisé
    return (Zx - Zx.mean(axis=0, keepdims=True)) @ A + Z0.mean(axis=0, keepdims=True)

# ================================ MAIN ================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', type=str, required=True, choices=list(ARCH_HYPERPARAMS.keys()))
    parser.add_argument('--dimred',  type=str, default="PCA", choices=["PCA", "UMAP"])
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--seed',    type=int, default=SEED)
    parser.add_argument('--save_dir', type=str, default=".")
    parser.add_argument('--combined_pickle', type=str, default=COMBINED_PICKLE_FILE)
    args = parser.parse_args()

    set_seed(args.seed)
    hp = ARCH_HYPERPARAMS[args.decoder]
    N_PCA, K_LAG, HIDDEN, NUM_EPOCHS, LR = hp["N_PCA"], hp["K_LAG"], hp["HIDDEN"], hp["NUM_EPOCHS"], hp["LR"]
    global WINDOW_STRIDE
    WINDOW_STRIDE = K_LAG if WINDOW_STRIDE is None else WINDOW_STRIDE  # stride large (évite near-dup)

    combined_df = pd.read_pickle(args.combined_pickle)
    if not np.issubdtype(combined_df["date"].dtype, np.datetime64):
        combined_df["date"] = pd.to_datetime(combined_df["date"])
    ALL_UNITS = get_all_unit_names(combined_df)
    unique_days = sorted(combined_df["date"].unique())
    if len(unique_days) == 0:
        print("[ERROR] No days found in combined_df!"); return
    day0 = unique_days[0]
    train_df = combined_df[combined_df["date"] == day0].reset_index(drop=True)

    # Detect EMG channels
    n_emg_channels = 0
    for _, row in combined_df.iterrows():
        emg_val = row.get("EMG", None)
        if emg_val is not None:
            if isinstance(emg_val, pd.DataFrame) and not emg_val.empty:
                n_emg_channels = emg_val.shape[1]; break
            elif isinstance(emg_val, np.ndarray) and emg_val.size > 0:
                n_emg_channels = emg_val.shape[1]; break
    if n_emg_channels == 0:
        print("[ERROR] Could not detect EMG channels."); return

    # --------- Day0 raw (pas de lissage global) ---------
    X0_raw, Y0_raw, cuts0 = build_continuous_dataset_raw(train_df, BIN_FACTOR, all_units=ALL_UNITS)
    if X0_raw.size == 0:
        print("[ERROR] empty day0"); return

    # Time K-fold splits (contigus)
    splits = time_kfold_splits(X0_raw.shape[0], args.n_folds)
    results = []

    for fold, (val_start, val_end) in enumerate(splits, 1):
        # Segments bruts
        X_left_raw  = X0_raw[:val_start];   Y_left_raw  = Y0_raw[:val_start]
        X_val_raw   = X0_raw[val_start:val_end]; Y_val_raw = Y0_raw[val_start:val_end]
        X_right_raw = X0_raw[val_end:];     Y_right_raw = Y0_raw[val_end:]

        # Prétraitement PAR SEGMENT
        X_left,  Y_left  = preprocess_segment(X_left_raw,  Y_left_raw,  BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
        X_val,   Y_val   = preprocess_segment(X_val_raw,   Y_val_raw,   BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)
        X_right, Y_right = preprocess_segment(X_right_raw, Y_right_raw, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)

        # Embargo
        EMB = embargo_bins(K_LAG, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH, GAUSS_TRUNCATE)
        if X_left.shape[0]  > EMB: X_left  = X_left[:-EMB];  Y_left  = Y_left[:-EMB]
        else:                      X_left  = np.empty((0, X_left.shape[1] if X_left.size else 0)); Y_left  = np.empty((0, Y_left.shape[1] if Y_left.size else n_emg_channels))
        if X_val.shape[0]   > 2*EMB: X_val = X_val[EMB:-EMB]; Y_val = Y_val[EMB:-EMB]
        else:                        X_val = np.empty((0, X_val.shape[1] if X_val.size else 0));   Y_val = np.empty((0, Y_val.shape[1] if Y_val.size else n_emg_channels))
        if X_right.shape[0] > EMB: X_right = X_right[EMB:];  Y_right = Y_right[EMB:]
        else:                     X_right = np.empty((0, X_right.shape[1] if X_right.size else 0)); Y_right = np.empty((0, Y_right.shape[1] if Y_right.size else n_emg_channels))

        # Train time concat
        if X_left.size and X_right.size:
            Xtr_time = np.vstack([X_left, X_right]); Ytr_time = np.vstack([Y_left, Y_right])
        elif X_left.size:
            Xtr_time, Ytr_time = X_left, Y_left
        else:
            Xtr_time, Ytr_time = X_right, Y_right

        if Xtr_time.shape[0] <= K_LAG or X_val.shape[0] <= K_LAG:
            print(f"[fold={fold}] not enough samples after embargo; skip"); continue

        # Fit manifold sur TRAIN uniquement
        dimred_model_day0 = get_dimred_model(Xtr_time, args.dimred, max(N_PCA, 2), args.seed + fold)
        Z_tr = transform_dimred(dimred_model_day0, Xtr_time, args.dimred)[:, :N_PCA]
        Z_va = transform_dimred(dimred_model_day0, X_val,   args.dimred)[:, :N_PCA]

        # Fenêtres non chevauchantes (stride = K_LAG)
        stride = WINDOW_STRIDE

        # Construire X/Y séquentiels
        def build_seq(Z, Y):
            if args.decoder == "Linear":
                # linear: concat fenetre (K*P)
                idx = list(range(K_LAG, Z.shape[0], stride))
                X = np.stack([Z[t-K_LAG:t, :].reshape(-1) for t in idx], axis=0) if idx else np.empty((0, K_LAG*N_PCA))
                Y = np.stack([Y[t, :] for t in idx], axis=0) if idx else np.empty((0, Y.shape[1]))
                return X, Y
            else:
                idx = list(range(K_LAG, Z.shape[0], stride))
                X = np.stack([Z[t-K_LAG:t, :] for t in idx], axis=0) if idx else np.empty((0, K_LAG, N_PCA))
                Y = np.stack([Y[t, :] for t in idx], axis=0) if idx else np.empty((0, Y.shape[1]))
                return X, Y

        X_tr, Y_tr = build_seq(Z_tr, Ytr_time)
        X_te, Y_te = build_seq(Z_va, Y_val)
        if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
            print(f"[fold={fold}] empty after windowing; skip"); continue

        # Modèle
        if args.decoder == "GRU":
            model = GRUDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
        elif args.decoder == "LSTM":
            model = LSTMDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
        elif args.decoder == "Linear":
            model = LinearLagDecoder(K_LAG * N_PCA, HIDDEN, n_emg_channels).to(DEVICE)
        else:
            model = LiGRUDecoder(N_PCA, HIDDEN, n_emg_channels).to(DEVICE)

        train_model(model, X_tr, Y_tr, NUM_EPOCHS, LR)
        vaf_te, vaf_ch_te = evaluate_model(model, X_te, Y_te)

        for ch_idx, vaf_single in enumerate(vaf_ch_te):
            results.append({
                "day": day0, "day_int": 0, "align": "crossval",
                "decoder": args.decoder, "dim_red": args.dimred,
                "fold": fold-1, "emg_channel": ch_idx, "vaf": vaf_single
            })

        # -------- Cross-days (direct + aligned) avec modèle fold --------
        for d_val in unique_days:
            if pd.to_datetime(d_val) == pd.to_datetime(day0): continue
            day_df = combined_df[combined_df["date"] == d_val].reset_index(drop=True)
            spikeX, emgX = build_continuous_dataset_raw(day_df, BIN_FACTOR, all_units=ALL_UNITS)[:2]  # raw
            if spikeX.shape[0] == 0: continue
            # preprocess tout le jour X (test-time only)
            spikeX, emgX = preprocess_segment(spikeX, emgX, BIN_FACTOR, BIN_SIZE, SMOOTHING_LENGTH)

            zx_direct = transform_dimred(dimred_model_day0, spikeX, args.dimred)[:, :N_PCA]
            dimred_model_dayX = get_dimred_model(spikeX, args.dimred, N_PCA, args.seed + fold)
            zx_dayX   = transform_dimred(dimred_model_dayX, spikeX, args.dimred)[:, :N_PCA]

            for align_mode in ["direct", "aligned"]:
                if align_mode == "direct":
                    zx_test = zx_direct
                else:
                    if args.dimred.upper() == "PCA":
                        V0 = dimred_model_day0.components_[:N_PCA, :].T
                        Vx = dimred_model_dayX.components_[:N_PCA, :].T
                        try:
                            R = pinv(Vx) @ V0      # Moore–Penrose sur bases PCA
                            zx_test = zx_dayX @ R
                        except Exception:
                            zx_test = zx_dayX
                    else:
                        try:
                            zx_test = align_linear_pinv(zx_dayX, zx_direct, lam=1e-6)  # Moore–Penrose LS
                        except Exception:
                            zx_test = zx_dayX

                # Windows (stride K_LAG)
                if args.decoder == "Linear":
                    idx = list(range(K_LAG, zx_test.shape[0], WINDOW_STRIDE))
                    X_seq = np.stack([zx_test[t-K_LAG:t, :].reshape(-1) for t in idx], axis=0) if idx else np.empty((0, K_LAG*N_PCA))
                    Y_seq = np.stack([emgX[t, :] for t in idx], axis=0) if idx else np.empty((0, emgX.shape[1]))
                else:
                    idx = list(range(K_LAG, zx_test.shape[0], WINDOW_STRIDE))
                    X_seq = np.stack([zx_test[t-K_LAG:t, :] for t in idx], axis=0) if idx else np.empty((0, K_LAG, N_PCA))
                    Y_seq = np.stack([emgX[t, :] for t in idx], axis=0) if idx else np.empty((0, emgX.shape[1]))

                if X_seq.shape[0] == 0: continue
                vaf, vaf_ch = evaluate_model(model, X_seq, Y_seq)
                for ch_idx, vaf_single in enumerate(vaf_ch):
                    results.append({
                        "day": d_val, "day_int": (pd.to_datetime(d_val) - pd.to_datetime(day0)).days,
                        "align": align_mode, "decoder": args.decoder, "dim_red": args.dimred,
                        "fold": fold-1, "emg_channel": ch_idx, "vaf": vaf_single, "mean_vaf": vaf
                    })

        print(f"[fold={fold}/{args.n_folds}] done.")

    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"crossday_results_{args.decoder}_{args.dimred}.pkl")
    pd.to_pickle(pd.DataFrame(results), save_path)
    print(f"\n[INFO] Saved all results to {save_path}")

if __name__ == "__main__":
    main()
