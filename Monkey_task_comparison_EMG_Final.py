#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-task EMG decoder validation (STRICT CV, no leakage), with optional deep debug.

- No normalization / no z-score.
- AMP uses torch.amp (autocast/GradScaler).
- Spike smoothing + EMG rectification + low-pass remain identical.
- Pass --debug to print rich diagnostics (per-muscle VAF, var ratios, correlations, etc.).
"""

import os, sys, argparse, random, warnings, datetime, time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from numpy.linalg import pinv
from collections import defaultdict

# UMAP optional
try:
    import umap
except Exception:
    umap = None

warnings.filterwarnings("ignore", category=UserWarning)

# ───────────────────────── Runtime / Flags ─────────────────────────
SEED   = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PERF_MODE = True   # True => speed (TF32, non-deterministic cudnn), False => stricter repro
DEBUG = False      # set by --debug

def dprint(msg: str):
    """Debug print only if --debug was passed."""
    if DEBUG:
        print(msg); sys.stdout.flush()

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
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
        if n > 0: return max(2, n-1)
    except Exception:
        pass
    return default

# ────────────────────── Debug helpers ──────────────────────
def _nz_pct(a):
    if a.size == 0: return 0.0
    return 100.0 * (np.count_nonzero(a) / a.size)

def _finite_ok(a):
    return np.isfinite(a).all()

def _nan_inf_counts(a):
    return np.isnan(a).sum(), np.isinf(a).sum()

def dbg_stats(arr, name):
    """Heavy array stats (only when DEBUG)."""
    if not DEBUG:
        return
    arr = np.asarray(arr)
    nans, infs = _nan_inf_counts(arr)
    print(f"[DBG] {name}: shape={arr.shape} | finite={_finite_ok(arr)} | NaNs={nans} Infs={infs}")
    if arr.size:
        try:
            mn = float(np.nanmin(arr)); mx = float(np.nanmax(arr)); med = float(np.nanmedian(arr))
            mean = float(np.nanmean(arr)); sd = float(np.nanstd(arr))
        except Exception:
            mn = mx = med = mean = sd = float('nan')
        print(f"[DBG] {name}: min/med/max={mn:.4g}/{med:.4g}/{mx:.4g} | mean±sd={mean:.4g}±{sd:.4g} | nz%={_nz_pct(arr):.1f}%")
        if arr.ndim == 2 and arr.shape[0] > 1 and arr.shape[1] > 0:
            try:
                v = np.var(arr, axis=0)
                vmin, vmed, vmax = float(np.nanmin(v)), float(np.nanmedian(v)), float(np.nanmax(v))
            except Exception:
                vmin = vmed = vmax = float('nan')
            print(f"[DBG] {name}: var[ch] min/med/max={vmin:.4g}/{vmed:.4g}/{vmax:.4g}")
    sys.stdout.flush()

def baseline_mean_vaf(Y):
    # predict per-channel mean; VAF should be ≈0
    if Y.size == 0: return np.nan, np.array([])
    Yp = np.tile(Y.mean(axis=0, keepdims=True), (Y.shape[0], 1))
    ch = np.array([compute_vaf_1d(Y[:,c], Yp[:,c]) for c in range(Y.shape[1])], dtype=np.float32)
    return float(np.nanmean(ch)), ch

def print_split_banner(seed, fold, EMB, K, STRIDE, len_tr, len_val):
    print(f"[DBG] seed={seed} fold={fold} | EMB={EMB}  K={K}  STRIDE={STRIDE}  len(TR)={len_tr} len(VAL)={len_val}")
    sys.stdout.flush()

# ─────────────────────── Hyperparams per decoder ───────────────────────
DECODER_CONFIG = {
    "GRU":    {"N_PCA": 32, "HIDDEN": 96,  "K_LAG": 25, "LR": 0.003, "EPOCHS": 60},
    "LSTM":   {"N_PCA": 24, "HIDDEN": 128, "K_LAG": 25, "LR": 0.003, "EPOCHS": 90},
    "Linear": {"N_PCA": 32, "HIDDEN": 64,  "K_LAG": 16, "LR": 0.003, "EPOCHS": 40},
    "LiGRU":  {"N_PCA": 32, "HIDDEN": 5,   "K_LAG": 16, "LR": 0.001, "EPOCHS": 60},
}
BATCH_SIZE_DEFAULT = 256
BIN_SIZE = 0.02   # 20 ms -> fs ~ 50Hz
SMOOTHING = 0.05  # 50 ms spike smoothing
GAUSS_TRUNCATE = 4.0  # embargo tail ~ truncate*sigma

# ─────────────────────── EMG mapping ───────────────────────
TARGET = {"FCR","FDS","FDP","FCU","ECR","EDC","ECU"}
GLOBAL_MUSCLE_MAP = {
    'ECR_1':'ECR','ECR_2':'ECR','EDC_1':'EDC','EDC_2':'EDC',
    'FCR_1':'FCR','FCU_1':'FCU','FDS_1':'FDS','FDS_2':'FDS',
    'FDP_1':'FDP','FDP_2':'FDP','ECU_1':'ECU'
}

def map_emg_labels(emg_df: pd.DataFrame) -> pd.DataFrame:
    out, cnt = {}, defaultdict(int)
    for col in emg_df.columns:
        raw = str(col).strip().upper(); tmp = raw
        while tmp and tmp not in GLOBAL_MUSCLE_MAP: tmp = tmp[:-1]
        base = GLOBAL_MUSCLE_MAP.get(tmp, None)
        if base and base in TARGET:
            cnt[base]+=1; out[f"{base}_{cnt[base]}"] = emg_df[col]
    return pd.DataFrame(out)

def filter_and_map_emg(df: pd.DataFrame):
    rows, cols = [], set()
    for _,r in df.iterrows():
        emg=r.get("EMG")
        if isinstance(emg, pd.DataFrame) and not emg.empty:
            m=map_emg_labels(emg); r=r.copy(); r["EMG"]=m; cols.update(m.columns)
        rows.append(r)
    df2=pd.DataFrame(rows); cols=sorted(cols)
    for i,r in df2.iterrows():
        emg=r.get("EMG")
        if isinstance(emg, pd.DataFrame):
            r["EMG"]=emg.reindex(cols, fill_value=0)
    return df2, cols

# ───────────────────────── Helpers for units ─────────────────────────
def get_all_unit_names(df):
    unit_set = set()
    for _, row in df.iterrows():
        sc = row.get("spike_counts", None)
        if isinstance(sc, pd.DataFrame):
            unit_set.update(sc.columns)
    return sorted(list(unit_set))

# ───────────────────────── Preprocessing (strict) ─────────────────────────
def smooth_spike_data(X, bin_size=BIN_SIZE, smoothing=SMOOTHING):
    sigma=(smoothing/bin_size)/2.0
    return gaussian_filter1d(X.astype(np.float32), sigma=sigma, axis=0)

def butter_lowpass_abs(Y, fs_hz, cutoff_hz=5.0, order=4):
    rect = np.abs(Y)
    nyq = 0.5*fs_hz
    b, a = butter(order, cutoff_hz/nyq, btype='low', analog=False)
    return filtfilt(b, a, rect, axis=0)

def downsample_pair(spike_df, emg, bin_factor=1):
    # If already binned -> bin_factor=1
    if bin_factor == 1:
        return spike_df, emg
    if spike_df.empty or spike_df.shape[0] < bin_factor:
        return spike_df, emg
    T_old, n_units = spike_df.shape
    T_new = T_old // bin_factor
    sp = spike_df.values[:T_new*bin_factor, :]
    sp = sp.reshape(T_new, bin_factor, n_units).sum(axis=1)
    ds_spike_df = pd.DataFrame(sp, columns=spike_df.columns)

    if isinstance(emg, pd.DataFrame):
        e = emg.values
        cols = emg.columns
    else:
        e = np.asarray(emg)
        cols = None
    if e.shape[0] < bin_factor:
        return ds_spike_df, emg
    e = e[:T_new*bin_factor, ...]
    if e.ndim == 2:
        e = e.reshape(T_new, bin_factor, e.shape[1]).mean(axis=1)
        ds_emg = pd.DataFrame(e, columns=cols) if cols is not None else e
    else:
        ds_emg = emg
    return ds_spike_df, ds_emg

def build_continuous_dataset_raw(df, bin_factor=1, all_units=None, ref_emg_cols=None):
    """Concat trials, keep trial boundaries as 'cuts'. Return X_raw, Y_raw, cuts."""
    Xs, Ys, lengths = [], [], []
    expected = all_units if all_units is not None else None
    n_rows_skipped = 0
    for _, r in df.iterrows():
        sp = r.get("spike_counts"); emg = r.get("EMG")
        if not isinstance(sp, pd.DataFrame) or sp.empty or emg is None:
            n_rows_skipped += 1; continue
        if expected is not None:
            sp = sp.reindex(columns=expected, fill_value=0)
        sp, emg = downsample_pair(sp, emg, bin_factor=bin_factor)
        if isinstance(emg, pd.DataFrame):
            if ref_emg_cols is not None:
                emg = emg.reindex(ref_emg_cols, axis=1, fill_value=0)
            e_val = emg.values.astype(np.float32)
        else:
            e_val = np.asarray(emg, dtype=np.float32)
        Xs.append(sp.values.astype(np.float32))
        Ys.append(e_val)
        lengths.append(len(sp))
    if not Xs:
        dprint(f"[DBG] build_continuous_dataset_raw: NO VALID ROWS (skipped={n_rows_skipped})")
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32), []
    cuts = np.cumsum(lengths)[:-1].tolist()
    X = np.concatenate(Xs, 0); Y = np.concatenate(Ys, 0)
    dprint(f"[DBG] build_raw: X={X.shape} Y={Y.shape} cuts={len(cuts)} skipped_rows={n_rows_skipped}")
    dbg_stats(X, "RAW_X(all trials)"); dbg_stats(Y, "RAW_Y(all trials)")
    return X, Y, cuts

def preprocess_segment(Xseg, Yseg, bin_factor=1, bin_size=BIN_SIZE, smoothing=SMOOTHING, emg_cutoff=5.0):
    fs_eff = int(round(1.0 / (bin_size * bin_factor)))
    dprint(f"[DBG] preprocess_segment: fs_eff≈{fs_eff}Hz  bin_factor={bin_factor}  bin_size={bin_size}s  smoothing={smoothing}s  emg_cutoff={emg_cutoff}Hz")
    Xs = smooth_spike_data(Xseg, bin_size * bin_factor, smoothing)
    Ys = butter_lowpass_abs(Yseg, fs_eff, cutoff_hz=emg_cutoff)
    return Xs, Ys

def preprocess_within_cuts(X_raw, Y_raw, cuts, bin_factor=1):
    """Apply preprocessing independently within each trial block => no bleed across trials."""
    if not cuts:
        Xs, Ys = preprocess_segment(X_raw, Y_raw, bin_factor)
        dbg_stats(Xs, "PREPROC_X(single)"); dbg_stats(Ys, "PREPROC_Y(single)")
        return Xs, Ys
    Xp, Yp = [], []
    start = 0
    for c in cuts + [len(X_raw)]:
        Xs, Ys = preprocess_segment(X_raw[start:c], Y_raw[start:c], bin_factor)
        Xp.append(Xs); Yp.append(Ys)
        start = c
    Xf, Yf = np.concatenate(Xp, 0), np.concatenate(Yp, 0)
    dbg_stats(Xf, "PREPROC_X(concat)"); dbg_stats(Yf, "PREPROC_Y(concat)")
    return Xf, Yf

def sigma_bins(bin_factor=1, bin_size=BIN_SIZE, smoothing=SMOOTHING):
    return (smoothing / (bin_size * bin_factor)) / 2.0

def embargo_bins(K_LAG, bin_factor=1, bin_size=BIN_SIZE, smoothing=SMOOTHING, truncate=GAUSS_TRUNCATE):
    emb = int(np.ceil(truncate * sigma_bins(bin_factor, bin_size, smoothing)))
    return max(K_LAG, emb)

def time_kfold_splits(n_time, n_splits):
    """Contiguous time folds (no shuffle)."""
    block = n_time // n_splits
    splits = []
    for k in range(n_splits):
        v0 = k * block
        v1 = (k + 1) * block if k < n_splits - 1 else n_time
        splits.append((v0, v1))
    return splits

def adjust_cuts_for_segment(start, end, cuts_global, trim_left=0, trim_right=0, seg_len=None):
    """Shift global cuts into [start,end), then into local indices after trimming."""
    local = [c - start for c in cuts_global if start < c < end]
    if seg_len is None:
        seg_len = end - start
    new_start = trim_left
    new_end = seg_len - trim_right
    return [c - new_start for c in local if new_start < c < new_end]

def valid_window_indices(n_time, k, cuts, stride=1, start=0, end=None):
    """Yield end indices t of windows [t-k, t) that DO NOT cross any cut."""
    end = n_time if end is None else end
    out = []
    for t in range(start + k, end, stride):
        if any(t - k < c < t for c in cuts):
            continue
        out.append(t)
    return out

def build_seq_with_cuts(Z, Y, K_LAG, cuts, stride, is_linear):
    idx = valid_window_indices(Z.shape[0], K_LAG, cuts, stride=stride)
    dprint(f"[DBG] build_seq_with_cuts: T={Z.shape[0]}  K={K_LAG}  stride={stride}  n_idx={len(idx)}  is_linear={is_linear}")
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

# ───────────────────── Decoders ─────────────────────
class GRUDecoder(nn.Module):
    def __init__(self, i, h, o): super().__init__(); self.r=nn.GRU(i,h,batch_first=True); self.f=nn.Linear(h,o)
    def forward(self,x): o,_=self.r(x); return self.f(o[:,-1,:])

class LSTMDecoder(nn.Module):
    def __init__(self, i, h, o): super().__init__(); self.r=nn.LSTM(i,h,batch_first=True); self.f=nn.Linear(h,o)
    def forward(self,x): o,_=self.r(x); return self.f(o[:,-1,:])

class LinearLagDecoder(nn.Module):
    def __init__(self, i, h, o): super().__init__(); self.l1=nn.Linear(i,h); self.a=nn.ReLU(); self.l2=nn.Linear(h,o)
    def forward(self,x): return self.l2(self.a(self.l1(x)))

class LiGRUCell(nn.Module):
    def __init__(self, i, h): super().__init__(); self.x2z=nn.Linear(i,h); self.h2z=nn.Linear(h,h,bias=False); self.x2h=nn.Linear(i,h); self.h2h=nn.Linear(h,h,bias=False)
    def forward(self,x,h): z=torch.sigmoid(self.x2z(x)+self.h2z(h)); hc=torch.relu(self.x2h(x)+self.h2h(h)); return (1-z)*h+z*hc

class LiGRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LiGRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        B, T, _ = x.size()
        h = x.new_zeros(B, self.hidden_size)
        for t in range(T):
            h = self.cell(x[:, t, :], h)
        return self.fc(h)

def get_model(name, n_in, h, n_out):
    if name=="GRU": return GRUDecoder(n_in,h,n_out).to(DEVICE)
    if name=="LSTM": return LSTMDecoder(n_in,h,n_out).to(DEVICE)
    if name=="Linear": return LinearLagDecoder(n_in,h,n_out).to(DEVICE)
    if name=="LiGRU": return LiGRUDecoder(n_in,h,n_out).to(DEVICE)
    raise ValueError("Unknown decoder")

# ───────────────────────── Train / Eval ─────────────────────────
def train_model(model, X_train, Y_train, epochs, lr, batch_size=BATCH_SIZE_DEFAULT,
                num_workers=None, use_amp=True):
    if num_workers is None: num_workers = auto_num_workers()
    ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                       torch.as_tensor(Y_train, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    pin_memory=True, persistent_workers=(num_workers>0), prefetch_factor=4)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    use_cuda = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and use_cuda))

    dprint(f"[DBG][TRAIN] device={DEVICE} use_amp={use_amp and use_cuda} epochs={epochs} bs={batch_size} workers={num_workers}  n_batches≈{max(1,len(dl))}  X={tuple(X_train.shape)} Y={tuple(Y_train.shape)}")

    model.train()
    for ep in range(1, epochs+1):
        s=0.0; n=0
        for xb, yb in dl:
            xb=xb.to(DEVICE, non_blocking=True); yb=yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if use_amp and use_cuda:
                with torch.amp.autocast('cuda'):
                    pred = model(xb); loss = crit(pred, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(xb); loss = crit(pred, yb)
                loss.backward(); opt.step()
            s += loss.item(); n += 1
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"[TRAIN] Epoch {ep}/{epochs} - mean_loss={s/max(1,n):.6f}")
            sys.stdout.flush()
    return model

def compute_vaf_1d(y, yp):
    vt=np.var(y)
    return np.nan if vt<1e-12 else 1.0 - np.var(y-yp)/vt

def eval_decoder(model, X, Y, batch_size=BATCH_SIZE_DEFAULT, use_amp=True):
    model.eval(); preds=[]
    use_cuda = torch.cuda.is_available()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            bx = torch.as_tensor(X[i:i+batch_size], dtype=torch.float32).to(DEVICE, non_blocking=True)
            if use_amp and use_cuda:
                with torch.amp.autocast('cuda'):
                    out = model(bx)
            else:
                out = model(bx)
            preds.append(out.cpu().numpy())
    P = np.concatenate(preds, axis=0) if preds else np.empty((0,0))
    if P.size == 0:
        return np.nan, np.array([]), P
    ch = np.array([compute_vaf_1d(Y[:,c], P[:,c]) for c in range(Y.shape[1])], dtype=np.float32)
    return float(np.nanmean(ch)), ch, P

# ─────────────────── DimRed: PCA / UMAP (+ align) ───────────────────
def fit_dimred(X, method='pca', n_components=10, random_state=SEED):
    if method == 'pca':
        dprint(f"[DBG][DIMRED] fit pca on X={tuple(X.shape)}  n_components={n_components}  seed={random_state}")
        m = PCA(n_components=n_components, random_state=random_state); m.fit(X); return m
    elif method == 'umap':
        if umap is None:
            raise RuntimeError("umap-learn not installed. `pip install umap-learn`")
        dprint(f"[DBG][DIMRED] fit umap on X={tuple(X.shape)}  n_components={n_components}  seed={random_state}")
        m = umap.UMAP(n_components=n_components, random_state=random_state, n_neighbors=30, min_dist=0.1)
        m.fit(X); return m
    else:
        raise ValueError("dimred must be 'pca' or 'umap'")

def transform_dimred(model, X, method='pca'):
    if method in ('pca','umap'):
        dprint(f"[DBG][DIMRED] transform {method}: X={tuple(X.shape)}")
        return model.transform(X)
    raise ValueError("unknown dimred method")

def align_latent_pinv(Zx: np.ndarray, Z0: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    """Moore–Penrose in latent space (works for PCA/UMAP). Shapes must match (T x d)."""
    if Zx.shape != Z0.shape:
        raise ValueError(f"latent pinv requires same shapes, got {Zx.shape} vs {Z0.shape}")
    X = Zx - Zx.mean(axis=0, keepdims=True)
    Y = Z0 - Z0.mean(axis=0, keepdims=True)
    d = X.shape[1]
    A = np.linalg.solve(X.T @ X + lam*np.eye(d), X.T @ Y)
    return (Zx - Zx.mean(axis=0, keepdims=True)) @ A + Z0.mean(axis=0, keepdims=True)

def align_pca_basis(pca_train: PCA, pca_test: PCA, X_test: np.ndarray, n_comp: int) -> np.ndarray:
    """True change of basis for PCA: project test with its own PCA, then map into train basis."""
    V0 = pca_train.components_[:n_comp, :].T   # p x k (train)
    Vx = pca_test.components_[:n_comp, :].T    # p x k (test)
    R  = pinv(Vx) @ V0                         # k x k
    Z_local = pca_test.transform(X_test)[:, :n_comp]
    return Z_local @ R

# ───────────────────────── Scenarios (full) ─────────────────────────
def build_scenarios():
    s=[]
    # Within-monkey Jango/JacB (iso/wm/spr)
    for m in ["Jango","JacB"]:
        for tr in ["iso","wm","spr"]:
            others=[t for t in ["iso","wm","spr"] if t!=tr]
            s.append({
                "name": f"{m}_{tr}",
                "train_filter": lambda r,m=m,tr=tr: (r.get("monkey")==m) and (str(r.get("task","")).strip().lower()==tr),
                "tests": [{"name":o, "test_filter": lambda r,m=m,o=o: (r.get("monkey")==m) and (str(r.get("task","")).strip().lower()==o)} for o in others],
                "force_same_day": True
            })
    # Jaco/Theo (mgpt/ball)
    s.append({"name":"Jaco_mgpt","train_filter":lambda r:(r.get("monkey")=="Jaco") and (str(r.get("task","")).strip().lower() in ["mgpt","mg-pt"]),
              "tests":[{"name":"ball","test_filter":lambda r:(r.get("monkey")=="Jaco") and (str(r.get("task","")).strip().lower()=="ball")}], "force_same_day":True})
    s.append({"name":"Jaco_ball","train_filter":lambda r:(r.get("monkey")=="Jaco") and (str(r.get("task","")).strip().lower()=="ball"),
              "tests":[{"name":"mgpt","test_filter":lambda r:(r.get("monkey")=="Jaco") and (str(r.get("task","")).strip().lower() in ["mgpt","mg-pt"])}], "force_same_day":True})
    s.append({"name":"Theo_mgpt","train_filter":lambda r:(r.get("monkey")=="Theo") and (str(r.get("task","")).strip().lower() in ["mgpt","mg-pt"]),
              "tests":[{"name":"ball","test_filter":lambda r:(r.get("monkey")=="Theo") and (str(r.get("task","")).strip().lower()=="ball")}], "force_same_day":True})
    s.append({"name":"Theo_ball","train_filter":lambda r:(r.get("monkey")=="Theo") and (str(r.get("task","")).strip().lower()=="ball"),
              "tests":[{"name":"mgpt","test_filter":lambda r:(r.get("monkey")=="Theo") and (str(r.get("task","")).strip().lower() in ["mgpt","mg-pt"])}], "force_same_day":True})
    return s

def parse_test_task(tname):
    tasks = ['iso','wm','spr','mgpt','ball','mg-pt']
    t = tname.lower()
    for k in tasks:
        if k in t: return 'mgpt' if k=='mg-pt' else k
    return "unknown"

# ───────────────────────── Main ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--scenario', required=True, choices=[s["name"] for s in build_scenarios()])
    ap.add_argument('--train_day_idx', type=int, required=True)
    ap.add_argument('--decoder', required=True, choices=list(DECODER_CONFIG.keys()))
    ap.add_argument('--dimred', default='pca', choices=['pca','umap'])
    ap.add_argument('--align', default='none', choices=['none','latent','pca_basis'])
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--seeds', type=str, default='42', help='comma-separated seeds, e.g., "42,43,44"')
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--stride_mul', type=float, default=1.0, help='stride = max(1, int(stride_mul*K_LAG))')
    ap.add_argument('--num_workers', type=int, default=-1)
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--perf_mode', action='store_true')
    ap.add_argument('--bin_factor', type=int, default=1)
    ap.add_argument('--output_dir', default='results_emg_cross_task_strict')
    ap.add_argument('--debug', action='store_true', help='print rich diagnostics')
    args = ap.parse_args()

    # Flags
    global PERF_MODE, DEBUG
    if args.perf_mode: PERF_MODE = True
    DEBUG = bool(args.debug)
    os.makedirs(args.output_dir, exist_ok=True)

    # Initial info
    cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    print(f"[INFO] DEVICE={DEVICE}  PERF_MODE={PERF_MODE}  torch.cuda.is_available={torch.cuda.is_available()}  TF32={torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else 'N/A'}  CUDA_VISIBLE_DEVICES={cuda_env}")
    sys.stdout.flush()

    # Load & EMG mapping
    df = pd.read_pickle(args.input)
    print(f"[INFO] loaded df rows={len(df)} columns={list(df.columns)}")
    if "date" not in df.columns:
        raise RuntimeError("input pickle must contain a 'date' column")
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
    df, emg_cols = filter_and_map_emg(df)
    print(f"[INFO] EMG mapped -> unified cols (n={len(emg_cols)}): {emg_cols}")
    print(f"[INFO] unique monkeys={sorted(df['monkey'].dropna().unique().tolist()) if 'monkey' in df.columns else 'N/A'}")
    print(f"[INFO] unique tasks={sorted(df['task'].dropna().unique().tolist()) if 'task' in df.columns else 'N/A'}")
    sys.stdout.flush()

    # Global EMG sanity (DEBUG)
    if DEBUG:
        try:
            emg_presence = {col: 0 for col in emg_cols}
            for _, r in df.iterrows():
                E = r.get("EMG")
                if isinstance(E, pd.DataFrame) and not E.empty:
                    for i, col in enumerate(E.columns):
                        v = E.values[:, i]
                        if np.any(np.isfinite(v)) and np.nanmax(np.abs(v)) > 0:
                            emg_presence[col] += 1
            print("[SANITY][EMG] rows with nonzero amplitude per muscle:")
            for k in emg_cols:
                print(f"   {k:>6s}: {emg_presence[k]} rows")

            all_vars = []
            for _, r in df.iterrows():
                E = r.get("EMG")
                if isinstance(E, pd.DataFrame) and not E.empty:
                    all_vars.append(np.var(E.values, axis=0))
            if all_vars:
                V = np.vstack(all_vars)
                print(f"[SANITY][EMG] per-muscle global var (median ± IQR):")
                for i, k in enumerate(emg_cols):
                    med = float(np.nanmedian(V[:, i]))
                    q1, q3 = float(np.nanpercentile(V[:, i], 25)), float(np.nanpercentile(V[:, i], 75))
                    print(f"   {k:>6s}: med={med:.6f}  IQR=({q1:.6f}, {q3:.6f})")
        except Exception as e:
            print(f"[SANITY][EMG] print failed: {e}")

    # Scenario selection
    scen_map = {s["name"]: s for s in build_scenarios()}
    sc = scen_map[args.scenario]
    test_names = [t["name"] for t in sc["tests"]]
    print(f"[INFO] scenario={args.scenario}  force_same_day={sc.get('force_same_day', True)}  test_names={test_names}")

    df_train_task = df[df.apply(sc["train_filter"], axis=1)].copy()
    days = sorted(df_train_task["date"].dropna().unique())
    print(f"[INFO] candidate train days for {args.scenario}: count={len(days)}  values={list(days)}")

    # No rows/dates at all → skip & write empty results
    if len(df_train_task) == 0 or len(days) == 0:
        print(f"[WARN] No training data/dates for scenario={args.scenario} in {args.input}. Skipping.")
        out = os.path.join(
            args.output_dir,
            f"strict_{args.scenario}_day{args.train_day_idx}_{args.decoder}_{args.dimred}_{args.align}_kf{args.n_folds}_seeds{args.seeds}.pkl"
        )
        pd.to_pickle(pd.DataFrame([]), out)
        print(f"[INFO] saved empty results → {out}")
        return

    # Index valid?
    if not (0 <= args.train_day_idx < len(days)):
        print(f"[WARN] train_day_idx={args.train_day_idx} out of range for scenario={args.scenario} (available days: {len(days)}). Skipping.")
        out = os.path.join(
            args.output_dir,
            f"strict_{args.scenario}_day{args.train_day_idx}_{args.decoder}_{args.dimred}_{args.align}_kf{args.n_folds}_seeds{args.seeds}.pkl"
        )
        pd.to_pickle(pd.DataFrame([]), out)
        print(f"[INFO] saved empty results → {out}")
        return

    day = days[args.train_day_idx]
    df_train_day = df_train_task[df_train_task["date"] == day].copy()
    if df_train_day.empty:
        print("[ERROR] no training rows for selected scenario/day"); sys.exit(1)

    print(f"[INFO] selected TRAIN day={day} n_rows={len(df_train_day)}")
    if DEBUG:
        present_tasks = sorted(df_train_day['task'].dropna().unique().tolist()) if 'task' in df_train_day.columns else []
        print(f"[INFO] train-day tasks present={present_tasks}")
    sys.stdout.flush()

    # decoder config
    cfg = DECODER_CONFIG[args.decoder]
    N = cfg["N_PCA"]; K = cfg["K_LAG"]; H = cfg["HIDDEN"]
    LR = args.lr if args.lr is not None else cfg["LR"]
    E  = args.epochs if args.epochs is not None else cfg["EPOCHS"]
    STRIDE = max(1, int(args.stride_mul * K))
    WORKERS = auto_num_workers() if args.num_workers == -1 else args.num_workers
    USE_AMP = (not args.no_amp)

    print(f"[INFO] decoder={args.decoder}  N_PCA={N}  K_LAG={K}  H={H}  LR={LR}  EPOCHS={E}  STRIDE={STRIDE}  bin_factor={args.bin_factor}  BATCH={args.batch_size}  WORKERS={WORKERS}  USE_AMP={USE_AMP}")

    # units set for consistency
    ALL_UNITS = get_all_unit_names(df_train_day)
    print(f"[INFO] ALL_UNITS count={len(ALL_UNITS)}  sample={[ALL_UNITS[0], ALL_UNITS[1], '...', ALL_UNITS[-1]] if ALL_UNITS else []}")
    if len(ALL_UNITS)==0:
        print("[ERROR] no units found on train day"); sys.exit(1)

    # Build raw (train day) with cuts
    X0_raw, Y0_raw, cuts0 = build_continuous_dataset_raw(df_train_day, bin_factor=args.bin_factor,
                                                         all_units=ALL_UNITS, ref_emg_cols=emg_cols)
    if X0_raw.size == 0:
        print("[ERROR] empty raw on train day"); sys.exit(1)

    # Train-day EMG var zero check + unit snapshot (DEBUG)
    if DEBUG:
        try:
            y0v = np.var(Y0_raw, axis=0)
            zero_mask = (y0v < 1e-12) | ~np.isfinite(y0v)
            zlist = [emg_cols[i] if i < len(emg_cols) else f"ch{i}" for i, z in enumerate(zero_mask) if z]
            print(f"[TRAIN-DAY] EMG var summary: min={float(np.nanmin(y0v)):.6f}  median={float(np.nanmedian(y0v)):.6f}  max={float(np.nanmax(y0v)):.6f}")
            if zlist:
                print(f"[TRAIN-DAY] ZERO-variance EMG channels (will give NaN VAF): {zlist}")
            else:
                print("[TRAIN-DAY] No zero-variance EMG channels.")
        except Exception as e:
            print(f"[TRAIN-DAY] EMG var check failed: {e}")

        try:
            mfr = X0_raw.mean(axis=0)                  # mean counts/bin
            nzp = (X0_raw>0).mean(axis=0)*100.0        # % nonzero bins
            idx = np.argsort(mfr)
            print("[UNITS] 5 least active (mean count/bin):")
            for j in idx[:5]:
                u = ALL_UNITS[j] if j < len(ALL_UNITS) else f"u{j}"
                print(f"   {u:>10s}  mean={mfr[j]:.5f}  nz%={nzp[j]:.1f}")
            print("[UNITS] 5 most active:")
            for j in idx[-5:]:
                u = ALL_UNITS[j] if j < len(ALL_UNITS) else f"u{j}"
                print(f"   {u:>10s}  mean={mfr[j]:.5f}  nz%={nzp[j]:.1f}")
        except Exception as e:
            print(f"[UNITS] snapshot failed: {e}")

    # Seeds × K-fold strict CV
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()!='']
    print(f"\n[INFO] ===== Seeds {seeds} =====")
    results = []
    for seed in seeds:
        set_seed(seed)
        splits = time_kfold_splits(X0_raw.shape[0], args.n_folds)
        EMB = embargo_bins(K, bin_factor=args.bin_factor, bin_size=BIN_SIZE, smoothing=SMOOTHING, truncate=GAUSS_TRUNCATE)
        print(f"[DBG] K-fold splits={splits}  EMB={EMB}")

        for fold_idx, (val_start, val_end) in enumerate(splits):
            print(f"\n[INFO] --- Fold {fold_idx+1}/{args.n_folds}  val=[{val_start}:{val_end}) ---")

            # raw segments
            X_left_raw  = X0_raw[:val_start];   Y_left_raw  = Y0_raw[:val_start]
            X_val_raw   = X0_raw[val_start:val_end]; Y_val_raw = Y0_raw[val_start:val_end]
            X_right_raw = X0_raw[val_end:];     Y_right_raw = Y0_raw[val_end:]

            # basic shapes for raw (DEBUG)
            if DEBUG:
                dbg_stats(X_left_raw,  "LEFT_raw.X");  dbg_stats(Y_left_raw,  "LEFT_raw.Y")
                dbg_stats(X_val_raw,   "VAL_raw.X");   dbg_stats(Y_val_raw,   "VAL_raw.Y")
                dbg_stats(X_right_raw, "RIGHT_raw.X"); dbg_stats(Y_right_raw, "RIGHT_raw.Y")

            # per-segment preprocessing (NO bleed)
            # LEFT: keep everything except right embargo
            trimL, trimR = 0, EMB
            X_left_p, Y_left_p = preprocess_segment(X_left_raw, Y_left_raw, args.bin_factor) if len(X_left_raw) else (np.empty((0,)), np.empty((0,)))
            if len(X_left_p) > trimR:
                X_left_p, Y_left_p = X_left_p[:len(X_left_p)-trimR], Y_left_p[:len(Y_left_p)-trimR]
                cuts_left = adjust_cuts_for_segment(0, len(X_left_raw), cuts0, trim_left=trimL, trim_right=trimR, seg_len=len(X_left_raw))
            else:
                X_left_p = np.empty((0, X0_raw.shape[1] if X0_raw.size else 0), dtype=np.float32)
                Y_left_p = np.empty((0, Y0_raw.shape[1] if Y0_raw.size else 0), dtype=np.float32)
                cuts_left = []
            if DEBUG:
                dbg_stats(X_left_p, "LEFT_preproc.X"); dbg_stats(Y_left_p, "LEFT_preproc.Y")
                print(f"[DBG] cuts_left={cuts_left}")

            # VAL: trim both sides by EMB
            trimL, trimR = EMB, EMB
            X_val_p, Y_val_p = preprocess_segment(X_val_raw, Y_val_raw, args.bin_factor) if len(X_val_raw) else (np.empty((0,)), np.empty((0,)))
            if len(X_val_p) > (trimL + trimR):
                X_val_p, Y_val_p = X_val_p[trimL:len(X_val_p)-trimR], Y_val_p[trimL:len(Y_val_p)-trimR]
                cuts_val = adjust_cuts_for_segment(val_start, val_end, cuts0, trim_left=trimL, trim_right=trimR, seg_len=len(X_val_raw))
            else:
                X_val_p = np.empty((0, X0_raw.shape[1] if X0_raw.size else 0), dtype=np.float32)
                Y_val_p = np.empty((0, Y0_raw.shape[1] if Y0_raw.size else 0), dtype=np.float32)
                cuts_val = []
            if DEBUG:
                dbg_stats(X_val_p, "VAL_preproc.X"); dbg_stats(Y_val_p, "VAL_preproc.Y")
                print(f"[DBG] cuts_val={cuts_val}")

            # RIGHT: drop left embargo
            trimL, trimR = EMB, 0
            X_right_p, Y_right_p = preprocess_segment(X_right_raw, Y_right_raw, args.bin_factor) if len(X_right_raw) else (np.empty((0,)), np.empty((0,)))
            if len(X_right_p) > trimL:
                X_right_p, Y_right_p = X_right_p[trimL:], Y_right_p[trimL:]
                cuts_right = adjust_cuts_for_segment(val_end, len(X0_raw), cuts0, trim_left=trimL, trim_right=trimR, seg_len=len(X_right_raw))
            else:
                X_right_p = np.empty((0, X0_raw.shape[1] if X0_raw.size else 0), dtype=np.float32)
                Y_right_p = np.empty((0, Y0_raw.shape[1] if Y0_raw.size else 0), dtype=np.float32)
                cuts_right = []
            if DEBUG:
                dbg_stats(X_right_p, "RIGHT_preproc.X"); dbg_stats(Y_right_p, "RIGHT_preproc.Y")
                print(f"[DBG] cuts_right={cuts_right}")

            # TRAIN = LEFT ⊕ RIGHT (keep a cut at the junction)
            if X_left_p.size and X_right_p.size:
                X_tr_time = np.vstack([X_left_p, X_right_p])
                Y_tr_time = np.vstack([Y_left_p, Y_right_p])
                cuts_train = cuts_left + [len(X_left_p)] + [c + len(X_left_p) for c in cuts_right]
            elif X_left_p.size:
                X_tr_time, Y_tr_time, cuts_train = X_left_p, Y_left_p, cuts_left
            else:
                X_tr_time, Y_tr_time, cuts_train = X_right_p, Y_right_p, cuts_right

            print_split_banner(seed, fold_idx, EMB, K, STRIDE, len(X_tr_time), len(X_val_p))
            dbg_stats(X_tr_time, "TR_X_preproc"); dbg_stats(Y_tr_time, "TR_Y_preproc(EMG)")
            dbg_stats(X_val_p, "VAL_X_preproc");  dbg_stats(Y_val_p,  "VAL_Y_preproc(EMG)")

            if X_tr_time.shape[0] <= K or X_val_p.shape[0] <= K:
                print(f"[WARN] seed={seed} fold={fold_idx}: not enough samples after embargo; skipping")
                continue

            # Fit manifold on TRAIN-only
            dim_model_train = fit_dimred(X_tr_time, method=args.dimred, n_components=max(N,2), random_state=seed)
            if args.dimred == 'pca':
                evr = getattr(dim_model_train, "explained_variance_ratio_", None)
                if evr is not None:
                    kept = float(100*np.sum(evr[:N])); first = float(100*evr[0])
                    print(f"[DBG] PCA kept variance (first {N}) = {kept:.2f}%  first_comp={first:.2f}%")
            else:
                dprint("[DBG] UMAP fitted on train segment")

            Z_tr = transform_dimred(dim_model_train, X_tr_time, method=args.dimred)[:, :N]
            Z_va = transform_dimred(dim_model_train, X_val_p,   method=args.dimred)[:, :N]
            dbg_stats(Z_tr, "Z_tr"); dbg_stats(Z_va, "Z_val")

            is_linear = (args.decoder == "Linear")
            X_tr, Y_tr = build_seq_with_cuts(Z_tr, Y_tr_time, K, cuts_train, STRIDE, is_linear)
            X_va, Y_va = build_seq_with_cuts(Z_va, Y_val_p,   K, cuts_val,   STRIDE, is_linear)
            print(f"[DBG] windowed shapes: X_tr={X_tr.shape}  Y_tr={Y_tr.shape} | X_val={X_va.shape}  Y_val={Y_va.shape}")
            dbg_stats(Y_tr, "Y_tr_win(EMG)"); dbg_stats(Y_va, "Y_val_win(EMG)")

            m0, _ = baseline_mean_vaf(Y_va)
            print(f"[DBG] baseline(mean) VAF on VAL: {m0:.4f}")

            # Model
            n_in = (N*K) if is_linear else N
            n_out = Y_tr.shape[1]
            model = get_model(args.decoder, n_in, H, n_out)
            print(f"[INFO] seed={seed} fold={fold_idx+1}/{args.n_folds} - training {args.decoder} ...")
            t0=time.time()
            model = train_model(model, X_tr, Y_tr, epochs=E, lr=LR, batch_size=args.batch_size,
                                num_workers=WORKERS, use_amp=USE_AMP)
            print(f"[INFO] train time: {time.time()-t0:.1f}s")

            # Internal CV eval
            mVAF_val, vaf_ch_val, P_val = eval_decoder(model, X_va, Y_va, batch_size=args.batch_size, use_amp=USE_AMP)
            if DEBUG:
                print(f"[DBG][EVAL] batches≈{max(1, len(Y_va)//args.batch_size)}  X={X_va.shape} Y={Y_va.shape} use_amp={USE_AMP}")
                dbg_stats(P_val, "PRED(eval)")
                finite = np.isfinite(vaf_ch_val)
                if finite.any():
                    vmed = float(np.nanmedian(vaf_ch_val))
                    vmin = float(np.nanmin(vaf_ch_val))
                    vmax = float(np.nanmax(vaf_ch_val))
                    print(f"[DBG][EVAL] per-channel VAF summary: mean={mVAF_val:.4f}  med={vmed:.4f}  min={vmin:.4f}  max={vmax:.4f}")

                # Per-muscle VAF sorted + corr metrics
                try:
                    # Per-muscle list
                    order = np.argsort(np.where(np.isfinite(vaf_ch_val), vaf_ch_val, -np.inf))
                    print("[EVAL][CV] per-muscle VAF (sorted low→high):")
                    for idxc in order:
                        name = emg_cols[idxc] if idxc < len(emg_cols) else f"ch{idxc}"
                        v = vaf_ch_val[idxc]
                        vv = float(np.var(Y_va[:, idxc])) if Y_va.size else float('nan')
                        mn = float(np.mean(Y_va[:, idxc])) if Y_va.size else float('nan')
                        sd = float(np.std(Y_va[:, idxc]))  if Y_va.size else float('nan')
                        print(f"     {name:>6s}  VAF={v: .4f}  Var(Y)={vv:.6f}  mean(Y)={mn: .4f}  std(Y)={sd: .4f}")
                    print(f"[EVAL][CV] stats: mean={mVAF_val:.4f}  median={np.nanmedian(vaf_ch_val):.4f}  min={np.nanmin(vaf_ch_val):.4f}  max={np.nanmax(vaf_ch_val):.4f}  n_ch={Y_va.shape[1]}")
                    # Correlations
                    from scipy.stats import pearsonr
                    r_list, r2_list = [], []
                    # Use P_val we already computed
                    for c in range(Y_va.shape[1]):
                        y = Y_va[:, c]; p = P_val[:, c]
                        if np.var(y) < 1e-12 or not np.isfinite(y).all() or not np.isfinite(p).all():
                            r, r2 = np.nan, np.nan
                        else:
                            r = pearsonr(y, p)[0]; r2 = r*r
                        r_list.append(r); r2_list.append(r2)
                    print(f"[EVAL][CV] Pearson r per muscle: {['{:.3f}'.format(x) if np.isfinite(x) else 'nan' for x in r_list]}")
                    print(f"[EVAL][CV] R^2 per muscle:      {['{:.3f}'.format(x) if np.isfinite(x) else 'nan' for x in r2_list]}")
                    print(f"[EVAL][CV] mean r={np.nanmean(r_list):.3f}  mean R^2={np.nanmean(r2_list):.3f}")

                    # Variance ratio VAL/TR
                    v_tr = np.var(Y_tr, axis=0); v_va = np.var(Y_va, axis=0)
                    ratio = np.divide(v_va, v_tr, out=np.full_like(v_va, np.nan), where=(v_tr>0))
                    rows = []
                    for i in range(len(v_tr)):
                        name = emg_cols[i] if i < len(emg_cols) else f"ch{i}"
                        rows.append((name, float(v_tr[i]), float(v_va[i]), float(ratio[i])))
                    rows.sort(key=lambda t: (np.isnan(t[3]), t[3]))
                    print("[DIAG] Var(val)/Var(train) per muscle (sorted low→high):")
                    for name, vt, vv, rr in rows:
                        print(f"   {name:>6s}  Var_tr={vt:.6f}  Var_val={vv:.6f}  ratio={rr:.3f}")
                except Exception as e:
                    print(f"[EVAL][CV] extra metrics failed: {e}")

            results.append({
                "scenario": args.scenario, "train_day": str(pd.to_datetime(day).date()),
                "decoder": args.decoder, "dimred": args.dimred, "align": "internal_cv",
                "seed": seed, "fold": fold_idx, "split": [int(val_start), int(val_end)],
                "mean_VAF": float(mVAF_val), "per_channel_VAF": vaf_ch_val,
                "timestamp": datetime.datetime.now(),
                "K_LAG": K, "stride": STRIDE, "embargo_bins": EMB
            })
            print(f"[RESULT] internal CV fold={fold_idx} | mean_VAF={mVAF_val:.4f} | per-ch med={np.nanmedian(vaf_ch_val):.4f}")
            sys.stdout.flush()

            # ── Cross-task off-diagonal on SAME day (strict preprocessing within cuts)
            df_same_day = df[df["date"]==day].copy() if sc.get("force_same_day", True) else df
            for tcfg in sc["tests"]:
                test_name = tcfg["name"]
                print(f"\n[INFO] Cross-task test={test_name}  rows={len(df_same_day)} (filtered below)")
                df_test = df_same_day[df_same_day.apply(tcfg["test_filter"], axis=1)].copy()
                if df_test.empty:
                    print(f"[WARN] no test data for {test_name}")
                    continue

                # Build raw+cuts and preprocess per trial (no embargo needed for inference)
                ALL_UNITS_TEST = get_all_unit_names(df_test)
                if DEBUG:
                    print(f"[DBG] test units count={len(ALL_UNITS_TEST)}  sample={[ALL_UNITS_TEST[0], '...', ALL_UNITS_TEST[-1]] if ALL_UNITS_TEST else []}")
                X_raw_te, Y_raw_te, cuts_te = build_continuous_dataset_raw(df_test, bin_factor=args.bin_factor,
                                                                           all_units=ALL_UNITS_TEST, ref_emg_cols=emg_cols)
                if X_raw_te.size == 0:
                    print(f"[WARN] empty X for {test_name}")
                    continue
                X_te_proc, Y_te_proc = preprocess_within_cuts(X_raw_te, Y_raw_te, cuts_te, bin_factor=args.bin_factor)
                if DEBUG:
                    dbg_stats(X_te_proc, f"{test_name}:X_proc")
                    dbg_stats(Y_te_proc, f"{test_name}:Y_proc")

                # Direct projection via TRAIN manifold
                Zte_direct = transform_dimred(dim_model_train, X_te_proc, method=args.dimred)[:, :N]
                if DEBUG:
                    dbg_stats(Zte_direct, f"{test_name}:Z_direct")

                # Select alignment
                if args.align == 'none':
                    Zte = Zte_direct
                    align_label = 'none'
                elif args.align == 'latent':
                    dim_model_test = fit_dimred(X_te_proc, method=args.dimred, n_components=max(N,2), random_state=seed)
                    Zte_test = transform_dimred(dim_model_test, X_te_proc, method=args.dimred)[:, :N]
                    Zte = align_latent_pinv(Zte_test, Zte_direct, lam=1e-6)
                    align_label = 'latent'
                elif args.align == 'pca_basis':
                    if args.dimred != 'pca':
                        dim_model_test = fit_dimred(X_te_proc, method=args.dimred, n_components=max(N,2), random_state=seed)
                        Zte_test = transform_dimred(dim_model_test, X_te_proc, method=args.dimred)[:, :N]
                        Zte = align_latent_pinv(Zte_test, Zte_direct, lam=1e-6)
                        align_label = 'latent(fallback)'
                    else:
                        pca_test = fit_dimred(X_te_proc, method='pca', n_components=max(N,2), random_state=seed)
                        Zte = align_pca_basis(dim_model_train, pca_test, X_te_proc, n_comp=N)
                        align_label = 'pca_basis'
                else:
                    raise ValueError("unknown align option")

                if DEBUG:
                    dbg_stats(Zte, f"{test_name}:Z_aligned({align_label})")

                # Window with cut respect
                X_te, Y_te = build_seq_with_cuts(Zte, Y_te_proc, K, cuts_te, STRIDE, is_linear)
                print(f"[DBG] {test_name}: windows => X={X_te.shape}  Y={Y_te.shape}")
                if X_te.shape[0] == 0:
                    print(f"[WARN] empty windows for {test_name} ({args.align})")
                    continue

                mVAF, vaf_ch, P_te = eval_decoder(model, X_te, Y_te, batch_size=args.batch_size, use_amp=USE_AMP)
                if DEBUG:
                    print(f"[DBG][EVAL] batches≈{max(1, len(Y_te)//args.batch_size)}  X={X_te.shape} Y={Y_te.shape} use_amp={USE_AMP}")
                    dbg_stats(P_te, "PRED(eval)")
                    finite = np.isfinite(vaf_ch)
                    if finite.any():
                        vmed = float(np.nanmedian(vaf_ch))
                        vmin = float(np.nanmin(vaf_ch))
                        vmax = float(np.nanmax(vaf_ch))
                        print(f"[DBG][EVAL] per-channel VAF summary: mean={mVAF:.4f}  med={vmed:.4f}  min={vmin:.4f}  max={vmax:.4f}")

                    # Extra cross-task diagnostics
                    try:
                        # var ratio test/train
                        v_tr = np.var(Y_tr, axis=0); v_te = np.var(Y_te, axis=0)
                        ratio = np.divide(v_te, v_tr, out=np.full_like(v_te, np.nan), where=(v_tr>0))
                        rows = []
                        for i in range(len(v_tr)):
                            name = emg_cols[i] if i < len(emg_cols) else f"ch{i}"
                            rows.append((name, float(v_tr[i]), float(v_te[i]), float(ratio[i])))
                        rows.sort(key=lambda t: (np.isnan(t[3]), t[3]))
                        print(f"[DIAG][XTASK:{test_name}] Var(test)/Var(train) per muscle (sorted low→high):")
                        for name, vt, vv, rr in rows:
                            print(f"   {name:>6s}  Var_tr={vt:.6f}  Var_te={vv:.6f}  ratio={rr:.3f}")

                        # correlations
                        from scipy.stats import pearsonr
                        r_list, r2_list = [], []
                        for c in range(Y_te.shape[1]):
                            y = Y_te[:, c]; p = P_te[:, c]
                            if np.var(y) < 1e-12 or not np.isfinite(y).all() or not np.isfinite(p).all():
                                r, r2 = np.nan, np.nan
                            else:
                                r = pearsonr(y, p)[0]; r2 = r*r
                            r_list.append(r); r2_list.append(r2)
                        print(f"[EVAL][XTASK:{test_name}] Pearson r per muscle: {['{:.3f}'.format(x) if np.isfinite(x) else 'nan' for x in r_list]}")
                        print(f"[EVAL][XTASK:{test_name}] R^2 per muscle:      {['{:.3f}'.format(x) if np.isfinite(x) else 'nan' for x in r2_list]}")
                        print(f"[EVAL][XTASK:{test_name}] mean r={np.nanmean(r_list):.3f}  mean R^2={np.nanmean(r2_list):.3f}")
                    except Exception as e:
                        print(f"[EVAL][XTASK:{test_name}] extra metrics failed: {e}")

                results.append({
                    "scenario": args.scenario, "train_day": str(pd.to_datetime(day).date()),
                    "decoder": args.decoder, "dimred": args.dimred, "align": args.align,
                    "seed": seed, "fold": fold_idx,
                    "test_task": parse_test_task(test_name),
                    "mean_VAF": float(mVAF), "per_channel_VAF": vaf_ch,
                    "timestamp": datetime.datetime.now(),
                    "K_LAG": K, "stride": STRIDE, "embargo_bins": EMB
                })
                print(f"[RESULT] cross-task {test_name} | {args.dimred}+{args.align} | seed={seed} fold={fold_idx} | mean_VAF={mVAF:.4f} | per-ch med={np.nanmedian(vaf_ch):.4f}")
                sys.stdout.flush()

    # save all
    out = os.path.join(
        args.output_dir,
        f"strict_{args.scenario}_day{args.train_day_idx}_{args.decoder}_{args.dimred}_{args.align}_kf{args.n_folds}_seeds{','.join(map(str,seeds))}.pkl"
    )
    pd.to_pickle(pd.DataFrame(results), out)
    print(f"\n[INFO] saved → {out}")

if __name__ == "__main__":
    main()
