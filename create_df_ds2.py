#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

ALLOWED_TASKS = {"wm", "iso", "iso8", "mgpt", "ball", "spr"}
TASK_MAP = {
    "wm":"wm","working memory":"wm","workmem":"wm","wf":"wm",
    "iso":"iso","isometric":"iso",
    "iso8":"iso8","iso 8":"iso8",
    "mgpt":"mgpt","mg-pt":"mgpt","mg pt":"mgpt",
    "ball":"ball",
}
def canon_task(t):
    return TASK_MAP.get(str(t).strip().lower(), str(t).strip().lower())

# ---------- EMG label mapping ----------
GLOBAL_MUSCLE_MAP = {
    "FDSR":"FDS","FDSU":"FDS","FDSM":"FDS","FDS2":"FDS","FDS":"FDS",
    "FDPR":"FDP","FDPU":"FDP","FDPM":"FDP","FDP3":"FDP","FDP":"FDP",
    "ECR1":"ECR","ECR2":"ECR","ECRB":"ECR","ER":"ECR","EM":"ECR","ECR":"ECR",
    "ECU1":"ECU","ECU2":"ECU","EU":"ECU","ECU":"ECU",
    "FCR1":"FCR","FCR2":"FCR","FR":"FCR","FCR":"FCR",
    "FCU1":"FCU","FCU2":"FCU","FU":"FCU","FCU":"FCU",
    "EDC1":"EDC","EDC2":"EDC","EDCU":"EDC","EDC":"EDC",
}
def replace_emg_columns(emg_df, muscle_map):
    new_cols, counts = {}, {}
    for col in emg_df.columns:
        raw = str(col).strip().upper()
        base = muscle_map.get(raw, raw)
        counts[base] = counts.get(base, 0) + 1
        new_cols[f"{base}_{counts[base]}"] = emg_df[col]
    return pd.DataFrame(new_cols)

# ---------- MATLAB string helpers ----------
def read_uint16_string(ds):
    arr = np.array(ds).flatten()
    return ''.join(chr(int(x)) for x in arr if int(x) != 0)

def read_object_string(ds):
    arr = np.array(ds).flatten()
    out = []
    for elem in arr:
        if isinstance(elem, h5py.Reference):
            resolved = ds.file[elem][()]
            if getattr(resolved, "dtype", None) == np.uint16:
                out.append(''.join(chr(int(x)) for x in resolved.flatten() if int(x) != 0))
            elif isinstance(resolved, bytes):
                out.append(resolved.decode('utf-8'))
            else:
                out.append(str(resolved))
        elif isinstance(elem, bytes):
            out.append(elem.decode('utf-8'))
        else:
            if hasattr(elem, 'dtype') and elem.dtype == np.uint16:
                out.append(''.join(chr(int(x)) for x in elem.flatten() if int(x) != 0))
            else:
                out.append(str(elem))
    return out

def neuron_ids_to_cols(nid_data):
    nid = np.asarray(nid_data)
    if nid.ndim == 2:
        if nid.shape[0] == 2: nid = nid[0, :]
        elif nid.shape[1] == 2: nid = nid[:, 0]
    return [f"neuron{int(x)}" for x in np.asarray(nid).flatten()]

# ---------- Task inference ----------
def infer_task_from_cell(sub_group):
    """Return one of {'wm','iso','iso8','mgpt','ball','unknown'}."""
    # 1) Trial table textual clues
    try:
        ttl = [s.lower() for s in read_object_string(sub_group['trialtablelabels'])]
        if any(("work" in s and "mem" in s) or "working memory" in s or "wm" in s for s in ttl):
            return "wm"
    except Exception:
        pass

    # 2) Number of target directions -> iso vs iso8
    try:
        tdir = np.asarray(sub_group['trial_target_dir'][()]).flatten()
        if tdir.size > 0:
            uniq = np.unique(np.round(tdir, 1))
            n_dirs = len(uniq)
            if 7 <= n_dirs <= 9:
                return "iso8"
            if 2 <= n_dirs <= 6:
                return "iso"
    except Exception:
        pass

    # 3) Weak clue via forcelabels
    try:
        fl = [s.lower() for s in read_object_string(sub_group['forcelabels'])]
        if any("ball" in s for s in fl):
            return "ball"
    except Exception:
        pass

    return "unknown"

# ---------- Core MAT -> DataFrame ----------
def build_dataframe(mat_file_path, verbose=True):
    rows = []
    with h5py.File(mat_file_path, 'r') as f:
        datasets_ds = f['datasets']
        drefs = list(np.array(datasets_ds).flatten())
        if verbose:
            print(f"[INFO] dataset structs: {len(drefs)}")

        for d_idx, dref in enumerate(drefs, 1):
            g = f[dref]
            date_str  = read_uint16_string(g['date'])
            date_only = date_str[:-4] if date_str.endswith('.mat') else date_str

            if "/" in date_only and len(date_only.split("/")) == 3:
                year, month, day = date_only.split("/")
            elif len(date_only) == 8 and date_only.isdigit():
                year, month, day = date_only[:4], date_only[4:6], date_only[6:]
            else:
                year = month = day = None

            monkey = read_uint16_string(g['monkey'])
            provided_labels = [canon_task(x) for x in read_object_string(g['labels'])]

            bd = g['binned_data']
            flat_cells = np.array(bd).flatten()
            n_cells = flat_cells.size

            # Build a multiset of provided labels but keep only allowed ones
            provided_counts = Counter(t for t in provided_labels if t in ALLOWED_TASKS)

            if verbose:
                print(f"\n[DATASET #{d_idx}] {date_only}  monkey={monkey}")
                print(f"  Provided labels (raw): {read_object_string(g['labels'])}")
                print(f"  Provided labels (canon, counts): {dict(provided_counts)}")
                if len(provided_labels) != n_cells:
                    print(f"  [WARN] labels length ({len(provided_labels)}) != cells ({n_cells})")

            # Pass 1: collect per-cell info + inference
            cell_infos = []
            for k, sub_ref in enumerate(flat_cells):
                sub = f[sub_ref]
                inferred = infer_task_from_cell(sub)

                timeframe = np.array(sub['timeframe']).flatten()
                bin_width = float(timeframe[1] - timeframe[0]) if timeframe.size >= 2 else 0.02

                # EMG
                emg_data  = np.array(sub['emgdatabin'])
                emg_guide = read_object_string(sub['emgguide'])
                emg_samples = emg_data if (emg_data.ndim==2 and emg_data.shape[1]==len(emg_guide)) else emg_data.T
                emg_df = pd.DataFrame(emg_samples, columns=emg_guide)
                emg_df = replace_emg_columns(emg_df, GLOBAL_MUSCLE_MAP)

                # Spikes: Hz -> counts
                spk_rate  = np.array(sub['spikeratedata'])
                neuronIDs = neuron_ids_to_cols(np.array(sub['neuronIDs']))
                spk_samples = spk_rate if (spk_rate.ndim==2 and spk_rate.shape[1]==len(neuronIDs)) else spk_rate.T
                spk_counts = np.rint(np.clip(spk_samples, 0, None) * bin_width).astype(np.int32)
                spike_df = pd.DataFrame(spk_counts, columns=neuronIDs)
                keep = [f"neuron{i}" for i in range(1,97) if f"neuron{i}" in spike_df.columns]
                if keep: spike_df = spike_df[keep]

                # Optional force
                try:
                    force_labels = read_object_string(sub['forcelabels'])
                    force_data   = np.array(sub['forcedatabin'])
                    if force_data.ndim==2 and force_data.shape[1]==len(force_labels):
                        force_df = pd.DataFrame(force_data, columns=force_labels)
                    elif force_data.ndim==2 and force_data.shape[0]==len(force_labels):
                        force_df = pd.DataFrame(force_data.T, columns=force_labels)
                    else:
                        force_df = None
                except Exception:
                    force_df = None

                # Trial start (optional)
                try:
                    tt_labels = read_object_string(sub['trialtablelabels'])
                    tt = np.array(sub['trialtable'])
                    idx_ts = next((i for i,s in enumerate(tt_labels) if "trial start" in str(s).lower()), None)
                    trial_start_time = tt[idx_ts,:] if (idx_ts is not None and tt.ndim==2) else None
                except Exception:
                    trial_start_time = None

                # Target dirs (optional)
                try:
                    trial_target_dir = np.asarray(sub['trial_target_dir'][()]).flatten()
                except Exception:
                    trial_target_dir = None

                # Keep raw label-at-index as a *weak* hint, but do not trust it blindly
                raw_label_at_index = canon_task(provided_labels[k]) if k < len(provided_labels) else None

                cell_infos.append(dict(
                    idx=k, inferred=inferred, raw_label_at_index=raw_label_at_index,
                    bin_width=bin_width,
                    EMG=emg_df, spike_counts=spike_df, force=force_df,
                    timeframe=timeframe, trial_start_time=trial_start_time,
                    trial_target_dir=trial_target_dir
                ))

            # Pass 2: assign tasks using the multiset conservatively
            assigned_counts = Counter()
            for ci in cell_infos:
                assigned = None
                reason = None

                # a) If inference is confident and available in pool -> take it
                if ci["inferred"] in ALLOWED_TASKS and provided_counts[ci["inferred"]] > 0:
                    assigned = ci["inferred"]; reason = "inferred"
                # b) else try raw label at this index if available in pool
                elif ci["raw_label_at_index"] in ALLOWED_TASKS and provided_counts[ci["raw_label_at_index"]] > 0:
                    assigned = ci["raw_label_at_index"]; reason = "raw_index_fallback"
                # c) else if pool has exactly one label left, take it
                elif sum(provided_counts.values()) == 1:
                    assigned = next(t for t,c in provided_counts.items() if c>0)
                    reason = "only_choice_left"
                # d) else we give up → unknown (will print)
                else:
                    assigned = "unknown"; reason = "unknown"

                ci["task"] = assigned
                ci["task_reason"] = reason
                assigned_counts[assigned] += 1
                if assigned in provided_counts and assigned in ALLOWED_TASKS:
                    provided_counts[assigned] -= 1

            # Report per-dataset reconciliation
            if verbose:
                provided_final = Counter(t for t in provided_labels if t in ALLOWED_TASKS)
                print(f"  Assigned counts: {dict(assigned_counts)}")
                # show any unknowns with quick context
                unk = [ci for ci in cell_infos if ci["task"]=="unknown"]
                if unk:
                    print("  [WARN] unknown cells:")
                    for ci in unk:
                        print(f"    cell#{ci['idx']}: inferred={ci['inferred']} raw_idx={ci['raw_label_at_index']}  EMG{ci['EMG'].shape}  SPIKES{ci['spike_counts'].shape}")

            # Materialize rows
            for ci in cell_infos:
                rows.append({
                    "year": year, "month": month, "day": day,
                    "date": date_only, "monkey": monkey, "task": ci["task"],
                    "task_inferred": ci["inferred"], "task_reason": ci["task_reason"],
                    "EMG": ci["EMG"], "spike_counts": ci["spike_counts"],
                    "bin_width": ci["bin_width"], "time_frame": ci["timeframe"],
                    "force": ci["force"],
                    "trial_start_time": ci["trial_start_time"],
                    "trial_target_dir": ci["trial_target_dir"],
                })

    if not rows:
        print("[ERROR] No rows parsed from MAT file.")
        return pd.DataFrame()

    # unify EMG columns
    all_emg_cols = sorted({c for r in rows for c in r["EMG"].columns})
    for r in rows:
        r["EMG"] = r["EMG"].reindex(columns=all_emg_cols, fill_value=0)

    df = pd.DataFrame(rows)
    try:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception:
        pass

    print("\n===== SUMMARY =====")
    print("Monkeys:", sorted(df["monkey"].astype(str).unique()))
    tasks = sorted(df["task"].astype(str).unique())
    print("Tasks  :", tasks)
    print(df["task"].value_counts())
    if not df.empty:
        print("Example EMG shape:", df.iloc[0]["EMG"].shape)
        print("Example Spike shape:", df.iloc[0]["spike_counts"].shape)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="all_manifold_datasets.mat")
    ap.add_argument("--out", default="output.pkl")
    ap.add_argument("--no-verbose", action="store_true")
    args = ap.parse_args()

    df = build_dataframe(args.mat, verbose=not args.no_verbose)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_pickle(args.out)
    print(f"\n[OK] Saved → {args.out}")

if __name__ == "__main__":
    main()
