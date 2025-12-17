# -*- coding: utf-8 -*-
"""
Batch GLM forward selection per session with one-sided Wilcoxon (spike-normalized LLHI),
UPDATED TO PURE POISSON (count GLM) AT 50 Hz (20 ms bins).

NEW in this version:
0) Before forward selection, fit FULL model (VARS_ALL) for ALL neurons (10-fold),
   and save FULL coefficients (incl. intercept) under:
     <session>/FULL_FIT/<ModelKey>/neuron_k/weights_mean.csv
     <session>/FULL_FIT/<ModelKey>/neuron_k/fold1/weights.csv ... fold10/weights.csv

Forward selection remains unchanged:
- only stores weights for accepted steps under <session>/<ModelKey>/neuron_k/fold*/weights.csv
"""

from __future__ import annotations

from datetime import datetime
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set

import h5py
import numpy as np
import pandas as pd

from dataclasses import dataclass
from scipy import sparse
from scipy.stats import wilcoxon
from scipy.ndimage import gaussian_filter1d

# Plotting (headless-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import PoissonRegressor
from joblib import Parallel, delayed
from tqdm import tqdm


# ===============================
# Configuration
# ===============================
# Input roots
IMU_ROOT      = Path(r"D:\Jiaqi\Projects\IMU_Preprocess\IMU_results")
SPIKE_ROOT    = Path(r"D:\Jiaqi\Projects\GLM_File\spike_binary")
DLC_ROOT      = Path(r"D:\Jiaqi\Projects\DLC_results_features")
POSITION_ROOT = Path(r"D:\Jiaqi\Projects\ACC_DATA\DLC_Process\position_50hz")

# Output root
WEIGHTS_BASE  = Path("weights_Poisson_forward")
WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)

# NEW: where to store full-model fits inside each session output
FULL_FIT_DIRNAME = "FULL_FIT"

# Parallel / CV
N_JOBS   = 29
SEED     = 0
CV_FOLDS = 10

# Bin definitions
BIN_MS   = 20
FS_HZ    = 50.0
BASE_FS  = 200.0
AGG_FACTOR = int(BASE_FS / FS_HZ)  # 4
MAX_MISMATCH_FRAMES_50HZ = 5       # tolerance in 50 Hz frames

# Forward-selection test threshold
ALPHA = 0.05

# PoissonRegressor params
MAX_ITER = 500
POISSON_ALPHA = 1e-6  # IMPORTANT: small alpha to avoid over-shrinking for one-hot high-dim X

# Candidate variable set
VARS_ALL = ["Position", "Speed", "roll", "yaw", "pitch"]

# Discretization bins
POSITION_CELL_CM = 8.0
SPEED_N_BINS = 20
ANGLE_N_BINS = 15  # roll/yaw/pitch bins

# Fitting-curve plots
PLOT_SMOOTH_MS = 1000
PLOT_START_SEC = 0.0
PLOT_END_SEC   = 600.0
PLOT_ZSCORE    = False


# ===============================
# Session enumeration utilities
# ===============================
def list_sessions_imu(root: Path) -> Set[str]:
    if not root.exists():
        return set()
    out = set()
    for sess_dir in root.iterdir():
        if not sess_dir.is_dir():
            continue
        stem = sess_dir.name
        f = sess_dir / f"{stem}_IMU_features.csv"
        if f.exists():
            out.add(stem)
    return out


def list_sessions_spike(root: Path) -> Set[str]:
    if not root.exists():
        return set()
    return {f.stem.replace("_200Hz", "") for f in root.glob("*_200Hz.h5")}


def list_sessions_dlc_final(root: Path) -> Set[str]:
    if not root.exists():
        return set()
    out = set()
    for sess_dir in root.iterdir():
        if not sess_dir.is_dir():
            continue
        stem = sess_dir.name
        f1 = sess_dir / f"final_filtered_{stem}_50hz.csv"
        if f1.exists():
            out.add(stem)
    return out


def list_sessions_position(root: Path) -> Set[str]:
    if not root.exists():
        return set()
    return {s.stem.replace("positions_", "") for s in root.glob("positions_*.csv")}


def session_paths(session: str) -> Dict[str, Path]:
    return {
        "imu":       IMU_ROOT / session / f"{session}_IMU_features.csv",
        "spike":     SPIKE_ROOT / f"{session}_200Hz.h5",
        "dlc_final": DLC_ROOT / session / f"final_filtered_{session}_50hz.csv",
        "position":  POSITION_ROOT / f"positions_{session}.csv",
    }


def is_session_done(session: str) -> bool:
    out_dir = WEIGHTS_BASE / session
    if not out_dir.exists():
        return False
    if (out_dir / "_SUCCESS").exists():
        return True
    sel = out_dir / "selected_models.csv"
    if sel.exists():
        try:
            df = pd.read_csv(sel)
            if df.shape[0] > 0:
                return True
        except Exception:
            pass
    return False


# ===============================
# Core utilities: discretization & design matrix
# ===============================
def bin_col(vals, n_bins: int, vmin=None, vmax=None) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float32)
    if vmin is None:
        vmin = np.nanmin(vals)
    if vmax is None:
        vmax = np.nanmax(vals)
    edges = np.linspace(vmin, vmax, n_bins + 1, dtype=np.float32)
    out = np.digitize(vals, edges) - 1
    out = np.clip(out, 0, n_bins - 1)
    return out.astype(np.int32)


def build_position_index(head_x_cm, head_y_cm) -> Tuple[np.ndarray, int]:
    cell = float(POSITION_CELL_CM)
    x_bin = (np.asarray(head_x_cm, dtype=np.float32) // cell).astype(int)
    y_bin = (np.asarray(head_y_cm, dtype=np.float32) // cell).astype(int)

    uniq = (
        pd.DataFrame({"x_bin": x_bin, "y_bin": y_bin})
        .drop_duplicates()
        .reset_index(drop=True)
    )
    uniq["pos_idx"] = np.arange(len(uniq), dtype=int)

    pos_idx = (
        pd.DataFrame({"x_bin": x_bin, "y_bin": y_bin})
        .merge(uniq, on=["x_bin", "y_bin"], how="left")["pos_idx"]
        .to_numpy(dtype=np.int32)
    )
    return pos_idx, int(uniq.shape[0])


def build_design_matrix(selected_vars: List[str], data_dict: Dict[str, np.ndarray]) -> Tuple[sparse.csr_matrix, List[str]]:
    cols, cats, order = [], [], []

    if "Position" in selected_vars:
        cols.append(data_dict["position"].astype(np.int32))
        cats.append(np.arange(data_dict["n_pos"], dtype=int))
        order.append("position")

    if "Speed" in selected_vars:
        cols.append(data_dict["head_v_bin"].astype(np.int32))
        cats.append(np.arange(SPEED_N_BINS, dtype=int))
        order.append("head_v")

    for ang in ["roll", "yaw", "pitch"]:
        if ang in selected_vars:
            cols.append(data_dict[f"{ang}_bin"].astype(np.int32))
            cats.append(np.arange(ANGLE_N_BINS, dtype=int))
            order.append(ang)

    if len(cols) == 0:
        X_zero = sparse.csr_matrix((len(data_dict["position"]), 0), dtype=np.float32)
        return X_zero, ["intercept"]

    cat_df = pd.DataFrame({name: col for name, col in zip(order, cols)})

    try:
        encoder = OneHotEncoder(
            categories=cats,
            sparse_output=True,
            handle_unknown="ignore",
            drop="first",
        )
    except TypeError:
        encoder = OneHotEncoder(
            categories=cats,
            sparse=True,
            handle_unknown="ignore",
            drop="first",
        )

    X_cat = encoder.fit_transform(cat_df).astype(np.float32).tocsr()
    feat_cat = encoder.get_feature_names_out(order).tolist()
    feature_names = feat_cat + ["intercept"]
    return X_cat, feature_names


def ensure_feature_mapping(model_dir: str, feature_names: List[str]):
    os.makedirs(model_dir, exist_ok=True)
    map_path = os.path.join(model_dir, "feature_mapping.txt")
    with open(map_path, "w", encoding="utf-8") as f:
        for j, nm in enumerate(feature_names):
            f.write(f"{j}: {nm}\n")


def model_key_from_vars(var_list: List[str]) -> str:
    return "_".join(var_list)


# ===============================
# Poisson LLHI + Wilcoxon utilities
# ===============================
def poisson_ll_noconst(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mu = np.clip(mu, 1e-12, None)
    return float(np.sum(y * np.log(mu) - mu))


def compute_llhi_bps_poisson(y_cnt: np.ndarray, mu_pred: np.ndarray) -> float:
    y = np.asarray(y_cnt, dtype=np.float64).ravel()
    mu = np.asarray(mu_pred, dtype=np.float64).ravel()
    if y.size == 0:
        return float("nan")
    mu0 = np.full_like(y, fill_value=np.mean(y), dtype=np.float64)

    ll_m = poisson_ll_noconst(y, mu)
    ll_b = poisson_ll_noconst(y, mu0)

    nsp = float(np.sum(y))
    if nsp <= 0:
        return float("nan")
    return (ll_m - ll_b) / (nsp * np.log(2))


def dll_bits_series_poisson(y_cnt: np.ndarray, mu_pred: np.ndarray) -> np.ndarray:
    y = np.asarray(y_cnt, dtype=np.float64).ravel()
    mu = np.asarray(mu_pred, dtype=np.float64).ravel()
    if y.size == 0:
        return np.array([], dtype=np.float32)

    mu = np.clip(mu, 1e-12, None)
    mean_rate = float(np.mean(y))
    mean_rate = max(mean_rate, 1e-12)

    ll_m = y * np.log(mu) - mu
    ll_b = y * np.log(mean_rate) - mean_rate
    dll = ll_m - ll_b
    return (dll / np.log(2)).astype(np.float32)


def wilcoxon_greater(a: np.ndarray, b: np.ndarray = None) -> Tuple[float, float, int]:
    if b is None:
        x = np.asarray(a, dtype=np.float64)
    else:
        x = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or np.allclose(x, 0):
        return 0.0, 1.0, 0
    stat, p = wilcoxon(
        x,
        alternative="greater",
        zero_method="wilcox",
        correction=False,
        mode="auto",
    )
    return float(stat), float(p), int(x.size)


# ===============================
# IO: load spikes (200Hz) -> counts at 50Hz, load covariates at 50Hz
# ===============================
def load_spikes_50hz_counts(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as hf:
        Y200 = hf["spike_binary"][:].astype(np.int16)  # (T200, N)

    T200, N = Y200.shape
    T200_trim = (T200 // AGG_FACTOR) * AGG_FACTOR
    if T200_trim <= 0:
        raise ValueError("Spike length too short after trimming.")

    Y200 = Y200[:T200_trim]
    Y50 = Y200.reshape(-1, AGG_FACTOR, N).sum(axis=1)  # (T50, N)
    return Y50.astype(np.int32)


def rebuild_inputs_50hz(session: str, paths: Dict[str, Path]) -> Dict[str, np.ndarray]:
    pos_df = pd.read_csv(paths["position"], usecols=["head_x", "head_y", "heading_deg"]).astype(np.float32)
    dlc_df = pd.read_csv(paths["dlc_final"], usecols=["head_v"]).astype(np.float32)
    imu_df = pd.read_csv(paths["imu"], usecols=["roll", "yaw", "pitch"]).astype(np.float32)

    yaw_rad = np.deg2rad(pos_df["heading_deg"].to_numpy(dtype=np.float32)).astype(np.float32)

    L = min(len(pos_df), len(dlc_df), len(imu_df), len(yaw_rad))
    pos_df = pos_df.iloc[:L].reset_index(drop=True)
    dlc_df = dlc_df.iloc[:L].reset_index(drop=True)
    imu_df = imu_df.iloc[:L].reset_index(drop=True)
    yaw_rad = yaw_rad[:L]

    imu_df["yaw"] = yaw_rad

    imu_df["yaw"] = np.mod(imu_df["yaw"].values, 2 * np.pi)
    imu_df["pitch"] = imu_df["pitch"].values + (np.pi / 2)
    imu_df["roll"] = np.mod(imu_df["roll"].values, 2 * np.pi)

    pos_idx, n_pos = build_position_index(pos_df["head_x"].values, pos_df["head_y"].values)

    head_v_bin = bin_col(dlc_df["head_v"].values, n_bins=SPEED_N_BINS)
    roll_bin   = bin_col(imu_df["roll"].values,  n_bins=ANGLE_N_BINS, vmin=0, vmax=2 * np.pi)
    yaw_bin    = bin_col(imu_df["yaw"].values,   n_bins=ANGLE_N_BINS, vmin=0, vmax=2 * np.pi)
    pitch_bin  = bin_col(imu_df["pitch"].values, n_bins=ANGLE_N_BINS, vmin=0, vmax=np.pi)

    return {
        "T": int(L),
        "position": pos_idx.astype(np.int32),
        "n_pos": int(n_pos),
        "head_v_bin": head_v_bin.astype(np.int32),
        "roll_bin": roll_bin.astype(np.int32),
        "yaw_bin": yaw_bin.astype(np.int32),
        "pitch_bin": pitch_bin.astype(np.int32),
    }


# ===============================
# Plotting: fitting curve (OOF)
# ===============================
def plot_fitting_curve(
    out_png: Path,
    title: str,
    y_cnt: np.ndarray,
    mu_cnt: np.ndarray,
    *,
    bin_ms: int = BIN_MS,
    smooth_ms: float = PLOT_SMOOTH_MS,
    start_sec: float = PLOT_START_SEC,
    end_sec: float = PLOT_END_SEC,
    do_zscore: bool = PLOT_ZSCORE,
    zscore_eps: float = 1e-8,
):
    y_cnt = np.asarray(y_cnt, dtype=np.float64).ravel()
    mu_cnt = np.asarray(mu_cnt, dtype=np.float64).ravel()
    assert y_cnt.shape == mu_cnt.shape

    bin_sec = bin_ms / 1000.0
    y_rate = y_cnt / bin_sec
    mu_rate = mu_cnt / bin_sec

    sigma_bins = float(smooth_ms) / float(bin_ms)
    if sigma_bins > 0:
        y_s = gaussian_filter1d(y_rate.astype(np.float32), sigma=sigma_bins)
        mu_s = gaussian_filter1d(mu_rate.astype(np.float32), sigma=sigma_bins)
    else:
        y_s = y_rate.astype(np.float32)
        mu_s = mu_rate.astype(np.float32)

    t = np.arange(len(y_s), dtype=np.float64) * bin_sec
    s0 = max(0, int(np.floor(start_sec / bin_sec)))
    s1 = min(len(y_s), int(np.ceil(end_sec / bin_sec))) if end_sec is not None else len(y_s)
    if s1 <= s0:
        return

    y_w = y_s[s0:s1]
    mu_w = mu_s[s0:s1]
    t_w = t[s0:s1]

    if do_zscore:
        def _z(x: np.ndarray) -> np.ndarray:
            m = float(np.mean(x))
            sd = float(np.std(x))
            return (x - m) / max(sd, zscore_eps)

        y_plot = _z(y_w)
        mu_plot = _z(mu_w)
        ylab = "Z-score (smoothed rate)"
        lab_true = "True rate (z-scored)"
        lab_pred = "Pred rate (z-scored)"
    else:
        y_plot = y_w
        mu_plot = mu_w
        ylab = "Spikes/s (smoothed)"
        lab_true = "True rate (smoothed)"
        lab_pred = "Pred rate (smoothed)"

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 4))
    plt.plot(t_w, y_plot, label=lab_true, linewidth=2)
    plt.plot(t_w, mu_plot, label=lab_pred, linewidth=1.6, alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def load_oof_from_neuron_dir(neuron_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    fold_dirs = sorted([p for p in neuron_dir.glob("fold*") if p.is_dir()])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold dirs under {neuron_dir}")

    max_idx = -1
    parts = []
    for fd in fold_dirs:
        h5p = fd / "pred.h5"
        with h5py.File(h5p, "r") as hf:
            va_idx = hf["va_idx"][:].astype(np.int64)
            pred_mu = hf["pred_mu"][:].astype(np.float64)
            true_cnt = hf["true_cnt"][:].astype(np.float64)
        max_idx = max(max_idx, int(np.max(va_idx)))
        parts.append((va_idx, pred_mu, true_cnt))

    T = max_idx + 1
    mu_oof = np.full(T, np.nan, dtype=np.float64)
    y_oof = np.full(T, np.nan, dtype=np.float64)

    for va_idx, pred_mu, true_cnt in parts:
        mu_oof[va_idx] = pred_mu
        y_oof[va_idx] = true_cnt

    if np.any(~np.isfinite(mu_oof)) or np.any(~np.isfinite(y_oof)):
        m = np.nanmean(y_oof)
        mu_oof = np.where(np.isfinite(mu_oof), mu_oof, m)
        y_oof = np.where(np.isfinite(y_oof), y_oof, m)

    return y_oof, mu_oof


# ===============================
# Training helpers
# ===============================
def fit_predict_one_fold_poisson(
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
) -> Tuple[np.ndarray, float]:
    Xtr, Xva = X_all[tr_idx], X_all[va_idx]
    ytr, yva = y_all[tr_idx].astype(np.float64), y_all[va_idx].astype(np.float64)

    mean_tr = float(np.mean(ytr))
    if mean_tr <= 0:
        mu_va = np.full_like(yva, 1e-12, dtype=np.float64)
        llhi = compute_llhi_bps_poisson(yva, mu_va)
        return mu_va.astype(np.float32), float(llhi)

    mdl = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=MAX_ITER, fit_intercept=True)
    mdl.fit(Xtr, ytr)
    mu_va = np.clip(mdl.predict(Xva).astype(np.float64), 1e-12, None)

    llhi = compute_llhi_bps_poisson(yva, mu_va)
    return mu_va.astype(np.float32), float(llhi)


def _fit_one_fold_weights_poisson(
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
) -> np.ndarray:
    """Return w = [coef..., intercept] for one fold (fit on train only)."""
    Xtr = X_all[tr_idx]
    ytr = y_all[tr_idx].astype(np.float64)

    mean_tr = float(np.mean(ytr))
    if mean_tr <= 0:
        w = np.zeros(Xtr.shape[1] + 1, dtype=np.float32)
        w[-1] = np.log(1e-12)
        return w

    mdl = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=MAX_ITER, fit_intercept=True)
    mdl.fit(Xtr, ytr)
    w = np.concatenate(
        [mdl.coef_.ravel().astype(np.float32), np.array([mdl.intercept_], dtype=np.float32)]
    )
    return w


def save_neuron_artifacts_for_model(
    model_vars: List[str],
    model_dir: Path,
    neuron_dir: Path,
    neuron_index: int,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    feature_names: List[str],
) -> Dict:
    neuron_dir.mkdir(parents=True, exist_ok=True)
    ensure_feature_mapping(str(model_dir), feature_names)

    fold_llhi: List[float] = []
    mu_oof = np.full_like(y_all, np.nan, dtype=np.float32)

    for k, (tr, va) in enumerate(folds, start=1):
        fold_dir = neuron_dir / f"fold{k}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        Xtr, Xva = X_all[tr], X_all[va]
        ytr, yva = y_all[tr].astype(np.float64), y_all[va].astype(np.float64)

        mean_tr = float(np.mean(ytr))
        if mean_tr <= 0:
            mu_va = np.full_like(yva, 1e-12, dtype=np.float64)
            w = np.zeros(Xtr.shape[1] + 1, dtype=np.float32)
            w[-1] = np.log(1e-12)
        else:
            mdl = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=MAX_ITER, fit_intercept=True)
            mdl.fit(Xtr, ytr)
            mu_va = np.clip(mdl.predict(Xva).astype(np.float64), 1e-12, None)
            w = np.concatenate(
                [mdl.coef_.ravel().astype(np.float32), np.array([mdl.intercept_], dtype=np.float32)]
            )

        pd.DataFrame(
            w.reshape(1, -1),
            index=[f"neuron_{neuron_index+1}"],
            columns=feature_names,
        ).to_csv(fold_dir / "weights.csv")

        with h5py.File(fold_dir / "pred.h5", "w") as hf:
            hf.create_dataset("pred_mu", data=mu_va.astype(np.float32), compression="gzip")
            hf.create_dataset("true_cnt", data=yva.astype(np.float32), compression="gzip")
            hf.create_dataset("va_idx", data=np.asarray(va, dtype=np.int64), compression="gzip")

        llhi_val = compute_llhi_bps_poisson(yva, mu_va)
        fold_llhi.append(float(llhi_val))
        pd.DataFrame({"fold": [k], "llhi_bits_per_spike": [float(llhi_val)]}).to_csv(
            fold_dir / "llhi.csv", index=False
        )

        mu_oof[va] = mu_va.astype(np.float32)

    llhi_oof = compute_llhi_bps_poisson(y_all, mu_oof)
    mean_llhi = float(np.nanmean(fold_llhi))

    with open(neuron_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_vars,
                "oof_llhi_bits_per_spike": float(llhi_oof),
                "mean_llhi_over_folds_bits_per_spike": float(mean_llhi),
                "fold_llhi_bits_per_spike": list(map(float, fold_llhi)),
                "poisson_alpha": float(POISSON_ALPHA),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "mean_llhi": float(mean_llhi),
        "fold_llhi": list(map(float, fold_llhi)),
        "oof_llhi": float(llhi_oof),
        "mu_oof": mu_oof,
    }


def save_full_fit_weights_for_all_neurons(
    out_root: Path,
    model_vars: List[str],
    X_all: sparse.csr_matrix,
    feature_names: List[str],
    Y_all: np.ndarray,
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
    n_jobs: int,
):
    """
    Pre-fit FULL model for every neuron and save ONLY weights:
      <out_root>/FULL_FIT/<ModelKey>/neuron_k/
        weights_mean.csv
        fold1/weights.csv ... fold10/weights.csv
    """
    model_key = model_key_from_vars(model_vars)
    model_dir = out_root / FULL_FIT_DIRNAME / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    ensure_feature_mapping(str(model_dir), feature_names)

    def _one_neuron(neuron_idx: int) -> Tuple[bool, str]:
        try:
            y = Y_all[:, neuron_idx].astype(np.float64)
            neuron_dir = model_dir / f"neuron_{neuron_idx+1}"
            neuron_dir.mkdir(parents=True, exist_ok=True)

            ws = []
            for k, (tr, _va) in enumerate(folds_idx, start=1):
                w = _fit_one_fold_weights_poisson(X_all, y, tr)
                ws.append(w)

                fold_dir = neuron_dir / f"fold{k}"
                fold_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    w.reshape(1, -1),
                    index=[f"neuron_{neuron_idx+1}"],
                    columns=feature_names,
                ).to_csv(fold_dir / "weights.csv")

            w_mean = np.mean(np.stack(ws, axis=0), axis=0).astype(np.float32)
            pd.DataFrame(
                w_mean.reshape(1, -1),
                index=[f"neuron_{neuron_idx+1}"],
                columns=feature_names,
            ).to_csv(neuron_dir / "weights_mean.csv")

            return True, "OK"
        except Exception as e:
            return False, str(e)

    N_NEURONS = Y_all.shape[1]
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_neuron)(i) for i in tqdm(range(N_NEURONS), desc=f"{out_root.name} | FULL_FIT ({model_key})")
    )

    # log failures (best-effort)
    bad = [(i, msg) for i, (ok, msg) in enumerate(res) if not ok]
    if bad:
        logp = model_dir / "full_fit_failures.txt"
        with open(logp, "w", encoding="utf-8") as f:
            for i, msg in bad:
                f.write(f"neuron_{i+1}\t{msg}\n")


# ===============================
# Single-session main flow
# ===============================
def run_one_session(session: str) -> Tuple[bool, str]:
    OUT_ROOT = WEIGHTS_BASE / session
    (OUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)

    paths = session_paths(session)
    for k in ["imu", "spike", "dlc_final", "position"]:
        if not paths[k].exists():
            return False, f"Missing input {k}: {paths[k]}"

    data_dict = rebuild_inputs_50hz(session, paths)

    Y50 = load_spikes_50hz_counts(paths["spike"])  # (T50_spk, N)
    T_spk, N_NEURONS = Y50.shape

    T_cov = int(data_dict["T"])
    T = min(T_cov, T_spk)
    if abs(T_cov - T_spk) > MAX_MISMATCH_FRAMES_50HZ:
        return False, f"Length mismatch @50Hz (> {MAX_MISMATCH_FRAMES_50HZ}): cov={T_cov}, spk={T_spk}"

    for k in ["position", "head_v_bin", "roll_bin", "yaw_bin", "pitch_bin"]:
        data_dict[k] = data_dict[k][:T]
    Y_all = Y50[:T].astype(np.float64)

    kf = KFold(n_splits=CV_FOLDS, shuffle=False)
    folds_idx = list(kf.split(np.arange(T)))

    X_cache: Dict[str, Tuple[sparse.csr_matrix, List[str]]] = {}

    def get_X_and_feats(model_vars: List[str]) -> Tuple[sparse.csr_matrix, List[str]]:
        mk = model_key_from_vars(model_vars)
        if mk in X_cache:
            return X_cache[mk]
        X, feats = build_design_matrix(model_vars, data_dict)
        X_cache[mk] = (X, feats)
        return X, feats

    # ============================
    # NEW: FULL FIT for all neurons
    # ============================
    X_full, feats_full = get_X_and_feats(VARS_ALL)
    save_full_fit_weights_for_all_neurons(
        out_root=OUT_ROOT,
        model_vars=VARS_ALL,
        X_all=X_full,
        feature_names=feats_full,
        Y_all=Y_all,
        folds_idx=folds_idx,
        n_jobs=N_JOBS,
    )

    # ============================
    # Forward selection (unchanged)
    # ============================
    def llhi_cv_for_neuron(model_vars: List[str], neuron_idx: int) -> Tuple[float, List[float], np.ndarray]:
        X_all_m, _feat = get_X_and_feats(model_vars)
        y = Y_all[:, neuron_idx].astype(np.float64)

        fold_llhi: List[float] = []
        mu_oof = np.full_like(y, np.nan, dtype=np.float32)

        for (tr, va) in folds_idx:
            mu_va, llhi = fit_predict_one_fold_poisson(X_all_m, y, tr, va)
            fold_llhi.append(float(llhi))
            mu_oof[va] = mu_va

        llhi_oof = compute_llhi_bps_poisson(y, mu_oof)
        dll_series_bits = dll_bits_series_poisson(y, mu_oof)
        return float(llhi_oof), fold_llhi, dll_series_bits

    def save_accepted_step(neuron_idx: int, model_vars: List[str]) -> Dict:
        model_key = model_key_from_vars(model_vars)
        model_dir = OUT_ROOT / model_key
        X_all_m, feat_names = get_X_and_feats(model_vars)
        y = Y_all[:, neuron_idx].astype(np.float64)
        neuron_dir = model_dir / f"neuron_{neuron_idx+1}"
        return save_neuron_artifacts_for_model(
            model_vars=model_vars,
            model_dir=model_dir,
            neuron_dir=neuron_dir,
            neuron_index=neuron_idx,
            folds=folds_idx,
            X_all=X_all_m,
            y_all=y,
            feature_names=feat_names,
        )

    @dataclass
    class StepRecord:
        step: int
        model: List[str]
        mean_llhi: float
        fold_llhi: List[float]
        p_value_vs_prev: float = None
        stat_vs_prev: float = None
        n_pairs: int = None
        accepted: bool = True

    def forward_select_one_neuron(neuron_idx: int) -> Dict:
        path_records: List[StepRecord] = []
        remaining = VARS_ALL.copy()

        single_candidates = []
        for v in remaining:
            oof_llhi, fold_llhi, dll_seq = llhi_cv_for_neuron([v], neuron_idx)
            single_candidates.append((v, oof_llhi, fold_llhi, dll_seq))

        single_candidates.sort(key=lambda x: (x[1] if np.isfinite(x[1]) else -np.inf), reverse=True)
        best_v, best_oof_llhi, best_fold, best_dll_seq = single_candidates[0]

        stat, p, n = wilcoxon_greater(best_dll_seq, b=None)
        accepted = (p < ALPHA)

        path_records.append(
            StepRecord(
                step=1,
                model=[best_v],
                mean_llhi=best_oof_llhi,
                fold_llhi=list(map(float, best_fold)),
                p_value_vs_prev=p,
                stat_vs_prev=stat,
                n_pairs=n,
                accepted=accepted,
            )
        )

        if not accepted:
            return {
                "neuron": f"neuron_{neuron_idx+1}",
                "final_model": [],
                "classified": False,
                "path": [vars(s) for s in path_records],
            }

        _ = save_accepted_step(neuron_idx, [best_v])

        selected = [best_v]
        remaining.remove(best_v)
        oof_dll_prev = best_dll_seq

        step = 2
        while remaining:
            cand_list = []
            for cand in remaining:
                trial_vars = selected + [cand]
                oof_llhi, fold_llhi, dll_seq = llhi_cv_for_neuron(trial_vars, neuron_idx)
                cand_list.append((cand, trial_vars, oof_llhi, fold_llhi, dll_seq))

            cand_list.sort(key=lambda x: (x[2] if np.isfinite(x[2]) else -np.inf), reverse=True)
            best_cand, best_trial_vars, best_trial_oof_llhi, best_trial_fold, best_trial_dll = cand_list[0]

            stat, p, n = wilcoxon_greater(best_trial_dll, oof_dll_prev)
            accepted = (p < ALPHA)

            path_records.append(
                StepRecord(
                    step=step,
                    model=best_trial_vars,
                    mean_llhi=best_trial_oof_llhi,
                    fold_llhi=list(map(float, best_trial_fold)),
                    p_value_vs_prev=p,
                    stat_vs_prev=stat,
                    n_pairs=n,
                    accepted=accepted,
                )
            )

            if not accepted:
                break

            _ = save_accepted_step(neuron_idx, best_trial_vars)
            selected = best_trial_vars
            remaining.remove(best_cand)
            oof_dll_prev = best_trial_dll
            step += 1

            if len(selected) == len(VARS_ALL):
                break

        return {
            "neuron": f"neuron_{neuron_idx+1}",
            "final_model": selected,
            "classified": True,
            "path": [vars(s) for s in path_records],
        }

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(forward_select_one_neuron)(i) for i in tqdm(range(N_NEURONS), desc=f"{session} | forward search (Poisson)")
    )

    logs_dir = OUT_ROOT / "logs"
    with open(logs_dir / "neuron_forward_paths.jsonl", "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rows, unclassified = [], []
    for rec in results:
        if rec["classified"]:
            rows.append({"neuron": rec["neuron"], "final_model": "_".join(rec["final_model"])})
        else:
            unclassified.append(rec["neuron"])

    pd.DataFrame(rows).to_csv(OUT_ROOT / "selected_models.csv", index=False)
    with open(OUT_ROOT / "unclassified_neurons.txt", "w", encoding="utf-8") as f:
        for n in unclassified:
            f.write(n + "\n")

    fig_dir = OUT_ROOT / "figures"
    for rec in rows:
        neuron_name = rec["neuron"]
        model_key = rec["final_model"]
        model_dir = OUT_ROOT / model_key
        neuron_dir = model_dir / neuron_name

        try:
            y_oof, mu_oof = load_oof_from_neuron_dir(neuron_dir)
            llhi = compute_llhi_bps_poisson(y_oof, mu_oof)
            title = f"{session} | {neuron_name} | PoissonGLM | vars={model_key.replace('_','+')} | ΔLL={llhi:.4f} bits/spk"
            out_png = fig_dir / f"{neuron_name}__{model_key}.png"
            plot_fitting_curve(
                out_png,
                title,
                y_oof,
                mu_oof,
                smooth_ms=PLOT_SMOOTH_MS,
                start_sec=PLOT_START_SEC,
                end_sec=PLOT_END_SEC,
                do_zscore=PLOT_ZSCORE,
            )
        except Exception:
            pass

    with open(OUT_ROOT / "_SUCCESS", "w", encoding="utf-8") as f:
        f.write(f"OK\t{datetime.now().isoformat(timespec='seconds')}\n")

    return True, f"OK (T50={T}, N={N_NEURONS}, poisson_alpha={POISSON_ALPHA})"


# ===============================
# Batch entry point
# ===============================
def main():
    set_imu = list_sessions_imu(IMU_ROOT)
    set_spk = list_sessions_spike(SPIKE_ROOT)
    set_dlc = list_sessions_dlc_final(DLC_ROOT)
    set_pos = list_sessions_position(POSITION_ROOT)

    all_present = sorted(list(set_imu & set_spk & set_dlc & set_pos))
    if not all_present:
        print("[FATAL] No sessions found with all required inputs present.")
        return

    with open(WEIGHTS_BASE / "sessions_all_present.txt", "w", encoding="utf-8") as f:
        for s in all_present:
            f.write(s + "\n")
    print(f"[INFO] Found {len(all_present)} sessions with all required inputs present.")

    already_done = [s for s in all_present if is_session_done(s)]
    todo = [s for s in all_present if s not in already_done]

    with open(WEIGHTS_BASE / "sessions_already_done.txt", "w", encoding="utf-8") as f:
        for s in already_done:
            f.write(s + "\n")

    with open(WEIGHTS_BASE / "sessions_todo.txt", "w", encoding="utf-8") as f:
        for s in todo:
            f.write(s + "\n")

    print(f"[INFO] Already done: {len(already_done)} (see sessions_already_done.txt)")
    print(f"[INFO] To compute:   {len(todo)} (see sessions_todo.txt)")

    if not todo:
        print("[INFO] No sessions left to compute. Exiting.")
        return

    processed, skipped = [], []
    for session in todo:
        try:
            ok, msg = run_one_session(session)
        except Exception as e:
            ok, msg = False, str(e)

        if ok:
            processed.append(session)
            print(f"[DONE] {session}: {msg}")
        else:
            skipped.append((session, msg))
            print(f"[SKIP] {session}: {msg}")

    with open(WEIGHTS_BASE / "sessions_processed.txt", "w", encoding="utf-8") as f:
        for s in processed:
            f.write(s + "\n")

    with open(WEIGHTS_BASE / "sessions_skipped.txt", "w", encoding="utf-8") as f:
        for s, reason in skipped:
            f.write(f"{s}\t{reason}\n")

    print("\n=== Batch complete ===")
    print(f"All-present list: {WEIGHTS_BASE / 'sessions_all_present.txt'}")
    print(f"Already done:     {WEIGHTS_BASE / 'sessions_already_done.txt'}")
    print(f"To compute:       {WEIGHTS_BASE / 'sessions_todo.txt'}")
    print(f"Processed:        {WEIGHTS_BASE / 'sessions_processed.txt'}")
    print(f"Skipped:          {WEIGHTS_BASE / 'sessions_skipped.txt'}")


if __name__ == "__main__":
    main()
