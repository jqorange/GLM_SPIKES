from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd

from .config import (
    AGG_FACTOR,
    ANGLE_N_BINS,
    DLC_ROOT,
    FS_HZ,
    IMU_ROOT,
    POSITION_ROOT,
    SPEED_N_BINS,
    SPIKE_ROOT,
)
from .design_matrix import bin_col, build_position_index


def list_sessions_imu(root):
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


def list_sessions_spike(root):
    if not root.exists():
        return set()
    return {f.stem.replace("_200Hz", "") for f in root.glob("*_200Hz.h5")}


def list_sessions_dlc_final(root):
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


def list_sessions_position(root):
    if not root.exists():
        return set()
    return {s.stem.replace("positions_", "") for s in root.glob("positions_*.csv")}


def session_paths(session: str) -> Dict[str, object]:
    return {
        "imu": IMU_ROOT / session / f"{session}_IMU_features.csv",
        "spike": SPIKE_ROOT / f"{session}_200Hz.h5",
        "dlc_final": DLC_ROOT / session / f"final_filtered_{session}_50hz.csv",
        "position": POSITION_ROOT / f"positions_{session}.csv",
    }


def is_session_done(session: str, weights_base) -> bool:
    out_dir = weights_base / session
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


def load_spikes_50hz_counts(h5_path) -> np.ndarray:
    with h5py.File(h5_path, "r") as hf:
        Y200 = hf["spike_binary"][:].astype(np.int16)  # (T200, N)

    T200, N = Y200.shape
    T200_trim = (T200 // AGG_FACTOR) * AGG_FACTOR
    if T200_trim <= 0:
        raise ValueError("Spike length too short after trimming.")

    Y200 = Y200[:T200_trim]
    Y50 = Y200.reshape(-1, AGG_FACTOR, N).sum(axis=1)  # (T50, N)
    return Y50.astype(np.int32)


def rebuild_inputs_50hz(session: str, paths: Dict[str, object]) -> Dict[str, np.ndarray]:
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
    imu_df["pitch"] = np.mod(imu_df["pitch"].values, 2 * np.pi)
    imu_df["roll"] = np.mod(imu_df["roll"].values, 2 * np.pi)

    pos_idx, n_pos = build_position_index(pos_df["head_x"].values, pos_df["head_y"].values)

    head_v = dlc_df["head_v"].values.astype(np.float32)
    head_v_bin = bin_col(head_v, n_bins=SPEED_N_BINS, vmin=0, vmax=1.5)
    roll_vals = imu_df["roll"].values.astype(np.float32)
    yaw_vals = imu_df["yaw"].values.astype(np.float32)
    pitch_vals = imu_df["pitch"].values.astype(np.float32)
    roll_d_vals = np.gradient(np.unwrap(roll_vals), edge_order=1) * FS_HZ
    yaw_d_vals = np.gradient(np.unwrap(yaw_vals), edge_order=1) * FS_HZ
    pitch_d_vals = np.gradient(np.unwrap(pitch_vals), edge_order=1) * FS_HZ
    roll_bin = bin_col(roll_vals, n_bins=ANGLE_N_BINS, vmin=0, vmax=2 * np.pi)
    yaw_bin = bin_col(yaw_vals, n_bins=ANGLE_N_BINS, vmin=0, vmax=2 * np.pi)
    pitch_bin = bin_col(pitch_vals, n_bins=ANGLE_N_BINS, vmin=0, vmax=2 * np.pi)
    roll_d_bin = bin_col(roll_d_vals, n_bins=ANGLE_N_BINS)
    yaw_d_bin = bin_col(yaw_d_vals, n_bins=ANGLE_N_BINS)
    pitch_d_bin = bin_col(pitch_d_vals, n_bins=ANGLE_N_BINS)

    return {
        "T": int(L),
        "position": pos_idx.astype(np.int32),
        "n_pos": int(n_pos),
        "head_v": head_v.astype(np.float32),
        "head_v_bin": head_v_bin.astype(np.int32),
        "roll": roll_vals,
        "yaw": yaw_vals,
        "pitch": pitch_vals,
        "roll_d": roll_d_vals.astype(np.float32),
        "yaw_d": yaw_d_vals.astype(np.float32),
        "pitch_d": pitch_d_vals.astype(np.float32),
        "roll_bin": roll_bin.astype(np.int32),
        "yaw_bin": yaw_bin.astype(np.int32),
        "pitch_bin": pitch_bin.astype(np.int32),
        "roll_d_bin": roll_d_bin.astype(np.int32),
        "yaw_d_bin": yaw_d_bin.astype(np.int32),
        "pitch_d_bin": pitch_d_bin.astype(np.int32),
    }


def filter_by_min_speed(
    data_dict: Dict[str, np.ndarray],
    Y_all: np.ndarray,
    min_speed_cm_s: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray | None]:
    if min_speed_cm_s <= 0:
        return data_dict, Y_all, None
    head_v = data_dict.get("head_v")
    if head_v is None:
        return data_dict, Y_all, None
    mask = head_v >= min_speed_cm_s
    if mask.ndim != 1:
        mask = mask.reshape(-1)
    filtered = {}
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray) and v.shape[0] == mask.shape[0]:
            filtered[k] = v[mask]
        else:
            filtered[k] = v
    filtered["T"] = int(np.sum(mask))
    Y_all = Y_all[mask]
    return filtered, Y_all, mask


def apply_residual_by_position(
    data_dict: Dict[str, np.ndarray],
    value_key: str,
    bin_key: str,
    n_bins: int,
) -> Dict[str, np.ndarray]:
    values = data_dict.get(value_key)
    pos_idx = data_dict.get("position")
    n_pos = data_dict.get("n_pos")
    if values is None or pos_idx is None or n_pos is None:
        return data_dict

    n_pos = int(n_pos)
    sums = np.bincount(pos_idx, weights=values, minlength=n_pos)
    counts = np.bincount(pos_idx, minlength=n_pos)
    mean_vals = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float32), where=counts > 0)
    vals_hat = mean_vals[pos_idx]
    vals_res = values - vals_hat

    updated = dict(data_dict)
    updated[f"{value_key}_raw"] = values.astype(np.float32)
    updated[value_key] = vals_res.astype(np.float32)
    updated[f"{value_key}_hat"] = vals_hat.astype(np.float32)
    updated[bin_key] = bin_col(vals_res, n_bins=n_bins)
    return updated


def apply_residual_speed(data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return apply_residual_by_position(data_dict, "head_v", "head_v_bin", SPEED_N_BINS)


def apply_residual_angle(data_dict: Dict[str, np.ndarray], angle_key: str) -> Dict[str, np.ndarray]:
    return apply_residual_by_position(data_dict, angle_key, f"{angle_key}_bin", ANGLE_N_BINS)
