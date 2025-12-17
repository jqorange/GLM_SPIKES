from typing import Dict

import h5py
import numpy as np
import pandas as pd

from .config import (
    AGG_FACTOR,
    ANGLE_N_BINS,
    DLC_ROOT,
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
    imu_df["pitch"] = imu_df["pitch"].values + (np.pi / 2)
    imu_df["roll"] = np.mod(imu_df["roll"].values, 2 * np.pi)

    pos_idx, n_pos = build_position_index(pos_df["head_x"].values, pos_df["head_y"].values)

    head_v_bin = bin_col(dlc_df["head_v"].values, n_bins=SPEED_N_BINS)
    roll_bin = bin_col(imu_df["roll"].values, n_bins=ANGLE_N_BINS, vmin=0, vmax=2 * np.pi)
    yaw_bin = bin_col(imu_df["yaw"].values, n_bins=ANGLE_N_BINS, vmin=0, vmax=2 * np.pi)
    pitch_bin = bin_col(imu_df["pitch"].values, n_bins=ANGLE_N_BINS, vmin=0, vmax=np.pi)

    return {
        "T": int(L),
        "position": pos_idx.astype(np.int32),
        "n_pos": int(n_pos),
        "head_v_bin": head_v_bin.astype(np.int32),
        "roll_bin": roll_bin.astype(np.int32),
        "yaw_bin": yaw_bin.astype(np.int32),
        "pitch_bin": pitch_bin.astype(np.int32),
    }
