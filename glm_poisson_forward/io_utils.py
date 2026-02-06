from typing import Dict, Tuple

import h5py
import numpy as np
import pandas as pd

from .config import (
    AGG_FACTOR,
    DLC_ROOT,
    IMU_ROOT,
    PITCH_N_BINS,
    POSITION_ROOT,
    ROLL_N_BINS,
    ROLL_PITCH_CLIP_PCTS,
    SPEED_N_BINS,
    SPIKE_ROOT,
    YAW_N_BINS,
)
from .design_matrix import bin_col, build_position_index


_GLOBAL_ROLL_PITCH_BOUNDS = None


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


def list_sessions_all():
    set_spk = list_sessions_spike(SPIKE_ROOT)
    set_dlc = list_sessions_dlc_final(DLC_ROOT)
    set_pos = list_sessions_position(POSITION_ROOT)
    set_imu = list_sessions_imu(IMU_ROOT)
    return sorted(list(set_spk & set_dlc & set_pos & set_imu))


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


def _circular_mean(angles: np.ndarray) -> float:
    if angles.size == 0:
        return 0.0
    return float(np.angle(np.mean(np.exp(1j * angles))))


def shift_angles(angles: np.ndarray, shift: float) -> np.ndarray:
    return np.mod(angles - shift, 2 * np.pi)


def _compute_shifted_bounds(angles: np.ndarray, pct_low: float, pct_high: float) -> Dict[str, float]:
    shift = np.mod(_circular_mean(angles), 2 * np.pi)
    shifted = shift_angles(angles, shift)
    vmin, vmax = np.percentile(shifted, [pct_low, pct_high])
    if np.isclose(vmin, vmax):
        vmax = float(vmin + 1e-3)
    return {"shift": float(shift), "vmin": float(vmin), "vmax": float(vmax)}


def get_global_roll_pitch_bounds() -> Dict[str, Dict[str, float]]:
    global _GLOBAL_ROLL_PITCH_BOUNDS
    if _GLOBAL_ROLL_PITCH_BOUNDS is not None:
        return _GLOBAL_ROLL_PITCH_BOUNDS

    sessions = list_sessions_all()
    if not sessions:
        default = {"shift": 0.0, "vmin": 0.0, "vmax": float(2 * np.pi)}
        _GLOBAL_ROLL_PITCH_BOUNDS = {"roll": default, "pitch": default}
        return _GLOBAL_ROLL_PITCH_BOUNDS

    # Shift angles by circular mean so 0/2π is not a hard boundary for percentiles.
    roll_vals = []
    pitch_vals = []
    for session in sessions:
        paths = session_paths(session)
        imu_df = pd.read_csv(paths["imu"], usecols=["roll", "pitch"]).astype(np.float32)
        roll_vals.append(np.mod(imu_df["roll"].values, 2 * np.pi))
        pitch_vals.append(np.mod(imu_df["pitch"].values, 2 * np.pi))

    all_roll = np.concatenate(roll_vals) if roll_vals else np.empty((0,), dtype=np.float32)
    all_pitch = np.concatenate(pitch_vals) if pitch_vals else np.empty((0,), dtype=np.float32)
    pct_low, pct_high = ROLL_PITCH_CLIP_PCTS

    _GLOBAL_ROLL_PITCH_BOUNDS = {
        "roll": _compute_shifted_bounds(all_roll, pct_low, pct_high),
        "pitch": _compute_shifted_bounds(all_pitch, pct_low, pct_high),
    }
    return _GLOBAL_ROLL_PITCH_BOUNDS


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
    bounds = get_global_roll_pitch_bounds()
    roll_shifted = shift_angles(imu_df["roll"].values, bounds["roll"]["shift"])
    pitch_shifted = shift_angles(imu_df["pitch"].values, bounds["pitch"]["shift"])
    roll_bin = bin_col(
        roll_shifted,
        n_bins=ROLL_N_BINS,
        vmin=bounds["roll"]["vmin"],
        vmax=bounds["roll"]["vmax"],
    )
    yaw_bin = bin_col(imu_df["yaw"].values, n_bins=YAW_N_BINS, vmin=0, vmax=2 * np.pi)
    pitch_bin = bin_col(
        pitch_shifted,
        n_bins=PITCH_N_BINS,
        vmin=bounds["pitch"]["vmin"],
        vmax=bounds["pitch"]["vmax"],
    )

    return {
        "T": int(L),
        "position": pos_idx.astype(np.int32),
        "n_pos": int(n_pos),
        "head_v": head_v.astype(np.float32),
        "head_v_bin": head_v_bin.astype(np.int32),
        "roll_bin": roll_bin.astype(np.int32),
        "yaw_bin": yaw_bin.astype(np.int32),
        "pitch_bin": pitch_bin.astype(np.int32),
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


def apply_residual_speed(data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    head_v = data_dict.get("head_v")
    pos_idx = data_dict.get("position")
    n_pos = data_dict.get("n_pos")
    if head_v is None or pos_idx is None or n_pos is None:
        return data_dict

    n_pos = int(n_pos)
    sums = np.bincount(pos_idx, weights=head_v, minlength=n_pos)
    counts = np.bincount(pos_idx, minlength=n_pos)
    mean_speed = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float32), where=counts > 0)
    speed_hat = mean_speed[pos_idx]
    speed_res = head_v - speed_hat

    updated = dict(data_dict)
    updated["head_v_raw"] = head_v.astype(np.float32)
    updated["head_v"] = speed_res.astype(np.float32)
    updated["speed_hat"] = speed_hat.astype(np.float32)
    updated["head_v_bin"] = bin_col(speed_res, n_bins=SPEED_N_BINS)
    return updated
