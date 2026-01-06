from typing import Dict

import numpy as np
import pandas as pd

from glm_poisson_forward.config import DLC_ROOT, IMU_ROOT, POSITION_ROOT, SPIKE_ROOT
from glm_poisson_forward.io_utils import (
    list_sessions_dlc_final,
    list_sessions_imu,
    list_sessions_position,
    list_sessions_spike,
)
from glm_poisson_forward.io_utils import load_spikes_50hz_counts


def list_sessions_all() -> list[str]:
    set_spk = list_sessions_spike(SPIKE_ROOT)
    set_dlc = list_sessions_dlc_final(DLC_ROOT)
    set_pos = list_sessions_position(POSITION_ROOT)
    set_imu = list_sessions_imu(IMU_ROOT)
    return sorted(list(set_spk & set_dlc & set_pos & set_imu))


def session_paths(session: str) -> Dict[str, object]:
    return {
        "spike": SPIKE_ROOT / f"{session}_200Hz.h5",
        "dlc_final": DLC_ROOT / session / f"final_filtered_{session}_50hz.csv",
        "position": POSITION_ROOT / f"positions_{session}.csv",
        "imu": IMU_ROOT / session / f"{session}_IMU_features.csv",
    }


def load_session_raw(session: str) -> Dict[str, np.ndarray]:
    paths = session_paths(session)
    pos_df = pd.read_csv(paths["position"], usecols=["head_x", "head_y", "heading_deg"]).astype(np.float32)
    dlc_df = pd.read_csv(paths["dlc_final"], usecols=["head_v"]).astype(np.float32)
    imu_df = pd.read_csv(paths["imu"], usecols=["roll", "yaw", "pitch"]).astype(np.float32)

    L = min(len(pos_df), len(dlc_df), len(imu_df))
    pos_df = pos_df.iloc[:L].reset_index(drop=True)
    dlc_df = dlc_df.iloc[:L].reset_index(drop=True)
    imu_df = imu_df.iloc[:L].reset_index(drop=True)

    heading_deg = pos_df["heading_deg"].to_numpy(dtype=np.float32)
    heading_rad = np.deg2rad(heading_deg)
    heading_rad = np.mod(heading_rad, 2 * np.pi)

    spikes_50hz = load_spikes_50hz_counts(paths["spike"])
    if spikes_50hz.shape[0] < L:
        L = spikes_50hz.shape[0]
        pos_df = pos_df.iloc[:L].reset_index(drop=True)
        dlc_df = dlc_df.iloc[:L].reset_index(drop=True)
        imu_df = imu_df.iloc[:L].reset_index(drop=True)
        heading_rad = heading_rad[:L]
        heading_deg = heading_deg[:L]
    elif spikes_50hz.shape[0] > L:
        spikes_50hz = spikes_50hz[:L]

    imu_df["yaw"] = heading_rad
    imu_df["roll"] = np.mod(imu_df["roll"].values, 2 * np.pi)
    imu_df["pitch"] = np.mod(imu_df["pitch"].values, 2 * np.pi)

    return {
        "T": int(L),
        "head_x": pos_df["head_x"].to_numpy(dtype=np.float32),
        "head_y": pos_df["head_y"].to_numpy(dtype=np.float32),
        "heading_rad": heading_rad.astype(np.float32),
        "heading_deg": heading_deg.astype(np.float32),
        "head_v": dlc_df["head_v"].to_numpy(dtype=np.float32),
        "roll": imu_df["roll"].to_numpy(dtype=np.float32),
        "pitch": imu_df["pitch"].to_numpy(dtype=np.float32),
        "spikes": spikes_50hz.astype(np.int32),
    }
