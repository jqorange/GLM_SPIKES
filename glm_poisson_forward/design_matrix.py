from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

from .config import (
    PITCH_N_KNOTS,
    POSITION_CELL_CM,
    ROLL_N_KNOTS,
    SPEED_N_KNOTS,
    SPLINE_TENSION,
    YAW_N_KNOTS,
)


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


def _cardinal_spline_weights(alpha: np.ndarray, tension: float) -> np.ndarray:
    # [a^3, a^2, a, 1] @ M(s)
    s = float(tension)
    a = alpha.astype(np.float32)
    a2 = a * a
    a3 = a2 * a

    w_m1 = (-s * a3) + (2.0 * s * a2) + (-s * a)
    w_0 = ((2.0 - s) * a3) + ((s - 3.0) * a2) + 1.0
    w_p1 = ((s - 2.0) * a3) + ((3.0 - 2.0 * s) * a2) + (s * a)
    w_p2 = (s * a3) + (-s * a2)
    return np.stack([w_m1, w_0, w_p1, w_p2], axis=1)


def _build_spline_basis(
    vals: np.ndarray,
    n_knots: int,
    vmin: float,
    vmax: float,
    *,
    tension: float,
    circular: bool,
) -> sparse.csr_matrix:
    vals = np.asarray(vals, dtype=np.float32)
    T = int(vals.shape[0])
    if T == 0:
        return sparse.csr_matrix((0, n_knots), dtype=np.float32)
    if n_knots < 4:
        raise ValueError("n_knots must be >= 4 for cardinal spline basis.")

    if circular:
        period = float(vmax - vmin)
        if period <= 0:
            raise ValueError("For circular spline, vmax must be > vmin.")
        step = period / float(n_knots)
        rel = np.mod(vals - vmin, period)
        seg = np.floor(rel / step).astype(np.int32)
        alpha = (rel / step) - seg.astype(np.float32)
        knot_idx = np.stack([seg - 1, seg, seg + 1, seg + 2], axis=1) % n_knots
    else:
        width = float(vmax - vmin)
        if width <= 0:
            return sparse.csr_matrix((T, n_knots), dtype=np.float32)
        vals_clip = np.clip(vals, vmin, vmax)
        step = width / float(n_knots - 1)
        rel = (vals_clip - vmin) / step
        seg = np.floor(rel).astype(np.int32)
        seg = np.clip(seg, 0, n_knots - 2)
        alpha = rel - seg.astype(np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)
        knot_idx = np.stack([seg - 1, seg, seg + 1, seg + 2], axis=1)
        knot_idx = np.clip(knot_idx, 0, n_knots - 1)

    weights = _cardinal_spline_weights(alpha, tension=tension)

    row = np.repeat(np.arange(T, dtype=np.int32), 4)
    col = knot_idx.reshape(-1)
    data = weights.reshape(-1).astype(np.float32)

    X = sparse.coo_matrix((data, (row, col)), shape=(T, n_knots), dtype=np.float32)
    return X.tocsr()


def _build_position_onehot(position: np.ndarray, n_pos: int) -> Tuple[sparse.csr_matrix, List[str]]:
    # Keep drop-first coding for Position to stay compatible with previous setup.
    pos = np.asarray(position, dtype=np.int32)
    T = int(pos.shape[0])
    if n_pos <= 1:
        return sparse.csr_matrix((T, 0), dtype=np.float32), []
    mask = pos > 0
    row = np.where(mask)[0].astype(np.int32)
    col = (pos[mask] - 1).astype(np.int32)
    data = np.ones(row.shape[0], dtype=np.float32)
    X = sparse.coo_matrix((data, (row, col)), shape=(T, n_pos - 1), dtype=np.float32).tocsr()
    names = [f"position_{i}" for i in range(1, n_pos)]
    return X, names


def build_design_matrix(selected_vars: List[str], data_dict: Dict[str, np.ndarray]) -> Tuple[sparse.csr_matrix, List[str]]:
    T = int(data_dict["T"])
    blocks: List[sparse.csr_matrix] = []
    feature_names: List[str] = []

    if "Position" in selected_vars:
        X_pos, f_pos = _build_position_onehot(data_dict["position"], int(data_dict["n_pos"]))
        blocks.append(X_pos)
        feature_names.extend(f_pos)

    if "Speed" in selected_vars:
        X = _build_spline_basis(
            data_dict["head_v"],
            n_knots=SPEED_N_KNOTS,
            vmin=0.0,
            vmax=1.5,
            tension=SPLINE_TENSION,
            circular=False,
        )
        blocks.append(X)
        feature_names.extend([f"head_v_knot_{i}" for i in range(SPEED_N_KNOTS)])

    if "roll" in selected_vars:
        X = _build_spline_basis(
            data_dict["roll"],
            n_knots=ROLL_N_KNOTS,
            vmin=0.0,
            vmax=float(data_dict["roll_width"]),
            tension=SPLINE_TENSION,
            circular=False,
        )
        blocks.append(X)
        feature_names.extend([f"roll_knot_{i}" for i in range(ROLL_N_KNOTS)])

    if "yaw" in selected_vars:
        X = _build_spline_basis(
            data_dict["yaw"],
            n_knots=YAW_N_KNOTS,
            vmin=0.0,
            vmax=2.0 * np.pi,
            tension=SPLINE_TENSION,
            circular=True,
        )
        blocks.append(X)
        feature_names.extend([f"yaw_knot_{i}" for i in range(YAW_N_KNOTS)])

    if "pitch" in selected_vars:
        X = _build_spline_basis(
            data_dict["pitch"],
            n_knots=PITCH_N_KNOTS,
            vmin=0.0,
            vmax=float(data_dict["pitch_width"]),
            tension=SPLINE_TENSION,
            circular=False,
        )
        blocks.append(X)
        feature_names.extend([f"pitch_knot_{i}" for i in range(PITCH_N_KNOTS)])

    if not blocks:
        X_zero = sparse.csr_matrix((T, 0), dtype=np.float32)
        return X_zero, ["intercept"]

    X_all = sparse.hstack(blocks, format="csr", dtype=np.float32)
    return X_all, feature_names + ["intercept"]


def ensure_feature_mapping(model_dir: str, feature_names: List[str]):
    import os

    os.makedirs(model_dir, exist_ok=True)
    map_path = os.path.join(model_dir, "feature_mapping.txt")
    with open(map_path, "w", encoding="utf-8") as f:
        for j, nm in enumerate(feature_names):
            f.write(f"{j}: {nm}\n")


def model_key_from_vars(var_list: List[str]) -> str:
    return "_".join(var_list)
