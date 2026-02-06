from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from .config import ANGLE_BIN_SETTINGS, POSITION_CELL_CM, SPEED_N_BINS, SMOOTH_LAMBDAS, SMOOTH_VARS


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


def _distribute_bins(ranges, n_bins: int, bins_per_range: Optional[List[int]] = None) -> List[int]:
    if bins_per_range is not None:
        if sum(bins_per_range) != n_bins:
            raise ValueError("bins_per_range must sum to n_bins.")
        return list(bins_per_range)
    lengths = np.asarray([float(end - start) for start, end in ranges], dtype=np.float64)
    if np.any(lengths <= 0):
        raise ValueError("All range lengths must be positive.")
    total = float(lengths.sum())
    raw = lengths / total * float(n_bins)
    floors = np.floor(raw).astype(int)
    bins = np.maximum(floors, 1)
    remainders = raw - floors
    diff = int(n_bins - bins.sum())
    while diff > 0:
        idx = int(np.argmax(remainders))
        bins[idx] += 1
        remainders[idx] = 0.0
        diff -= 1
    while diff < 0:
        idx = int(np.argmin(remainders + (bins <= 1) * 10.0))
        if bins[idx] <= 1:
            break
        bins[idx] -= 1
        diff += 1
    if bins.sum() != n_bins:
        raise ValueError("Could not distribute bins across ranges.")
    return bins.tolist()


def _clamp_to_ranges(vals: np.ndarray, ranges) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float32)
    clamped = vals.copy()
    in_any = np.zeros(vals.shape, dtype=bool)
    boundaries = np.asarray([b for start, end in ranges for b in (start, end)], dtype=np.float32)
    for start, end in ranges:
        mask = (vals >= start) & (vals <= end)
        in_any |= mask
        clamped[mask] = vals[mask]
    missing = ~in_any
    if np.any(missing):
        diffs = np.abs(vals[missing][:, None] - boundaries[None, :])
        nearest = np.argmin(diffs, axis=1)
        clamped[missing] = boundaries[nearest]
    return clamped


def bin_angle(vals, *, ranges, n_bins: int, bins_per_range: Optional[List[int]] = None) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float32)
    clamped = _clamp_to_ranges(vals, ranges)
    range_bins = _distribute_bins(ranges, n_bins, bins_per_range=bins_per_range)
    out = np.full(vals.shape, -1, dtype=np.int32)
    offset = 0
    for (start, end), n_sub in zip(ranges, range_bins):
        edges = np.linspace(start, end, n_sub + 1, dtype=np.float32)
        mask = (clamped >= start) & (clamped <= end) & (out < 0)
        if np.any(mask):
            bin_idx = np.searchsorted(edges, clamped[mask], side="right") - 1
            bin_idx = np.clip(bin_idx, 0, n_sub - 1)
            out[mask] = bin_idx + offset
        offset += n_sub
    out[out < 0] = 0
    return out.astype(np.int32)


def angle_bin_centers(*, ranges, n_bins: int, bins_per_range: Optional[List[int]] = None) -> np.ndarray:
    range_bins = _distribute_bins(ranges, n_bins, bins_per_range=bins_per_range)
    centers = []
    for (start, end), n_sub in zip(ranges, range_bins):
        edges = np.linspace(start, end, n_sub + 1, dtype=np.float64)
        centers.append(0.5 * (edges[:-1] + edges[1:]))
    return np.concatenate(centers)


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
            n_bins = int(ANGLE_BIN_SETTINGS[ang]["n_bins"])
            cats.append(np.arange(n_bins, dtype=int))
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


def build_smoothness_rows(
    feature_names: List[str],
    *,
    smooth_lambda: Optional[float] = None,
    smooth_vars: Optional[List[str]] = None,
    smooth_lambdas: Optional[Dict[str, float]] = None,
) -> sparse.csr_matrix:
    if smooth_lambdas is None:
        smooth_lambdas = SMOOTH_LAMBDAS
    if smooth_vars is None:
        smooth_vars = SMOOTH_VARS
    n_features = len(feature_names)
    if feature_names and feature_names[-1] == "intercept":
        n_features -= 1
    if smooth_lambda is not None and smooth_lambda <= 0 and not smooth_lambdas:
        return sparse.csr_matrix((0, n_features), dtype=np.float32)
    if not smooth_vars or n_features <= 0:
        return sparse.csr_matrix((0, n_features), dtype=np.float32)

    var_to_prefix = {
        "Position": "position",
        "Speed": "head_v",
        "roll": "roll",
        "yaw": "yaw",
        "pitch": "pitch",
    }

    rows = []
    cols = []
    data = []
    row_idx = 0
    feature_prefixes = feature_names[:n_features]

    for var in smooth_vars:
        lambda_val = smooth_lambda
        if smooth_lambdas is not None:
            lambda_val = smooth_lambdas.get(var, lambda_val)
        if lambda_val is None or lambda_val <= 0:
            continue
        sqrt_lambda = float(np.sqrt(lambda_val))
        prefix = var_to_prefix.get(var, var)
        idx = [
            i
            for i, name in enumerate(feature_prefixes)
            if name.startswith(f"{prefix}_")
        ]
        if len(idx) < 2:
            continue
        for j in range(len(idx) - 1):
            rows.extend([row_idx, row_idx])
            cols.extend([idx[j], idx[j + 1]])
            data.extend([-sqrt_lambda, sqrt_lambda])
            row_idx += 1

    if row_idx == 0:
        return sparse.csr_matrix((0, n_features), dtype=np.float32)
    smooth_rows = sparse.coo_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(row_idx, n_features),
    ).tocsr()
    return smooth_rows


def ensure_feature_mapping(model_dir: str, feature_names: List[str]):
    import os

    os.makedirs(model_dir, exist_ok=True)
    map_path = os.path.join(model_dir, "feature_mapping.txt")
    with open(map_path, "w", encoding="utf-8") as f:
        for j, nm in enumerate(feature_names):
            f.write(f"{j}: {nm}\n")


def model_key_from_vars(var_list: List[str]) -> str:
    return "_".join(var_list)
