from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from .config import ANGLE_N_BINS, POSITION_CELL_CM, SPEED_N_BINS


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


def build_position_index(head_x_cm, head_y_cm) -> Tuple[np.ndarray, int, np.ndarray]:
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
    pos_bins = uniq[["x_bin", "y_bin"]].to_numpy(dtype=np.int32)
    return pos_idx, int(uniq.shape[0]), pos_bins


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
    import os

    os.makedirs(model_dir, exist_ok=True)
    map_path = os.path.join(model_dir, "feature_mapping.txt")
    with open(map_path, "w", encoding="utf-8") as f:
        for j, nm in enumerate(feature_names):
            f.write(f"{j}: {nm}\n")


def model_key_from_vars(var_list: List[str]) -> str:
    return "_".join(var_list)
