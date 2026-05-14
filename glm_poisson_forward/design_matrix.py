from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from .config import (
    SMOOTH_LAMBDAS,
    SMOOTH_VARS,
    VARIABLE_SPECS,
)


def _design_channel_specs(var: str, spec: Dict) -> List[Dict]:
    col_spec = spec.get("column")
    if col_spec is None:
        return []

    n_bins_spec = spec.get("n_bins")

    def _pick_n_bins(i: int, key: str, fallback: int) -> int:
        if isinstance(n_bins_spec, dict):
            if key in n_bins_spec:
                return int(n_bins_spec[key])
            if i in n_bins_spec:
                return int(n_bins_spec[i])
            return int(fallback)
        if isinstance(n_bins_spec, (list, tuple)):
            if i < len(n_bins_spec):
                return int(n_bins_spec[i])
            return int(fallback)
        return int(fallback)

    if isinstance(col_spec, str):
        default_design = spec.get("design_key", var)
        default_bin = spec.get("bin_key", f"{var}_bin")
        return [{
            "design_key": default_design,
            "bin_key": default_bin,
            "n_bins": int(spec.get("n_bins", 0)),
        }]

    if isinstance(col_spec, dict):
        names = [str(k) for k in col_spec.keys()]
    elif isinstance(col_spec, (list, tuple)):
        names = [str(k) for k in col_spec]
    elif isinstance(col_spec, set):
        names = [str(k) for k in sorted(col_spec)]
    else:
        return []

    out = []
    for i, name in enumerate(names):
        out.append({
            "design_key": name,
            "bin_key": f"{name}_bin",
            "n_bins": _pick_n_bins(i, name, int(spec.get("n_bins", 0))),
        })
    return out


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
    cell = float(VARIABLE_SPECS.get("Position", {}).get("cell_cm", 8.0))
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
    pos_xy_by_idx = uniq[["x_bin", "y_bin"]].to_numpy(dtype=np.int32)
    return pos_idx, int(uniq.shape[0]), pos_xy_by_idx


def build_design_matrix(selected_vars: List[str], data_dict: Dict[str, np.ndarray]) -> Tuple[sparse.csr_matrix, List[str]]:
    cat_cols, cats, cat_order = [], [], []
    cont_cols, cont_order = [], []
    for var in selected_vars:
        spec = VARIABLE_SPECS.get(var, {})
        kind = spec.get("kind", "continuous")
        if kind == "position2d":
            design_key = spec.get("design_key", var)
            cat_cols.append(data_dict["position"].astype(np.int32))
            cats.append(np.arange(int(data_dict["n_pos"]), dtype=int))
            cat_order.append(design_key)
            continue

        if kind == "time":
            design_key = spec.get("design_key", var)
            value_key = spec.get("value_key", design_key)
            if value_key not in data_dict:
                continue
            cont_cols.append(data_dict[value_key].astype(np.float32).reshape(-1, 1))
            cont_order.append(design_key)
            continue

        channels = _design_channel_specs(var, spec)
        for c in channels:
            bin_key = c["bin_key"]
            if bin_key not in data_dict:
                continue
            n_bins = int(c["n_bins"]) if int(c["n_bins"]) > 0 else int(np.max(data_dict[bin_key]) + 1)
            cat_cols.append(data_dict[bin_key].astype(np.int32))
            cats.append(np.arange(n_bins, dtype=int))
            cat_order.append(c["design_key"])

    n_samples = int(data_dict.get("T", 0))
    if n_samples <= 0:
        if cat_cols:
            n_samples = len(cat_cols[0])
        elif cont_cols:
            n_samples = cont_cols[0].shape[0]
        else:
            n_samples = len(data_dict.get("position", []))

    if len(cat_cols) == 0:
        X_cat = sparse.csr_matrix((n_samples, 0), dtype=np.float32)
        feat_cat = []
    else:
        cat_df = pd.DataFrame({name: col for name, col in zip(cat_order, cat_cols)})

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
        feat_cat = encoder.get_feature_names_out(cat_order).tolist()

    if len(cont_cols) == 0:
        X = X_cat
    else:
        X_cont_dense = np.hstack(cont_cols).astype(np.float32)
        X_cont = sparse.csr_matrix(X_cont_dense)
        X = sparse.hstack([X_cat, X_cont], format="csr")

    feature_names = feat_cat + cont_order + ["intercept"]
    return X, feature_names


def build_smoothness_rows(
    feature_names: List[str],
    *,
    smooth_lambda: Optional[float] = None,
    smooth_vars: Optional[List[str]] = None,
    smooth_lambdas: Optional[Dict[str, float]] = None,
    position_xy_by_idx: Optional[np.ndarray] = None,
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

    var_to_prefixes = {}
    for var, spec in VARIABLE_SPECS.items():
        kind = spec.get("kind", "continuous")
        if kind == "position2d":
            var_to_prefixes[var] = [spec.get("design_key", var)]
            continue
        if kind == "time":
            var_to_prefixes[var] = [spec.get("design_key", var)]
            continue
        channels = _design_channel_specs(var, spec)
        var_to_prefixes[var] = [c["design_key"] for c in channels]

    rows = []
    cols = []
    data = []
    row_idx = 0
    feature_prefixes = feature_names[:n_features]

    def _append_symmetric_difference_row(a: int, b: int, scale: float) -> None:
        nonlocal row_idx
        # Add both directions so the pseudo-observation penalty is symmetric in
        # the coefficient difference rather than favoring one sign.
        rows.extend([row_idx, row_idx])
        cols.extend([a, b])
        data.extend([-scale, scale])
        row_idx += 1

        rows.extend([row_idx, row_idx])
        cols.extend([a, b])
        data.extend([scale, -scale])
        row_idx += 1

    for var in smooth_vars:
        lambda_val = smooth_lambda
        if smooth_lambdas is not None:
            lambda_val = smooth_lambdas.get(var, lambda_val)
        if lambda_val is None or lambda_val <= 0:
            continue
        # Each edge contributes two mirrored pseudo-observations; split lambda
        # across them so the local curvature near zero stays comparable.
        sqrt_lambda = float(np.sqrt(lambda_val / 2.0))
        for prefix in var_to_prefixes.get(var, [var]):
            idx = [
                i
                for i, name in enumerate(feature_prefixes)
                if name == prefix or name.startswith(f"{prefix}_")
            ]
            if len(idx) < 2:
                continue

            if var == "Position" and position_xy_by_idx is not None:
                feat_to_pos_idx = {}
                for feat_i in idx:
                    token = feature_prefixes[feat_i]
                    try:
                        feat_to_pos_idx[feat_i] = int(token.split("_")[-1])
                    except ValueError:
                        continue

                if len(feat_to_pos_idx) < 2:
                    continue

                pos_to_feat_idx = {p: f for f, p in feat_to_pos_idx.items()}
                xy_to_pos_idx = {
                    (int(x), int(y)): p
                    for p, (x, y) in enumerate(np.asarray(position_xy_by_idx, dtype=np.int32))
                }

                edge_pairs = set()
                for pos_i, feat_i in pos_to_feat_idx.items():
                    x_i, y_i = position_xy_by_idx[pos_i]
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nb_pos = xy_to_pos_idx.get((int(x_i) + dx, int(y_i) + dy))
                            if nb_pos is None:
                                continue
                            feat_j = pos_to_feat_idx.get(nb_pos)
                            if feat_j is None:
                                continue
                            a, b = sorted((feat_i, feat_j))
                            if a != b:
                                edge_pairs.add((a, b))

                for a, b in sorted(edge_pairs):
                    _append_symmetric_difference_row(a, b, sqrt_lambda)
                continue

            for j in range(len(idx) - 1):
                _append_symmetric_difference_row(idx[j], idx[j + 1], sqrt_lambda)

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
