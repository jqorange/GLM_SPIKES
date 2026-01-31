import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy import sparse
from tqdm import tqdm

from .config import (
    ANGLE_N_BINS,
    FULL_FIT_DIRNAME,
    MAX_ITER,
    N_JOBS,
    POISSON_ALPHA,
    REG_ANGLE_RIDGE,
    REG_ANGLE_SMOOTH,
    REG_POSITION_RIDGE,
    REG_POSITION_SMOOTH,
    REG_SPEED_RIDGE,
    REG_SPEED_SMOOTH,
    SPEED_N_BINS,
    USE_TORCH,
)
from .design_matrix import ensure_feature_mapping, model_key_from_vars
from .metrics import compute_llhi_bps_poisson


@dataclass(frozen=True)
class RidgePenalty:
    name: str
    lambda_: float
    indices: np.ndarray


@dataclass(frozen=True)
class SmoothPenalty:
    name: str
    lambda_: float
    edge_i: np.ndarray
    edge_j: np.ndarray


@dataclass(frozen=True)
class RegularizationSpec:
    ridge: Tuple[RidgePenalty, ...]
    smooth: Tuple[SmoothPenalty, ...]


def _build_position_edges(position_bins: np.ndarray) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    if position_bins is None or position_bins.size == 0:
        return edges
    lookup = {(int(x), int(y)): i for i, (x, y) in enumerate(position_bins)}
    for idx, (x_bin, y_bin) in enumerate(position_bins):
        right = (int(x_bin) + 1, int(y_bin))
        up = (int(x_bin), int(y_bin) + 1)
        if right in lookup:
            edges.append((idx, lookup[right]))
        if up in lookup:
            edges.append((idx, lookup[up]))
    return edges


def _build_1d_edges(n_bins: int, circular: bool) -> List[Tuple[int, int]]:
    edges = [(i, i - 1) for i in range(1, n_bins)]
    if circular and n_bins > 1:
        edges.append((0, n_bins - 1))
    return edges


def _edges_to_feature_indices(
    edges: List[Tuple[int, int]],
    bin_to_feature: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    edge_i: List[int] = []
    edge_j: List[int] = []
    for bin_a, bin_b in edges:
        idx_a = int(bin_to_feature[bin_a])
        idx_b = int(bin_to_feature[bin_b])
        if idx_a < 0 and idx_b < 0:
            continue
        if idx_a < 0:
            edge_i.append(idx_b)
            edge_j.append(-1)
        elif idx_b < 0:
            edge_i.append(idx_a)
            edge_j.append(-1)
        else:
            edge_i.append(idx_a)
            edge_j.append(idx_b)
    return np.asarray(edge_i, dtype=np.int64), np.asarray(edge_j, dtype=np.int64)


def build_group_regularization(
    feature_names: List[str],
    data_dict: Dict[str, np.ndarray],
) -> RegularizationSpec:
    prefix_map = {
        "position": "position",
        "head_v": "speed",
        "roll": "roll",
        "yaw": "yaw",
        "pitch": "pitch",
    }
    group_bins: Dict[str, Dict[int, int]] = {g: {} for g in prefix_map.values()}
    for idx, name in enumerate(feature_names):
        if name == "intercept":
            continue
        for prefix, group in prefix_map.items():
            if name.startswith(f"{prefix}_"):
                bin_idx = int(name[len(prefix) + 1 :])
                group_bins[group][bin_idx] = idx
                break

    ridge_penalties: List[RidgePenalty] = []
    smooth_penalties: List[SmoothPenalty] = []

    def add_ridge(group: str, lam: float):
        if lam <= 0:
            return
        bins = group_bins.get(group, {})
        if not bins:
            return
        indices = np.array(sorted(bins.values()), dtype=np.int64)
        ridge_penalties.append(RidgePenalty(group, lam, indices))

    def add_smooth(group: str, lam: float, edges: List[Tuple[int, int]], n_bins: int):
        if lam <= 0:
            return
        bins = group_bins.get(group, {})
        if not bins:
            return
        bin_to_feature = np.full(n_bins, -1, dtype=np.int64)
        for b, feat_idx in bins.items():
            bin_to_feature[int(b)] = int(feat_idx)
        edge_i, edge_j = _edges_to_feature_indices(edges, bin_to_feature)
        if edge_i.size == 0:
            return
        smooth_penalties.append(SmoothPenalty(group, lam, edge_i, edge_j))

    n_pos = int(data_dict.get("n_pos", 0))
    pos_bins = data_dict.get("position_bins")
    if n_pos > 0 and pos_bins is not None:
        pos_edges = _build_position_edges(pos_bins)
        add_smooth("position", REG_POSITION_SMOOTH, pos_edges, n_pos)
    add_ridge("position", REG_POSITION_RIDGE)

    add_smooth("speed", REG_SPEED_SMOOTH, _build_1d_edges(SPEED_N_BINS, circular=False), SPEED_N_BINS)
    add_ridge("speed", REG_SPEED_RIDGE)

    angle_edges = _build_1d_edges(ANGLE_N_BINS, circular=True)
    for ang in ["roll", "yaw", "pitch"]:
        add_smooth(ang, REG_ANGLE_SMOOTH, angle_edges, ANGLE_N_BINS)
        add_ridge(ang, REG_ANGLE_RIDGE)

    return RegularizationSpec(tuple(ridge_penalties), tuple(smooth_penalties))


def _torch_sparse_from_csr(X: sparse.csr_matrix, device: torch.device) -> torch.Tensor:
    X_coo = X.tocoo()
    indices = np.vstack((X_coo.row, X_coo.col))
    i_t = torch.tensor(indices, dtype=torch.int64, device=device)
    v_t = torch.tensor(X_coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(i_t, v_t, size=X_coo.shape, device=device)


def _torch_penalty(
    w: torch.Tensor,
    ridge_terms: List[Tuple[float, torch.Tensor]],
    smooth_terms: List[Tuple[float, torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=w.device)
    for lam, idx in ridge_terms:
        penalty = penalty + 0.5 * lam * torch.sum(w[idx] ** 2)
    for lam, idx_i, idx_j in smooth_terms:
        w_i = w[idx_i]
        mask = idx_j >= 0
        if torch.any(mask):
            w_j = torch.zeros_like(w_i)
            w_j[mask] = w[idx_j[mask]]
        else:
            w_j = torch.zeros_like(w_i)
        diff = w_i - w_j
        penalty = penalty + 0.5 * lam * torch.sum(diff**2)
    return penalty


def _fit_poisson_torch(
    X: sparse.csr_matrix,
    y: np.ndarray,
    reg_spec: RegularizationSpec,
) -> Tuple[np.ndarray, float]:
    if y.size == 0:
        return np.zeros(X.shape[1], dtype=np.float32), float(np.log(1e-12))
    device = torch.device("cuda" if USE_TORCH and torch.cuda.is_available() else "cpu")
    X_t = _torch_sparse_from_csr(X, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    w = torch.zeros(X.shape[1], dtype=torch.float32, device=device, requires_grad=True)
    intercept = torch.tensor(float(np.log(np.mean(y) + 1e-12)), device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, intercept], max_iter=MAX_ITER, line_search_fn="strong_wolfe")
    ridge_terms = [
        (ridge.lambda_, torch.tensor(ridge.indices, dtype=torch.int64, device=device))
        for ridge in reg_spec.ridge
        if ridge.lambda_ > 0 and ridge.indices.size > 0
    ]
    smooth_terms = [
        (
            smooth.lambda_,
            torch.tensor(smooth.edge_i, dtype=torch.int64, device=device),
            torch.tensor(smooth.edge_j, dtype=torch.int64, device=device),
        )
        for smooth in reg_spec.smooth
        if smooth.lambda_ > 0 and smooth.edge_i.size > 0
    ]

    def closure():
        opt.zero_grad(set_to_none=True)
        eta = torch.sparse.mm(X_t, w.unsqueeze(1)).squeeze(1) + intercept
        mu = torch.exp(eta)
        loss = torch.sum(mu - y_t * eta)
        loss = loss + _torch_penalty(w, ridge_terms, smooth_terms)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().cpu().numpy(), float(intercept.detach().cpu().numpy())


def _fit_poisson_weights(
    X: sparse.csr_matrix,
    y: np.ndarray,
    reg_spec: RegularizationSpec,
) -> Tuple[np.ndarray, float]:
    return _fit_poisson_torch(X, y, reg_spec)


def _predict_mu(
    X: sparse.csr_matrix,
    w: np.ndarray,
    intercept: float,
) -> np.ndarray:
    eta = X @ w + float(intercept)
    mu = np.exp(np.clip(eta, -20.0, 20.0))
    return np.clip(mu.astype(np.float64), 1e-12, None)


def fit_predict_one_fold_poisson(
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    reg_spec: RegularizationSpec,
) -> Tuple[np.ndarray, float]:
    Xtr, Xva = X_all[tr_idx], X_all[va_idx]
    ytr, yva = y_all[tr_idx].astype(np.float64), y_all[va_idx].astype(np.float64)

    mean_tr = float(np.mean(ytr))
    if mean_tr <= 0:
        mu_va = np.full_like(yva, 1e-12, dtype=np.float64)
        llhi = compute_llhi_bps_poisson(yva, mu_va)
        return mu_va.astype(np.float32), float(llhi)

    w, intercept = _fit_poisson_weights(Xtr, ytr, reg_spec)
    mu_va = _predict_mu(Xva, w, intercept)

    llhi = compute_llhi_bps_poisson(yva, mu_va)
    return mu_va.astype(np.float32), float(llhi)


def _fit_one_fold_weights_poisson(
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    reg_spec: RegularizationSpec,
) -> np.ndarray:
    """Return w = [coef..., intercept] for one fold (fit on train only)."""
    Xtr = X_all[tr_idx]
    ytr = y_all[tr_idx].astype(np.float64)

    mean_tr = float(np.mean(ytr))
    if mean_tr <= 0:
        w = np.zeros(Xtr.shape[1] + 1, dtype=np.float32)
        w[-1] = np.log(1e-12)
        return w

    coef, intercept = _fit_poisson_weights(Xtr, ytr, reg_spec)
    w = np.concatenate([coef.astype(np.float32), np.array([intercept], dtype=np.float32)])
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
    reg_spec: RegularizationSpec,
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
            coef, intercept = _fit_poisson_weights(Xtr, ytr, reg_spec)
            mu_va = _predict_mu(Xva, coef, intercept)
            w = np.concatenate([coef.astype(np.float32), np.array([intercept], dtype=np.float32)])

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
                "regularization": {
                    "position": {
                        "smooth": float(REG_POSITION_SMOOTH),
                        "ridge": float(REG_POSITION_RIDGE),
                    },
                    "speed": {
                        "smooth": float(REG_SPEED_SMOOTH),
                        "ridge": float(REG_SPEED_RIDGE),
                    },
                    "angle": {
                        "smooth": float(REG_ANGLE_SMOOTH),
                        "ridge": float(REG_ANGLE_RIDGE),
                    },
                },
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
    reg_spec: RegularizationSpec,
    n_jobs: int = N_JOBS,
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

    def _one_neuron(neuron_idx: int):
        try:
            y = Y_all[:, neuron_idx].astype(np.float64)
            neuron_dir = model_dir / f"neuron_{neuron_idx+1}"
            neuron_dir.mkdir(parents=True, exist_ok=True)

            ws = []
            for k, (tr, _va) in enumerate(folds_idx, start=1):
                w = _fit_one_fold_weights_poisson(X_all, y, tr, reg_spec)
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
        except Exception as e:  # pragma: no cover - logging path
            return False, str(e)

    N_NEURONS = Y_all.shape[1]
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_neuron)(i) for i in tqdm(range(N_NEURONS), desc=f"{out_root.name} | FULL_FIT ({model_key})")
    )

    bad = [(i, msg) for i, (ok, msg) in enumerate(res) if not ok]
    if bad:
        logp = model_dir / "full_fit_failures.txt"
        with open(logp, "w", encoding="utf-8") as f:
            for i, msg in bad:
                f.write(f"neuron_{i+1}\t{msg}\n")
