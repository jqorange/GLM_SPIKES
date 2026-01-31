import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm

from .config import FULL_FIT_DIRNAME, MAX_ITER, N_JOBS, POISSON_ALPHA, POSITION_SMOOTH_LAMBDA
from .design_matrix import ensure_feature_mapping, model_key_from_vars
from .metrics import compute_llhi_bps_poisson


def _position_feature_map(feature_names: List[str]) -> Tuple[List[Tuple[int, int]], int]:
    pos_pairs: List[Tuple[int, int]] = []
    max_cat = -1
    for idx, name in enumerate(feature_names):
        if not name.startswith("position_"):
            continue
        try:
            cat = int(name.split("_", 1)[1])
        except ValueError:
            continue
        pos_pairs.append((idx, cat))
        max_cat = max(max_cat, cat)
    n_pos = max_cat + 1
    return pos_pairs, n_pos


def _build_position_laplacian(pos_bins: np.ndarray, n_pos: int) -> sparse.csr_matrix:
    if pos_bins is None or n_pos <= 1:
        return sparse.csr_matrix((n_pos, n_pos), dtype=np.float32)
    coords = {(int(x), int(y)): i for i, (x, y) in enumerate(pos_bins)}
    rows = []
    cols = []
    data = []
    for (x, y), idx in coords.items():
        for dx, dy in ((1, 0), (0, 1)):
            nbr = (x + dx, y + dy)
            jdx = coords.get(nbr)
            if jdx is None:
                continue
            rows.extend([idx, jdx, idx, jdx])
            cols.extend([idx, jdx, jdx, idx])
            data.extend([1.0, 1.0, -1.0, -1.0])
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_pos, n_pos), dtype=np.float32)


def _fit_poisson_model(
    Xtr: sparse.csr_matrix,
    ytr: np.ndarray,
    feature_names: List[str] | None = None,
    pos_bins: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    mean_tr = float(np.mean(ytr))
    if mean_tr <= 0:
        w = np.zeros(Xtr.shape[1], dtype=np.float32)
        b = float(np.log(1e-12))
        return w, b

    smooth_lambda = float(POSITION_SMOOTH_LAMBDA)
    pos_pairs, n_pos = _position_feature_map(feature_names or [])
    use_smoothing = smooth_lambda > 0 and pos_pairs and n_pos > 1 and pos_bins is not None
    if not use_smoothing:
        mdl = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=MAX_ITER, fit_intercept=True)
        mdl.fit(Xtr, ytr)
        return mdl.coef_.ravel().astype(np.float32), float(mdl.intercept_)

    pos_laplacian = _build_position_laplacian(pos_bins, n_pos)

    def objective(params: np.ndarray) -> Tuple[float, np.ndarray]:
        w = params[:-1]
        b = params[-1]
        eta = Xtr.dot(w) + b
        mu = np.exp(eta)
        residual = mu - ytr
        nll = float(np.sum(mu - ytr * eta))
        grad_w = Xtr.T.dot(residual) + 2.0 * POISSON_ALPHA * w
        grad_b = float(np.sum(residual))

        if pos_laplacian.shape[0] > 1:
            w_full = np.zeros(n_pos, dtype=np.float64)
            for feat_idx, cat in pos_pairs:
                if cat < n_pos:
                    w_full[cat] = w[feat_idx]
            lap_w = pos_laplacian.dot(w_full)
            nll += smooth_lambda * float(w_full.dot(lap_w))
            grad_full = 2.0 * smooth_lambda * lap_w
            for feat_idx, cat in pos_pairs:
                if cat < n_pos:
                    grad_w[feat_idx] += grad_full[cat]

        grad = np.concatenate([np.asarray(grad_w).ravel(), np.array([grad_b])])
        return nll, grad

    w0 = np.zeros(Xtr.shape[1] + 1, dtype=np.float64)
    w0[-1] = np.log(mean_tr + 1e-12)
    res = minimize(
        fun=lambda p: objective(p),
        x0=w0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": MAX_ITER},
    )
    w_opt = res.x.astype(np.float32)
    return w_opt[:-1], float(w_opt[-1])


def fit_predict_one_fold_poisson(
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    feature_names: List[str] | None = None,
    pos_bins: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:
    Xtr, Xva = X_all[tr_idx], X_all[va_idx]
    ytr, yva = y_all[tr_idx].astype(np.float64), y_all[va_idx].astype(np.float64)

    w, b = _fit_poisson_model(Xtr, ytr, feature_names=feature_names, pos_bins=pos_bins)
    mu_va = np.clip(np.exp(Xva.dot(w) + b).astype(np.float64), 1e-12, None)

    llhi = compute_llhi_bps_poisson(yva, mu_va)
    return mu_va.astype(np.float32), float(llhi)


def _fit_one_fold_weights_poisson(
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    feature_names: List[str] | None = None,
    pos_bins: np.ndarray | None = None,
) -> np.ndarray:
    """Return w = [coef..., intercept] for one fold (fit on train only)."""
    Xtr = X_all[tr_idx]
    ytr = y_all[tr_idx].astype(np.float64)

    w_coef, w_intercept = _fit_poisson_model(
        Xtr,
        ytr,
        feature_names=feature_names,
        pos_bins=pos_bins,
    )
    w = np.concatenate(
        [w_coef.astype(np.float32), np.array([w_intercept], dtype=np.float32)]
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
    pos_bins: np.ndarray | None = None,
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

        w_coef, w_intercept = _fit_poisson_model(
            Xtr,
            ytr,
            feature_names=feature_names,
            pos_bins=pos_bins,
        )
        mu_va = np.clip(np.exp(Xva.dot(w_coef) + w_intercept).astype(np.float64), 1e-12, None)
        w = np.concatenate(
            [w_coef.astype(np.float32), np.array([w_intercept], dtype=np.float32)]
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
    pos_bins: np.ndarray | None = None,
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
                w = _fit_one_fold_weights_poisson(
                    X_all,
                    y,
                    tr,
                    feature_names=feature_names,
                    pos_bins=pos_bins,
                )
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
