import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import optimize, sparse
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm

from .config import (
    FULL_FIT_DIRNAME,
    L1_LAMBDA,
    L1_PROX_LR,
    L1_PROX_STEPS,
    MAX_ITER,
    N_JOBS,
    POISSON_ALPHA,
    SKIP_SMALL_ARTIFACT_WRITES,
)
from .design_matrix import build_smoothness_rows, ensure_feature_mapping, model_key_from_vars
from .metrics import build_oof_constant_mu, compute_deviance_explained_poisson_vs_baseline


def _build_smoothness_matrix(
    feature_names: List[str],
    position_xy_by_idx: np.ndarray | None = None,
) -> sparse.csr_matrix:
    return build_smoothness_rows(feature_names, position_xy_by_idx=position_xy_by_idx)


def _soft_threshold(w: np.ndarray, thr: float) -> np.ndarray:
    return np.sign(w) * np.maximum(np.abs(w) - thr, 0.0)


def _prox_refine_poisson_l1(
    X_data: sparse.csr_matrix,
    y_data: np.ndarray,
    smooth_rows: sparse.csr_matrix,
    w_init: np.ndarray,
    b_init: float,
) -> Tuple[np.ndarray, float]:
    """Warm-start proximal-gradient refinement for Poisson + L1 on coefficients."""
    if L1_PROX_STEPS <= 0 or L1_LAMBDA <= 0:
        return w_init.astype(np.float64, copy=False), float(b_init)

    w = w_init.astype(np.float64, copy=True)
    b = float(b_init)
    n = float(max(X_data.shape[0] + smooth_rows.shape[0], 1))
    step = float(L1_PROX_LR)

    for _ in range(int(L1_PROX_STEPS)):
        eta_data = np.asarray(X_data.dot(w)).ravel() + b
        np.clip(eta_data, -20.0, 20.0, out=eta_data)
        mu_data = np.exp(eta_data)
        residual_data = mu_data - y_data

        grad_w = np.asarray(X_data.T.dot(residual_data)).ravel()
        if smooth_rows.shape[0] > 0:
            eta_smooth = np.asarray(smooth_rows.dot(w)).ravel()
            np.clip(eta_smooth, -20.0, 20.0, out=eta_smooth)
            mu_smooth = np.exp(eta_smooth)
            grad_w += np.asarray(smooth_rows.T.dot(mu_smooth)).ravel()

        grad_w = grad_w / n + float(POISSON_ALPHA) * w
        grad_b = float(np.sum(residual_data) / n)

        w = _soft_threshold(w - step * grad_w, step * float(L1_LAMBDA))
        b = b - step * grad_b

    return w, b


def _fit_poisson_with_prox_l1(
    X_data: sparse.csr_matrix,
    y_data: np.ndarray,
    smooth_rows: sparse.csr_matrix,
) -> Tuple[np.ndarray, float]:
    mdl = PoissonRegressor(
        alpha=POISSON_ALPHA,
        max_iter=MAX_ITER,
        fit_intercept=True,
    )
    mdl.fit(X_data, y_data)
    w0 = mdl.coef_.ravel().astype(np.float64, copy=False)
    b0 = float(mdl.intercept_)

    n_total = float(max(X_data.shape[0] + smooth_rows.shape[0], 1))
    alpha = float(POISSON_ALPHA)

    def _objective_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
        w = params[:-1]
        b = float(params[-1])

        eta_data = np.asarray(X_data.dot(w)).ravel() + b
        np.clip(eta_data, -20.0, 20.0, out=eta_data)
        mu_data = np.exp(eta_data)

        loss = float(np.sum(mu_data - y_data * eta_data))
        grad_w = np.asarray(X_data.T.dot(mu_data - y_data)).ravel()
        grad_b = float(np.sum(mu_data - y_data))

        if smooth_rows.shape[0] > 0:
            eta_smooth = np.asarray(smooth_rows.dot(w)).ravel()
            np.clip(eta_smooth, -20.0, 20.0, out=eta_smooth)
            mu_smooth = np.exp(eta_smooth)
            loss += float(np.sum(mu_smooth))
            grad_w += np.asarray(smooth_rows.T.dot(mu_smooth)).ravel()

        loss = loss / n_total + 0.5 * alpha * float(np.dot(w, w))
        grad_w = grad_w / n_total + alpha * w
        grad_b = grad_b / n_total
        grad = np.concatenate([grad_w, np.array([grad_b], dtype=np.float64)])
        return loss, grad

    x0 = np.concatenate([w0, np.array([b0], dtype=np.float64)])
    opt = optimize.minimize(
        fun=_objective_and_grad,
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(MAX_ITER)},
    )
    if not opt.success and not np.isfinite(opt.fun):
        raise RuntimeError(f"Poisson fit failed: {opt.message}")

    w_opt = np.asarray(opt.x[:-1], dtype=np.float64)
    b_opt = float(opt.x[-1])
    return _prox_refine_poisson_l1(
        X_data,
        y_data.astype(np.float64, copy=False),
        smooth_rows,
        w_opt,
        b_opt,
    )


def fit_predict_one_fold_poisson(
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    feature_names: List[str],
    position_xy_by_idx: np.ndarray | None = None,
) -> Tuple[np.ndarray, float]:

    Xtr, Xva = X_all[tr_idx], X_all[va_idx]
    ytr, yva = y_all[tr_idx].astype(np.float64), y_all[va_idx].astype(np.float64)
    mean_tr = float(np.mean(ytr))
    base_rate = max(mean_tr, 1e-12)
    if mean_tr <= 0:
        mu_va = np.full_like(yva, 1e-12, dtype=np.float64)
        mu_base = np.full_like(yva, base_rate, dtype=np.float64)
        dev_exp = compute_deviance_explained_poisson_vs_baseline(yva, mu_va, mu_base)
        return mu_va.astype(np.float32), float(dev_exp)

    smooth_rows = _build_smoothness_matrix(
        feature_names,
        position_xy_by_idx=position_xy_by_idx,
    )
    coef, intercept = _fit_poisson_with_prox_l1(Xtr, ytr, smooth_rows)
    mu_va = np.clip(np.exp(Xva.dot(coef) + intercept).astype(np.float64), 1e-12, None)

    mu_base = np.full_like(yva, base_rate, dtype=np.float64)
    dev_exp = compute_deviance_explained_poisson_vs_baseline(yva, mu_va, mu_base)
    return mu_va.astype(np.float32), float(dev_exp)


def _fit_one_fold_weights_poisson(
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    feature_names: List[str],
    position_xy_by_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Return w = [coef..., intercept] for one fold (fit on train only)."""
    Xtr = X_all[tr_idx]
    ytr = y_all[tr_idx].astype(np.float64)

    mean_tr = float(np.mean(ytr))
    if mean_tr <= 0:
        w = np.zeros(Xtr.shape[1] + 1, dtype=np.float32)
        w[-1] = np.log(1e-12)
        return w

    smooth_rows = _build_smoothness_matrix(
        feature_names,
        position_xy_by_idx=position_xy_by_idx,
    )
    coef, intercept = _fit_poisson_with_prox_l1(Xtr, ytr, smooth_rows)
    w = np.concatenate(
        [coef.astype(np.float32), np.array([intercept], dtype=np.float32)]
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
    position_xy_by_idx: np.ndarray | None = None,
) -> Dict:
    neuron_dir.mkdir(parents=True, exist_ok=True)
    ensure_feature_mapping(str(model_dir), feature_names)

    fold_dev_exp: List[float] = []
    mu_oof = np.full_like(y_all, np.nan, dtype=np.float32)

    for k, (tr, va) in enumerate(folds, start=1):
        fold_dir = neuron_dir / f"fold{k}"
        if not SKIP_SMALL_ARTIFACT_WRITES:
            fold_dir.mkdir(parents=True, exist_ok=True)

        Xtr, Xva = X_all[tr], X_all[va]
        ytr, yva = y_all[tr].astype(np.float64), y_all[va].astype(np.float64)

        mean_tr = float(np.mean(ytr))
        base_rate = max(mean_tr, 1e-12)
        if mean_tr <= 0:
            mu_va = np.full_like(yva, 1e-12, dtype=np.float64)
            w = np.zeros(Xtr.shape[1] + 1, dtype=np.float32)
            w[-1] = np.log(1e-12)
        else:
            smooth_rows = _build_smoothness_matrix(
                feature_names,
                position_xy_by_idx=position_xy_by_idx,
            )
            coef, intercept = _fit_poisson_with_prox_l1(Xtr, ytr, smooth_rows)
            mu_va = np.clip(np.exp(Xva.dot(coef) + intercept).astype(np.float64), 1e-12, None)
            w = np.concatenate(
                [coef.astype(np.float32), np.array([intercept], dtype=np.float32)]
            )

        if not SKIP_SMALL_ARTIFACT_WRITES:
            pd.DataFrame(
                w.reshape(1, -1),
                index=[f"neuron_{neuron_index+1}"],
                columns=feature_names,
            ).to_csv(fold_dir / "weights.csv")

            with h5py.File(fold_dir / "pred.h5", "w") as hf:
                hf.create_dataset("pred_mu", data=mu_va.astype(np.float32), compression="gzip")
                hf.create_dataset("true_cnt", data=yva.astype(np.float32), compression="gzip")
                hf.create_dataset("va_idx", data=np.asarray(va, dtype=np.int64), compression="gzip")

        mu_base = np.full_like(yva, base_rate, dtype=np.float64)
        dev_exp_val = compute_deviance_explained_poisson_vs_baseline(yva, mu_va, mu_base)
        fold_dev_exp.append(float(dev_exp_val))
        if not SKIP_SMALL_ARTIFACT_WRITES:
            pd.DataFrame({"fold": [k], "deviance_explained": [float(dev_exp_val)]}).to_csv(
                fold_dir / "deviance_explained.csv", index=False
            )

        mu_oof[va] = mu_va.astype(np.float32)

    mu_base_oof = build_oof_constant_mu(y_all, folds)
    dev_exp_oof = compute_deviance_explained_poisson_vs_baseline(y_all, mu_oof, mu_base_oof)
    mean_dev_exp = float(np.nanmean(fold_dev_exp))

    with open(neuron_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_vars,
                "oof_deviance_explained": float(dev_exp_oof),
                "mean_deviance_explained_over_folds": float(mean_dev_exp),
                "fold_deviance_explained": list(map(float, fold_dev_exp)),
                "poisson_alpha": float(POISSON_ALPHA),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "mean_deviance_explained": float(mean_dev_exp),
        "fold_deviance_explained": list(map(float, fold_dev_exp)),
        "oof_deviance_explained": float(dev_exp_oof),
        "mu_oof": mu_oof,
    }


def save_full_fit_weights_for_all_neurons(
    out_root: Path,
    model_vars: List[str],
    X_all: sparse.csr_matrix,
    feature_names: List[str],
    position_xy_by_idx: np.ndarray | None,
    Y_all: np.ndarray,
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
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
                    feature_names,
                    position_xy_by_idx=position_xy_by_idx,
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
