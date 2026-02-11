import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm

from .config import FULL_FIT_DIRNAME, MAX_ITER, N_JOBS, POISSON_ALPHA
from .design_matrix import build_smoothness_rows, ensure_feature_mapping, model_key_from_vars
from .metrics import build_oof_constant_mu, compute_deviance_explained_poisson_vs_baseline


def _augment_with_smoothness(
    X: sparse.csr_matrix,
    y: np.ndarray,
    feature_names: List[str],
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    smooth_rows = build_smoothness_rows(feature_names)
    if smooth_rows.shape[0] == 0:
        return X, y
    y_smooth = np.zeros(smooth_rows.shape[0], dtype=y.dtype)
    X_aug = sparse.vstack([X, smooth_rows], format="csr")
    y_aug = np.concatenate([y, y_smooth])
    return X_aug, y_aug


def fit_predict_one_fold_poisson(
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, float]:
    Xtr, Xva = X_all[tr_idx], X_all[va_idx]
    ytr, yva = y_all[tr_idx].astype(np.float64), y_all[va_idx].astype(np.float64)

    mean_tr = float(np.mean(ytr))
    base_rate = max(mean_tr, 1e-12)
    if mean_tr <= 0:
        mu_va = np.full_like(yva, 1e-12, dtype=np.float64)
        mu_base = np.full_like(yva, base_rate, dtype=np.float64)
        dev_expl = compute_deviance_explained_poisson_vs_baseline(yva, mu_va, mu_base)
        return mu_va.astype(np.float32), float(dev_expl)

    Xtr_aug, ytr_aug = _augment_with_smoothness(Xtr, ytr, feature_names)
    mdl = PoissonRegressor(
        alpha=POISSON_ALPHA,
        max_iter=MAX_ITER,
        fit_intercept=True,  # intercept is not part of X/feature_names; smoothing rows exclude it
    )
    mdl.fit(Xtr_aug, ytr_aug)
    mu_va = np.clip(mdl.predict(Xva).astype(np.float64), 1e-12, None)

    mu_base = np.full_like(yva, base_rate, dtype=np.float64)
    dev_expl = compute_deviance_explained_poisson_vs_baseline(yva, mu_va, mu_base)
    return mu_va.astype(np.float32), float(dev_expl)


def _fit_one_fold_weights_poisson(
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
    feature_names: List[str],
) -> np.ndarray:
    """Return w = [coef..., intercept] for one fold (fit on train only)."""
    Xtr = X_all[tr_idx]
    ytr = y_all[tr_idx].astype(np.float64)

    mean_tr = float(np.mean(ytr))
    if mean_tr <= 0:
        w = np.zeros(Xtr.shape[1] + 1, dtype=np.float32)
        w[-1] = np.log(1e-12)
        return w

    Xtr_aug, ytr_aug = _augment_with_smoothness(Xtr, ytr, feature_names)
    mdl = PoissonRegressor(
        alpha=POISSON_ALPHA,
        max_iter=MAX_ITER,
        fit_intercept=True,  # intercept is not part of X/feature_names; smoothing rows exclude it
    )
    mdl.fit(Xtr_aug, ytr_aug)
    w = np.concatenate(
        [mdl.coef_.ravel().astype(np.float32), np.array([mdl.intercept_], dtype=np.float32)]
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
) -> Dict:
    neuron_dir.mkdir(parents=True, exist_ok=True)
    ensure_feature_mapping(str(model_dir), feature_names)

    fold_dev_expl: List[float] = []
    mu_oof = np.full_like(y_all, np.nan, dtype=np.float32)

    for k, (tr, va) in enumerate(folds, start=1):
        fold_dir = neuron_dir / f"fold{k}"
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
            Xtr_aug, ytr_aug = _augment_with_smoothness(Xtr, ytr, feature_names)
            mdl = PoissonRegressor(
                alpha=POISSON_ALPHA,
                max_iter=MAX_ITER,
                fit_intercept=True,  # intercept is not part of X/feature_names; smoothing rows exclude it
            )
            mdl.fit(Xtr_aug, ytr_aug)
            mu_va = np.clip(mdl.predict(Xva).astype(np.float64), 1e-12, None)
            w = np.concatenate(
                [mdl.coef_.ravel().astype(np.float32), np.array([mdl.intercept_], dtype=np.float32)]
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

        mu_base = np.full_like(yva, base_rate, dtype=np.float64)
        dev_expl_val = compute_deviance_explained_poisson_vs_baseline(yva, mu_va, mu_base)
        fold_dev_expl.append(float(dev_expl_val))
        pd.DataFrame({"fold": [k], "deviance_explained": [float(dev_expl_val)]}).to_csv(
            fold_dir / "deviance_explained.csv", index=False
        )

        mu_oof[va] = mu_va.astype(np.float32)

    mu_base_oof = build_oof_constant_mu(y_all, folds)
    dev_expl_oof = compute_deviance_explained_poisson_vs_baseline(y_all, mu_oof, mu_base_oof)
    mean_dev_expl = float(np.nanmean(fold_dev_expl))

    with open(neuron_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_vars,
                "oof_deviance_explained": float(dev_expl_oof),
                "mean_deviance_explained_over_folds": float(mean_dev_expl),
                "fold_deviance_explained": list(map(float, fold_dev_expl)),
                "poisson_alpha": float(POISSON_ALPHA),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "mean_deviance_explained": float(mean_dev_expl),
        "fold_deviance_explained": list(map(float, fold_dev_expl)),
        "oof_deviance_explained": float(dev_expl_oof),
        "mu_oof": mu_oof,
    }


def save_full_fit_weights_for_all_neurons(
    out_root: Path,
    model_vars: List[str],
    X_all: sparse.csr_matrix,
    feature_names: List[str],
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
                w = _fit_one_fold_weights_poisson(X_all, y, tr, feature_names)
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
