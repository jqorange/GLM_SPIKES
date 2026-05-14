from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm

from glm_poisson_forward.config import (
    CONTRIB_FIT_SIGNATURE,
    CV_FOLDS,
    MAX_ITER,
    POISSON_ALPHA,
)
from glm_poisson_forward.design_matrix import ensure_feature_mapping
from glm_poisson_forward.training import _fit_one_fold_weights_poisson

from .constants import MU_EPS


def ensure_feature_names_file(model_dir: Path, feature_names: List[str]):
    model_dir.mkdir(parents=True, exist_ok=True)
    p = model_dir / "feature_names.json"
    if not p.exists():
        with open(p, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2)


def _load_feature_mapping_file(model_dir: Path) -> List[str] | None:
    mapping_path = model_dir / "feature_mapping.txt"
    if not mapping_path.exists():
        return None
    mapping: Dict[int, str] = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            idx_str, name = line.split(":", 1)
            try:
                idx = int(idx_str.strip())
            except ValueError:
                continue
            mapping[idx] = name.strip()
    if not mapping:
        return None
    return [mapping[i] for i in sorted(mapping)]


def load_feature_names_file(model_dir: Path) -> List[str] | None:
    p = model_dir / "feature_names.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return _load_feature_mapping_file(model_dir)


def _fit_spec_path(model_dir: Path) -> Path:
    return Path(model_dir) / "fit_spec.json"


def load_fit_signature(model_dir: Path) -> str | None:
    fit_spec_path = _fit_spec_path(model_dir)
    if not fit_spec_path.exists():
        return None
    try:
        with open(fit_spec_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    sig = payload.get("fit_signature")
    return str(sig) if sig else None


def ensure_fit_signature_file(model_dir: Path, fit_signature: str) -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    fit_spec_path = _fit_spec_path(model_dir)
    payload = {"fit_signature": str(fit_signature)}
    with open(fit_spec_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def fit_signature_matches(
    model_dir: Path,
    expected_fit_signature: str | None,
    *,
    allow_legacy_forward: bool = False,
) -> bool:
    if expected_fit_signature is None:
        return True
    current = load_fit_signature(model_dir)
    if current is None:
        return bool(allow_legacy_forward and expected_fit_signature == CONTRIB_FIT_SIGNATURE)
    return current == expected_fit_signature


def fit_one_fold_weights(
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
) -> np.ndarray:
    """
    Return w = [coef..., intercept] fitted on train indices only.
    """
    Xtr = X_all[tr_idx]
    ytr = y_all[tr_idx].astype(np.float64)

    mean_tr = float(np.mean(ytr))
    if mean_tr <= 0:
        w = np.zeros(Xtr.shape[1] + 1, dtype=np.float32)
        w[-1] = np.log(MU_EPS)
        return w

    mdl = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=MAX_ITER, fit_intercept=True)
    mdl.fit(Xtr, ytr)
    w = np.concatenate([mdl.coef_.ravel().astype(np.float32), np.array([mdl.intercept_], dtype=np.float32)])
    return w


def weights_exist_for_neuron(
    model_dir: Path,
    neuron_idx1: int,
    folds_count: int,
    *,
    expected_fit_signature: str | None = None,
    allow_legacy_forward: bool = False,
) -> bool:
    if not fit_signature_matches(
        model_dir,
        expected_fit_signature,
        allow_legacy_forward=allow_legacy_forward,
    ):
        return False
    nd = model_dir / f"neuron_{neuron_idx1}"
    if not nd.exists():
        return False
    for k in range(1, folds_count + 1):
        if not (nd / f"fold{k}" / "weights.csv").exists():
            return False
    return True


def save_weights_for_model(
    model_dir: Path,
    feature_names: List[str],
    X_all: sparse.csr_matrix,
    Y_all: np.ndarray,
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
    neuron_indices: np.ndarray,
    n_jobs: int,
    folds_count: int | None = None,
    position_xy_by_idx: np.ndarray | None = None,
    use_forward_fit: bool = False,
    fit_signature: str | None = None,
    force_recompute: bool = False,
):
    """
    Ensure weights exist for all neuron_indices for this model.
    Only trains missing neurons (caches per neuron).

    When ``use_forward_fit`` is True, reuse the same per-fold fitting logic as
    glm_poisson_forward so auto-backfilled full-model weights stay compatible
    with the original forward-search pipeline.
    """
    ensure_feature_names_file(model_dir, feature_names)
    ensure_feature_mapping(str(model_dir), feature_names)
    folds_count = folds_count or CV_FOLDS
    if fit_signature is None:
        fit_signature = CONTRIB_FIT_SIGNATURE if use_forward_fit else "legacy_poisson_regressor"
    ensure_fit_signature_file(model_dir, fit_signature)

    def _one_neuron(neuron_idx: int) -> Tuple[bool, str]:
        idx1 = neuron_idx + 1
        if (not force_recompute) and weights_exist_for_neuron(
            model_dir,
            idx1,
            folds_count,
            expected_fit_signature=fit_signature,
        ):
            return True, "CACHED"

        try:
            y = Y_all[:, neuron_idx].astype(np.float64)
            nd = model_dir / f"neuron_{idx1}"
            nd.mkdir(parents=True, exist_ok=True)

            ws = []
            for k, (tr, _va) in enumerate(folds_idx, start=1):
                if use_forward_fit:
                    w = _fit_one_fold_weights_poisson(
                        X_all,
                        y,
                        tr,
                        feature_names,
                        position_xy_by_idx=position_xy_by_idx,
                    )
                else:
                    w = fit_one_fold_weights(X_all, y, tr)
                ws.append(w)

                fd = nd / f"fold{k}"
                fd.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    w.reshape(1, -1),
                    index=[f"neuron_{idx1}"],
                    columns=feature_names,
                ).to_csv(fd / "weights.csv")

            w_mean = np.mean(np.stack(ws, axis=0), axis=0).astype(np.float32)
            pd.DataFrame(
                w_mean.reshape(1, -1),
                index=[f"neuron_{idx1}"],
                columns=feature_names,
            ).to_csv(nd / "weights_mean.csv")

            return True, "OK"
        except Exception as e:  # pylint: disable=broad-except
            return False, str(e)

    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_neuron)(i) for i in tqdm(neuron_indices.tolist(), desc=f"fit weights | {model_dir.parent.name}/{model_dir.name}")
    )

    bad = [(i, msg) for i, (ok, msg) in zip(neuron_indices.tolist(), res) if not ok]
    if bad:
        with open(model_dir / "failures.txt", "w", encoding="utf-8") as f:
            for i, msg in bad:
                f.write(f"neuron_{i+1}\t{msg}\n")


def load_fold_weights(csv_path: Path, feature_names: Iterable[str]) -> np.ndarray:
    """
    Read weights saved by pandas.DataFrame(...).to_csv("weights.csv") (wide 1-row format),
    OR long key/value format. Return ordered weight vector aligned to feature_names.

    Expected feature_names includes intercept as the last name.
    """
    csv_path = Path(csv_path)
    if (not csv_path.exists()) or csv_path.stat().st_size < 8:
        raise ValueError(f"Missing or too-small weights file: {csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if len(r) > 0]

    if len(rows) < 2:
        raise ValueError(f"Corrupted weights CSV (need >=2 rows): {csv_path}")

    header = [c.strip() for c in rows[0]]
    first_data = [c.strip() for c in rows[1]]

    def to_float(x: str) -> float:
        if x is None or x == "":
            return 0.0
        return float(x)

    out: Dict[str, float] = {}

    header_lc = [h.lower() for h in header]
    is_key_value = (
        len(header) == 2
        and len(first_data) == 2
        and (
            ("name" in header_lc[0] and "value" in header_lc[1])
            or ("feature" in header_lc[0] and "value" in header_lc[1])
            or ("key" in header_lc[0] and "value" in header_lc[1])
        )
    )

    if is_key_value:
        for r in rows[1:]:
            if len(r) < 2:
                continue
            k = r[0].strip()
            v = r[1].strip()
            if k == "":
                continue
            out[k] = to_float(v)
    else:
        if len(first_data) != len(header):
            if len(first_data) < len(header):
                first_data = first_data + [""] * (len(header) - len(first_data))
            else:
                first_data = first_data[: len(header)]

        for k, v in zip(header, first_data):
            if k == "":
                continue
            if k.lower().startswith("unnamed"):
                continue
            out[k] = to_float(v)

    names: List[str] = list(feature_names)
    for n in names:
        if n not in out:
            out[n] = 0.0

    w_vec = np.asarray([out[n] for n in names], dtype=np.float64)
    return w_vec


def predict_oof_from_saved_weights(
    model_dir: Path,
    X_all: sparse.csr_matrix,
    feature_names_now: List[str],
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
    neuron_idx: int,
) -> np.ndarray:
    idx1 = neuron_idx + 1
    neuron_dir = model_dir / f"neuron_{idx1}"

    saved_feats = load_feature_names_file(model_dir)
    if saved_feats is None:
        raise FileNotFoundError(f"Missing feature_names.json in {model_dir}")
    if saved_feats != feature_names_now:
        raise ValueError(
            f"Feature name mismatch for {model_dir}.\n"
            f"Saved: {len(saved_feats)} cols, Now: {len(feature_names_now)} cols."
        )

    T = X_all.shape[0]
    mu_oof = np.full(T, np.nan, dtype=np.float64)

    for k, (_tr, va) in enumerate(folds_idx, start=1):
        csv_path = neuron_dir / f"fold{k}" / "weights.csv"
        w_vec = load_fold_weights(csv_path, feature_names=saved_feats)  # 1D vector

        coef = w_vec[:-1]
        intercept = w_vec[-1]

        Xva = X_all[va]
        eta = (Xva @ coef).astype(np.float64) + float(intercept)
        mu_va = np.exp(eta)
        mu_va = np.clip(mu_va, MU_EPS, None)
        mu_oof[va] = mu_va

    if np.any(~np.isfinite(mu_oof)):
        mu_oof = np.where(np.isfinite(mu_oof), mu_oof, MU_EPS)
    return mu_oof
