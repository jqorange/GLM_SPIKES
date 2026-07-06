#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Iterable

# Prevent BLAS/OpenMP oversubscription before importing numpy/scipy.
for _env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_env_name, "1")

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import gammaln

from contribution_rllr_utils.selection import load_forward_selected_models
from contribution_rllr_utils.stats import build_oof_intercept_mu
from contribution_rllr_utils.weights import (
    load_feature_names_file,
    load_fold_weights,
    weights_exist_for_neuron,
)
from glm_poisson_forward import build_cv_folds
from glm_poisson_forward.config import FS_HZ, MAX_MISMATCH_FRAMES_50HZ
from glm_poisson_forward.design_matrix import build_design_matrix, model_key_from_vars
from glm_poisson_forward.io_utils import (
    load_spikes_50hz_counts,
    prepare_session_for_modeling,
    rebuild_inputs_50hz,
    session_paths,
)
from glm_poisson_forward.training import _fit_one_fold_weights_poisson

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_BASE = REPO_ROOT / "weights_Poisson_forward"
DEFAULT_OUTPUT_DIR = DEFAULT_WEIGHTS_BASE / "MS_SCORE_CONTINUOUS"
FILTERED_NEURON_CSV = "filtered_neuron_ids_ALL.csv"
EPS = 1e-12
CANONICAL_FEATURES = ("Position", "Speed", "roll", "pitch", "yaw")
FULL_LENGTH_SCORE_TOL_FRAMES = 50


def _default_n_jobs() -> int:
    return max(1, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute continuous-time MS-score sequences using full models refit "
            "with the exact glm_poisson_forward training path and saved drop-one weights."
        )
    )
    parser.add_argument(
        "--weights-base",
        type=Path,
        default=DEFAULT_WEIGHTS_BASE,
        help=f"Base weights directory (default: {DEFAULT_WEIGHTS_BASE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Optional session name. Pass multiple times to restrict processing.",
    )
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Only write neurons that appear in selected_models.csv.",
    )
    parser.add_argument(
        "--write-long-csv",
        action="store_true",
        help="Also write a long CSV with one row per neuron per bin. This can be very large.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise an error if any selected neuron fails full-model refit or is missing saved drop-one weights.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=_default_n_jobs(),
        help=(
            "Deprecated for session scheduling. The script now uses one process "
            "per session, so the effective worker count equals the number of sessions. "
            "This flag is kept only for backward compatibility."
            f"(default: {_default_n_jobs()})"
        ),
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=EPS,
        help=f"Small epsilon used in the MS-score formula (default: {EPS})",
    )
    return parser.parse_args()


def discover_sessions(weights_base: Path, requested_sessions: list[str] | None) -> list[str]:
    if requested_sessions:
        return sorted(set(requested_sessions))

    sessions: list[str] = []
    for session_dir in sorted(weights_base.iterdir()):
        if not session_dir.is_dir():
            continue
        if (session_dir / "selected_models.csv").exists():
            sessions.append(session_dir.name)
    return sorted(set(sessions))


def split_model_string(model_str: str) -> list[str]:
    model_str = str(model_str).strip()
    if not model_str:
        return []
    return [token for token in model_str.replace(",", "_").split("_") if token]


def load_selected_model_neurons(session_dir: Path) -> set[int]:
    selected_csv = session_dir / "selected_models.csv"
    if not selected_csv.exists():
        return set()

    neuron_ids: set[int] = set()
    with open(selected_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            neuron_name = str(row.get("neuron", "")).strip()
            model_str = str(row.get("final_model", "")).strip()
            if not neuron_name or not model_str:
                continue
            if not neuron_name.lower().startswith("neuron_"):
                continue
            if not split_model_string(model_str):
                continue
            try:
                neuron_idx = int(neuron_name.split("_", maxsplit=1)[1]) - 1
            except Exception:
                continue
            if neuron_idx >= 0:
                neuron_ids.add(neuron_idx)
    return neuron_ids


def load_unclassified_neurons(session_dir: Path) -> set[int]:
    txt_path = session_dir / "unclassified_neurons.txt"
    if not txt_path.exists():
        return set()

    neuron_ids: set[int] = set()
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("neuron_"):
            continue
        try:
            neuron_idx = int(line.split("_", maxsplit=1)[1]) - 1
        except Exception:
            continue
        if neuron_idx >= 0:
            neuron_ids.add(neuron_idx)
    return neuron_ids


def load_filtered_neurons(weights_base: Path, session: str) -> set[int]:
    filtered_csv = weights_base / FILTERED_NEURON_CSV
    if not filtered_csv.exists():
        return set()

    df = pd.read_csv(filtered_csv)
    if "session" not in df.columns:
        return set()

    df = df[df["session"].astype(str) == str(session)]
    if df.empty:
        return set()

    idx_col = "neuron_idx_0based" if "neuron_idx_0based" in df.columns else "neuron_idx"
    ids = pd.to_numeric(df[idx_col], errors="coerce").dropna().astype(int)
    return set(ids.tolist())


def infer_all_neuron_ids(weights_base: Path, session: str, session_dir: Path) -> list[int]:
    known_ids: set[int] = set()
    known_ids |= load_selected_model_neurons(session_dir)
    known_ids |= load_filtered_neurons(weights_base, session)
    known_ids |= load_unclassified_neurons(session_dir)

    if not known_ids:
        raise ValueError(f"{session}: could not infer neuron ids from selected/filtered/unclassified metadata.")

    max_idx = max(known_ids)
    return list(range(max_idx + 1))


def ordered_unique(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        label = str(value).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered


def infer_group(session: str) -> str:
    if "_" in session:
        return session.rsplit("_", maxsplit=1)[-1]
    return ""


def poisson_loglik_series(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    mu = np.clip(mu, EPS, None)
    return y * np.log(mu) - mu - gammaln(y + 1.0)


def ensure_saved_weights(
    model_dir: Path | None,
    neuron_idx: int,
    folds_count: int,
) -> bool:
    if model_dir is None:
        return False
    return weights_exist_for_neuron(
        model_dir,
        neuron_idx + 1,
        folds_count,
        expected_fit_signature=None,
    )


def predict_oof_by_refitting_full_model(
    x_train,
    y_train: np.ndarray,
    x_score,
    feature_names: list[str],
    folds_idx,
    *,
    position_xy_by_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Fit each fold on training data and average predictions on the full scoring session."""
    t_score = int(x_score.shape[0])
    mu_sum = np.zeros(t_score, dtype=np.float64)
    n_folds = 0
    for tr_idx, _va_idx in folds_idx:
        w = _fit_one_fold_weights_poisson(
            x_train,
            y_train,
            tr_idx,
            feature_names,
            position_xy_by_idx=position_xy_by_idx,
        )
        w = np.asarray(w, dtype=np.float64)
        coef = w[:-1]
        intercept = float(w[-1])
        eta = (x_score @ coef).astype(np.float64) + intercept
        mu_fold = np.exp(eta)
        mu_fold = np.clip(mu_fold, EPS, None)
        mu_sum += mu_fold
        n_folds += 1
    if n_folds <= 0:
        return np.full(t_score, EPS, dtype=np.float64)
    mu_mean = mu_sum / float(n_folds)
    if np.any(~np.isfinite(mu_mean)):
        mu_mean = np.where(np.isfinite(mu_mean), mu_mean, EPS)
    return mu_mean


def predict_mean_from_saved_weights(
    model_dir: Path,
    x_score,
    feature_names_now: list[str],
    folds_count: int,
    neuron_idx: int,
) -> np.ndarray:
    idx1 = neuron_idx + 1
    neuron_dir = model_dir / f"neuron_{idx1}"

    saved_feats = load_feature_names_file(model_dir)
    if saved_feats is None:
        raise FileNotFoundError(f"Missing feature_names.json in {model_dir}")
    if saved_feats != feature_names_now:
        raise ValueError(
            f"Feature name mismatch for {model_dir}. "
            f"Saved {len(saved_feats)} cols vs current {len(feature_names_now)} cols."
        )

    t_score = int(x_score.shape[0])
    mu_sum = np.zeros(t_score, dtype=np.float64)
    for k in range(1, int(folds_count) + 1):
        csv_path = neuron_dir / f"fold{k}" / "weights.csv"
        w_vec = load_fold_weights(csv_path, feature_names=saved_feats)
        coef = w_vec[:-1]
        intercept = float(w_vec[-1])
        eta = (x_score @ coef).astype(np.float64) + intercept
        mu_fold = np.exp(eta)
        mu_fold = np.clip(mu_fold, EPS, None)
        mu_sum += mu_fold
    mu_mean = mu_sum / float(folds_count)
    if np.any(~np.isfinite(mu_mean)):
        mu_mean = np.where(np.isfinite(mu_mean), mu_mean, EPS)
    return mu_mean


def _threadpool_limit_context():
    if threadpool_limits is None:
        class _NoOpContext:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _NoOpContext()
    return threadpool_limits(limits=1)


def compute_ms_components(
    positive_delta_by_feature: dict[str, np.ndarray],
    *,
    features: list[str],
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not features:
        raise ValueError("No features available for MS-score computation.")

    shape_ref = positive_delta_by_feature[features[0]].shape
    t_i = np.zeros(shape_ref, dtype=np.float64)
    positive_feature_count = np.zeros(shape_ref, dtype=np.int16)
    p_by_feature: dict[str, np.ndarray] = {}

    for feat in features:
        arr = np.asarray(positive_delta_by_feature[feat], dtype=np.float64)
        t_i += arr
        positive_feature_count += (arr > 0).astype(np.int16)

    if len(features) > 1:
        d_i = np.zeros(shape_ref, dtype=np.float64)
        denom = t_i + float(eps)
        for feat in features:
            p = np.divide(
                positive_delta_by_feature[feat],
                denom,
                out=np.zeros(shape_ref, dtype=np.float64),
                where=t_i > 0,
            )
            p_by_feature[feat] = p.astype(np.float32, copy=False)
            d_i -= p * np.log(p + float(eps))
        d_i /= math.log(len(features))
        d_i = np.where(t_i > 0, d_i, 0.0)
    else:
        d_i = np.zeros(shape_ref, dtype=np.float64)
        p_by_feature[features[0]] = np.where(t_i > 0, 1.0, 0.0).astype(np.float32, copy=False)

    d_i = np.clip(d_i, 0.0, None)
    d_i = np.where(positive_feature_count > 1, d_i, 0.0)
    ms_score = np.clip(t_i * d_i, 0.0, None)
    return (
        t_i.astype(np.float32, copy=False),
        d_i.astype(np.float32, copy=False),
        ms_score.astype(np.float32, copy=False),
        p_by_feature,
    )


def trim_data_dict_time_axis(data_dict: dict[str, np.ndarray], target_len: int) -> dict[str, np.ndarray]:
    original_t = int(data_dict.get("T", 0))
    out: dict[str, np.ndarray] = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == original_t:
            out[key] = value[:target_len]
        else:
            out[key] = value
    out["T"] = int(target_len)
    return out


def prepare_session_for_full_length_scoring(
    session: str,
    paths: dict[str, object],
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    data_dict = rebuild_inputs_50hz(session, paths)
    y_all = load_spikes_50hz_counts(paths["spike"]).astype(np.float64)
    t_spk, n_neurons = y_all.shape
    t_cov = int(data_dict["T"])
    if abs(t_cov - t_spk) > MAX_MISMATCH_FRAMES_50HZ:
        raise ValueError(
            f"{session}: full-length scoring mismatch @50Hz (> {MAX_MISMATCH_FRAMES_50HZ}): "
            f"cov={t_cov}, spk={t_spk}"
        )

    # For tiny end-of-session mismatches, keep start alignment and score on the
    # shared prefix instead of failing the whole session.
    if t_cov != t_spk:
        diff = abs(t_cov - t_spk)
        if diff <= FULL_LENGTH_SCORE_TOL_FRAMES:
            t_common = min(t_cov, t_spk)
            print(
                f"[INFO] {session}: full-length scoring start-aligned with small mismatch "
                f"(cov={t_cov}, spk={t_spk}, using {t_common})"
            )
            if t_cov != t_common:
                data_dict = trim_data_dict_time_axis(data_dict, t_common)
            if t_spk != t_common:
                y_all = y_all[:t_common]
            t_cov = int(data_dict["T"])
            t_spk = int(y_all.shape[0])
        elif t_cov > t_spk:
            data_dict = trim_data_dict_time_axis(data_dict, t_spk)
            t_cov = int(data_dict["T"])
        else:
            raise ValueError(
                f"{session}: covariate length {t_cov} is shorter than spike length {t_spk}; "
                f"cannot score full original spike length safely when mismatch exceeds "
                f"{FULL_LENGTH_SCORE_TOL_FRAMES} frames."
            )

    meta = {
        "t_cov_full": int(t_cov),
        "t_spk_full": int(t_spk),
        "t_score": int(t_spk),
        "n_neurons": int(n_neurons),
    }
    return data_dict, y_all, meta


def build_score_matrix_from_feature_names(
    feature_names: list[str],
    data_dict: dict[str, np.ndarray],
) -> sparse.csr_matrix:
    t = int(data_dict["T"])
    cols: list[sparse.csr_matrix] = []
    for name in feature_names:
        if name == "intercept":
            continue

        arr_name = str(name)
        prefix, suffix = arr_name.rsplit("_", 1) if "_" in arr_name else (arr_name, "")
        is_cat = suffix.isdigit()

        if is_cat:
            level = int(suffix)
            source_key = "position" if prefix == "position" else f"{prefix}_bin"
            if source_key not in data_dict:
                if prefix in data_dict:
                    source_key = prefix
                else:
                    raise KeyError(f"Missing categorical source '{source_key}' for feature '{arr_name}'")
            src = np.asarray(data_dict[source_key]).reshape(-1)
            col = (src == level).astype(np.float32).reshape(t, 1)
            cols.append(sparse.csr_matrix(col))
            continue

        if arr_name not in data_dict:
            raise KeyError(f"Missing continuous source '{arr_name}' in scoring data.")
        src = np.asarray(data_dict[arr_name], dtype=np.float32).reshape(t, 1)
        cols.append(sparse.csr_matrix(src))

    if not cols:
        return sparse.csr_matrix((t, 0), dtype=np.float32)
    return sparse.hstack(cols, format="csr")


def build_score_matrix_for_saved_model(
    model_dir: Path,
    data_dict: dict[str, np.ndarray],
) -> tuple[sparse.csr_matrix, list[str]]:
    saved_feature_names = load_feature_names_file(model_dir)
    if saved_feature_names is None:
        raise FileNotFoundError(f"Missing feature_names.json in {model_dir}")
    x_score = build_score_matrix_from_feature_names(saved_feature_names, data_dict)
    return x_score, list(saved_feature_names)


def write_long_csv(
    out_csv: Path,
    *,
    session: str,
    group: str,
    neuron_ids: np.ndarray,
    time_sec: np.ndarray,
    features: list[str],
    fitted: np.ndarray,
    full_models: np.ndarray,
    status: np.ndarray,
    raw_delta_by_feature: dict[str, np.ndarray],
    positive_delta_by_feature: dict[str, np.ndarray],
    p_by_feature: dict[str, np.ndarray],
    t_i: np.ndarray,
    d_i: np.ndarray,
    ms_score: np.ndarray,
) -> None:
    rows: list[pd.DataFrame] = []
    for row_idx, neuron_idx in enumerate(neuron_ids.tolist()):
        df = pd.DataFrame(
            {
                "session": session,
                "group": group,
                "neuron_idx": int(neuron_idx),
                "time_bin": np.arange(time_sec.shape[0], dtype=int),
                "time_sec": time_sec,
                "fitted": int(fitted[row_idx]),
                "full_model": str(full_models[row_idx]),
                "status": str(status[row_idx]),
                "ms_score": ms_score[row_idx],
                "T_i": t_i[row_idx],
                "D_i": d_i[row_idx],
            }
        )
        for feat in features:
            df[f"delta_ll_{feat}"] = raw_delta_by_feature[feat][row_idx]
            df[f"d_{feat}"] = positive_delta_by_feature[feat][row_idx]
            df[f"p_{feat}"] = p_by_feature[feat][row_idx]
        rows.append(df)
    pd.concat(rows, axis=0, ignore_index=True).to_csv(out_csv, index=False)


def compute_session_continuous_ms(
    weights_base: Path,
    session: str,
    *,
    output_dir: Path,
    eps: float,
    selected_only: bool,
    write_long_csv_flag: bool,
    strict: bool,
) -> tuple[Path, Path, Path | None]:
    session_dir = weights_base / session
    if not session_dir.exists():
        raise FileNotFoundError(f"{session}: missing session directory {session_dir}")

    selected_models = load_forward_selected_models(session_dir)
    if not selected_models:
        raise ValueError(f"{session}: selected_models.csv has no usable forward-selected neurons.")

    features = [
        feat
        for feat in CANONICAL_FEATURES
        if any(feat in model_vars for model_vars in selected_models.values())
    ]
    if not features:
        raise ValueError(f"{session}: no canonical features found in selected models.")

    paths = session_paths(session)
    data_train, y_train, prep_meta_train = prepare_session_for_modeling(session, paths)
    data_score, y_score, prep_meta_score = prepare_session_for_full_length_scoring(session, paths)
    t_final = int(prep_meta_train["t_final"])
    folds_idx = build_cv_folds(t_final)
    group = infer_group(session)

    if selected_only:
        neuron_ids = np.asarray(sorted(selected_models.keys()), dtype=int)
    else:
        neuron_ids = np.asarray(infer_all_neuron_ids(weights_base, session, session_dir), dtype=int)
    n_neurons = int(neuron_ids.size)
    n_time = int(y_score.shape[0])
    neuron_to_row = {int(ni): idx for idx, ni in enumerate(neuron_ids.tolist())}

    time_bin = np.arange(n_time, dtype=np.int32)
    time_sec = (time_bin.astype(np.float64) / float(FS_HZ)).astype(np.float32)

    x_cache: dict[str, tuple[object, object, list[str], np.ndarray | None, str]] = {}
    saved_score_cache: dict[Path, tuple[sparse.csr_matrix, list[str]]] = {}

    def build_and_store_design(model_vars: list[str]):
        model_key = model_key_from_vars(model_vars)
        if model_key in x_cache:
            return
        x_train, feature_names = build_design_matrix(model_vars, data_train)
        x_score = build_score_matrix_from_feature_names(feature_names, data_score)
        position_xy_by_idx = data_train.get("position_xy_by_idx") if "Position" in model_vars else None
        x_cache[model_key] = (x_train, x_score, feature_names, position_xy_by_idx, model_key)

    all_model_specs: set[tuple[str, ...]] = set()
    for model_vars in selected_models.values():
        all_model_specs.add(tuple(model_vars))
    for model_vars in sorted(all_model_specs):
        build_and_store_design(list(model_vars))

    def get_design(model_vars: list[str]):
        model_key = model_key_from_vars(model_vars)
        return x_cache[model_key]

    def get_saved_score_design(model_dir: Path):
        model_dir = Path(model_dir)
        if model_dir not in saved_score_cache:
            saved_score_cache[model_dir] = build_score_matrix_for_saved_model(model_dir, data_score)
        return saved_score_cache[model_dir]

    raw_delta_by_feature = {
        feat: np.zeros((n_neurons, n_time), dtype=np.float32) for feat in features
    }
    positive_delta_by_feature = {
        feat: np.zeros((n_neurons, n_time), dtype=np.float32) for feat in features
    }
    fitted = np.zeros(n_neurons, dtype=np.int8)
    full_ll_gain = np.full(n_neurons, np.nan, dtype=np.float64)
    full_models = np.full(n_neurons, "", dtype=object)
    status = np.full(n_neurons, "not_selected", dtype=object)
    n_selected_features = np.zeros(n_neurons, dtype=np.int16)
    neuron_items = sorted(selected_models.items())

    def fail_or_warn(message: str) -> None:
        if strict:
            raise FileNotFoundError(message)
        print(f"[WARN] {message}")

    def process_one_neuron(neuron_idx: int, model_vars: list[str]) -> dict[str, object]:
        started = perf_counter()
        result: dict[str, object] = {
            "neuron_idx": int(neuron_idx),
            "status": "not_selected",
            "fitted": 0,
            "full_model": model_key_from_vars(model_vars),
            "n_selected_features": int(sum(1 for feat in features if feat in model_vars)),
            "full_ll_gain": np.nan,
            "raw_delta": {feat: None for feat in features},
            "positive_delta": {feat: None for feat in features},
            "full_refit_sec": 0.0,
            "dropone_sec": 0.0,
            "total_sec": 0.0,
            "error": None,
        }

        if neuron_idx not in neuron_to_row:
            result["status"] = "not_requested"
            result["total_sec"] = perf_counter() - started
            return result
        if neuron_idx >= y_train.shape[1] or neuron_idx >= y_score.shape[1]:
            result["status"] = "neuron_idx_out_of_range"
            result["error"] = (
                f"{session}: neuron_idx={neuron_idx} exceeds available spike columns "
                f"(train={y_train.shape[1]}, score={y_score.shape[1]})."
            )
            result["total_sec"] = perf_counter() - started
            return result

        y_fit = y_train[:, neuron_idx].astype(np.float64)
        y_full = y_score[:, neuron_idx].astype(np.float64)
        mu0_fit = build_oof_intercept_mu(y_fit, folds_idx)
        mu0_full = np.full_like(y_full, fill_value=max(float(np.mean(y_fit)), EPS), dtype=np.float64)
        ll0_series = poisson_loglik_series(y_full, mu0_full)

        x_train_full, x_score_full, feats_full, pos_xy_full, mk_full = get_design(model_vars)
        try:
            t0 = perf_counter()
            with _threadpool_limit_context():
                mu_full = predict_oof_by_refitting_full_model(
                    x_train_full,
                    y_fit,
                    x_score_full,
                    feats_full,
                    folds_idx,
                    position_xy_by_idx=pos_xy_full,
                )
            result["full_refit_sec"] = perf_counter() - t0
        except Exception as exc:
            result["status"] = "failed_full_refit"
            result["error"] = (
                f"{session}: failed refitting full model for neuron_{neuron_idx + 1} model {mk_full}: {exc}"
            )
            result["total_sec"] = perf_counter() - started
            return result

        ll_full_series = poisson_loglik_series(y_full, mu_full)
        ll_gain = float(np.sum(ll_full_series - ll0_series))
        result["full_ll_gain"] = ll_gain
        if not np.isfinite(ll_gain) or ll_gain < 0:
            result["status"] = "negative_full_ll_gain_zeroed"
            result["total_sec"] = perf_counter() - started
            return result

        result["fitted"] = 1
        result["status"] = "ok"

        t1 = perf_counter()
        for feat in features:
            if feat not in model_vars:
                continue

            drop_vars = [var for var in model_vars if var != feat]
            if not drop_vars:
                mu_drop = mu0_full
            else:
                mk_drop = model_key_from_vars(drop_vars)
                drop_model_dir = weights_base / "drop_one" / session / f"neuron_{neuron_idx + 1}" / mk_drop
                if not ensure_saved_weights(drop_model_dir, neuron_idx, len(folds_idx)):
                    result["status"] = f"missing_dropone_{feat}"
                    result["fitted"] = 0
                    result["error"] = (
                        f"{session}: missing drop-one weights for neuron_{neuron_idx + 1}, "
                        f"feature={feat}, model={mk_drop}"
                    )
                    break
                try:
                    x_score_drop, feats_drop = get_saved_score_design(drop_model_dir)
                    mu_drop = predict_mean_from_saved_weights(
                        drop_model_dir,
                        x_score_drop,
                        feats_drop,
                        len(folds_idx),
                        neuron_idx,
                    )
                except Exception as exc:
                    result["status"] = f"failed_dropone_{feat}"
                    result["fitted"] = 0
                    result["error"] = (
                        f"{session}: failed loading drop-one prediction for neuron_{neuron_idx + 1}, "
                        f"feature={feat}, model={mk_drop}: {exc}"
                    )
                    break

            ll_drop_series = poisson_loglik_series(y_full, mu_drop)
            delta_series = ll_full_series - ll_drop_series
            result["raw_delta"][feat] = delta_series.astype(np.float32, copy=False)
            result["positive_delta"][feat] = np.clip(delta_series, 0.0, None).astype(
                np.float32,
                copy=False,
            )
        result["dropone_sec"] = perf_counter() - t1
        result["total_sec"] = perf_counter() - started
        return result

    worker_count = 1
    print(
        f"[INFO] {session}: {len(neuron_items)} selected neurons, {len(features)} canonical features, "
        f"{len(folds_idx)} CV folds, neuron workers={worker_count}, "
        f"train_len={prep_meta_train['t_final']}, score_len={prep_meta_score['t_score']}"
    )

    neuron_results = [process_one_neuron(ni, model_vars) for ni, model_vars in neuron_items]

    total_full_refit_sec = 0.0
    total_dropone_sec = 0.0
    total_worker_sec = 0.0

    for result in neuron_results:
        neuron_idx = int(result["neuron_idx"])
        row_idx = neuron_to_row.get(neuron_idx)
        if row_idx is None:
            continue

        full_models[row_idx] = str(result["full_model"])
        n_selected_features[row_idx] = int(result["n_selected_features"])
        full_ll_gain[row_idx] = float(result["full_ll_gain"]) if np.isfinite(result["full_ll_gain"]) else np.nan
        fitted[row_idx] = int(result["fitted"])
        status[row_idx] = str(result["status"])

        total_full_refit_sec += float(result["full_refit_sec"])
        total_dropone_sec += float(result["dropone_sec"])
        total_worker_sec += float(result["total_sec"])

        if result["error"]:
            fail_or_warn(str(result["error"]))

        if int(result["fitted"]) == 0:
            continue
        for feat in features:
            arr_raw = result["raw_delta"].get(feat)
            arr_pos = result["positive_delta"].get(feat)
            if arr_raw is not None:
                raw_delta_by_feature[feat][row_idx, :] = arr_raw
            if arr_pos is not None:
                positive_delta_by_feature[feat][row_idx, :] = arr_pos

    print(
        f"[TIMING] {session}: summed worker time full_refit={total_full_refit_sec:.1f}s, "
        f"dropone={total_dropone_sec:.1f}s, total={total_worker_sec:.1f}s"
    )

    t_i, d_i, ms_score, p_by_feature = compute_ms_components(
        positive_delta_by_feature,
        features=features,
        eps=float(eps),
    )
    t_i[fitted == 0, :] = 0.0
    d_i[fitted == 0, :] = 0.0
    ms_score[fitted == 0, :] = 0.0
    for feat in features:
        positive_delta_by_feature[feat][fitted == 0, :] = 0.0
        raw_delta_by_feature[feat][fitted == 0, :] = 0.0
        p_by_feature[feat][fitted == 0, :] = 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    session_out_dir = output_dir / session
    session_out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = session_out_dir / f"{session}_continuous_ms_score.npz"
    savez_payload: dict[str, object] = {
        "session": np.asarray(session, dtype=str),
        "group": np.asarray(group, dtype=str),
        "full_model_source": np.asarray("refit_exact_glm_poisson_forward", dtype=str),
        "feature_names": np.asarray(features, dtype=str),
        "neuron_idx": neuron_ids.astype(np.int32),
        "time_bin": time_bin,
        "time_sec": time_sec,
        "fitted": fitted.astype(np.int8),
        "status": status.astype(str),
        "full_model": full_models.astype(str),
        "full_ll_gain": full_ll_gain.astype(np.float32),
        "n_selected_features": n_selected_features.astype(np.int16),
        "T_i": t_i,
        "D_i": d_i,
        "ms_score": ms_score,
    }
    for feat in features:
        savez_payload[f"delta_ll_{feat}"] = raw_delta_by_feature[feat]
        savez_payload[f"d_{feat}"] = positive_delta_by_feature[feat]
        savez_payload[f"p_{feat}"] = p_by_feature[feat]
    np.savez_compressed(npz_path, **savez_payload)

    summary_df = pd.DataFrame(
        {
            "session": session,
            "group": group,
            "neuron_idx": neuron_ids.astype(int),
            "fitted": fitted.astype(int),
            "status": status.astype(str),
            "full_model": full_models.astype(str),
            "full_model_source": "refit_exact_glm_poisson_forward",
            "full_ll_gain": full_ll_gain,
            "K": int(len(features)),
            "n_selected_features": n_selected_features.astype(int),
            "n_time_bins": int(n_time),
            "mean_ms_score": ms_score.mean(axis=1).astype(np.float32),
            "sum_ms_score": ms_score.sum(axis=1).astype(np.float32),
            "max_ms_score": ms_score.max(axis=1).astype(np.float32),
            "mean_T_i": t_i.mean(axis=1).astype(np.float32),
            "mean_D_i": d_i.mean(axis=1).astype(np.float32),
        }
    )
    for feat in features:
        summary_df[f"sum_d_{feat}"] = positive_delta_by_feature[feat].sum(axis=1).astype(np.float32)
    summary_csv = session_out_dir / f"{session}_continuous_ms_score_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    long_csv: Path | None = None
    if write_long_csv_flag:
        long_csv = session_out_dir / f"{session}_continuous_ms_score_long.csv"
        write_long_csv(
            long_csv,
            session=session,
            group=group,
            neuron_ids=neuron_ids,
            time_sec=time_sec,
            features=features,
            fitted=fitted,
            full_models=full_models,
            status=status,
            raw_delta_by_feature=raw_delta_by_feature,
            positive_delta_by_feature=positive_delta_by_feature,
            p_by_feature=p_by_feature,
            t_i=t_i,
            d_i=d_i,
            ms_score=ms_score,
        )

    return npz_path, summary_csv, long_csv


def main() -> None:
    args = parse_args()
    weights_base = args.weights_base.resolve()
    output_dir = args.output_dir.resolve()

    if not weights_base.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_base}")

    sessions = discover_sessions(weights_base, args.session)
    if not sessions:
        raise RuntimeError(f"No sessions found under {weights_base}")

    print(f"[INFO] weights base: {weights_base}")
    print(f"[INFO] output dir:   {output_dir}")
    print(f"[INFO] sessions:     {len(sessions)}")

    session_workers = max(1, len(sessions))
    print(f"[INFO] session workers: {session_workers}")

    written_npz: list[Path] = []
    if session_workers > 1:
        executor = ProcessPoolExecutor(max_workers=session_workers)
        futures = {
            executor.submit(
                compute_session_continuous_ms,
                weights_base,
                session,
                output_dir=output_dir,
                eps=float(args.eps),
                selected_only=bool(args.selected_only),
                write_long_csv_flag=bool(args.write_long_csv),
                strict=bool(args.strict),
            ): session
            for session in sessions
        }
        try:
            for future in as_completed(futures):
                session = futures[future]
                npz_path, summary_csv, long_csv = future.result()
                written_npz.append(npz_path)
                print(f"[OK] {session} -> {npz_path}")
                print(f"[OK] {session} -> {summary_csv}")
                if long_csv is not None:
                    print(f"[OK] {session} -> {long_csv}")
        except KeyboardInterrupt:
            print("[INTERRUPT] Cancelling remaining session workers...")
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=False)
    else:
        for session in sessions:
            npz_path, summary_csv, long_csv = compute_session_continuous_ms(
                weights_base,
                session,
                output_dir=output_dir,
                eps=float(args.eps),
                selected_only=bool(args.selected_only),
                write_long_csv_flag=bool(args.write_long_csv),
                strict=bool(args.strict),
            )
            written_npz.append(npz_path)
            print(f"[OK] {session} -> {npz_path}")
            print(f"[OK] {session} -> {summary_csv}")
            if long_csv is not None:
                print(f"[OK] {session} -> {long_csv}")

    print(f"[DONE] Wrote {len(written_npz)} session continuous MS-score file(s).")


if __name__ == "__main__":
    main()
