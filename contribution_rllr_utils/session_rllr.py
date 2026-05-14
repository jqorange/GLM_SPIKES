from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from glm_poisson_forward import build_cv_folds
from glm_poisson_forward.config import (
    CONTRIB_FIT_SIGNATURE,
    CONTRIB_FORCE_RECOMPUTE_FULL_MODEL_WEIGHTS,
    CONTRIB_STATS_VERSION,
    CV_FOLDS,
    INPUT_FILES,
    N_JOBS,
    SEED,
    SPIKE_INPUT_ROOT,
    VARIABLE_COMPOSITES,
    VARS_ALL,
    WEIGHTS_BASE,
)
from glm_poisson_forward.design_matrix import build_design_matrix, model_key_from_vars as forward_model_key
from glm_poisson_forward.io_utils import (
    list_sessions_dlc_final,
    list_sessions_imu,
    list_sessions_position,
    list_sessions_spike,
    prepare_session_for_modeling,
    session_paths,
)
from glm_poisson_forward.metrics import (
    compute_deviance_explained_poisson_vs_baseline,
    compute_llhi_bps_poisson_vs_baseline,
)

from .constants import MU_EPS, RLLR_FITS_DIRNAME, RLLR_STATS_DIRNAME
from .selection import load_forward_selected_models
from .stats import build_oof_intercept_mu, poisson_loglik
from .weights import (
    fit_signature_matches,
    load_feature_names_file,
    load_fold_weights,
    predict_oof_from_saved_weights,
    save_weights_for_model,
    weights_exist_for_neuron,
)
from .cell_metrics import pyramidal_indices_for_session


DROP_COMPOSITES = {str(k): [str(v) for v in vals] for k, vals in VARIABLE_COMPOSITES.items() if vals}
DROP_FEATURES = VARS_ALL + [k for k in DROP_COMPOSITES if k not in VARS_ALL]


@dataclass
class SessionResult:
    session: str
    group: str
    full_ll_gain_by_neuron: Dict[int, float]
    contrib_rllr_by_feature_by_neuron: Dict[str, Dict[int, float]]
    shuf_mean_rllr_by_feature_by_neuron: Dict[str, Dict[int, float]]
    shuf_std_rllr_by_feature_by_neuron: Dict[str, Dict[int, float]]
    full_llhi_by_neuron: Dict[int, float]
    contrib_delta_llhi_by_feature_by_neuron: Dict[str, Dict[int, float]]
    shuf_mean_delta_llhi_by_feature_by_neuron: Dict[str, Dict[int, float]]
    shuf_std_delta_llhi_by_feature_by_neuron: Dict[str, Dict[int, float]]
    full_deviance_explained_by_neuron: Dict[int, float]
    contrib_delta_deviance_explained_by_feature_by_neuron: Dict[str, Dict[int, float]]
    all_selected_neurons: np.ndarray
    pyramidal_neurons: np.ndarray
    unfit_neurons: List[int]


def list_required_sessions() -> List[str]:
    set_spk = list_sessions_spike(SPIKE_INPUT_ROOT)
    required_sets = [set_spk]
    for key, spec in INPUT_FILES.items():
        root = spec["root"]
        if key == "imu":
            required_sets.append(list_sessions_imu(root))
        elif key == "dlc_final":
            required_sets.append(list_sessions_dlc_final(root))
        elif key == "position":
            required_sets.append(list_sessions_position(root))
    common = set.intersection(*required_sets) if required_sets else set()
    return sorted(list(common))


def circular_shift(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = arr.size
    if n <= 1:
        return arr
    shift = int(rng.integers(1, n))
    return np.roll(arr, shift)


def zscore(value: float, mu: float, sigma: float) -> float:
    if not (np.isfinite(value) and np.isfinite(mu) and np.isfinite(sigma)):
        return float("nan")
    if sigma <= MU_EPS:
        return float("nan")
    return float((value - mu) / sigma)


def right_tail_pvalue(z_val: float) -> float:
    if not np.isfinite(z_val):
        return float("nan")
    return float(0.5 * math.erfc(float(z_val) / math.sqrt(2.0)))


def build_contribution_folds(T: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Use the exact same CV fold construction as glm_poisson_forward."""
    return build_cv_folds(T)


def ensure_z_pvalue_columns(
    csv_path: Path,
    *,
    value_col: str,
    mean_col: str,
    std_col: str,
    z_col: str,
    p_col: str,
) -> bool:
    if (not csv_path.exists()) or csv_path.stat().st_size < 1:
        return False
    df = pd.read_csv(csv_path)
    if value_col not in df.columns:
        return False
    if mean_col not in df.columns or std_col not in df.columns:
        return False
    updated = False
    if z_col not in df.columns:
        df[z_col] = np.nan
        updated = True
    if p_col not in df.columns:
        df[p_col] = np.nan
        updated = True

    needs_z = df[z_col].isna()
    if needs_z.any():
        z_vals = (df[value_col] - df[mean_col]) / df[std_col]
        z_vals = z_vals.where(df[std_col] > MU_EPS)
        df.loc[needs_z, z_col] = z_vals[needs_z]
        updated = True

    needs_p = df[p_col].isna()
    if needs_p.any():
        df.loc[needs_p, p_col] = df.loc[needs_p, z_col].apply(right_tail_pvalue)
        updated = True

    if updated:
        df.to_csv(csv_path, index=False)
    return updated


def ensure_rscc_column(csv_path: Path) -> bool:
    if (not csv_path.exists()) or csv_path.stat().st_size < 1:
        return False
    df = pd.read_csv(csv_path)
    if "delta_llhi" not in df.columns:
        return False

    if "rSCC" not in df.columns:
        df["rSCC"] = np.nan

    updated = False
    group_cols = [c for c in ["session", "group", "neuron_idx"] if c in df.columns]
    if not group_cols:
        group_cols = ["neuron_idx"]

    for _, idx in df.groupby(group_cols).groups.items():
        sub = df.loc[idx]
        delta = sub["delta_llhi"].to_numpy(dtype=float)
        finite = np.isfinite(delta)
        norm = float(np.sqrt(np.sum(np.square(delta[finite])))) if np.any(finite) else 0.0
        if np.isfinite(norm) and norm > MU_EPS:
            rscc = np.where(finite, delta / norm, 0.0)
        else:
            rscc = np.zeros_like(delta, dtype=float)
        old = sub["rSCC"].to_numpy(dtype=float)
        if np.any(~np.isclose(np.nan_to_num(old, nan=-999.0), np.nan_to_num(rscc, nan=-999.0), atol=1e-12)):
            df.loc[idx, "rSCC"] = rscc
            updated = True

    if updated:
        df.to_csv(csv_path, index=False)
    return updated


def _stats_paths(stats_root: Path) -> Dict[str, Path]:
    return {
        "meta": stats_root / "stats_meta.json",
        "full_rllr": stats_root / "full_rllr.csv",
        "dropone_rllr": stats_root / "dropone_rllr.csv",
        "full_llhi": stats_root / "full_llhi.csv",
        "dropone_llhi": stats_root / "dropone_llhi.csv",
        "full_devexp": stats_root / "full_deviance_explained.csv",
        "dropone_devexp": stats_root / "dropone_deviance_explained.csv",
    }


def _stats_meta_is_current(meta_path: Path, *, n_shuffle: int) -> bool:
    if not meta_path.exists():
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False
    return (
        int(payload.get("stats_version", -1)) == int(CONTRIB_STATS_VERSION)
        and str(payload.get("fit_signature", "")) == str(CONTRIB_FIT_SIGNATURE)
        and int(payload.get("n_shuffle", -1)) == int(n_shuffle)
        and bool(payload.get("force_recompute_full_model_weights", False))
        == bool(CONTRIB_FORCE_RECOMPUTE_FULL_MODEL_WEIGHTS)
    )


def _load_session_result_from_stats(
    session: str,
    group: str,
    stats_root: Path,
    pyramidal_neurons: np.ndarray,
    all_selected_neurons: np.ndarray,
) -> SessionResult:
    paths = _stats_paths(stats_root)
    df_full = pd.read_csv(paths["full_rllr"])
    df_rllr = pd.read_csv(paths["dropone_rllr"])
    df_full_llhi = pd.read_csv(paths["full_llhi"])
    df_llhi = pd.read_csv(paths["dropone_llhi"])
    df_full_devexp = pd.read_csv(paths["full_devexp"])
    df_devexp = pd.read_csv(paths["dropone_devexp"])

    full_map = {int(r["neuron_idx"]): float(r["ll_gain"]) for _, r in df_full.iterrows()}
    unfit = sorted(ni for ni, val in full_map.items() if np.isfinite(val) and val < 0)
    unfit_set = set(unfit)

    def _feature_map(df: pd.DataFrame, value_col: str) -> Dict[str, Dict[int, float]]:
        out: Dict[str, Dict[int, float]] = {feat: {} for feat in DROP_FEATURES}
        for _, row in df.iterrows():
            feat = str(row["feature"])
            if feat not in out:
                out[feat] = {}
            ni = int(row["neuron_idx"])
            val = row.get(value_col, np.nan)
            if np.isfinite(val):
                out[feat][ni] = float(val)
        return out

    def _shuffle_map(df: pd.DataFrame, value_col: str) -> Dict[str, Dict[int, float]]:
        out: Dict[str, Dict[int, float]] = {feat: {} for feat in DROP_FEATURES}
        for _, row in df.iterrows():
            feat = str(row["feature"])
            if feat not in out:
                out[feat] = {}
            ni = int(row["neuron_idx"])
            val = row.get(value_col, np.nan)
            if np.isfinite(val):
                out[feat][ni] = float(val)
        return out

    return SessionResult(
        session=session,
        group=group,
        full_ll_gain_by_neuron={ni: v for ni, v in full_map.items() if ni not in unfit_set},
        contrib_rllr_by_feature_by_neuron={
            feat: {ni: val for ni, val in vals.items() if ni not in unfit_set}
            for feat, vals in _feature_map(df_rllr, "rllr").items()
        },
        shuf_mean_rllr_by_feature_by_neuron=_shuffle_map(df_rllr, "rllr_shuf_mean"),
        shuf_std_rllr_by_feature_by_neuron=_shuffle_map(df_rllr, "rllr_shuf_std"),
        full_llhi_by_neuron={
            int(r["neuron_idx"]): float(r["llhi_full"])
            for _, r in df_full_llhi.iterrows()
            if int(r["neuron_idx"]) not in unfit_set and np.isfinite(r["llhi_full"])
        },
        contrib_delta_llhi_by_feature_by_neuron={
            feat: {ni: val for ni, val in vals.items() if ni not in unfit_set}
            for feat, vals in _feature_map(df_llhi, "delta_llhi").items()
        },
        shuf_mean_delta_llhi_by_feature_by_neuron=_shuffle_map(df_llhi, "delta_llhi_shuf_mean"),
        shuf_std_delta_llhi_by_feature_by_neuron=_shuffle_map(df_llhi, "delta_llhi_shuf_std"),
        full_deviance_explained_by_neuron={
            int(r["neuron_idx"]): float(r["deviance_explained_full"])
            for _, r in df_full_devexp.iterrows()
            if int(r["neuron_idx"]) not in unfit_set and np.isfinite(r["deviance_explained_full"])
        },
        contrib_delta_deviance_explained_by_feature_by_neuron={
            feat: {ni: val for ni, val in vals.items() if ni not in unfit_set}
            for feat, vals in _feature_map(df_devexp, "delta_deviance_explained").items()
        },
        all_selected_neurons=all_selected_neurons,
        pyramidal_neurons=pyramidal_neurons,
        unfit_neurons=unfit,
    )


def _write_pyramidal_views(stats_root: Path, pyramidal_neurons: np.ndarray) -> None:
    pyr_set = set(map(int, pyramidal_neurons.tolist()))
    for base_name in [
        "full_rllr",
        "dropone_rllr",
        "full_llhi",
        "dropone_llhi",
        "full_deviance_explained",
        "dropone_deviance_explained",
    ]:
        src = stats_root / f"{base_name}.csv"
        dst = stats_root / f"{base_name}_pyr.csv"
        if not src.exists():
            continue
        df = pd.read_csv(src)
        if "neuron_idx" in df.columns:
            df = df[df["neuron_idx"].isin(sorted(pyr_set))]
        df.to_csv(dst, index=False)


def compute_session_rllr(
    session: str,
    dayid2cellinfo: Dict[str, Path],
    *,
    n_jobs: int = N_JOBS,
    n_shuffle: int = 200,
) -> Optional[SessionResult]:
    sess_dir = WEIGHTS_BASE / session
    sess_dir.mkdir(parents=True, exist_ok=True)
    stats_root = sess_dir / RLLR_STATS_DIRNAME
    stats_root.mkdir(parents=True, exist_ok=True)
    stats_paths = _stats_paths(stats_root)

    paths = session_paths(session)
    for key in ["imu", "spike", "dlc_final", "position"]:
        if not paths[key].exists():
            print(f"[SKIP] {session}: missing input {key}: {paths[key]}")
            return None

    s_lower = session.lower()
    if "indoor" in s_lower:
        group = "indoor"
    elif "outdoor" in s_lower:
        group = "outdoor"
    else:
        print(f"[SKIP] {session}: cannot infer indoor/outdoor from name")
        return None

    selected_models = load_forward_selected_models(sess_dir)
    if not selected_models:
        print(f"[SKIP] {session}: no forward-selected models found")
        return None

    try:
        _, _, prep_meta = prepare_session_for_modeling(session, paths)
    except Exception as exc:
        print(f"[SKIP] {session}: {exc}")
        return None
    n_neurons_total = int(prep_meta["n_neurons"])

    pyr_idx = pyramidal_indices_for_session(session, dayid2cellinfo, n_neurons_total)
    if pyr_idx is None:
        pyr_idx = np.array([], dtype=int)
        print(f"[WARN] {session}: pyramidal cell info not found; writing all-neuron stats only")
    else:
        pyr_idx = np.asarray(sorted(set(map(int, pyr_idx.tolist())) & set(selected_models.keys())), dtype=int)

    all_selected_neurons = np.asarray(sorted(selected_models.keys()), dtype=int)
    print(
        f"[INFO] {session}: selected neurons={all_selected_neurons.size}, "
        f"pyramidal-selected={pyr_idx.size}",
    )

    current_stats = (
        _stats_meta_is_current(stats_paths["meta"], n_shuffle=n_shuffle)
        and all(path.exists() for key, path in stats_paths.items() if key != "meta")
    )
    if current_stats:
        try:
            cached_neurons = set(
                pd.read_csv(stats_paths["full_rllr"], usecols=["neuron_idx"])["neuron_idx"]
                .dropna()
                .astype(int)
                .tolist()
            )
        except Exception:
            current_stats = False
        else:
            current_stats = cached_neurons == set(all_selected_neurons.tolist())
    if current_stats:
        _write_pyramidal_views(stats_root, pyr_idx)
        return _load_session_result_from_stats(
            session,
            group,
            stats_root,
            pyramidal_neurons=pyr_idx,
            all_selected_neurons=all_selected_neurons,
        )

    try:
        data_dict, Y_all, prep_meta = prepare_session_for_modeling(session, paths)
    except Exception as exc:
        print(f"[SKIP] {session}: {exc}")
        return None

    matched_len = prep_meta.get("matched_len")
    if matched_len is not None:
        print(f"[pair_match] {session}: truncated to matched indoor/outdoor length {matched_len}")

    folds_idx = build_contribution_folds(int(prep_meta["t_final"]))
    X_cache: Dict[str, Tuple[np.ndarray, List[str], np.ndarray | None, str]] = {}
    x_cache_lock = Lock()

    def get_X(model_vars: List[str]):
        mk = forward_model_key(model_vars)
        with x_cache_lock:
            if mk in X_cache:
                X, feats, pos_xy, _ = X_cache[mk]
                return X, feats, pos_xy, mk
        X, feats = build_design_matrix(model_vars, data_dict)
        pos_xy = data_dict.get("position_xy_by_idx") if "Position" in model_vars else None
        with x_cache_lock:
            X_cache[mk] = (X, feats, pos_xy, mk)
        return X, feats, pos_xy, mk

    def find_saved_model_dir(model_key: str) -> Optional[Path]:
        for cand in [sess_dir / model_key, sess_dir / RLLR_FITS_DIRNAME / model_key]:
            if cand.exists():
                return cand
        return None

    def neuron_weights_ready_local(
        model_dir: Path,
        neuron_idx: int,
        *,
        expected_fit_signature: str | None,
        allow_legacy_forward: bool = False,
    ) -> bool:
        idx1 = neuron_idx + 1
        if not weights_exist_for_neuron(
            model_dir,
            idx1,
            len(folds_idx),
            expected_fit_signature=expected_fit_signature,
            allow_legacy_forward=allow_legacy_forward,
        ):
            return False
        neuron_dir = model_dir / f"neuron_{idx1}"
        weights_mean = neuron_dir / "weights_mean.csv"
        if weights_mean.exists() and weights_mean.stat().st_size >= 8:
            return True
        saved_feats = load_feature_names_file(model_dir)
        if not saved_feats:
            return False
        try:
            weights = []
            for k in range(1, len(folds_idx) + 1):
                csv_path = neuron_dir / f"fold{k}" / "weights.csv"
                weights.append(load_fold_weights(csv_path, feature_names=saved_feats))
            w_mean = np.mean(np.stack(weights, axis=0), axis=0).astype(np.float32)
            pd.DataFrame(
                w_mean.reshape(1, -1),
                index=[f"neuron_{idx1}"],
                columns=saved_feats,
            ).to_csv(weights_mean)
            return True
        except Exception as exc:
            print(f"[WARN] {session}: failed computing weights_mean for neuron_{idx1}: {exc}")
            return False

    dropone_root = WEIGHTS_BASE / "drop_one" / session
    dropone_root.mkdir(parents=True, exist_ok=True)

    def _process_one_neuron(ni: int, full_vars: List[str]) -> Dict[str, object]:
        out: Dict[str, object] = {
            "ni": ni,
            "skip": False,
            "ll0": np.nan,
            "ll_full": np.nan,
            "ll_gain": np.nan,
            "llhi_full": np.nan,
            "devexp_full": np.nan,
            "unfit": False,
            "contrib_rllr": {feat: float("nan") for feat in DROP_FEATURES},
            "contrib_delta_llhi": {feat: float("nan") for feat in DROP_FEATURES},
            "contrib_delta_devexp": {feat: float("nan") for feat in DROP_FEATURES},
            "contrib_rscc": {feat: 0.0 for feat in DROP_FEATURES},
            "shuf_mean_rllr": {},
            "shuf_std_rllr": {},
            "shuf_mean_llhi": {},
            "shuf_std_llhi": {},
            "missing_full_model_dir": 0,
            "missing_full_weights": 0,
            "failed_full_predict": 0,
            "full_processed": 0,
            "total_drop_models": 0,
            "missing_drop_model_dir": 0,
            "missing_drop_weights": 0,
            "failed_drop_predict": 0,
        }

        y = Y_all[:, ni].astype(np.float64)
        mu0_oof = build_oof_intercept_mu(y, folds_idx)
        out["ll0"] = float(poisson_loglik(y, mu0_oof))

        X_full, feats_full, pos_xy_full, mk_full = get_X(full_vars)
        full_forward_dir = sess_dir / mk_full
        if CONTRIB_FORCE_RECOMPUTE_FULL_MODEL_WEIGHTS:
            model_dir_full = sess_dir / RLLR_FITS_DIRNAME / mk_full
            allow_legacy_full = False
        else:
            model_dir_full = find_saved_model_dir(mk_full) or (sess_dir / RLLR_FITS_DIRNAME / mk_full)
            allow_legacy_full = model_dir_full == full_forward_dir
            if model_dir_full == sess_dir / RLLR_FITS_DIRNAME / mk_full and not model_dir_full.exists():
                out["missing_full_model_dir"] = 1

        if not neuron_weights_ready_local(
            model_dir_full,
            ni,
            expected_fit_signature=CONTRIB_FIT_SIGNATURE,
            allow_legacy_forward=allow_legacy_full,
        ):
            print(f"[INFO] {session}: backfilling full model {mk_full} weights for neuron_{ni+1}")
            out["missing_full_weights"] = 1
            force_full_refit = CONTRIB_FORCE_RECOMPUTE_FULL_MODEL_WEIGHTS or not fit_signature_matches(
                model_dir_full,
                CONTRIB_FIT_SIGNATURE,
                allow_legacy_forward=allow_legacy_full,
            )
            save_weights_for_model(
                model_dir=model_dir_full,
                feature_names=feats_full,
                X_all=X_full,
                Y_all=Y_all,
                folds_idx=folds_idx,
                neuron_indices=np.array([ni], dtype=int),
                n_jobs=1,
                folds_count=len(folds_idx),
                position_xy_by_idx=pos_xy_full,
                use_forward_fit=True,
                fit_signature=CONTRIB_FIT_SIGNATURE,
                force_recompute=force_full_refit,
            )
        if not neuron_weights_ready_local(
            model_dir_full,
            ni,
            expected_fit_signature=CONTRIB_FIT_SIGNATURE,
            allow_legacy_forward=allow_legacy_full,
        ):
            print(f"[SKIP] {session}: incomplete full-model weights {mk_full} (neuron_{ni+1})")
            out["skip"] = True
            return out

        try:
            mu_oof_full = predict_oof_from_saved_weights(model_dir_full, X_full, feats_full, folds_idx, ni)
        except Exception as exc:
            print(f"[SKIP] {session}: failed loading full model {mk_full} (neuron_{ni+1}): {exc}")
            out["skip"] = True
            out["failed_full_predict"] = 1
            return out

        ll_full = float(poisson_loglik(y, mu_oof_full))
        ll_gain = float(ll_full - float(out["ll0"]))
        out["ll_full"] = ll_full
        out["ll_gain"] = ll_gain
        out["llhi_full"] = float(compute_llhi_bps_poisson_vs_baseline(y, mu_oof_full, mu0_oof))
        out["devexp_full"] = float(
            compute_deviance_explained_poisson_vs_baseline(y, mu_oof_full, mu0_oof)
        )
        out["full_processed"] = 1
        if np.isfinite(ll_gain) and ll_gain < 0:
            out["unfit"] = True
            return out

        drop_specs: Dict[str, List[str]] = {v: [x for x in full_vars if x != v] for v in full_vars}
        for comp_name, members in DROP_COMPOSITES.items():
            member_set = set(members)
            if any(v in member_set for v in full_vars):
                drop_specs[comp_name] = [x for x in full_vars if x not in member_set]

        mu_oof_red_by_feat: Dict[str, np.ndarray] = {}
        for feat_name, drop_vars in drop_specs.items():
            if not drop_vars:
                mu_oof_red_by_feat[feat_name] = mu0_oof.copy()
                continue
            X_red, feats_red, pos_xy_red, mk_red = get_X(drop_vars)
            model_dir_red = dropone_root / f"neuron_{ni + 1}" / mk_red
            if not model_dir_red.exists():
                out["missing_drop_model_dir"] = int(out["missing_drop_model_dir"]) + 1
            drop_force_refit = not fit_signature_matches(model_dir_red, CONTRIB_FIT_SIGNATURE)
            if not neuron_weights_ready_local(
                model_dir_red,
                ni,
                expected_fit_signature=CONTRIB_FIT_SIGNATURE,
            ):
                save_weights_for_model(
                    model_dir=model_dir_red,
                    feature_names=feats_red,
                    X_all=X_red,
                    Y_all=Y_all,
                    folds_idx=folds_idx,
                    neuron_indices=np.array([ni], dtype=int),
                    n_jobs=1,
                    folds_count=len(folds_idx),
                    position_xy_by_idx=pos_xy_red,
                    use_forward_fit=True,
                    fit_signature=CONTRIB_FIT_SIGNATURE,
                    force_recompute=drop_force_refit,
                )
            if not neuron_weights_ready_local(
                model_dir_red,
                ni,
                expected_fit_signature=CONTRIB_FIT_SIGNATURE,
            ):
                out["missing_drop_weights"] = int(out["missing_drop_weights"]) + 1
                continue
            try:
                mu_oof_red_by_feat[feat_name] = predict_oof_from_saved_weights(
                    model_dir_red,
                    X_red,
                    feats_red,
                    folds_idx,
                    ni,
                )
                out["total_drop_models"] = int(out["total_drop_models"]) + 1
            except Exception as exc:
                print(f"[WARN] {session}: failed loading drop model {mk_red} (neuron_{ni+1}): {exc}")
                out["failed_drop_predict"] = int(out["failed_drop_predict"]) + 1

        rng = np.random.default_rng(SEED + int(ni))
        shuf_llhi_vals: Dict[str, List[float]] = {feat: [] for feat in drop_specs}
        shuf_rllr_vals: Dict[str, List[float]] = {feat: [] for feat in drop_specs}
        for _ in range(n_shuffle):
            y_shuf = circular_shift(y, rng)
            ll0_shuf = poisson_loglik(y_shuf, mu0_oof)
            ll_full_shuf = poisson_loglik(y_shuf, mu_oof_full)
            denom_shuf = ll_full_shuf - ll0_shuf
            llhi_full_shuf = compute_llhi_bps_poisson_vs_baseline(y_shuf, mu_oof_full, mu0_oof)
            for feat_name, mu_red in mu_oof_red_by_feat.items():
                llhi_red_shuf = compute_llhi_bps_poisson_vs_baseline(y_shuf, mu_red, mu0_oof)
                shuf_llhi_vals[feat_name].append(float(llhi_full_shuf - llhi_red_shuf))
                if np.isfinite(denom_shuf) and denom_shuf > MU_EPS:
                    ll_red_shuf = poisson_loglik(y_shuf, mu_red)
                    if np.isfinite(ll_red_shuf) and np.isfinite(ll_full_shuf):
                        shuf_rllr_vals[feat_name].append(float((ll_full_shuf - ll_red_shuf) / denom_shuf))

        for feat_name in drop_specs:
            arr = np.asarray(shuf_llhi_vals.get(feat_name, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out["shuf_mean_llhi"][feat_name] = float(np.mean(arr))
                out["shuf_std_llhi"][feat_name] = float(np.std(arr))
            arr = np.asarray(shuf_rllr_vals.get(feat_name, []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out["shuf_mean_rllr"][feat_name] = float(np.mean(arr))
                out["shuf_std_rllr"][feat_name] = float(np.std(arr))

        for feat_name, mu_red in mu_oof_red_by_feat.items():
            if np.isfinite(ll_gain) and ll_gain > MU_EPS:
                ll_red = poisson_loglik(y, mu_red)
                out["contrib_rllr"][feat_name] = float((ll_full - ll_red) / ll_gain)
            llhi_red = compute_llhi_bps_poisson_vs_baseline(y, mu_red, mu0_oof)
            devexp_red = compute_deviance_explained_poisson_vs_baseline(y, mu_red, mu0_oof)
            if np.isfinite(out["llhi_full"]) and np.isfinite(llhi_red):
                out["contrib_delta_llhi"][feat_name] = float(float(out["llhi_full"]) - llhi_red)
            if np.isfinite(out["devexp_full"]) and np.isfinite(devexp_red):
                out["contrib_delta_devexp"][feat_name] = float(float(out["devexp_full"]) - devexp_red)

        llhi_deltas = np.array([out["contrib_delta_llhi"].get(v, np.nan) for v in full_vars], dtype=float)
        finite = np.isfinite(llhi_deltas)
        l2_norm = float(np.sqrt(np.sum(np.square(llhi_deltas[finite])))) if np.any(finite) else 0.0
        for feat_name in DROP_FEATURES:
            delta_val = out["contrib_delta_llhi"].get(feat_name, np.nan)
            if np.isfinite(delta_val) and l2_norm > MU_EPS:
                out["contrib_rscc"][feat_name] = float(delta_val / l2_norm)
        return out

    neuron_items = list(selected_models.items())
    session_workers = max(1, min(int(n_jobs), len(neuron_items)))
    if session_workers > 1:
        print(f"[INFO] {session}: processing {len(neuron_items)} neurons with {session_workers} worker threads")
        with ThreadPoolExecutor(max_workers=session_workers) as executor:
            futures = {executor.submit(_process_one_neuron, ni, full_vars): ni for ni, full_vars in neuron_items}
            neuron_results = []
            for fut in as_completed(futures):
                try:
                    neuron_results.append(fut.result())
                except Exception as exc:
                    ni = futures[fut]
                    print(f"[WARN] {session}: neuron_{ni+1} worker failed: {exc}")
    else:
        neuron_results = [_process_one_neuron(ni, full_vars) for ni, full_vars in neuron_items]

    full_ll_gain_by_neuron: Dict[int, float] = {}
    ll_full_by_neuron: Dict[int, float] = {}
    ll0_by_neuron: Dict[int, float] = {}
    full_llhi_by_neuron: Dict[int, float] = {}
    full_devexp_by_neuron: Dict[int, float] = {}
    contrib_rllr: Dict[str, Dict[int, float]] = {feat: {} for feat in DROP_FEATURES}
    contrib_delta_llhi: Dict[str, Dict[int, float]] = {feat: {} for feat in DROP_FEATURES}
    contrib_delta_devexp: Dict[str, Dict[int, float]] = {feat: {} for feat in DROP_FEATURES}
    contrib_rscc: Dict[str, Dict[int, float]] = {feat: {} for feat in DROP_FEATURES}
    shuf_mean_rllr: Dict[str, Dict[int, float]] = {feat: {} for feat in DROP_FEATURES}
    shuf_std_rllr: Dict[str, Dict[int, float]] = {feat: {} for feat in DROP_FEATURES}
    shuf_mean_llhi: Dict[str, Dict[int, float]] = {feat: {} for feat in DROP_FEATURES}
    shuf_std_llhi: Dict[str, Dict[int, float]] = {feat: {} for feat in DROP_FEATURES}
    unfit_neurons: List[int] = []
    missing_full_model_dir = 0
    missing_full_weights = 0
    failed_full_predict = 0
    full_processed = 0
    total_drop_models = 0
    missing_drop_model_dir = 0
    missing_drop_weights = 0
    failed_drop_predict = 0

    for out in neuron_results:
        ni = int(out["ni"])
        missing_full_model_dir += int(out["missing_full_model_dir"])
        missing_full_weights += int(out["missing_full_weights"])
        failed_full_predict += int(out["failed_full_predict"])
        full_processed += int(out["full_processed"])
        total_drop_models += int(out["total_drop_models"])
        missing_drop_model_dir += int(out["missing_drop_model_dir"])
        missing_drop_weights += int(out["missing_drop_weights"])
        failed_drop_predict += int(out["failed_drop_predict"])

        if out["skip"]:
            continue
        ll0_by_neuron[ni] = float(out["ll0"])
        ll_full_by_neuron[ni] = float(out["ll_full"])
        full_ll_gain_by_neuron[ni] = float(out["ll_gain"])
        if out["unfit"]:
            unfit_neurons.append(ni)
            continue
        full_llhi_by_neuron[ni] = float(out["llhi_full"])
        full_devexp_by_neuron[ni] = float(out["devexp_full"])
        for feat_name in DROP_FEATURES:
            contrib_rllr[feat_name][ni] = float(out["contrib_rllr"].get(feat_name, np.nan))
            contrib_delta_llhi[feat_name][ni] = float(out["contrib_delta_llhi"].get(feat_name, np.nan))
            contrib_delta_devexp[feat_name][ni] = float(out["contrib_delta_devexp"].get(feat_name, np.nan))
            contrib_rscc[feat_name][ni] = float(out["contrib_rscc"].get(feat_name, 0.0))
            mu_rllr = out["shuf_mean_rllr"].get(feat_name, np.nan)
            std_rllr = out["shuf_std_rllr"].get(feat_name, np.nan)
            mu_llhi = out["shuf_mean_llhi"].get(feat_name, np.nan)
            std_llhi = out["shuf_std_llhi"].get(feat_name, np.nan)
            if np.isfinite(mu_rllr):
                shuf_mean_rllr[feat_name][ni] = float(mu_rllr)
            if np.isfinite(std_rllr):
                shuf_std_rllr[feat_name][ni] = float(std_rllr)
            if np.isfinite(mu_llhi):
                shuf_mean_llhi[feat_name][ni] = float(mu_llhi)
            if np.isfinite(std_llhi):
                shuf_std_llhi[feat_name][ni] = float(std_llhi)

    all_selected_list = sorted(all_selected_neurons.tolist())
    full_rows = [
        {
            "session": session,
            "group": group,
            "neuron_idx": ni,
            "ll_full": ll_full_by_neuron.get(ni, np.nan),
            "ll0": ll0_by_neuron.get(ni, np.nan),
            "ll_gain": full_ll_gain_by_neuron.get(ni, np.nan),
        }
        for ni in all_selected_list
    ]
    pd.DataFrame(full_rows).to_csv(stats_paths["full_rllr"], index=False)

    rllr_rows = []
    llhi_rows = []
    devexp_rows = []
    for feat_name in DROP_FEATURES:
        for ni in all_selected_list:
            rllr_val = contrib_rllr[feat_name].get(ni, np.nan)
            mu_rllr = shuf_mean_rllr[feat_name].get(ni, np.nan)
            std_rllr = shuf_std_rllr[feat_name].get(ni, np.nan)
            rllr_z = zscore(rllr_val, mu_rllr, std_rllr)
            rllr_rows.append(
                {
                    "session": session,
                    "group": group,
                    "feature": feat_name,
                    "neuron_idx": ni,
                    "rllr": rllr_val,
                    "rllr_shuf_mean": mu_rllr,
                    "rllr_shuf_std": std_rllr,
                    "rllr_z": rllr_z,
                    "rllr_pvalue": right_tail_pvalue(rllr_z),
                }
            )
            llhi_val = contrib_delta_llhi[feat_name].get(ni, np.nan)
            mu_llhi = shuf_mean_llhi[feat_name].get(ni, np.nan)
            std_llhi = shuf_std_llhi[feat_name].get(ni, np.nan)
            llhi_z = zscore(llhi_val, mu_llhi, std_llhi)
            llhi_rows.append(
                {
                    "session": session,
                    "group": group,
                    "feature": feat_name,
                    "neuron_idx": ni,
                    "delta_llhi": llhi_val,
                    "rSCC": contrib_rscc[feat_name].get(ni, 0.0),
                    "delta_llhi_shuf_mean": mu_llhi,
                    "delta_llhi_shuf_std": std_llhi,
                    "delta_llhi_z": llhi_z,
                    "delta_llhi_pvalue": right_tail_pvalue(llhi_z),
                }
            )
            devexp_rows.append(
                {
                    "session": session,
                    "group": group,
                    "feature": feat_name,
                    "neuron_idx": ni,
                    "delta_deviance_explained": contrib_delta_devexp[feat_name].get(ni, np.nan),
                }
            )

    pd.DataFrame(rllr_rows).to_csv(stats_paths["dropone_rllr"], index=False)
    pd.DataFrame(
        [
            {
                "session": session,
                "group": group,
                "neuron_idx": ni,
                "llhi_full": full_llhi_by_neuron.get(ni, np.nan),
            }
            for ni in all_selected_list
        ]
    ).to_csv(stats_paths["full_llhi"], index=False)
    pd.DataFrame(llhi_rows).to_csv(stats_paths["dropone_llhi"], index=False)
    pd.DataFrame(
        [
            {
                "session": session,
                "group": group,
                "neuron_idx": ni,
                "deviance_explained_full": full_devexp_by_neuron.get(ni, np.nan),
            }
            for ni in all_selected_list
        ]
    ).to_csv(stats_paths["full_devexp"], index=False)
    pd.DataFrame(devexp_rows).to_csv(stats_paths["dropone_devexp"], index=False)
    _write_pyramidal_views(stats_root, pyr_idx)

    with open(stats_paths["meta"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "stats_version": int(CONTRIB_STATS_VERSION),
                "fit_signature": CONTRIB_FIT_SIGNATURE,
                "force_recompute_full_model_weights": bool(CONTRIB_FORCE_RECOMPUTE_FULL_MODEL_WEIGHTS),
                "n_shuffle": int(n_shuffle),
            },
            f,
            indent=2,
        )

    print(
        f"[INFO] {session}: full processed={full_processed}, "
        f"missing full model dir={missing_full_model_dir}, "
        f"missing full weights={missing_full_weights}, "
        f"failed full predict={failed_full_predict}"
    )
    print(
        f"[INFO] {session}: drop models attempted={total_drop_models}, "
        f"missing drop model dir={missing_drop_model_dir}, "
        f"missing drop weights={missing_drop_weights}, "
        f"failed drop predict={failed_drop_predict}"
    )

    return _load_session_result_from_stats(
        session,
        group,
        stats_root,
        pyramidal_neurons=pyr_idx,
        all_selected_neurons=all_selected_neurons,
    )
