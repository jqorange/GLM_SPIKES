from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from glm_poisson_forward.config import (
    CV_FOLDS,
    INPUT_FILES,
    MAX_MISMATCH_FRAMES_50HZ,
    MIN_SPEED_CM_S,
    N_JOBS,
    SEED,
    SPIKE_ROOT,
    VARIABLE_COMPOSITES,
    VARS_ALL,
    WEIGHTS_BASE,
)
from glm_poisson_forward.design_matrix import build_design_matrix, model_key_from_vars as forward_model_key
from glm_poisson_forward.io_utils import (
    filter_by_min_speed,
    list_sessions_dlc_final,
    list_sessions_imu,
    list_sessions_position,
    list_sessions_spike,
    load_spikes_50hz_counts,
    rebuild_inputs_50hz,
    session_paths,
)
from glm_poisson_forward.metrics import compute_llhi_bps_poisson_vs_baseline

from .constants import MU_EPS, RLLR_FITS_DIRNAME, RLLR_STATS_DIRNAME
from .selection import load_forward_selected_models
from .stats import build_oof_intercept_mu, poisson_loglik
from .weights import (
    load_feature_names_file,
    load_fold_weights,
    predict_oof_from_saved_weights,
    save_weights_for_model,
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
    pyramidal_neurons: np.ndarray
    unfit_neurons: List[int]


def list_required_sessions() -> List[str]:
    set_spk = list_sessions_spike(SPIKE_ROOT)
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


def build_minute_block_shuffled_folds(T: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Match forward-search CV split: shuffle 1-minute blocks, then do KFold."""
    rng = np.random.default_rng(SEED)
    fs = 50
    block_size = fs * 60
    idx = np.arange(T)
    blocks = [idx[i:i + block_size] for i in range(0, T, block_size)]
    rng.shuffle(blocks)
    permuted_idx = np.concatenate(blocks) if blocks else idx
    kf = KFold(n_splits=CV_FOLDS, shuffle=False)
    return [(permuted_idx[tr], permuted_idx[va]) for tr, va in kf.split(permuted_idx)]


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


def compute_session_rllr(
    session: str,
    dayid2cellinfo: Dict[str, Path],
    *,
    n_jobs: int = N_JOBS,
    n_shuffle: int = 200,
) -> Optional[SessionResult]:

    sess_dir = WEIGHTS_BASE / session
    sess_dir.mkdir(parents=True, exist_ok=True)

    paths = session_paths(session)
    for k in ["imu", "spike", "dlc_final", "position"]:
        if not paths[k].exists():
            print(f"[SKIP] {session}: missing input {k}: {paths[k]}")
            return None

    s_lower = session.lower()
    if "indoor" in s_lower:
        group = "indoor"
    elif "outdoor" in s_lower:
        group = "outdoor"
    else:
        print(f"[SKIP] {session}: cannot infer indoor/outdoor from name")
        return None

    data_dict = rebuild_inputs_50hz(session, paths)
    Y50 = load_spikes_50hz_counts(paths["spike"])
    T_spk, N_NEURONS = Y50.shape
    T_cov = int(data_dict["T"])
    if abs(T_cov - T_spk) > MAX_MISMATCH_FRAMES_50HZ:
        print(f"[SKIP] {session}: length mismatch @50Hz cov={T_cov} spk={T_spk}")
        return None

    T = min(T_cov, T_spk)
    for k, v in list(data_dict.items()):
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T_cov:
            data_dict[k] = v[:T]
    Y_all = Y50[:T].astype(np.float64)
    data_dict, Y_all, speed_mask = filter_by_min_speed(data_dict, Y_all, MIN_SPEED_CM_S)
    if speed_mask is not None and not speed_mask.any():
        print(f"[SKIP] {session}: no samples >= min speed {MIN_SPEED_CM_S:g} cm/s")
        return None
    # T = int(data_dict["T"])

    pyr_idx = pyramidal_indices_for_session(session, dayid2cellinfo, N_NEURONS)
    if pyr_idx is None or pyr_idx.size == 0:
        print(f"[SKIP] {session}: pyramidal cell info not found or empty")
        return None

    selected_models = load_forward_selected_models(sess_dir)
    if not selected_models:
        print(f"[SKIP] {session}: no forward-selected models found")
        return None

    pyr_models = {ni: selected_models[ni] for ni in pyr_idx.tolist() if ni in selected_models}
    if not pyr_models:
        print(f"[SKIP] {session}: no pyramidal neurons with forward-selected models")
        return None

    print(
        f"[INFO] {session}: pyramidal neurons={pyr_idx.size}, "
        f"with forward-selected models={len(pyr_models)}",
    )

    T_valid = int(min(T_cov, T_spk))
    folds_idx = build_minute_block_shuffled_folds(T_valid)

    X_cache: Dict[str, Tuple[np.ndarray, List[str], str]] = {}
    x_cache_lock = Lock()

    def get_X(model_vars: List[str]):
        mk = forward_model_key(model_vars)
        with x_cache_lock:
            if mk in X_cache:
                X, feats, _ = X_cache[mk]
                return X, feats, mk
        X, feats = build_design_matrix(model_vars, data_dict)
        with x_cache_lock:
            if mk not in X_cache:
                X_cache[mk] = (X, feats, mk)
            else:
                X, feats, _ = X_cache[mk]
        return X, feats, mk

    def find_saved_model_dir(model_key: str) -> Optional[Path]:
        candidates = [sess_dir / model_key, sess_dir / RLLR_FITS_DIRNAME / model_key]
        for cand in candidates:
            if cand.exists():
                return cand
        return None

    def neuron_weights_ready(model_dir: Path, neuron_idx: int) -> bool:
        idx1 = neuron_idx + 1
        neuron_dir = model_dir / f"neuron_{idx1}"
        if not neuron_dir.exists():
            return False
        for k in range(1, CV_FOLDS + 1):
            csv_path = neuron_dir / f"fold{k}" / "weights.csv"
            if (not csv_path.exists()) or csv_path.stat().st_size < 8:
                return False
        weights_mean = neuron_dir / "weights_mean.csv"
        if (not weights_mean.exists()) or weights_mean.stat().st_size < 8:
            saved_feats = load_feature_names_file(model_dir)
            if not saved_feats:
                return False
            try:
                weights = []
                for k in range(1, CV_FOLDS + 1):
                    csv_path = neuron_dir / f"fold{k}" / "weights.csv"
                    weights.append(load_fold_weights(csv_path, feature_names=saved_feats))
                w_mean = np.mean(np.stack(weights, axis=0), axis=0).astype(np.float32)
                pd.DataFrame(
                    w_mean.reshape(1, -1),
                    index=[f"neuron_{idx1}"],
                    columns=saved_feats,
                ).to_csv(weights_mean)
            except Exception as exc:
                print(f"[WARN] {session}: failed computing weights_mean for neuron_{idx1}: {exc}")
                return False
        return True

    stats_root = sess_dir / RLLR_STATS_DIRNAME
    stats_root.mkdir(parents=True, exist_ok=True)

    full_csv = stats_root / "full_rllr_pyr.csv"
    contrib_csv = stats_root / "dropone_rllr_pyr.csv"
    full_llhi_csv = stats_root / "full_llhi_pyr.csv"
    contrib_llhi_csv = stats_root / "dropone_llhi_pyr.csv"

    full_map: Dict[int, float] = {}
    unfit_neurons: List[int] = []
    contrib: Dict[str, Dict[int, float]] = {v: {} for v in DROP_FEATURES}
    shuf_mean_rllr: Dict[str, Dict[int, float]] = {v: {} for v in DROP_FEATURES}
    shuf_std_rllr: Dict[str, Dict[int, float]] = {v: {} for v in DROP_FEATURES}
    full_llhi_map: Dict[int, float] = {}
    contrib_llhi: Dict[str, Dict[int, float]] = {v: {} for v in DROP_FEATURES}
    shuf_mean_llhi: Dict[str, Dict[int, float]] = {v: {} for v in DROP_FEATURES}
    shuf_std_llhi: Dict[str, Dict[int, float]] = {v: {} for v in DROP_FEATURES}

    has_rllr = False
    has_rllr_shuffle = False
    has_rllr_z = False
    has_rllr_pvalue = False
    if full_csv.exists() and contrib_csv.exists():
        df_full = pd.read_csv(full_csv)
        df_con = pd.read_csv(contrib_csv)
        required_composites = set(DROP_COMPOSITES.keys())
        present_features = set(df_con["feature"].astype(str).unique()) if "feature" in df_con.columns else set()
        has_rllr = ("ll_gain" in df_full.columns) and ("rllr" in df_con.columns)
        if required_composites and not required_composites.issubset(present_features):
            has_rllr = False
        has_rllr_shuffle = ("rllr_shuf_mean" in df_con.columns) and ("rllr_shuf_std" in df_con.columns)
        has_rllr_z = "rllr_z" in df_con.columns
        has_rllr_pvalue = "rllr_pvalue" in df_con.columns
        if has_rllr:
            full_map = {int(r["neuron_idx"]): float(r["ll_gain"]) for _, r in df_full.iterrows()}
            for _, r in df_con.iterrows():
                feat = str(r["feature"])
                if feat not in contrib:
                    contrib[feat] = {}
                    shuf_mean_rllr[feat] = {}
                    shuf_std_rllr[feat] = {}
                ni = int(r["neuron_idx"])
                contrib[feat][ni] = float(r["rllr"])
                if has_rllr_shuffle:
                    mu = r.get("rllr_shuf_mean", np.nan)
                    std = r.get("rllr_shuf_std", np.nan)
                    if np.isfinite(mu):
                        shuf_mean_rllr[feat][ni] = float(mu)
                    if np.isfinite(std):
                        shuf_std_rllr[feat][ni] = float(std)

    has_llhi = False
    has_llhi_shuffle = False
    has_llhi_z = False
    has_llhi_pvalue = False
    has_llhi_rscc = False
    if full_llhi_csv.exists() and contrib_llhi_csv.exists():
        df_full_llhi = pd.read_csv(full_llhi_csv)
        df_con_llhi = pd.read_csv(contrib_llhi_csv)
        required_composites = set(DROP_COMPOSITES.keys())
        present_features = set(df_con_llhi["feature"].astype(str).unique()) if "feature" in df_con_llhi.columns else set()
        has_llhi = ("llhi_full" in df_full_llhi.columns) and ("delta_llhi" in df_con_llhi.columns)
        if required_composites and not required_composites.issubset(present_features):
            has_llhi = False
        has_llhi_shuffle = ("delta_llhi_shuf_mean" in df_con_llhi.columns) and ("delta_llhi_shuf_std" in df_con_llhi.columns)
        has_llhi_z = "delta_llhi_z" in df_con_llhi.columns
        has_llhi_pvalue = "delta_llhi_pvalue" in df_con_llhi.columns
        has_llhi_rscc = "rSCC" in df_con_llhi.columns
        if has_llhi:
            full_llhi_map = {int(r["neuron_idx"]): float(r["llhi_full"]) for _, r in df_full_llhi.iterrows()}
            for _, r in df_con_llhi.iterrows():
                feat = str(r["feature"])
                if feat not in contrib_llhi:
                    contrib_llhi[feat] = {}
                    shuf_mean_llhi[feat] = {}
                    shuf_std_llhi[feat] = {}
                ni = int(r["neuron_idx"])
                contrib_llhi[feat][ni] = float(r["delta_llhi"])
                if has_llhi_shuffle:
                    mu = r.get("delta_llhi_shuf_mean", np.nan)
                    std = r.get("delta_llhi_shuf_std", np.nan)
                    if np.isfinite(mu):
                        shuf_mean_llhi[feat][ni] = float(mu)
                    if np.isfinite(std):
                        shuf_std_llhi[feat][ni] = float(std)

    if has_rllr and has_rllr_shuffle:
        if ensure_z_pvalue_columns(
            contrib_csv,
            value_col="rllr",
            mean_col="rllr_shuf_mean",
            std_col="rllr_shuf_std",
            z_col="rllr_z",
            p_col="rllr_pvalue",
        ):
            df_con = pd.read_csv(contrib_csv)
            has_rllr_z = "rllr_z" in df_con.columns
            has_rllr_pvalue = "rllr_pvalue" in df_con.columns

    if has_llhi and has_llhi_shuffle:
        if ensure_z_pvalue_columns(
            contrib_llhi_csv,
            value_col="delta_llhi",
            mean_col="delta_llhi_shuf_mean",
            std_col="delta_llhi_shuf_std",
            z_col="delta_llhi_z",
            p_col="delta_llhi_pvalue",
        ):
            df_con_llhi = pd.read_csv(contrib_llhi_csv)
            has_llhi_z = "delta_llhi_z" in df_con_llhi.columns
            has_llhi_pvalue = "delta_llhi_pvalue" in df_con_llhi.columns

    if has_llhi:
        if ensure_rscc_column(contrib_llhi_csv):
            df_con_llhi = pd.read_csv(contrib_llhi_csv)
        has_llhi_rscc = "rSCC" in df_con_llhi.columns if "df_con_llhi" in locals() else has_llhi_rscc

    if has_rllr and has_llhi and has_llhi_shuffle and has_rllr_shuffle and has_rllr_z and has_rllr_pvalue and has_llhi_z and has_llhi_pvalue and has_llhi_rscc:
        unfit_neurons = sorted([ni for ni, val in full_map.items() if np.isfinite(val) and val < 0])
        return SessionResult(
            session=session,
            group=group,
            full_ll_gain_by_neuron={ni: v for ni, v in full_map.items() if ni not in unfit_neurons},
            contrib_rllr_by_feature_by_neuron={
                feat: {ni: val for ni, val in vals.items() if ni not in unfit_neurons}
                for feat, vals in contrib.items()
            },
            shuf_mean_rllr_by_feature_by_neuron=shuf_mean_rllr,
            shuf_std_rllr_by_feature_by_neuron=shuf_std_rllr,
            full_llhi_by_neuron={ni: v for ni, v in full_llhi_map.items() if ni not in unfit_neurons},
            contrib_delta_llhi_by_feature_by_neuron={
                feat: {ni: val for ni, val in vals.items() if ni not in unfit_neurons}
                for feat, vals in contrib_llhi.items()
            },
            shuf_mean_delta_llhi_by_feature_by_neuron=shuf_mean_llhi,
            shuf_std_delta_llhi_by_feature_by_neuron=shuf_std_llhi,
            pyramidal_neurons=pyr_idx,
            unfit_neurons=unfit_neurons,
        )

    need_rllr = not has_rllr
    need_llhi = not has_llhi
    need_rllr_shuffle = not has_rllr_shuffle
    need_llhi_shuffle = not has_llhi_shuffle

    existing_unfit = {ni for ni, val in full_map.items() if np.isfinite(val) and val < 0}
    if existing_unfit:
        unfit_neurons.extend(sorted(existing_unfit))
    full_ll_gain_by_neuron: Dict[int, float] = full_map.copy()
    ll_full_by_neuron: Dict[int, float] = {}
    ll0_by_neuron: Dict[int, float] = {}
    full_llhi_by_neuron: Dict[int, float] = full_llhi_map.copy()
    contrib_rllr: Dict[str, Dict[int, float]] = {v: {} for v in DROP_FEATURES}
    contrib_delta_llhi: Dict[str, Dict[int, float]] = {v: {} for v in DROP_FEATURES}
    contrib_rscc: Dict[str, Dict[int, float]] = {v: {} for v in DROP_FEATURES}
    shuf_mean_rllr = {v: dict(shuf_mean_rllr.get(v, {})) for v in DROP_FEATURES}
    shuf_std_rllr = {v: dict(shuf_std_rllr.get(v, {})) for v in DROP_FEATURES}
    shuf_mean_llhi = {v: dict(shuf_mean_llhi.get(v, {})) for v in DROP_FEATURES}
    shuf_std_llhi = {v: dict(shuf_std_llhi.get(v, {})) for v in DROP_FEATURES}
    if has_rllr:
        contrib_rllr = contrib
    if has_llhi:
        contrib_delta_llhi = contrib_llhi

    missing_full_model_dir = 0
    missing_full_weights = 0
    failed_full_predict = 0
    full_processed = 0
    total_drop_models = 0
    missing_drop_model_dir = 0
    missing_drop_weights = 0
    failed_drop_predict = 0

    dropone_root = WEIGHTS_BASE / "drop_one" / session
    dropone_root.mkdir(parents=True, exist_ok=True)

    def _process_one_neuron(ni: int, full_vars: List[str]) -> Dict:
        out = {
            "skip": False,
            "skip_reason": None,
            "ni": ni,
            "ll0": np.nan,
            "ll_full": np.nan,
            "ll_gain": np.nan,
            "llhi_full": np.nan,
            "unfit": False,
            "drop_specs": {},
            "contrib_rllr": {},
            "contrib_delta_llhi": {},
            "contrib_rscc": {},
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
        if ni in existing_unfit:
            out["skip"] = True
            out["skip_reason"] = "existing_unfit"
            return out

        y = Y_all[:, ni].astype(np.float64)
        mu0_oof = None
        if need_rllr or need_rllr_shuffle or need_llhi or need_llhi_shuffle:
            mu0_oof = build_oof_intercept_mu(y, folds_idx)
            if need_rllr:
                out["ll0"] = float(poisson_loglik(y, mu0_oof))

        X_full, feats_full, mk_full = get_X(full_vars)
        model_dir_full = find_saved_model_dir(mk_full)
        if model_dir_full is None:
            print(f"[SKIP] {session}: missing saved weights for full model {mk_full} (neuron_{ni+1})")
            out["skip"] = True
            out["skip_reason"] = "missing_full_model_dir"
            out["missing_full_model_dir"] = 1
            return out
        if not neuron_weights_ready(model_dir_full, ni):
            print(f"[SKIP] {session}: incomplete weights for full model {mk_full} (neuron_{ni+1})")
            out["skip"] = True
            out["skip_reason"] = "missing_full_weights"
            out["missing_full_weights"] = 1
            return out
        try:
            mu_oof_full = predict_oof_from_saved_weights(model_dir_full, X_full, feats_full, folds_idx, ni)
        except Exception as exc:
            print(f"[SKIP] {session}: failed loading full model {mk_full} (neuron_{ni+1}): {exc}")
            out["skip"] = True
            out["skip_reason"] = "failed_full_predict"
            out["failed_full_predict"] = 1
            return out

        ll_full = np.nan
        if need_rllr:
            ll_full = float(poisson_loglik(y, mu_oof_full))
            out["ll_full"] = ll_full
            out["ll_gain"] = float(ll_full - out["ll0"])
            if np.isfinite(out["ll_gain"]) and out["ll_gain"] < 0:
                out["unfit"] = True
                return out

        if need_llhi and mu0_oof is not None:
            out["llhi_full"] = float(compute_llhi_bps_poisson_vs_baseline(y, mu_oof_full, mu0_oof))
        out["full_processed"] = 1

        drop_specs: Dict[str, List[str]] = {}
        for v in full_vars:
            drop_specs[v] = [x for x in full_vars if x != v]
        for comp_name, members in DROP_COMPOSITES.items():
            member_set = set(members)
            if not member_set:
                continue
            if not any(v in member_set for v in full_vars):
                continue
            drop_specs[comp_name] = [x for x in full_vars if x not in member_set]
        out["drop_specs"] = drop_specs

        mu_oof_red_by_feat: Dict[str, np.ndarray] = {}
        for v, drop_vars in drop_specs.items():
            if not drop_vars:
                if mu0_oof is not None:
                    mu_oof_red_by_feat[v] = mu0_oof.copy()
                continue
            X_red, feats_red, mk_red = get_X(drop_vars)
            model_dir_red = dropone_root / f"neuron_{ni + 1}" / mk_red
            if not model_dir_red.exists():
                out["missing_drop_model_dir"] += 1
            if not neuron_weights_ready(model_dir_red, ni):
                save_weights_for_model(
                    model_dir=model_dir_red,
                    feature_names=feats_red,
                    X_all=X_red,
                    Y_all=Y_all,
                    folds_idx=folds_idx,
                    neuron_indices=np.array([ni], dtype=int),
                    n_jobs=1,
                    folds_count=len(folds_idx),
                )
            if not neuron_weights_ready(model_dir_red, ni):
                out["missing_drop_weights"] += 1
                continue
            try:
                mu_red = predict_oof_from_saved_weights(model_dir_red, X_red, feats_red, folds_idx, ni)
                mu_oof_red_by_feat[v] = mu_red
            except Exception as exc:
                print(f"[WARN] {session}: failed loading drop model {mk_red} (neuron_{ni+1}): {exc}")
                out["failed_drop_predict"] += 1
            out["total_drop_models"] += 1

        if need_llhi_shuffle or need_rllr_shuffle:
            rng = np.random.default_rng(SEED + int(ni))
            shuf_llhi_vals: Dict[str, List[float]] = {v: [] for v in drop_specs}
            shuf_rllr_vals: Dict[str, List[float]] = {v: [] for v in drop_specs}
            for _ in range(n_shuffle):
                y_shuf = circular_shift(y, rng)
                llhi_full_shuf = np.nan
                ll0_shuf = np.nan
                ll_full_shuf = np.nan
                denom_shuf = np.nan
                if need_llhi_shuffle and mu0_oof is not None:
                    llhi_full_shuf = compute_llhi_bps_poisson_vs_baseline(y_shuf, mu_oof_full, mu0_oof)
                if need_rllr_shuffle and mu0_oof is not None:
                    ll0_shuf = poisson_loglik(y_shuf, mu0_oof)
                    ll_full_shuf = poisson_loglik(y_shuf, mu_oof_full)
                    denom_shuf = ll_full_shuf - ll0_shuf

                for v, mu_red in mu_oof_red_by_feat.items():
                    if need_llhi_shuffle and mu0_oof is not None:
                        llhi_red_shuf = compute_llhi_bps_poisson_vs_baseline(y_shuf, mu_red, mu0_oof)
                        shuf_llhi_vals[v].append(float(llhi_full_shuf - llhi_red_shuf))
                    if need_rllr_shuffle and np.isfinite(denom_shuf) and denom_shuf > MU_EPS:
                        ll_red_shuf = poisson_loglik(y_shuf, mu_red)
                        if np.isfinite(ll_red_shuf) and np.isfinite(ll_full_shuf):
                            shuf_rllr_vals[v].append(float((ll_full_shuf - ll_red_shuf) / denom_shuf))

            for v in drop_specs:
                if need_llhi_shuffle:
                    arr = np.asarray(shuf_llhi_vals.get(v, []), dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        out["shuf_mean_llhi"][v] = float(np.mean(arr))
                        out["shuf_std_llhi"][v] = float(np.std(arr))
                if need_rllr_shuffle:
                    arr = np.asarray(shuf_rllr_vals.get(v, []), dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        out["shuf_mean_rllr"][v] = float(np.mean(arr))
                        out["shuf_std_rllr"][v] = float(np.std(arr))

        for v in DROP_FEATURES:
            if need_rllr:
                out["contrib_rllr"][v] = float("nan")
            if need_llhi:
                out["contrib_delta_llhi"][v] = float("nan")

        for v in drop_specs:
            if v not in mu_oof_red_by_feat:
                if need_rllr:
                    out["contrib_rllr"][v] = float("nan")
                if need_llhi:
                    out["contrib_delta_llhi"][v] = float("nan")
                continue

            if need_rllr:
                denom = out["ll_gain"]
                if denom <= 0 or not np.isfinite(denom) or not np.isfinite(ll_full):
                    out["contrib_rllr"][v] = float("nan")
                else:
                    ll_red = poisson_loglik(y, mu_oof_red_by_feat[v])
                    out["contrib_rllr"][v] = float((ll_full - ll_red) / denom)

            if need_llhi and mu0_oof is not None:
                llhi_full = out["llhi_full"]
                llhi_red = compute_llhi_bps_poisson_vs_baseline(y, mu_oof_red_by_feat[v], mu0_oof)
                if not np.isfinite(llhi_full) or not np.isfinite(llhi_red):
                    out["contrib_delta_llhi"][v] = float("nan")
                else:
                    out["contrib_delta_llhi"][v] = float(llhi_full - llhi_red)

        if need_llhi:
            deltas = np.array([out["contrib_delta_llhi"].get(v, np.nan) for v in full_vars], dtype=float)
            finite = np.isfinite(deltas)
            l2_norm = float(np.sqrt(np.sum(np.square(deltas[finite])))) if np.any(finite) else 0.0
            for v in DROP_FEATURES:
                delta_v = out["contrib_delta_llhi"].get(v, np.nan)
                if np.isfinite(delta_v) and np.isfinite(l2_norm) and l2_norm > MU_EPS:
                    out["contrib_rscc"][v] = float(delta_v / l2_norm)
                else:
                    out["contrib_rscc"][v] = 0.0

        return out

    neuron_items = list(pyr_models.items())
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

    for out in neuron_results:
        ni = out["ni"]
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
        if out["unfit"]:
            unfit_neurons.append(ni)
            if need_rllr and np.isfinite(out["ll_gain"]):
                ll0_by_neuron[ni] = float(out["ll0"])
                ll_full_by_neuron[ni] = float(out["ll_full"])
                full_ll_gain_by_neuron[ni] = float(out["ll_gain"])
            continue

        if need_rllr:
            ll0_by_neuron[ni] = float(out["ll0"])
            ll_full_by_neuron[ni] = float(out["ll_full"])
            full_ll_gain_by_neuron[ni] = float(out["ll_gain"])
        if need_llhi and np.isfinite(out["llhi_full"]):
            full_llhi_by_neuron[ni] = float(out["llhi_full"])

        for v in DROP_FEATURES:
            if need_rllr:
                contrib_rllr[v][ni] = out["contrib_rllr"].get(v, float("nan"))
            if need_llhi:
                contrib_delta_llhi[v][ni] = out["contrib_delta_llhi"].get(v, float("nan"))
                contrib_rscc[v][ni] = out["contrib_rscc"].get(v, 0.0)
            if need_rllr_shuffle:
                mu = out["shuf_mean_rllr"].get(v, np.nan)
                std = out["shuf_std_rllr"].get(v, np.nan)
                if np.isfinite(mu):
                    shuf_mean_rllr[v][ni] = float(mu)
                if np.isfinite(std):
                    shuf_std_rllr[v][ni] = float(std)
            if need_llhi_shuffle:
                mu = out["shuf_mean_llhi"].get(v, np.nan)
                std = out["shuf_std_llhi"].get(v, np.nan)
                if np.isfinite(mu):
                    shuf_mean_llhi[v][ni] = float(mu)
                if np.isfinite(std):
                    shuf_std_llhi[v][ni] = float(std)

    need_rllr_zp = not (has_rllr_z and has_rllr_pvalue)
    if need_rllr:
        df_full_rows = [
            {
                "session": session,
                "group": group,
                "neuron_idx": ni,
                "ll_full": ll_full_by_neuron[ni],
                "ll0": ll0_by_neuron[ni],
                "ll_gain": full_ll_gain_by_neuron[ni],
            }
            for ni in sorted(full_ll_gain_by_neuron.keys())
        ]
        if not df_full_rows:
            print(f"[WARN] {session}: no full-model rows produced; check weights and inputs")
        df_full = pd.DataFrame(
            df_full_rows,
            columns=[
                "session",
                "group",
                "neuron_idx",
                "ll_full",
                "ll0",
                "ll_gain",
            ],
        )
        df_full.to_csv(full_csv, index=False)

    if need_rllr or need_rllr_shuffle or need_rllr_zp:
        rows = []
        all_neurons = sorted(set(full_ll_gain_by_neuron.keys()) | set(full_map.keys()))
        for v in DROP_FEATURES:
            for ni in all_neurons:
                frac = contrib_rllr[v].get(ni, float("nan"))
                mu = shuf_mean_rllr[v].get(ni, float("nan"))
                std = shuf_std_rllr[v].get(ni, float("nan"))
                rllr_z = zscore(frac, mu, std)
                rows.append(
                    {
                        "session": session,
                        "group": group,
                        "feature": v,
                        "neuron_idx": ni,
                        "rllr": frac,
                        "rllr_shuf_mean": mu,
                        "rllr_shuf_std": std,
                        "rllr_z": rllr_z,
                        "rllr_pvalue": right_tail_pvalue(rllr_z),
                    }
                )
        if not rows:
            print(f"[WARN] {session}: no drop-one rows produced; check reduced models and weights")
        pd.DataFrame(
            rows,
            columns=[
                "session",
                "group",
                "feature",
                "neuron_idx",
                "rllr",
                "rllr_shuf_mean",
                "rllr_shuf_std",
                "rllr_z",
                "rllr_pvalue",
            ],
        ).to_csv(contrib_csv, index=False)

    if need_llhi:
        df_full_llhi_rows = [
            {
                "session": session,
                "group": group,
                "neuron_idx": ni,
                "llhi_full": full_llhi_by_neuron[ni],
            }
            for ni in sorted(full_llhi_by_neuron.keys())
        ]
        if not df_full_llhi_rows:
            print(f"[WARN] {session}: no full LLHI rows produced; check weights and inputs")
        pd.DataFrame(
            df_full_llhi_rows,
            columns=[
                "session",
                "group",
                "neuron_idx",
                "llhi_full",
            ],
        ).to_csv(full_llhi_csv, index=False)

    need_llhi_pvalue = not has_llhi_pvalue
    if need_llhi or need_llhi_shuffle or need_llhi_pvalue:
        llhi_rows = []
        all_neurons = sorted(set(full_llhi_by_neuron.keys()) | set(full_llhi_map.keys()))
        for v in DROP_FEATURES:
            for ni in all_neurons:
                delta = contrib_delta_llhi[v].get(ni, float("nan"))
                rscc = contrib_rscc[v].get(ni, 0.0)
                mu = shuf_mean_llhi[v].get(ni, float("nan"))
                std = shuf_std_llhi[v].get(ni, float("nan"))
                delta_z = zscore(delta, mu, std)
                llhi_rows.append(
                    {
                        "session": session,
                        "group": group,
                        "feature": v,
                        "neuron_idx": ni,
                        "delta_llhi": delta,
                        "rSCC": rscc,
                        "delta_llhi_shuf_mean": mu,
                        "delta_llhi_shuf_std": std,
                        "delta_llhi_z": delta_z,
                        "delta_llhi_pvalue": right_tail_pvalue(delta_z),
                    }
                )
        if not llhi_rows:
            print(f"[WARN] {session}: no drop-one LLHI rows produced; check reduced models and weights")
        pd.DataFrame(
            llhi_rows,
            columns=[
                "session",
                "group",
                "feature",
                "neuron_idx",
                "delta_llhi",
                "rSCC",
                "delta_llhi_shuf_mean",
                "delta_llhi_shuf_std",
                "delta_llhi_z",
                "delta_llhi_pvalue",
            ],
        ).to_csv(contrib_llhi_csv, index=False)

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

    unfit_set = set(unfit_neurons)
    return SessionResult(
        session=session,
        group=group,
        full_ll_gain_by_neuron={ni: v for ni, v in full_ll_gain_by_neuron.items() if ni not in unfit_neurons},
        contrib_rllr_by_feature_by_neuron={
            feat: {ni: val for ni, val in vals.items() if ni not in unfit_set}
            for feat, vals in contrib_rllr.items()
        },
        shuf_mean_rllr_by_feature_by_neuron=shuf_mean_rllr,
        shuf_std_rllr_by_feature_by_neuron=shuf_std_rllr,
        full_llhi_by_neuron={ni: v for ni, v in full_llhi_by_neuron.items() if ni not in unfit_neurons},
        contrib_delta_llhi_by_feature_by_neuron={
            feat: {ni: val for ni, val in vals.items() if ni not in unfit_set}
            for feat, vals in contrib_delta_llhi.items()
        },
        shuf_mean_delta_llhi_by_feature_by_neuron=shuf_mean_llhi,
        shuf_std_delta_llhi_by_feature_by_neuron=shuf_std_llhi,
        pyramidal_neurons=pyr_idx,
        unfit_neurons=sorted(set(unfit_neurons)),
    )
