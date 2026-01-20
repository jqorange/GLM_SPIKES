#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute Poisson-GLM drop-one contributions using rLLR normalization with forward-selected
(full) models. Full model is the forward-search-selected subset per neuron; drop-one
models remove a single covariate from that subset. Contributions are computed as the
relative log-likelihood ratio (rLLR) and include label-shuffle z-scoring. This pipeline
reuses saved 10-fold weights from GLM_Poisson_forward. Drop-one models are (re)fit per
neuron and cached under a drop_one/ folder for reuse.
"""

from __future__ import annotations

import faulthandler

faulthandler.enable()
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from contribution_rllr_utils import (
    CI_HI,
    CI_LO,
    HEAD_POSE_COMPONENTS,
    HEAD_POSE_FEATURE,
    MU_EPS,
    N_BOOT,
    RLLR_FITS_DIRNAME,
    RLLR_STATS_DIRNAME,
    DroponeSessionStats,
    build_oof_intercept_mu,
    compute_head_pose_map,
    collect_dropone_plot_data,
    hierarchical_bootstrap_mean,
    load_forward_selected_models,
    plot_dropone_suite,
    plot_summary_figure,
    poisson_loglik,
    build_dayid_to_cellinfo,
    load_feature_names_file,
    load_fold_weights,
    pyramidal_indices_for_session,
    predict_oof_from_saved_weights,
    save_weights_for_model,
)
from glm_poisson_forward.metrics import compute_llhi_bps_poisson
from glm_poisson_forward.config import (
    CV_FOLDS,
    DLC_ROOT,
    IMU_ROOT,
    MAX_MISMATCH_FRAMES_50HZ,
    MIN_SPEED_CM_S,
    N_JOBS,
    POSITION_ROOT,
    SEED,
    SPIKE_ROOT,
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

MIN_FULL_LL_GAIN = 0.0
MIN_FULL_LLHI = 0.0
POSITIVE_CONTRIB_ONLY = False
INCLUDE_UNFIT_CELLS = False
LINK_INDOOR_OUTDOOR_PAIRS = False
INCLUDE_HEAD_POSE = True
N_SHUFFLE = 200


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


def circular_shift(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = arr.size
    if n <= 1:
        return arr
    shift = int(rng.integers(1, n))
    return np.roll(arr, shift)


def compute_session_rllr(
    session: str,
    dayid2cellinfo: Dict[str, Path],
    *,
    n_jobs: int = N_JOBS,
) -> Optional[SessionResult]:
    def zscore(value: float, mu: float, sigma: float) -> float:
        if not (np.isfinite(value) and np.isfinite(mu) and np.isfinite(sigma)):
            return float("nan")
        if sigma <= MU_EPS:
            return float("nan")
        return float((value - mu) / sigma)
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
    for k in ["position", "head_v", "head_v_bin", "roll_bin", "yaw_bin", "pitch_bin"]:
        data_dict[k] = data_dict[k][:T]
    Y_all = Y50[:T].astype(np.float64)
    data_dict, Y_all, speed_mask = filter_by_min_speed(data_dict, Y_all, MIN_SPEED_CM_S)
    if speed_mask is not None and not speed_mask.any():
        print(f"[SKIP] {session}: no samples >= min speed {MIN_SPEED_CM_S:g} cm/s")
        return None
    T = int(data_dict["T"])

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

    kf = KFold(n_splits=CV_FOLDS, shuffle=False)
    folds_idx = list(kf.split(np.arange(T)))

    X_cache: Dict[str, Tuple[np.ndarray, List[str], str]] = {}

    def get_X(model_vars: List[str]):
        mk = forward_model_key(model_vars)
        if mk in X_cache:
            X, feats, _ = X_cache[mk]
            return X, feats, mk
        X, feats = build_design_matrix(model_vars, data_dict)
        X_cache[mk] = (X, feats, mk)
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
    contrib: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
    shuf_mean_rllr: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
    shuf_std_rllr: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
    full_llhi_map: Dict[int, float] = {}
    contrib_llhi: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
    shuf_mean_llhi: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
    shuf_std_llhi: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}

    has_rllr = False
    has_rllr_shuffle = False
    if full_csv.exists() and contrib_csv.exists():
        df_full = pd.read_csv(full_csv)
        df_con = pd.read_csv(contrib_csv)
        has_rllr = ("ll_gain" in df_full.columns) and ("rllr" in df_con.columns)
        has_rllr_shuffle = ("rllr_shuf_mean" in df_con.columns) and ("rllr_shuf_std" in df_con.columns)
        if has_rllr:
            full_map = {int(r["neuron_idx"]): float(r["ll_gain"]) for _, r in df_full.iterrows()}
            for _, r in df_con.iterrows():
                feat = str(r["feature"])
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
    if full_llhi_csv.exists() and contrib_llhi_csv.exists():
        df_full_llhi = pd.read_csv(full_llhi_csv)
        df_con_llhi = pd.read_csv(contrib_llhi_csv)
        has_llhi = ("llhi_full" in df_full_llhi.columns) and ("delta_llhi" in df_con_llhi.columns)
        has_llhi_shuffle = ("delta_llhi_shuf_mean" in df_con_llhi.columns) and ("delta_llhi_shuf_std" in df_con_llhi.columns)
        if has_llhi:
            full_llhi_map = {int(r["neuron_idx"]): float(r["llhi_full"]) for _, r in df_full_llhi.iterrows()}
            for _, r in df_con_llhi.iterrows():
                feat = str(r["feature"])
                ni = int(r["neuron_idx"])
                contrib_llhi[feat][ni] = float(r["delta_llhi"])
                if has_llhi_shuffle:
                    mu = r.get("delta_llhi_shuf_mean", np.nan)
                    std = r.get("delta_llhi_shuf_std", np.nan)
                    if np.isfinite(mu):
                        shuf_mean_llhi[feat][ni] = float(mu)
                    if np.isfinite(std):
                        shuf_std_llhi[feat][ni] = float(std)

    if has_rllr and has_llhi and has_rllr_shuffle and has_llhi_shuffle:
        return SessionResult(
            session=session,
            group=group,
            full_ll_gain_by_neuron=full_map,
            contrib_rllr_by_feature_by_neuron=contrib,
            shuf_mean_rllr_by_feature_by_neuron=shuf_mean_rllr,
            shuf_std_rllr_by_feature_by_neuron=shuf_std_rllr,
            full_llhi_by_neuron=full_llhi_map,
            contrib_delta_llhi_by_feature_by_neuron=contrib_llhi,
            shuf_mean_delta_llhi_by_feature_by_neuron=shuf_mean_llhi,
            shuf_std_delta_llhi_by_feature_by_neuron=shuf_std_llhi,
            pyramidal_neurons=pyr_idx,
        )

    need_rllr = not has_rllr
    need_llhi = not has_llhi
    need_rllr_shuffle = not has_rllr_shuffle
    need_llhi_shuffle = not has_llhi_shuffle

    full_ll_gain_by_neuron: Dict[int, float] = full_map.copy()
    ll_full_by_neuron: Dict[int, float] = {}
    ll0_by_neuron: Dict[int, float] = {}
    full_llhi_by_neuron: Dict[int, float] = full_llhi_map.copy()
    contrib_rllr: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
    contrib_delta_llhi: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
    shuf_mean_rllr = {v: dict(shuf_mean_rllr.get(v, {})) for v in VARS_ALL}
    shuf_std_rllr = {v: dict(shuf_std_rllr.get(v, {})) for v in VARS_ALL}
    shuf_mean_llhi = {v: dict(shuf_mean_llhi.get(v, {})) for v in VARS_ALL}
    shuf_std_llhi = {v: dict(shuf_std_llhi.get(v, {})) for v in VARS_ALL}
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

    for ni, full_vars in pyr_models.items():
        y = Y_all[:, ni].astype(np.float64)
        if need_rllr:
            mu0_oof = build_oof_intercept_mu(y, folds_idx)
            ll0 = poisson_loglik(y, mu0_oof)
            ll0_by_neuron[ni] = float(ll0)

        X_full, feats_full, mk_full = get_X(full_vars)
        model_dir_full = find_saved_model_dir(mk_full)
        if model_dir_full is None:
            print(f"[SKIP] {session}: missing saved weights for full model {mk_full} (neuron_{ni+1})")
            missing_full_model_dir += 1
            continue
        if not neuron_weights_ready(model_dir_full, ni):
            print(f"[SKIP] {session}: incomplete weights for full model {mk_full} (neuron_{ni+1})")
            missing_full_weights += 1
            continue
        try:
            mu_oof_full = predict_oof_from_saved_weights(model_dir_full, X_full, feats_full, folds_idx, ni)
        except Exception as exc:
            print(f"[SKIP] {session}: failed loading full model {mk_full} (neuron_{ni+1}): {exc}")
            failed_full_predict += 1
            continue
        ll_full = None
        if need_rllr:
            ll_full = poisson_loglik(y, mu_oof_full)
            ll_full_by_neuron[ni] = float(ll_full)
            ll_gain = float(ll_full - ll0_by_neuron[ni])
            full_ll_gain_by_neuron[ni] = ll_gain
        if need_llhi:
            llhi_full = compute_llhi_bps_poisson(y, mu_oof_full)
            full_llhi_by_neuron[ni] = float(llhi_full)
        full_processed += 1

        mu_oof_red_by_feat: Dict[str, np.ndarray] = {}
        for v in full_vars:
            drop_vars = [x for x in full_vars if x != v]
            if not drop_vars:
                continue
            X_red, feats_red, mk_red = get_X(drop_vars)
            model_dir_red = dropone_root / f"neuron_{ni + 1}" / mk_red
            if not model_dir_red.exists():
                missing_drop_model_dir += 1
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
                missing_drop_weights += 1
                continue
            try:
                if need_rllr or need_llhi:
                    mu_red = predict_oof_from_saved_weights(model_dir_red, X_red, feats_red, folds_idx, ni)
                    mu_oof_red_by_feat[v] = mu_red
            except Exception as exc:
                print(f"[WARN] {session}: failed loading drop model {mk_red} (neuron_{ni+1}): {exc}")
                failed_drop_predict += 1
            total_drop_models += 1

        if need_rllr_shuffle or need_llhi_shuffle:
            rng = np.random.default_rng(SEED + int(ni))
            shuf_rllr_vals: Dict[str, List[float]] = {v: [] for v in full_vars}
            shuf_llhi_vals: Dict[str, List[float]] = {v: [] for v in full_vars}
            for _ in range(N_SHUFFLE):
                y_shuf = circular_shift(y, rng)
                if need_rllr_shuffle:
                    mu0_shuf = build_oof_intercept_mu(y_shuf, folds_idx)
                    ll0_shuf = poisson_loglik(y_shuf, mu0_shuf)
                    ll_full_shuf = poisson_loglik(y_shuf, mu_oof_full)
                else:
                    ll0_shuf = np.nan
                    ll_full_shuf = np.nan

                if need_llhi_shuffle:
                    llhi_full_shuf = compute_llhi_bps_poisson(y_shuf, mu_oof_full)
                else:
                    llhi_full_shuf = np.nan

                for v, mu_red in mu_oof_red_by_feat.items():
                    if need_rllr_shuffle:
                        ll_red_shuf = poisson_loglik(y_shuf, mu_red)
                        denom = ll_full_shuf - ll0_shuf
                        if not np.isfinite(denom) or denom <= 0:
                            rllr_shuf = float("nan")
                        else:
                            rllr_shuf = float((ll_full_shuf - ll_red_shuf) / denom)
                        shuf_rllr_vals[v].append(rllr_shuf)

                    if need_llhi_shuffle:
                        llhi_red_shuf = compute_llhi_bps_poisson(y_shuf, mu_red)
                        shuf_llhi_vals[v].append(float(llhi_full_shuf - llhi_red_shuf))

            for v in full_vars:
                if need_rllr_shuffle:
                    arr = np.asarray(shuf_rllr_vals.get(v, []), dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        shuf_mean_rllr[v][ni] = float(np.mean(arr))
                        shuf_std_rllr[v][ni] = float(np.std(arr))
                if need_llhi_shuffle:
                    arr = np.asarray(shuf_llhi_vals.get(v, []), dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size:
                        shuf_mean_llhi[v][ni] = float(np.mean(arr))
                        shuf_std_llhi[v][ni] = float(np.std(arr))

        for v in full_vars:
            if v not in mu_oof_red_by_feat:
                if need_rllr:
                    contrib_rllr[v][ni] = float("nan")
                if need_llhi:
                    contrib_delta_llhi[v][ni] = float("nan")
                continue

            if need_rllr:
                denom = full_ll_gain_by_neuron.get(ni, float("nan"))
                if denom <= 0 or not np.isfinite(denom) or ll_full is None:
                    contrib_rllr[v][ni] = float("nan")
                else:
                    ll_red = poisson_loglik(y, mu_oof_red_by_feat[v])
                    contrib_rllr[v][ni] = float((ll_full - ll_red) / denom)

            if need_llhi:
                llhi_full = full_llhi_by_neuron.get(ni, float("nan"))
                llhi_red = compute_llhi_bps_poisson(y, mu_oof_red_by_feat[v])
                if not np.isfinite(llhi_full) or not np.isfinite(llhi_red):
                    contrib_delta_llhi[v][ni] = float("nan")
                else:
                    contrib_delta_llhi[v][ni] = float(llhi_full - llhi_red)

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

        rows = []
        for v in VARS_ALL:
            for ni, frac in contrib_rllr[v].items():
                mu = shuf_mean_rllr[v].get(ni, float("nan"))
                std = shuf_std_rllr[v].get(ni, float("nan"))
                rows.append(
                    {
                        "session": session,
                        "group": group,
                        "feature": v,
                        "neuron_idx": ni,
                        "rllr": frac,
                        "rllr_shuf_mean": mu,
                        "rllr_shuf_std": std,
                        "rllr_z": zscore(frac, mu, std),
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

        llhi_rows = []
        for v in VARS_ALL:
            for ni, delta in contrib_delta_llhi[v].items():
                mu = shuf_mean_llhi[v].get(ni, float("nan"))
                std = shuf_std_llhi[v].get(ni, float("nan"))
                llhi_rows.append(
                    {
                        "session": session,
                        "group": group,
                        "feature": v,
                        "neuron_idx": ni,
                        "delta_llhi": delta,
                        "delta_llhi_shuf_mean": mu,
                        "delta_llhi_shuf_std": std,
                        "delta_llhi_z": zscore(delta, mu, std),
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
                "delta_llhi_shuf_mean",
                "delta_llhi_shuf_std",
                "delta_llhi_z",
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

    return SessionResult(
        session=session,
        group=group,
        full_ll_gain_by_neuron=full_ll_gain_by_neuron,
        contrib_rllr_by_feature_by_neuron=contrib_rllr,
        shuf_mean_rllr_by_feature_by_neuron=shuf_mean_rllr,
        shuf_std_rllr_by_feature_by_neuron=shuf_std_rllr,
        full_llhi_by_neuron=full_llhi_by_neuron,
        contrib_delta_llhi_by_feature_by_neuron=contrib_delta_llhi,
        shuf_mean_delta_llhi_by_feature_by_neuron=shuf_mean_llhi,
        shuf_std_delta_llhi_by_feature_by_neuron=shuf_std_llhi,
        pyramidal_neurons=pyr_idx,
    )


def main():
    WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)

    set_imu = list_sessions_imu(IMU_ROOT)
    set_spk = list_sessions_spike(SPIKE_ROOT)
    set_dlc = list_sessions_dlc_final(DLC_ROOT)
    set_pos = list_sessions_position(POSITION_ROOT)
    sessions = sorted(list(set_imu & set_spk & set_dlc & set_pos))

    if not sessions:
        print("[FATAL] No sessions found with all required inputs.")
        return

    dayid2cellinfo = build_dayid_to_cellinfo()

    results: List[SessionResult] = []
    for s in sessions:
        try:
            r = compute_session_rllr(s, dayid2cellinfo, n_jobs=N_JOBS)
        except Exception as e:  # pylint: disable=broad-except
            print(f"[SKIP] {s}: exception {e}")
            r = None
        if r is not None:
            results.append(r)

    if not results:
        print("[FATAL] No sessions processed successfully.")
        return

    filtered_results = []
    if POSITIVE_CONTRIB_ONLY:
        for r in results:
            filtered = {
                feat: {ni: val for ni, val in contrib.items() if np.isfinite(val) and val > 0}
                for feat, contrib in r.contrib_rllr_by_feature_by_neuron.items()
            }
            filtered_llhi = {
                feat: {ni: val for ni, val in contrib.items() if np.isfinite(val) and val > 0}
                for feat, contrib in r.contrib_delta_llhi_by_feature_by_neuron.items()
            }
            filtered_results.append(
                SessionResult(
                    session=r.session,
                    group=r.group,
                    full_ll_gain_by_neuron=r.full_ll_gain_by_neuron,
                    contrib_rllr_by_feature_by_neuron=filtered,
                    shuf_mean_rllr_by_feature_by_neuron=r.shuf_mean_rllr_by_feature_by_neuron,
                    shuf_std_rllr_by_feature_by_neuron=r.shuf_std_rllr_by_feature_by_neuron,
                    full_llhi_by_neuron=r.full_llhi_by_neuron,
                    contrib_delta_llhi_by_feature_by_neuron=filtered_llhi,
                    shuf_mean_delta_llhi_by_feature_by_neuron=r.shuf_mean_delta_llhi_by_feature_by_neuron,
                    shuf_std_delta_llhi_by_feature_by_neuron=r.shuf_std_delta_llhi_by_feature_by_neuron,
                    pyramidal_neurons=r.pyramidal_neurons,
                )
            )
    else:
        filtered_results = results

    plot_features = VARS_ALL[:]
    if INCLUDE_HEAD_POSE and HEAD_POSE_FEATURE not in plot_features:
        plot_features.append(HEAD_POSE_FEATURE)

    rllr_stats = [
        DroponeSessionStats(
            session=r.session,
            group=r.group,
            full_ll_gain=r.full_ll_gain_by_neuron,
            frac_by_feature=r.contrib_rllr_by_feature_by_neuron,
            shuf_mean_by_feature=r.shuf_mean_rllr_by_feature_by_neuron,
            shuf_std_by_feature=r.shuf_std_rllr_by_feature_by_neuron,
            all_neuron_ids=r.pyramidal_neurons.tolist(),
        )
        for r in filtered_results
        if r.full_ll_gain_by_neuron and any(r.contrib_rllr_by_feature_by_neuron.values())
    ]
    if rllr_stats:
        dropone_plot_data = collect_dropone_plot_data(
            rllr_stats,
            features=plot_features,
            min_full_ll_gain=MIN_FULL_LL_GAIN,
            compute_delta=True,
            include_missing_cells=INCLUDE_UNFIT_CELLS,
            include_head_pose=INCLUDE_HEAD_POSE,
            include_paired_points=LINK_INDOOR_OUTDOOR_PAIRS,
            head_pose_components=HEAD_POSE_COMPONENTS,
        )
        summary_csv = plot_dropone_suite(
            WEIGHTS_BASE / "RLLR_SUMMARY",
            features=plot_features,
            plot_data=dropone_plot_data,
            min_full_ll_gain=MIN_FULL_LL_GAIN,
            seed=SEED,
            max_scatter_points=0,
            ylim_pad_frac=0.08,
            use_zscore=False,
            metric_tag="rllr",
            ylabel="rLLR",
            title_metric="drop-one contribution",
            summary_metric="dropone_rllr",
            include_delta_plot=True,
            use_shuffle_line=False,
            paired_points=dropone_plot_data.paired_points if LINK_INDOOR_OUTDOOR_PAIRS else None,
        )
        print(
            f"[OK] Drop-one boxplots (full LL gain ≥ {MIN_FULL_LL_GAIN:g}): "
            f"indoor {dropone_plot_data.kept_counts['indoor']}/{dropone_plot_data.total_counts['indoor']}, "
            f"outdoor {dropone_plot_data.kept_counts['outdoor']}/{dropone_plot_data.total_counts['outdoor']}",
        )
        print(f"[OK] Summary CSV: {summary_csv}")

        rllr_z_stats = []
        for r in filtered_results:
            z_by_feature: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
            for feat in VARS_ALL:
                for ni, val in r.contrib_rllr_by_feature_by_neuron.get(feat, {}).items():
                    mu = r.shuf_mean_rllr_by_feature_by_neuron.get(feat, {}).get(ni, np.nan)
                    std = r.shuf_std_rllr_by_feature_by_neuron.get(feat, {}).get(ni, np.nan)
                    if np.isfinite(val) and np.isfinite(mu) and np.isfinite(std) and std > MU_EPS:
                        z_by_feature[feat][ni] = float((val - mu) / std)
            if any(z_by_feature.values()):
                rllr_z_stats.append(
                    DroponeSessionStats(
                        session=r.session,
                        group=r.group,
                        full_ll_gain=r.full_ll_gain_by_neuron,
                        frac_by_feature=z_by_feature,
                        shuf_mean_by_feature=r.shuf_mean_rllr_by_feature_by_neuron,
                        shuf_std_by_feature=r.shuf_std_rllr_by_feature_by_neuron,
                        all_neuron_ids=r.pyramidal_neurons.tolist(),
                    )
                )

        if rllr_z_stats:
            rllr_z_plot_data = collect_dropone_plot_data(
                rllr_z_stats,
                features=plot_features,
                min_full_ll_gain=MIN_FULL_LL_GAIN,
                compute_delta=False,
                include_missing_cells=INCLUDE_UNFIT_CELLS,
                include_head_pose=INCLUDE_HEAD_POSE,
                include_paired_points=LINK_INDOOR_OUTDOOR_PAIRS,
                head_pose_components=HEAD_POSE_COMPONENTS,
            )
            rllr_z_summary_csv = plot_dropone_suite(
                WEIGHTS_BASE / "RLLR_SUMMARY",
                features=plot_features,
                plot_data=rllr_z_plot_data,
                min_full_ll_gain=MIN_FULL_LL_GAIN,
                seed=SEED,
                max_scatter_points=0,
                ylim_pad_frac=0.08,
                use_zscore=True,
                metric_tag="rllr_z",
                ylabel="rLLR",
                title_metric="drop-one contribution",
                summary_metric="dropone_rllr_z",
                include_delta_plot=False,
                use_shuffle_line=True,
                paired_points=rllr_z_plot_data.paired_points if LINK_INDOOR_OUTDOOR_PAIRS else None,
            )
            print(f"[OK] rLLR z-score Summary CSV: {rllr_z_summary_csv}")
    else:
        print("[WARN] No rLLR stats available for plotting.")

    llhi_stats = [
        DroponeSessionStats(
            session=r.session,
            group=r.group,
            full_ll_gain=r.full_llhi_by_neuron,
            frac_by_feature=r.contrib_delta_llhi_by_feature_by_neuron,
            shuf_mean_by_feature=r.shuf_mean_delta_llhi_by_feature_by_neuron,
            shuf_std_by_feature=r.shuf_std_delta_llhi_by_feature_by_neuron,
            all_neuron_ids=r.pyramidal_neurons.tolist(),
        )
        for r in filtered_results
        if r.full_llhi_by_neuron and any(r.contrib_delta_llhi_by_feature_by_neuron.values())
    ]
    if llhi_stats:
        dropone_llhi_data = collect_dropone_plot_data(
            llhi_stats,
            features=plot_features,
            min_full_ll_gain=MIN_FULL_LLHI,
            compute_delta=False,
            include_missing_cells=INCLUDE_UNFIT_CELLS,
            include_head_pose=INCLUDE_HEAD_POSE,
            include_paired_points=LINK_INDOOR_OUTDOOR_PAIRS,
            head_pose_components=HEAD_POSE_COMPONENTS,
        )
        llhi_summary_csv = plot_dropone_suite(
            WEIGHTS_BASE / "LLHI_SUMMARY",
            features=plot_features,
            plot_data=dropone_llhi_data,
            min_full_ll_gain=MIN_FULL_LLHI,
            seed=SEED,
            max_scatter_points=0,
            ylim_pad_frac=0.08,
            use_zscore=False,
            metric_tag="llhi",
            ylabel="ΔLLHI (bits/spike)",
            title_metric="drop-one absolute contribution",
            summary_metric="dropone_delta_llhi",
            include_delta_plot=False,
            use_shuffle_line=False,
            paired_points=dropone_llhi_data.paired_points if LINK_INDOOR_OUTDOOR_PAIRS else None,
        )
        print(
            f"[OK] Drop-one LLHI boxplots (full LLHI ≥ {MIN_FULL_LLHI:g}): "
            f"indoor {dropone_llhi_data.kept_counts['indoor']}/{dropone_llhi_data.total_counts['indoor']}, "
            f"outdoor {dropone_llhi_data.kept_counts['outdoor']}/{dropone_llhi_data.total_counts['outdoor']}",
        )
        print(f"[OK] LLHI Summary CSV: {llhi_summary_csv}")
    else:
        print("[WARN] No ΔLLHI stats available for plotting.")

    rllhi_stats = []
    for r in filtered_results:
        if not r.full_llhi_by_neuron or not any(r.contrib_delta_llhi_by_feature_by_neuron.values()):
            continue
        rllhi_by_feature: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
        for feat in VARS_ALL:
            for ni, delta_val in r.contrib_delta_llhi_by_feature_by_neuron.get(feat, {}).items():
                full_val = r.full_llhi_by_neuron.get(ni, np.nan)
                if not np.isfinite(full_val) or not np.isfinite(delta_val) or full_val == 0:
                    continue
                rllhi_by_feature[feat][ni] = float(delta_val) / float(full_val)
        if any(rllhi_by_feature.values()):
            rllhi_stats.append(
                DroponeSessionStats(
                    session=r.session,
                    group=r.group,
                    full_ll_gain=r.full_llhi_by_neuron,
                    frac_by_feature=rllhi_by_feature,
                    shuf_mean_by_feature={v: {} for v in VARS_ALL},
                    shuf_std_by_feature={v: {} for v in VARS_ALL},
                    all_neuron_ids=r.pyramidal_neurons.tolist(),
                )
            )

    if rllhi_stats:
        dropone_rllhi_data = collect_dropone_plot_data(
            rllhi_stats,
            features=plot_features,
            min_full_ll_gain=MIN_FULL_LLHI,
            compute_delta=False,
            include_missing_cells=INCLUDE_UNFIT_CELLS,
            include_head_pose=INCLUDE_HEAD_POSE,
            include_paired_points=LINK_INDOOR_OUTDOOR_PAIRS,
            head_pose_components=HEAD_POSE_COMPONENTS,
        )
        rllhi_summary_csv = plot_dropone_suite(
            WEIGHTS_BASE / "RLLHI_SUMMARY",
            features=plot_features,
            plot_data=dropone_rllhi_data,
            min_full_ll_gain=MIN_FULL_LLHI,
            seed=SEED,
            max_scatter_points=0,
            ylim_pad_frac=0.08,
            use_zscore=False,
            metric_tag="rllhi",
            ylabel="rLLHI (ΔLLHI / LLHI_full)",
            title_metric="drop-one relative contribution",
            summary_metric="dropone_rllhi",
            include_delta_plot=False,
            use_shuffle_line=False,
            paired_points=dropone_rllhi_data.paired_points if LINK_INDOOR_OUTDOOR_PAIRS else None,
        )
        print(
            f"[OK] Drop-one rLLHI boxplots (full LLHI ≥ {MIN_FULL_LLHI:g}): "
            f"indoor {dropone_rllhi_data.kept_counts['indoor']}/{dropone_rllhi_data.total_counts['indoor']}, "
            f"outdoor {dropone_rllhi_data.kept_counts['outdoor']}/{dropone_rllhi_data.total_counts['outdoor']}",
        )
        print(f"[OK] rLLHI Summary CSV: {rllhi_summary_csv}")

        rllhi_z_stats = []
        for r in filtered_results:
            rllhi_z_by_feature: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
            rllhi_shuf_mean: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
            rllhi_shuf_std: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
            for feat in VARS_ALL:
                for ni, delta_val in r.contrib_delta_llhi_by_feature_by_neuron.get(feat, {}).items():
                    full_val = r.full_llhi_by_neuron.get(ni, np.nan)
                    if not np.isfinite(full_val) or full_val == 0 or not np.isfinite(delta_val):
                        continue
                    rllhi_val = float(delta_val) / float(full_val)
                    mu_delta = r.shuf_mean_delta_llhi_by_feature_by_neuron.get(feat, {}).get(ni, np.nan)
                    std_delta = r.shuf_std_delta_llhi_by_feature_by_neuron.get(feat, {}).get(ni, np.nan)
                    if np.isfinite(mu_delta) and np.isfinite(std_delta):
                        mu_rllhi = float(mu_delta) / float(full_val)
                        std_rllhi = float(std_delta) / float(abs(full_val))
                        rllhi_shuf_mean[feat][ni] = mu_rllhi
                        rllhi_shuf_std[feat][ni] = std_rllhi
                        if std_rllhi > MU_EPS:
                            rllhi_z_by_feature[feat][ni] = float((rllhi_val - mu_rllhi) / std_rllhi)
            if any(rllhi_z_by_feature.values()):
                rllhi_z_stats.append(
                    DroponeSessionStats(
                        session=r.session,
                        group=r.group,
                        full_ll_gain=r.full_llhi_by_neuron,
                        frac_by_feature=rllhi_z_by_feature,
                        shuf_mean_by_feature=rllhi_shuf_mean,
                        shuf_std_by_feature=rllhi_shuf_std,
                        all_neuron_ids=r.pyramidal_neurons.tolist(),
                    )
                )

        if rllhi_z_stats:
            rllhi_z_plot_data = collect_dropone_plot_data(
                rllhi_z_stats,
                features=plot_features,
                min_full_ll_gain=MIN_FULL_LLHI,
                compute_delta=False,
                include_missing_cells=INCLUDE_UNFIT_CELLS,
                include_head_pose=INCLUDE_HEAD_POSE,
                include_paired_points=LINK_INDOOR_OUTDOOR_PAIRS,
                head_pose_components=HEAD_POSE_COMPONENTS,
            )
            rllhi_z_summary_csv = plot_dropone_suite(
                WEIGHTS_BASE / "RLLHI_SUMMARY",
                features=plot_features,
                plot_data=rllhi_z_plot_data,
                min_full_ll_gain=MIN_FULL_LLHI,
                seed=SEED,
                max_scatter_points=0,
                ylim_pad_frac=0.08,
                use_zscore=True,
                metric_tag="rllhi_z",
                ylabel="rLLHI (ΔLLHI / LLHI_full)",
                title_metric="drop-one contribution",
                summary_metric="dropone_rllhi_z",
                include_delta_plot=False,
                use_shuffle_line=True,
                paired_points=rllhi_z_plot_data.paired_points if LINK_INDOOR_OUTDOOR_PAIRS else None,
            )
            print(f"[OK] rLLHI z-score Summary CSV: {rllhi_z_summary_csv}")
    else:
        print("[WARN] No rLLHI stats available for plotting.")

    for group in ["indoor", "outdoor"]:
        group_res = [r for r in filtered_results if r.group == group and r.full_ll_gain_by_neuron]
        if not group_res:
            print(f"[WARN] No sessions for group={group}")
            continue

        sess_full: Dict[str, np.ndarray] = {}
        for r in group_res:
            arr = np.array(list(r.full_ll_gain_by_neuron.values()), dtype=np.float64)
            sess_full[r.session] = arr

        full_stat = hierarchical_bootstrap_mean(sess_full, n_boot=N_BOOT, ci_lo=CI_LO, ci_hi=CI_HI, seed=SEED)

        feature_stats: Dict[str, Tuple[float, float, float]] = {}
        for feat in plot_features:
            sess_feat: Dict[str, np.ndarray] = {}
            for r in group_res:
                if feat == HEAD_POSE_FEATURE:
                    head_pose_map = compute_head_pose_map(
                        r.contrib_rllr_by_feature_by_neuron,
                        components=HEAD_POSE_COMPONENTS,
                        include_missing=INCLUDE_UNFIT_CELLS,
                        neuron_ids=r.pyramidal_neurons.tolist(),
                    )
                    arr = np.array(list(head_pose_map.values()), dtype=np.float64)
                else:
                    arr = np.array(list(r.contrib_rllr_by_feature_by_neuron[feat].values()), dtype=np.float64)
                sess_feat[r.session] = arr
            feature_stats[feat] = hierarchical_bootstrap_mean(sess_feat, n_boot=N_BOOT, ci_lo=CI_LO, ci_hi=CI_HI, seed=SEED)

        out_png = WEIGHTS_BASE / f"RLLR_DROPONE_PYR_{group}.png"
        plot_summary_figure(
            out_png=out_png,
            title=f"Poisson GLM | Pyramidal only | {group} | mean ± {CI_LO}-{CI_HI}th (hier bootstrap)",
            full_stat=full_stat,
            feature_stats=feature_stats,
            full_ylabel="LL gain (full - intercept)",
            feature_ylabel="rLLR",
            features=plot_features,
        )
        print(f"[OK] Saved: {out_png}")

        rows = [
            {
                "group": group,
                "metric": "full_ll_gain",
                "feature": "FULL",
                "mean": full_stat[0],
                "ci_lo": full_stat[1],
                "ci_hi": full_stat[2],
            }
        ]
        for feat in plot_features:
            m, lo, hi = feature_stats[feat]
            rows.append({"group": group, "metric": "rllr", "feature": feat, "mean": m, "ci_lo": lo, "ci_hi": hi})
        pd.DataFrame(rows).to_csv(WEIGHTS_BASE / f"RLLR_DROPONE_PYR_{group}_summary.csv", index=False)

    for group in ["indoor", "outdoor"]:
        group_res = [r for r in filtered_results if r.group == group and r.full_llhi_by_neuron]
        if not group_res:
            print(f"[WARN] No sessions with LLHI stats for group={group}")
            continue

        sess_full: Dict[str, np.ndarray] = {}
        for r in group_res:
            arr = np.array(list(r.full_llhi_by_neuron.values()), dtype=np.float64)
            sess_full[r.session] = arr

        full_stat = hierarchical_bootstrap_mean(sess_full, n_boot=N_BOOT, ci_lo=CI_LO, ci_hi=CI_HI, seed=SEED)

        feature_stats = {}
        for feat in plot_features:
            sess_feat = {}
            for r in group_res:
                if feat == HEAD_POSE_FEATURE:
                    head_pose_map = compute_head_pose_map(
                        r.contrib_delta_llhi_by_feature_by_neuron,
                        components=HEAD_POSE_COMPONENTS,
                        include_missing=INCLUDE_UNFIT_CELLS,
                        neuron_ids=r.pyramidal_neurons.tolist(),
                    )
                    arr = np.array(list(head_pose_map.values()), dtype=np.float64)
                else:
                    arr = np.array(list(r.contrib_delta_llhi_by_feature_by_neuron[feat].values()), dtype=np.float64)
                sess_feat[r.session] = arr
            feature_stats[feat] = hierarchical_bootstrap_mean(sess_feat, n_boot=N_BOOT, ci_lo=CI_LO, ci_hi=CI_HI, seed=SEED)

        out_png = WEIGHTS_BASE / f"LLHI_DROPONE_PYR_{group}.png"
        plot_summary_figure(
            out_png=out_png,
            title=f"Poisson GLM | Pyramidal only | {group} | mean ± {CI_LO}-{CI_HI}th (hier bootstrap)",
            full_stat=full_stat,
            feature_stats=feature_stats,
            full_ylabel="LLHI (full - mean-rate)",
            feature_ylabel="ΔLLHI (bits/spike)",
            features=plot_features,
        )
        print(f"[OK] Saved: {out_png}")

        rows = [
            {
                "group": group,
                "metric": "full_llhi",
                "feature": "FULL",
                "mean": full_stat[0],
                "ci_lo": full_stat[1],
                "ci_hi": full_stat[2],
            }
        ]
        for feat in plot_features:
            m, lo, hi = feature_stats[feat]
            rows.append({"group": group, "metric": "delta_llhi", "feature": feat, "mean": m, "ci_lo": lo, "ci_hi": hi})
        pd.DataFrame(rows).to_csv(WEIGHTS_BASE / f"LLHI_DROPONE_PYR_{group}_summary.csv", index=False)

    print("[DONE]")


if __name__ == "__main__":
    main()
