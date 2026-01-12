#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute Poisson-GLM drop-one contributions using rLLR normalization with forward-selected
(full) models. Full model is the forward-search-selected subset per neuron; drop-one
models remove a single covariate from that subset. Contributions are optionally z-scored
against a label-shuffle baseline.
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
    MU_EPS,
    N_BOOT,
    RLLR_FITS_DIRNAME,
    RLLR_STATS_DIRNAME,
    DroponeSessionStats,
    build_oof_intercept_mu,
    collect_dropone_plot_data,
    hierarchical_bootstrap_mean,
    load_forward_selected_models,
    plot_dropone_suite,
    plot_summary_figure,
    poisson_loglik,
)
from contribution_utils import (
    build_dayid_to_cellinfo,
    pyramidal_indices_for_session,
    predict_oof_from_saved_weights,
    save_weights_for_model,
)
from contribution_utils.weights import model_key_from_vars
from glm_poisson_forward.config import (
    CV_FOLDS,
    DLC_ROOT,
    IMU_ROOT,
    MAX_MISMATCH_FRAMES_50HZ,
    N_JOBS,
    POSITION_ROOT,
    SEED,
    SPIKE_ROOT,
    VARS_ALL,
    WEIGHTS_BASE,
)
from glm_poisson_forward.design_matrix import build_design_matrix
from glm_poisson_forward.io_utils import (
    list_sessions_dlc_final,
    list_sessions_imu,
    list_sessions_position,
    list_sessions_spike,
    load_spikes_50hz_counts,
    rebuild_inputs_50hz,
    session_paths,
)

MIN_FULL_LL_GAIN = 0.0
N_SHUFFLE = 100
POSITIVE_CONTRIB_ONLY = False


@dataclass
class SessionResult:
    session: str
    group: str
    full_ll_gain_by_neuron: Dict[int, float]
    contrib_rllr_by_feature_by_neuron: Dict[str, Dict[int, float]]


def compute_session_rllr(
    session: str,
    dayid2cellinfo: Dict[str, Path],
    *,
    n_jobs: int = N_JOBS,
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
    for k in ["position", "head_v_bin", "roll_bin", "yaw_bin", "pitch_bin"]:
        data_dict[k] = data_dict[k][:T]
    Y_all = Y50[:T].astype(np.float64)

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

    kf = KFold(n_splits=CV_FOLDS, shuffle=False)
    folds_idx = list(kf.split(np.arange(T)))

    X_cache: Dict[str, Tuple[np.ndarray, List[str], str]] = {}

    def get_X(model_vars: List[str]):
        mk = model_key_from_vars(model_vars)
        if mk in X_cache:
            X, feats, _ = X_cache[mk]
            return X, feats, mk
        X, feats = build_design_matrix(model_vars, data_dict)
        X_cache[mk] = (X, feats, mk)
        return X, feats, mk

    model_neurons: Dict[str, Dict[str, object]] = {}

    for ni, full_vars in pyr_models.items():
        if not full_vars:
            continue
        _, _, mk_full = get_X(full_vars)
        model_neurons.setdefault(mk_full, {"vars": full_vars, "neurons": set()})["neurons"].add(ni)
        for v in full_vars:
            drop_vars = [x for x in full_vars if x != v]
            if not drop_vars:
                continue
            _, _, mk_drop = get_X(drop_vars)
            model_neurons.setdefault(mk_drop, {"vars": drop_vars, "neurons": set()})["neurons"].add(ni)

    fits_root = sess_dir / RLLR_FITS_DIRNAME
    fits_root.mkdir(parents=True, exist_ok=True)

    for mk, meta in model_neurons.items():
        model_vars = meta["vars"]
        neuron_indices = np.array(sorted(meta["neurons"]), dtype=int)
        X, feats, _ = get_X(model_vars)
        model_dir = fits_root / mk
        save_weights_for_model(
            model_dir=model_dir,
            feature_names=feats,
            X_all=X,
            Y_all=Y_all,
            folds_idx=folds_idx,
            neuron_indices=neuron_indices,
            n_jobs=n_jobs,
            folds_count=len(folds_idx),
        )

    stats_root = sess_dir / RLLR_STATS_DIRNAME
    stats_root.mkdir(parents=True, exist_ok=True)

    full_csv = stats_root / "full_rllr_pyr.csv"
    contrib_csv = stats_root / "dropone_rllr_pyr.csv"

    if full_csv.exists() and contrib_csv.exists():
        df_full = pd.read_csv(full_csv)
        df_con = pd.read_csv(contrib_csv)
        has_shuffle = ("ll_gain_shuf_mean" in df_full.columns) and ("rllr_z" in df_con.columns)
        if has_shuffle:
            full_map = {int(r["neuron_idx"]): float(r["ll_gain"]) for _, r in df_full.iterrows()}
            contrib: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
            for _, r in df_con.iterrows():
                contrib[str(r["feature"])][int(r["neuron_idx"])] = float(r["rllr_z"])
            return SessionResult(
                session=session,
                group=group,
                full_ll_gain_by_neuron=full_map,
                contrib_rllr_by_feature_by_neuron=contrib,
            )

    full_ll_gain_by_neuron: Dict[int, float] = {}
    ll_full_by_neuron: Dict[int, float] = {}
    ll0_by_neuron: Dict[int, float] = {}
    full_shuffle_stats: Dict[int, Tuple[float, float, float]] = {}

    contrib: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
    contrib_shuffle_stats: Dict[str, Dict[int, Tuple[float, float, float]]] = {v: {} for v in VARS_ALL}

    rng = np.random.default_rng(SEED)

    for ni, full_vars in pyr_models.items():
        y = Y_all[:, ni].astype(np.float64)
        mu0_oof = build_oof_intercept_mu(y, folds_idx)
        ll0 = poisson_loglik(y, mu0_oof)
        ll0_by_neuron[ni] = float(ll0)

        X_full, feats_full, mk_full = get_X(full_vars)
        model_dir_full = fits_root / mk_full
        mu_oof_full = predict_oof_from_saved_weights(model_dir_full, X_full, feats_full, folds_idx, ni)
        ll_full = poisson_loglik(y, mu_oof_full)
        ll_full_by_neuron[ni] = float(ll_full)

        ll_gain = float(ll_full - ll0)
        full_ll_gain_by_neuron[ni] = ll_gain

        mu_oof_red_by_feat: Dict[str, np.ndarray] = {}
        for v in full_vars:
            drop_vars = [x for x in full_vars if x != v]
            if not drop_vars:
                continue
            X_red, feats_red, mk_red = get_X(drop_vars)
            model_dir_red = fits_root / mk_red
            mu_red = predict_oof_from_saved_weights(model_dir_red, X_red, feats_red, folds_idx, ni)
            mu_oof_red_by_feat[v] = mu_red

        denom = ll_full - ll0
        for v in full_vars:
            if denom <= 0 or not np.isfinite(denom):
                contrib[v][ni] = float("nan")
                continue
            if v in mu_oof_red_by_feat:
                ll_red = poisson_loglik(y, mu_oof_red_by_feat[v])
            else:
                ll_red = ll0
            contrib[v][ni] = float((ll_full - ll_red) / denom)

        shuf_full = np.full(N_SHUFFLE, np.nan, dtype=np.float64)
        shuf_frac: Dict[str, np.ndarray] = {v: np.full(N_SHUFFLE, np.nan, dtype=np.float64) for v in VARS_ALL}

        for s in range(N_SHUFFLE):
            y_shuf = rng.permutation(y)
            ll_full_shuf = poisson_loglik(y_shuf, mu_oof_full)
            ll0_shuf = poisson_loglik(y_shuf, mu0_oof)
            shuf_full[s] = ll_full_shuf - ll0_shuf

            denom_shuf = ll_full_shuf - ll0_shuf
            for v in full_vars:
                if v in mu_oof_red_by_feat:
                    ll_red_shuf = poisson_loglik(y_shuf, mu_oof_red_by_feat[v])
                else:
                    ll_red_shuf = ll0_shuf
                if not np.isfinite(denom_shuf) or denom_shuf <= 0:
                    shuf_frac[v][s] = float("nan")
                else:
                    shuf_frac[v][s] = float((ll_full_shuf - ll_red_shuf) / denom_shuf)

        full_mu = float(np.nanmean(shuf_full))
        full_std = float(np.nanstd(shuf_full, ddof=1))
        full_z = float("nan") if (not np.isfinite(full_std) or full_std <= 0) else float((ll_gain - full_mu) / full_std)
        full_shuffle_stats[ni] = (full_mu, full_std, full_z)

        for v in full_vars:
            arr = shuf_frac[v]
            mu = float(np.nanmean(arr))
            std = float(np.nanstd(arr, ddof=1))
            real = contrib[v][ni]
            z = float("nan") if (not np.isfinite(std) or std <= 0 or not np.isfinite(real)) else float((real - mu) / std)
            contrib_shuffle_stats[v][ni] = (mu, std, z)

    df_full = pd.DataFrame(
        [
            {
                "session": session,
                "group": group,
                "neuron_idx": ni,
                "ll_full": ll_full_by_neuron[ni],
                "ll0": ll0_by_neuron[ni],
                "ll_gain": full_ll_gain_by_neuron[ni],
                "ll_gain_shuf_mean": full_shuffle_stats[ni][0],
                "ll_gain_shuf_std": full_shuffle_stats[ni][1],
                "ll_gain_z": full_shuffle_stats[ni][2],
            }
            for ni in sorted(full_ll_gain_by_neuron.keys())
        ]
    )
    df_full.to_csv(full_csv, index=False)

    rows = []
    for v in VARS_ALL:
        for ni, frac in contrib[v].items():
            mu, std, z = contrib_shuffle_stats[v].get(ni, (float("nan"), float("nan"), float("nan")))
            rows.append(
                {
                    "session": session,
                    "group": group,
                    "feature": v,
                    "neuron_idx": ni,
                    "rllr": frac,
                    "rllr_shuf_mean": mu,
                    "rllr_shuf_std": std,
                    "rllr_z": z,
                }
            )
    pd.DataFrame(rows).to_csv(contrib_csv, index=False)

    contrib_z: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
    for v in VARS_ALL:
        for ni in contrib[v]:
            contrib_z[v][ni] = contrib_shuffle_stats[v].get(ni, (float("nan"), float("nan"), float("nan")))[2]

    return SessionResult(
        session=session,
        group=group,
        full_ll_gain_by_neuron=full_ll_gain_by_neuron,
        contrib_rllr_by_feature_by_neuron=contrib_z,
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
            filtered_results.append(
                SessionResult(
                    session=r.session,
                    group=r.group,
                    full_ll_gain_by_neuron=r.full_ll_gain_by_neuron,
                    contrib_rllr_by_feature_by_neuron=filtered,
                )
            )
    else:
        filtered_results = results

    plot_stats = [
        DroponeSessionStats(
            session=r.session,
            group=r.group,
            full_ll_gain=r.full_ll_gain_by_neuron,
            frac_by_feature=r.contrib_rllr_by_feature_by_neuron,
            shuf_mean_by_feature={v: {} for v in VARS_ALL},
            shuf_std_by_feature={v: {} for v in VARS_ALL},
        )
        for r in filtered_results
    ]
    dropone_plot_data = collect_dropone_plot_data(
        plot_stats,
        features=VARS_ALL,
        min_full_ll_gain=MIN_FULL_LL_GAIN,
        compute_delta=False,
    )
    summary_csv = plot_dropone_suite(
        WEIGHTS_BASE / "RLLR_SUMMARY",
        features=VARS_ALL,
        plot_data=dropone_plot_data,
        min_full_ll_gain=MIN_FULL_LL_GAIN,
        seed=SEED,
        max_scatter_points=0,
        ylim_pad_frac=0.08,
        use_zscore=True,
    )
    print(
        f"[OK] Drop-one boxplots (full LL gain ≥ {MIN_FULL_LL_GAIN:g}): "
        f"indoor {dropone_plot_data.kept_counts['indoor']}/{dropone_plot_data.total_counts['indoor']}, "
        f"outdoor {dropone_plot_data.kept_counts['outdoor']}/{dropone_plot_data.total_counts['outdoor']}",
    )
    print(f"[OK] Summary CSV: {summary_csv}")

    for group in ["indoor", "outdoor"]:
        group_res = [r for r in filtered_results if r.group == group]
        if not group_res:
            print(f"[WARN] No sessions for group={group}")
            continue

        sess_full: Dict[str, np.ndarray] = {}
        for r in group_res:
            arr = np.array(list(r.full_ll_gain_by_neuron.values()), dtype=np.float64)
            sess_full[r.session] = arr

        full_stat = hierarchical_bootstrap_mean(sess_full, n_boot=N_BOOT, ci_lo=CI_LO, ci_hi=CI_HI, seed=SEED)

        feature_stats: Dict[str, Tuple[float, float, float]] = {}
        for feat in VARS_ALL:
            sess_feat: Dict[str, np.ndarray] = {}
            for r in group_res:
                arr = np.array(list(r.contrib_rllr_by_feature_by_neuron[feat].values()), dtype=np.float64)
                sess_feat[r.session] = arr
            feature_stats[feat] = hierarchical_bootstrap_mean(sess_feat, n_boot=N_BOOT, ci_lo=CI_LO, ci_hi=CI_HI, seed=SEED)

        out_png = WEIGHTS_BASE / f"RLLR_DROPONE_PYR_{group}.png"
        plot_summary_figure(
            out_png=out_png,
            title=f"Poisson GLM | Pyramidal only | {group} | mean ± {CI_LO}-{CI_HI}th (hier bootstrap)",
            full_stat=full_stat,
            feature_stats=feature_stats,
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
        for feat in VARS_ALL:
            m, lo, hi = feature_stats[feat]
            rows.append({"group": group, "metric": "rllr", "feature": feat, "mean": m, "ci_lo": lo, "ci_hi": hi})
        pd.DataFrame(rows).to_csv(WEIGHTS_BASE / f"RLLR_DROPONE_PYR_{group}_summary.csv", index=False)

    print("[DONE]")


if __name__ == "__main__":
    main()
