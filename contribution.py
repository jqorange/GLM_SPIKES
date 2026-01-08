#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute Poisson-GLM Deviance Explained (DevExpl) and drop-one (reduced-model) contributions
for PYRAMIDAL cells only, separately for indoor vs outdoor sessions. Contributions are
z-scored against a label-shuffle baseline.

The heavy lifting (cell-metrics parsing, weight caching, statistics, plotting) lives in
``contribution_utils`` so this file can stay focused on the high-level pipeline:

1) For each session:
   - load 50 Hz covariates and spike counts
   - identify pyramidal cells via cell_metrics.putativeCellType
   - fit/reuse Poisson GLM weights for the full model and each drop-one variant
   - compute neuron-level DevExpl and per-feature contribution fractions
   - shuffle labels to estimate null distributions and z-score contributions
2) Aggregate across sessions with hierarchical bootstraps (indoor vs outdoor)
3) Save summary plots and CSVs, and write the drop-one boxplots (same logic as plot_contribution.py)
   for neurons with full DevExpl >= MIN_FULL_DEVEXPL.
"""

from __future__ import annotations

import faulthandler

faulthandler.enable()
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.special import gammaln

from contribution_utils import (
    CI_HI,
    CI_LO,
    DROPONE_FITS_DIRNAME,
    DROPONE_STATS_DIRNAME,
    DroponeSessionStats,
    MU_EPS,
    N_BOOT,
    build_dayid_to_cellinfo,
    collect_dropone_plot_data,
    devexpl_from_deviances,
    deviance_from_ll,
    hierarchical_bootstrap_mean,
    model_key_from_vars,
    plot_dropone_suite,
    plot_summary_figure,
    poisson_loglik,
    poisson_loglik_saturated,
    predict_oof_from_saved_weights,
    pyramidal_indices_for_session,
    save_weights_for_model,
)
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


MIN_FULL_DEVEXPL = 0.1
N_SHUFFLE = 100
POSITIVE_CONTRIB_ONLY = False  # if True, only contributions > 0 are kept per feature


@dataclass
class SessionResult:
    session: str
    group: str  # indoor/outdoor
    full_devexpl_by_neuron: Dict[int, float]  # neuron_idx -> DevExpl
    contrib_frac_by_feature_by_neuron: Dict[str, Dict[int, float]]  # feature -> neuron_idx -> frac


def compute_session_dropone(
    session: str,
    dayid2cellinfo: Dict[str, Path],
    *,
    n_jobs: int = N_JOBS,
) -> Optional[SessionResult]:

    sess_dir = WEIGHTS_BASE / session
    if not sess_dir.exists():
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
    Y50 = load_spikes_50hz_counts(paths["spike"])  # (T50_spk, N)
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

    model_vars_list: List[List[str]] = [VARS_ALL]
    for v in VARS_ALL:
        model_vars_list.append([x for x in VARS_ALL if x != v])

    fits_root = sess_dir / DROPONE_FITS_DIRNAME
    fits_root.mkdir(parents=True, exist_ok=True)

    for mv in model_vars_list:
        X, feats, mk = get_X(mv)
        model_dir = fits_root / mk
        save_weights_for_model(
            model_dir=model_dir,
            feature_names=feats,
            X_all=X,
            Y_all=Y_all,
            folds_idx=folds_idx,
            neuron_indices=pyr_idx,
            n_jobs=n_jobs,
            folds_count=len(folds_idx),
        )

    stats_root = sess_dir / DROPONE_STATS_DIRNAME
    stats_root.mkdir(parents=True, exist_ok=True)

    full_csv = stats_root / "full_devexpl_pyr.csv"
    contrib_csv = stats_root / "dropone_contrib_pyr.csv"

    if full_csv.exists() and contrib_csv.exists():
        df_full = pd.read_csv(full_csv)
        df_con = pd.read_csv(contrib_csv)
        has_shuffle = ("devexpl_shuf_mean" in df_full.columns) and ("frac_z" in df_con.columns)
        if has_shuffle:
            full_map = {int(r["neuron_idx"]): float(r["devexpl_full"]) for _, r in df_full.iterrows()}
            contrib: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
            for _, r in df_con.iterrows():
                contrib[str(r["feature"])][int(r["neuron_idx"])] = float(r["frac_z"])
            return SessionResult(session=session, group=group, full_devexpl_by_neuron=full_map, contrib_frac_by_feature_by_neuron=contrib)

    X_full, feats_full, mk_full = get_X(VARS_ALL)
    model_dir_full = fits_root / mk_full

    red_models: Dict[str, Tuple[np.ndarray, List[str], Path]] = {}
    for v in VARS_ALL:
        mv = [x for x in VARS_ALL if x != v]
        X_red, feats_red, mk_red = get_X(mv)
        red_models[v] = (X_red, feats_red, fits_root / mk_red)

    full_devexpl_by_neuron: Dict[int, float] = {}
    D_full_by_neuron: Dict[int, float] = {}
    D_null_by_neuron: Dict[int, float] = {}
    full_shuffle_stats: Dict[int, Tuple[float, float, float]] = {}

    contrib: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
    contrib_shuffle_stats: Dict[str, Dict[int, Tuple[float, float, float]]] = {v: {} for v in VARS_ALL}

    rng = np.random.default_rng(SEED)

    for ni in pyr_idx.tolist():
        y = Y_all[:, ni].astype(np.float64)
        mu_oof_full = predict_oof_from_saved_weights(model_dir_full, X_full, feats_full, folds_idx, ni)

        log_factorial_sum = float(np.sum(gammaln(y + 1.0)))
        ll_sat = poisson_loglik_saturated(y)
        log_mu_full = np.log(np.clip(mu_oof_full, MU_EPS, None))
        sum_mu_full = float(np.sum(mu_oof_full))
        ll_full = float(np.dot(y, log_mu_full) - sum_mu_full - log_factorial_sum)
        mu0 = np.full_like(y, fill_value=max(float(np.mean(y)), MU_EPS), dtype=np.float64)
        log_mu0 = np.log(np.clip(mu0, MU_EPS, None))
        sum_mu0 = float(np.sum(mu0))
        ll_null = float(np.dot(y, log_mu0) - sum_mu0 - log_factorial_sum)

        D_full = deviance_from_ll(ll_sat, ll_full)
        D_null = deviance_from_ll(ll_sat, ll_null)

        D_full_by_neuron[ni] = float(D_full)
        D_null_by_neuron[ni] = float(D_null)
        full_devexpl_by_neuron[ni] = devexpl_from_deviances(D_full, D_null)

        mu_oof_red_by_feat: Dict[str, np.ndarray] = {}
        log_mu_red_by_feat: Dict[str, np.ndarray] = {}
        sum_mu_red_by_feat: Dict[str, float] = {}
        for v, (X_red, feats_red, model_dir_red) in red_models.items():
            mu_red = predict_oof_from_saved_weights(model_dir_red, X_red, feats_red, folds_idx, ni)
            mu_oof_red_by_feat[v] = mu_red
            log_mu_red_by_feat[v] = np.log(np.clip(mu_red, MU_EPS, None))
            sum_mu_red_by_feat[v] = float(np.sum(mu_red))

        for v, mu_oof_red in mu_oof_red_by_feat.items():
            ll_red = float(np.dot(y, log_mu_red_by_feat[v]) - sum_mu_red_by_feat[v] - log_factorial_sum)
            D_red = deviance_from_ll(ll_sat, ll_red)

            denom = (D_null - D_full)
            frac = float((D_red - D_full) / denom)
            contrib[v][ni] = frac

        shuf_full = np.full(N_SHUFFLE, np.nan, dtype=np.float64)
        shuf_frac: Dict[str, np.ndarray] = {v: np.full(N_SHUFFLE, np.nan, dtype=np.float64) for v in VARS_ALL}

        for s in range(N_SHUFFLE):
            y_shuf = rng.permutation(y)
            ll_full_shuf = float(np.dot(y_shuf, log_mu_full) - sum_mu_full - log_factorial_sum)
            D_full_shuf = deviance_from_ll(ll_sat, ll_full_shuf)
            shuf_full[s] = devexpl_from_deviances(D_full_shuf, D_null)

            denom = D_null - D_full_shuf
            for v in VARS_ALL:
                ll_red_shuf = float(np.dot(y_shuf, log_mu_red_by_feat[v]) - sum_mu_red_by_feat[v] - log_factorial_sum)
                D_red_shuf = deviance_from_ll(ll_sat, ll_red_shuf)
                if not np.isfinite(D_red_shuf) or not np.isfinite(denom) :
                    shuf_frac[v][s] = float("nan")
                else:
                    shuf_frac[v][s] = float((D_red_shuf - D_full_shuf) / denom)

        full_mu = float(np.nanmean(shuf_full))
        full_std = float(np.nanstd(shuf_full, ddof=1))
        full_z = float("nan") if (not np.isfinite(full_std) or full_std <= 0) else float((full_devexpl_by_neuron[ni] - full_mu) / full_std)
        full_shuffle_stats[ni] = (full_mu, full_std, full_z)

        for v in VARS_ALL:
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
                "devexpl_full": full_devexpl_by_neuron[ni],
                "devexpl_shuf_mean": full_shuffle_stats[ni][0],
                "devexpl_shuf_std": full_shuffle_stats[ni][1],
                "devexpl_z": full_shuffle_stats[ni][2],
            }
            for ni in sorted(full_devexpl_by_neuron.keys())
        ]
    )
    df_full.to_csv(full_csv, index=False)

    rows = []
    for v in VARS_ALL:
        for ni, frac in contrib[v].items():
            mu, std, z = contrib_shuffle_stats[v][ni]
            rows.append(
                {
                    "session": session,
                    "group": group,
                    "feature": v,
                    "neuron_idx": ni,
                    "frac_full_dev": frac,
                    "frac_shuf_mean": mu,
                    "frac_shuf_std": std,
                    "frac_z": z,
                }
            )
    pd.DataFrame(rows).to_csv(contrib_csv, index=False)

    contrib_z: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
    for v in VARS_ALL:
        for ni in contrib[v]:
            contrib_z[v][ni] = contrib_shuffle_stats[v][ni][2]

    return SessionResult(session=session, group=group, full_devexpl_by_neuron=full_devexpl_by_neuron, contrib_frac_by_feature_by_neuron=contrib_z)


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
            r = compute_session_dropone(s, dayid2cellinfo, n_jobs=N_JOBS)
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
                for feat, contrib in r.contrib_frac_by_feature_by_neuron.items()
            }
            filtered_results.append(
                SessionResult(
                    session=r.session,
                    group=r.group,
                    full_devexpl_by_neuron=r.full_devexpl_by_neuron,
                    contrib_frac_by_feature_by_neuron=filtered,
                )
            )
    else:
        filtered_results = results

    plot_stats = [
        DroponeSessionStats(
            session=r.session,
            group=r.group,
            full_devexpl=r.full_devexpl_by_neuron,
            frac_by_feature=r.contrib_frac_by_feature_by_neuron,
            shuf_mean_by_feature={v: {} for v in VARS_ALL},
            shuf_std_by_feature={v: {} for v in VARS_ALL},
        )
        for r in filtered_results
    ]
    dropone_plot_data = collect_dropone_plot_data(
        plot_stats,
        features=VARS_ALL,
        min_full_devexpl=MIN_FULL_DEVEXPL,
        compute_delta=False,
    )
    summary_csv = plot_dropone_suite(
        WEIGHTS_BASE / "DROPONE_SUMMARY",
        features=VARS_ALL,
        plot_data=dropone_plot_data,
        min_full_devexpl=MIN_FULL_DEVEXPL,
        seed=SEED,
        max_scatter_points=0,
        ylim_pad_frac=0.08,
        use_zscore=True,
    )
    print(
        f"[OK] Drop-one boxplots (full DevExpl ≥ {MIN_FULL_DEVEXPL:g}): "
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
            arr = np.array(list(r.full_devexpl_by_neuron.values()), dtype=np.float64)
            sess_full[r.session] = arr

        full_stat = hierarchical_bootstrap_mean(sess_full, n_boot=N_BOOT, ci_lo=CI_LO, ci_hi=CI_HI, seed=SEED)

        feature_stats: Dict[str, Tuple[float, float, float]] = {}
        for feat in VARS_ALL:
            sess_feat: Dict[str, np.ndarray] = {}
            for r in group_res:
                arr = np.array(list(r.contrib_frac_by_feature_by_neuron[feat].values()), dtype=np.float64)
                sess_feat[r.session] = arr
            feature_stats[feat] = hierarchical_bootstrap_mean(sess_feat, n_boot=N_BOOT, ci_lo=CI_LO, ci_hi=CI_HI, seed=SEED)

        out_png = WEIGHTS_BASE / f"DEVEXPL_DROPONE_PYR_{group}.png"
        plot_summary_figure(
            out_png=out_png,
            title=f"Poisson GLM | Pyramidal only | {group} | mean ± {CI_LO}-{CI_HI}th (hier bootstrap)",
            full_stat=full_stat,
            feature_stats=feature_stats,
        )
        print(f"[OK] Saved: {out_png}")

        rows = [{"group": group, "metric": "full_devexpl", "feature": "FULL", "mean": full_stat[0], "ci_lo": full_stat[1], "ci_hi": full_stat[2]}]
        for feat in VARS_ALL:
            m, lo, hi = feature_stats[feat]
            rows.append({"group": group, "metric": "frac_full_dev", "feature": feat, "mean": m, "ci_lo": lo, "ci_hi": hi})
        pd.DataFrame(rows).to_csv(WEIGHTS_BASE / f"DEVEXPL_DROPONE_PYR_{group}_summary.csv", index=False)

    print("[DONE]")


if __name__ == "__main__":
    main()
