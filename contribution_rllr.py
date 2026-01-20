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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from contribution_rllr_utils import (
    CI_HI,
    CI_LO,
    HEAD_POSE_COMPONENTS,
    HEAD_POSE_FEATURE,
    MU_EPS,
    N_BOOT,
    DroponeSessionStats,
    compute_head_pose_map,
    collect_dropone_plot_data,
    hierarchical_bootstrap_mean,
    plot_dropone_suite,
    plot_summary_figure,
    compute_session_rllr,
    list_required_sessions,
    SessionResult,
    build_dayid_to_cellinfo,
)
from glm_poisson_forward.config import (
    N_JOBS,
    SEED,
    VARS_ALL,
    WEIGHTS_BASE,
)

MIN_FULL_LL_GAIN = 0.0
MIN_FULL_LLHI = 0.0
POSITIVE_CONTRIB_ONLY = False
INCLUDE_UNFIT_CELLS = False
LINK_INDOOR_OUTDOOR_PAIRS = False
INCLUDE_HEAD_POSE = True
N_SHUFFLE = 200

def _filter_positive_results(results: List[SessionResult]) -> List[SessionResult]:
    if not POSITIVE_CONTRIB_ONLY:
        return results

    filtered_results = []
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
    return filtered_results


def _plot_features() -> List[str]:
    plot_features = VARS_ALL[:]
    if INCLUDE_HEAD_POSE and HEAD_POSE_FEATURE not in plot_features:
        plot_features.append(HEAD_POSE_FEATURE)
    return plot_features


def _build_rllr_stats(results: List[SessionResult]) -> List[DroponeSessionStats]:
    return [
        DroponeSessionStats(
            session=r.session,
            group=r.group,
            full_ll_gain=r.full_ll_gain_by_neuron,
            frac_by_feature=r.contrib_rllr_by_feature_by_neuron,
            shuf_mean_by_feature=r.shuf_mean_rllr_by_feature_by_neuron,
            shuf_std_by_feature=r.shuf_std_rllr_by_feature_by_neuron,
            all_neuron_ids=r.pyramidal_neurons.tolist(),
        )
        for r in results
        if r.full_ll_gain_by_neuron and any(r.contrib_rllr_by_feature_by_neuron.values())
    ]


def _plot_rllr_suite(results: List[SessionResult], plot_features: List[str]) -> None:
    rllr_stats = _build_rllr_stats(results)
    if not rllr_stats:
        print("[WARN] No rLLR stats available for plotting.")
        return

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
    for r in results:
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

    if not rllr_z_stats:
        return

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


def _plot_llhi_suite(results: List[SessionResult], plot_features: List[str]) -> None:
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
        for r in results
        if r.full_llhi_by_neuron and any(r.contrib_delta_llhi_by_feature_by_neuron.values())
    ]
    if not llhi_stats:
        print("[WARN] No ΔLLHI stats available for plotting.")
        return

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


def _plot_rllhi_suite(results: List[SessionResult], plot_features: List[str]) -> None:
    rllhi_stats = []
    for r in results:
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

    if not rllhi_stats:
        print("[WARN] No rLLHI stats available for plotting.")
        return

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
    for r in results:
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

    if not rllhi_z_stats:
        return

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


def _plot_group_summaries(
    results: List[SessionResult],
    plot_features: List[str],
    *,
    full_metric_name: str,
    feature_metric_name: str,
    full_ylabel: str,
    feature_ylabel: str,
    output_prefix: str,
    full_metric_attr: str,
    feature_metric_attr: str,
) -> None:
    for group in ["indoor", "outdoor"]:
        group_res = [r for r in results if r.group == group and getattr(r, full_metric_attr)]
        if not group_res:
            print(f"[WARN] No sessions for group={group}")
            continue

        sess_full: Dict[str, np.ndarray] = {}
        for r in group_res:
            arr = np.array(list(getattr(r, full_metric_attr).values()), dtype=np.float64)
            sess_full[r.session] = arr

        full_stat = hierarchical_bootstrap_mean(sess_full, n_boot=N_BOOT, ci_lo=CI_LO, ci_hi=CI_HI, seed=SEED)

        feature_stats: Dict[str, Tuple[float, float, float]] = {}
        for feat in plot_features:
            sess_feat: Dict[str, np.ndarray] = {}
            for r in group_res:
                if feat == HEAD_POSE_FEATURE:
                    head_pose_map = compute_head_pose_map(
                        getattr(r, feature_metric_attr),
                        components=HEAD_POSE_COMPONENTS,
                        include_missing=INCLUDE_UNFIT_CELLS,
                        neuron_ids=r.pyramidal_neurons.tolist(),
                    )
                    arr = np.array(list(head_pose_map.values()), dtype=np.float64)
                else:
                    arr = np.array(list(getattr(r, feature_metric_attr)[feat].values()), dtype=np.float64)
                sess_feat[r.session] = arr
            feature_stats[feat] = hierarchical_bootstrap_mean(
                sess_feat,
                n_boot=N_BOOT,
                ci_lo=CI_LO,
                ci_hi=CI_HI,
                seed=SEED,
            )

        out_png = WEIGHTS_BASE / f"{output_prefix}_{group}.png"
        plot_summary_figure(
            out_png=out_png,
            title=f"Poisson GLM | Pyramidal only | {group} | mean ± {CI_LO}-{CI_HI}th (hier bootstrap)",
            full_stat=full_stat,
            feature_stats=feature_stats,
            full_ylabel=full_ylabel,
            feature_ylabel=feature_ylabel,
            features=plot_features,
        )
        print(f"[OK] Saved: {out_png}")

        rows = [
            {
                "group": group,
                "metric": full_metric_name,
                "feature": "FULL",
                "mean": full_stat[0],
                "ci_lo": full_stat[1],
                "ci_hi": full_stat[2],
            }
        ]
        for feat in plot_features:
            m, lo, hi = feature_stats[feat]
            rows.append({"group": group, "metric": feature_metric_name, "feature": feat, "mean": m, "ci_lo": lo, "ci_hi": hi})
        pd.DataFrame(rows).to_csv(WEIGHTS_BASE / f"{output_prefix}_{group}_summary.csv", index=False)


def main():
    WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)

    sessions = list_required_sessions()
    if not sessions:
        print("[FATAL] No sessions found with all required inputs.")
        return

    dayid2cellinfo = build_dayid_to_cellinfo()

    results: List[SessionResult] = []
    for s in sessions:
        try:
            r = compute_session_rllr(s, dayid2cellinfo, n_jobs=N_JOBS, n_shuffle=N_SHUFFLE)
        except Exception as e:  # pylint: disable=broad-except
            print(f"[SKIP] {s}: exception {e}")
            r = None
        if r is not None:
            results.append(r)

    if not results:
        print("[FATAL] No sessions processed successfully.")
        return

    filtered_results = _filter_positive_results(results)
    plot_features = _plot_features()

    _plot_rllr_suite(filtered_results, plot_features)
    _plot_llhi_suite(filtered_results, plot_features)
    _plot_rllhi_suite(filtered_results, plot_features)

    _plot_group_summaries(
        filtered_results,
        plot_features,
        full_metric_name="full_ll_gain",
        feature_metric_name="rllr",
        full_ylabel="LL gain (full - intercept)",
        feature_ylabel="rLLR",
        output_prefix="RLLR_DROPONE_PYR",
        full_metric_attr="full_ll_gain_by_neuron",
        feature_metric_attr="contrib_rllr_by_feature_by_neuron",
    )
    _plot_group_summaries(
        filtered_results,
        plot_features,
        full_metric_name="full_llhi",
        feature_metric_name="delta_llhi",
        full_ylabel="LLHI (full - mean-rate)",
        feature_ylabel="ΔLLHI (bits/spike)",
        output_prefix="LLHI_DROPONE_PYR",
        full_metric_attr="full_llhi_by_neuron",
        feature_metric_attr="contrib_delta_llhi_by_feature_by_neuron",
    )

    print("[DONE]")


if __name__ == "__main__":
    main()
