#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drop-one contribution plotting (pyramidal-only) with:
  - indoor vs outdoor combined plot (per-feature, side-by-side boxplots + jitter)
  - PLUS indoor-only and outdoor-only plots (per-feature), where feature order is
    sorted by group mean (high -> low)
  - Feature selection via --features (comma-separated). Features not listed are ignored.
  - Optional: use --forward_modulated_only to limit each feature to neurons whose forward-search
    final model includes that feature (still pyramidal only).
  - Filter: only neurons with full LL gain/LLHI >= threshold enter downstream analysis
  - Use --metric rllr to plot contribution_rllr outputs (uses RLLR_STATS + full LL gain filter).
  - Use --metric llhi or rllhi to plot contribution_rllr LLHI outputs (full_llhi_pyr.csv/dropone_llhi_pyr.csv).
  - By default, uses shuffle-normalized z-scores for rllr/rllhi when available (use --use_raw for fractions)
  - No bootstrap CI bars. Use boxplot whiskers/caps; y-lims auto from whiskers/caps.

Inputs per session:
  - metric=rllr:
      <WEIGHTS_BASE>/<session>/RLLR_STATS/
          full_rllr_pyr.csv
          dropone_rllr_pyr.csv  columns include: feature, neuron_idx, rllr
  - metric=llhi or rllhi:
      <WEIGHTS_BASE>/<session>/RLLR_STATS/
          full_llhi_pyr.csv
          dropone_llhi_pyr.csv  columns include: feature, neuron_idx, delta_llhi

Outputs:
  - metric=rllr:
      <WEIGHTS_BASE>/RLLR_SUMMARY/
          BOX_dropone_rllr_indoor_vs_outdoor_*.png
          BOX_dropone_rllr_indoor_only_sorted_*.png
          BOX_dropone_rllr_outdoor_only_sorted_*.png
          (Optional if full exists) BOX_full_ll_gain_*.png, BOX_dropone_delta_*.png
          boxplot_dropone_rllr_summary_*.csv
  - metric=llhi:
      <WEIGHTS_BASE>/LLHI_SUMMARY/
          BOX_dropone_llhi_indoor_vs_outdoor_*.png
          BOX_dropone_llhi_indoor_only_sorted_*.png
          BOX_dropone_llhi_outdoor_only_sorted_*.png
          (Optional if full exists) BOX_full_ll_gain_*.png
          boxplot_dropone_delta_llhi_summary_*.csv
  - metric=rllhi:
      <WEIGHTS_BASE>/RLLHI_SUMMARY/
          BOX_dropone_rllhi_indoor_vs_outdoor_*.png
          BOX_dropone_rllhi_indoor_only_sorted_*.png
          BOX_dropone_rllhi_outdoor_only_sorted_*.png
          (Optional if full exists) BOX_full_ll_gain_*.png
          boxplot_dropone_rllhi_summary_*.csv

Run example:
  python plot_contribution.py ^
    --weights_base "D:\\Jiaqi\\Projects\\GLM_File\\GLM_Poisson_Forward\\weights_Poisson_forward" ^
    --forward_modulated_only ^
    --features "Position,Speed,roll,yaw,pitch"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from contribution_rllr_utils import (
    HEAD_POSE_COMPONENTS,
    HEAD_POSE_FEATURE,
    collect_dropone_plot_data as collect_dropone_plot_data_rllr,
    load_dropone_llhi_session_stats as load_dropone_llhi_session_stats_rllr,
    load_dropone_rllhi_session_stats as load_dropone_rllhi_session_stats_rllr,
    load_forward_selected_neurons as load_forward_selected_neurons_rllr,
    load_dropone_session_stats as load_dropone_session_stats_rllr,
    plot_dropone_suite as plot_dropone_suite_rllr,
)


DEFAULT_FEATURES = ["Speed", "roll", "yaw", "pitch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_base", type=str,
                    default=r"D:\Jiaqi\Projects\GLM_File\GLM_Poisson_Forward\weights_Poisson_forward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_scatter_points", type=int, default=0, help="0 = no subsampling")
    ap.add_argument("--ylim_pad_frac", type=float, default=0.08)
    ap.add_argument(
        "--metric",
        type=str,
        default="rllr",
        choices=["rllr", "llhi", "rllhi"],
        help=(
            "Which contribution metric to plot: rllr (RLLR_STATS), "
            "llhi (dropone ΔLLHI), or rllhi (ΔLLHI / LLHI_full)."
        ),
    )
    ap.add_argument("--min_full_ll_gain", type=float, default=0.0,
                    help="Only neurons with full LL gain >= this threshold enter downstream analyses (metric=rllr).")
    ap.add_argument("--min_full_llhi", type=float, default=0.0,
                    help="Only neurons with full LLHI >= this threshold enter downstream analyses (metric=llhi/rllhi).")
    ap.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES),
                    help="Comma-separated feature names to plot; others are ignored.")
    ap.add_argument(
        "--use_raw",
        action="store_true",
        help="If set, plot raw drop-one fractions instead of shuffle-normalized z-scores.",
    )
    ap.add_argument(
        "--forward_modulated_only",
        action="store_true",
        help="If set, per-feature neurons are restricted to those whose forward-search final model includes that feature.",
    )
    ap.add_argument(
        "--positive_only",
        action="store_true",
        help="If set, only contributions > 0 are kept for each feature.",
    )
    args = ap.parse_args()

    features = [s.strip() for s in args.features.split(",") if s.strip()]
    if not features:
        raise SystemExit("[FATAL] --features parsed to empty list.")

    use_zscore = not args.use_raw
    if args.metric == "llhi":
        use_zscore = False

    weights_base = Path(args.weights_base)
    if args.metric == "rllr":
        out_dir = weights_base / "RLLR_SUMMARY"
    elif args.metric == "llhi":
        out_dir = weights_base / "LLHI_SUMMARY"
    else:
        out_dir = weights_base / "RLLHI_SUMMARY"
    out_dir.mkdir(parents=True, exist_ok=True)

    sess_stats: List = []
    allowed_session_tokens = ["F5D10", "F5D2", "F5D3", "F6D10", "F6D2", "F6D3"]
    session_dirs = [p for p in weights_base.iterdir() if p.is_dir()]
    session_dirs = [
        p for p in session_dirs
        if any(token in p.name for token in allowed_session_tokens)
    ]
    for sess_dir in sorted(session_dirs):
        whitelist = None
        if args.forward_modulated_only:
            whitelist = load_forward_selected_neurons_rllr(sess_dir, features=features)
        if args.metric == "rllr":
            st = load_dropone_session_stats_rllr(
                sess_dir,
                features=features,
                feature_neuron_whitelist=whitelist,
                use_zscore=use_zscore,
            )
        elif args.metric == "llhi":
            st = load_dropone_llhi_session_stats_rllr(
                sess_dir,
                features=features,
                feature_neuron_whitelist=whitelist,
            )
        elif args.metric == "rllhi":
            st = load_dropone_rllhi_session_stats_rllr(
                sess_dir,
                features=features,
                feature_neuron_whitelist=whitelist,
                use_zscore=use_zscore,
            )
        if st is not None and args.positive_only:
            filtered_frac = {}
            filtered_shuf_mean = {}
            filtered_shuf_std = {}
            for feat, contrib in st.frac_by_feature.items():
                kept = {ni: val for ni, val in contrib.items() if np.isfinite(val) and val > 0}
                filtered_frac[feat] = kept
                filtered_shuf_mean[feat] = {ni: val for ni, val in st.shuf_mean_by_feature.get(feat, {}).items() if ni in kept}
                filtered_shuf_std[feat] = {ni: val for ni, val in st.shuf_std_by_feature.get(feat, {}).items() if ni in kept}
            st = st.__class__(
                session=st.session,
                group=st.group,
                full_ll_gain=st.full_ll_gain,
                frac_by_feature=filtered_frac,
                shuf_mean_by_feature=filtered_shuf_mean,
                shuf_std_by_feature=filtered_shuf_std,
                all_neuron_ids=st.all_neuron_ids,
            )
        if st is not None:
            sess_stats.append(st)
        elif args.forward_modulated_only:
            print(f"[SKIP] {sess_dir.name}: no neurons matched forward-search filter for requested features.")
    if not sess_stats:
        raise SystemExit(f"[FATAL] No sessions found with valid drop-one stats under: {weights_base}")

    plot_features = features[:]
    if HEAD_POSE_FEATURE not in plot_features and any(f in HEAD_POSE_COMPONENTS for f in plot_features):
        plot_features.append(HEAD_POSE_FEATURE)

    if args.metric == "rllr":
        plot_data = collect_dropone_plot_data_rllr(
            sess_stats,
            features=plot_features,
            min_full_ll_gain=args.min_full_ll_gain,
            compute_delta=not use_zscore,
            include_head_pose=HEAD_POSE_FEATURE in plot_features,
            head_pose_components=HEAD_POSE_COMPONENTS,
        )
        summary_csv = plot_dropone_suite_rllr(
            out_dir,
            features=plot_features,
            plot_data=plot_data,
            min_full_ll_gain=args.min_full_ll_gain,
            seed=args.seed,
            max_scatter_points=args.max_scatter_points,
            ylim_pad_frac=args.ylim_pad_frac,
            use_zscore=use_zscore,
            metric_tag="rllr",
            ylabel="rLLR",
            title_metric="drop-one contribution",
            summary_metric="dropone_rllr",
            include_delta_plot=not use_zscore,
            use_shuffle_line=use_zscore,
        )
    elif args.metric == "llhi":
        plot_data = collect_dropone_plot_data_rllr(
            sess_stats,
            features=plot_features,
            min_full_ll_gain=args.min_full_llhi,
            compute_delta=False,
            include_head_pose=HEAD_POSE_FEATURE in plot_features,
            head_pose_components=HEAD_POSE_COMPONENTS,
        )
        summary_csv = plot_dropone_suite_rllr(
            out_dir,
            features=plot_features,
            plot_data=plot_data,
            min_full_ll_gain=args.min_full_llhi,
            seed=args.seed,
            max_scatter_points=args.max_scatter_points,
            ylim_pad_frac=args.ylim_pad_frac,
            use_zscore=False,
            metric_tag="llhi",
            ylabel="ΔLLHI (bits/spike)",
            title_metric="drop-one contribution",
            summary_metric="dropone_delta_llhi",
            include_delta_plot=False,
            use_shuffle_line=False,
        )
    elif args.metric == "rllhi":
        plot_data = collect_dropone_plot_data_rllr(
            sess_stats,
            features=plot_features,
            min_full_ll_gain=args.min_full_llhi,
            compute_delta=False,
            include_head_pose=HEAD_POSE_FEATURE in plot_features,
            head_pose_components=HEAD_POSE_COMPONENTS,
        )
        summary_csv = plot_dropone_suite_rllr(
            out_dir,
            features=plot_features,
            plot_data=plot_data,
            min_full_ll_gain=args.min_full_llhi,
            seed=args.seed,
            max_scatter_points=args.max_scatter_points,
            ylim_pad_frac=args.ylim_pad_frac,
            use_zscore=use_zscore,
            metric_tag="rllhi_z" if use_zscore else "rllhi",
            ylabel="rLLHI (ΔLLHI / LLHI_full)",
            title_metric="drop-one contribution",
            summary_metric="dropone_rllhi_z" if use_zscore else "dropone_rllhi",
            include_delta_plot=False,
            use_shuffle_line=use_zscore,
        )

    print(f"[OK] Sessions loaded: {len(sess_stats)}")
    print(f"[OK] Features used: {plot_features}")
    if args.metric == "rllr":
        print(f"[OK] Filter: full LL gain >= {args.min_full_ll_gain:g}")
    else:
        print(f"[OK] Filter: full LLHI >= {args.min_full_llhi:g}")
    if use_zscore:
        metric_label = "z-score"
    elif args.metric in {"llhi", "rllhi"}:
        metric_label = "raw values"
    else:
        metric_label = "raw fraction"
    print(f"[OK] Contribution metric: {metric_label}")
    print(
        f"[OK] Kept neurons: indoor {plot_data.kept_counts['indoor']}/{plot_data.total_counts['indoor']}, "
        f"outdoor {plot_data.kept_counts['outdoor']}/{plot_data.total_counts['outdoor']}",
    )
    print(f"[OK] Summary CSV: {summary_csv}")
    print(f"[OK] Wrote plots/summary to: {out_dir}")


if __name__ == "__main__":
    main()
