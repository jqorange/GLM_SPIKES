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
  - Plots rLLR, ΔLLHI, and rLLHI in one run.
  - By default, uses shuffle-normalized z-scores for rLLR and ΔLLHI (use --use_raw to plot raw ΔLLHI).
  - No bootstrap CI bars. Use boxplot whiskers/caps; y-lims auto from whiskers/caps.

Inputs per session:
  - rllr:
      <WEIGHTS_BASE>/<session>/RLLR_STATS/
          full_rllr_pyr.csv
          dropone_rllr_pyr.csv  columns include: feature, neuron_idx, rllr
  - delta_llhi or rllhi:
      <WEIGHTS_BASE>/<session>/RLLR_STATS/
          full_llhi_pyr.csv
          dropone_llhi_pyr.csv  columns include: feature, neuron_idx, delta_llhi

Outputs:
  - rllr:
      <WEIGHTS_BASE>/RLLR_SUMMARY/
          BOX_rllr_indoor_outdoor_*.png
          BOX_rllr_indoor_sorted_*.png
          BOX_rllr_outdoor_sorted_*.png
          (Optional if full exists) BOX_full_ll_gain_*.png, BOX_dropone_delta_*.png
          boxplot_dropone_rllr_summary_*.csv
  - delta_llhi:
      <WEIGHTS_BASE>/LLHI_SUMMARY/
          BOX_delta_llhi_indoor_outdoor_*.png
          BOX_delta_llhi_indoor_sorted_*.png
          BOX_delta_llhi_outdoor_sorted_*.png
          (Optional if full exists) BOX_full_ll_gain_*.png
          boxplot_dropone_delta_llhi_summary_*.csv (and z-score if enabled)
  - rllhi:
      <WEIGHTS_BASE>/RLLHI_SUMMARY/
          BOX_rllhi_indoor_outdoor_*.png
          BOX_rllhi_indoor_sorted_*.png
          BOX_rllhi_outdoor_sorted_*.png
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
    ap.add_argument("--min_full_ll_gain", type=float, default=0.0,
                    help="Only neurons with full LL gain >= this threshold enter downstream analyses (metric=rllr).")
    ap.add_argument("--min_full_llhi", type=float, default=0.0,
                    help="Only neurons with full LLHI >= this threshold enter downstream analyses (metric=llhi/rllhi).")
    ap.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES),
                    help="Comma-separated feature names to plot; others are ignored.")
    ap.add_argument(
        "--use_raw",
        action="store_true",
        help="If set, plot raw ΔLLHI instead of shuffle-normalized z-scores.",
    )
    ap.add_argument(
        "--forward_modulated_only",
        action="store_true",
        help="If set, per-feature neurons are restricted to those whose forward-search final model includes that feature.",
    )
    ap.add_argument(
        "--include_unfit_cells",
        action="store_true",
        help="If set, include unfit cells as zeros in aggregation; otherwise exclude them.",
    )
    args = ap.parse_args()

    features = [s.strip() for s in args.features.split(",") if s.strip()]
    if not features:
        raise SystemExit("[FATAL] --features parsed to empty list.")

    weights_base = Path(args.weights_base)
    sess_stats_by_metric = {}
    allowed_session_tokens = ["F5D10", "F5D2", "F5D3", "F6D10", "F6D2", "F6D3"]
    session_dirs = [p for p in weights_base.iterdir() if p.is_dir()]
    session_dirs = [
        p for p in session_dirs
        if any(token in p.name for token in allowed_session_tokens)
    ]

    for metric in ["rllr", "delta_llhi", "rllhi"]:
        sess_stats = []
        for sess_dir in sorted(session_dirs):
            whitelist = None
            if args.forward_modulated_only:
                whitelist = load_forward_selected_neurons_rllr(sess_dir, features=features)
            if metric == "rllr":
                st = load_dropone_session_stats_rllr(
                    sess_dir,
                    features=features,
                    feature_neuron_whitelist=whitelist,
                    use_zscore=True,
                )
            elif metric == "delta_llhi":
                st = load_dropone_llhi_session_stats_rllr(
                    sess_dir,
                    features=features,
                    feature_neuron_whitelist=whitelist,
                    use_zscore=not args.use_raw,
                )
            else:
                st = load_dropone_rllhi_session_stats_rllr(
                    sess_dir,
                    features=features,
                    feature_neuron_whitelist=whitelist,
                )
            if st is not None:
                sess_stats.append(st)
            elif args.forward_modulated_only:
                print(f"[SKIP] {sess_dir.name}: no neurons matched forward-search filter for requested features.")
        if not sess_stats:
            raise SystemExit(f"[FATAL] No sessions found with valid drop-one stats under: {weights_base}")
        sess_stats_by_metric[metric] = sess_stats

    plot_features = features[:]
    if HEAD_POSE_FEATURE not in plot_features and any(f in HEAD_POSE_COMPONENTS for f in plot_features):
        plot_features.append(HEAD_POSE_FEATURE)

    summaries = []

    for metric in ["rllr", "delta_llhi", "rllhi"]:
        sess_stats = sess_stats_by_metric[metric]
        if metric == "rllr":
            out_dir = weights_base / "RLLR_SUMMARY"
            plot_data = collect_dropone_plot_data_rllr(
                sess_stats,
                features=plot_features,
                min_full_ll_gain=args.min_full_ll_gain,
                compute_delta=True,
                include_missing_cells=args.include_unfit_cells,
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
                use_zscore=True,
                metric_tag="rllr_z",
                ylabel="rLLR",
                title_metric="drop-one rLLR",
                summary_metric="dropone_rllr_z",
                include_delta_plot=True,
                use_shuffle_line=True,
            )
            filter_label = f"full LL gain ≥ {args.min_full_ll_gain:g}"
        elif metric == "delta_llhi":
            out_dir = weights_base / "LLHI_SUMMARY"
            plot_data = collect_dropone_plot_data_rllr(
                sess_stats,
                features=plot_features,
                min_full_ll_gain=args.min_full_llhi,
                compute_delta=False,
                include_missing_cells=args.include_unfit_cells,
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
                use_zscore=not args.use_raw,
                metric_tag="delta_llhi_z" if not args.use_raw else "delta_llhi",
                ylabel="ΔLLHI (bits/spike)",
                title_metric="drop-one ΔLLHI",
                summary_metric="dropone_delta_llhi_z" if not args.use_raw else "dropone_delta_llhi",
                include_delta_plot=False,
                use_shuffle_line=not args.use_raw,
            )
            filter_label = f"full LLHI ≥ {args.min_full_llhi:g}"
        else:
            out_dir = weights_base / "RLLHI_SUMMARY"
            plot_data = collect_dropone_plot_data_rllr(
                sess_stats,
                features=plot_features,
                min_full_ll_gain=args.min_full_llhi,
                compute_delta=False,
                include_missing_cells=args.include_unfit_cells,
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
                metric_tag="rllhi",
                ylabel="rLLHI (ΔLLHI / LLHI_full)",
                title_metric="drop-one rLLHI",
                summary_metric="dropone_rllhi",
                include_delta_plot=False,
                use_shuffle_line=False,
            )
            filter_label = f"full LLHI ≥ {args.min_full_llhi:g}"

        summaries.append((metric, out_dir, plot_data, summary_csv, filter_label))

    for metric, out_dir, plot_data, summary_csv, filter_label in summaries:
        print(f"[OK] Metric: {metric}")
        print(f"[OK] Sessions loaded: {len(sess_stats_by_metric[metric])}")
        print(f"[OK] Features used: {plot_features}")
        print(f"[OK] Filter: {filter_label}")
        print(
            f"[OK] Kept neurons: indoor {plot_data.kept_counts['indoor']}/{plot_data.total_counts['indoor']}, "
            f"outdoor {plot_data.kept_counts['outdoor']}/{plot_data.total_counts['outdoor']}",
        )
        print(f"[OK] Summary CSV: {summary_csv}")
        print(f"[OK] Wrote plots/summary to: {out_dir}")


if __name__ == "__main__":
    main()
