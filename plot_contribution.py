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
  - Filter: only neurons with full DevExpl >= --min_full_devexpl enter downstream analysis
  - Use --metric rllr to plot contribution_rllr outputs (uses RLLR_STATS + full LL gain filter).
  - By default, uses shuffle-normalized z-scores when available (use --use_raw for fractions)
  - No bootstrap CI bars. Use boxplot whiskers/caps; y-lims auto from whiskers/caps.

Inputs per session:
  - metric=devexpl:
      <WEIGHTS_BASE>/<session>/DROPONE_STATS/
          full_devexpl_pyr.csv
          dropone_contrib_pyr.csv  columns include: feature, neuron_idx, frac_full_dev
  - metric=rllr:
      <WEIGHTS_BASE>/<session>/RLLR_STATS/
          full_rllr_pyr.csv
          dropone_rllr_pyr.csv  columns include: feature, neuron_idx, rllr

Outputs:
  - metric=devexpl:
      <WEIGHTS_BASE>/DROPONE_SUMMARY/
          BOX_dropone_frac_indoor_vs_outdoor_*.png
          BOX_dropone_frac_indoor_only_*.png
          BOX_dropone_frac_outdoor_only_*.png
          (Optional if full exists) BOX_full_devexpl_*.png, BOX_dropone_delta_*.png
          boxplot_dropone_summary_*.csv
  - metric=rllr:
      <WEIGHTS_BASE>/RLLR_SUMMARY/
          BOX_dropone_rllr_indoor_vs_outdoor_*.png
          BOX_dropone_rllr_indoor_only_sorted_*.png
          BOX_dropone_rllr_outdoor_only_sorted_*.png
          (Optional if full exists) BOX_full_ll_gain_*.png, BOX_dropone_delta_*.png
          boxplot_dropone_rllr_summary_*.csv

Run example:
  python plot_contribution.py ^
    --weights_base "D:\\Jiaqi\\Projects\\GLM_File\\GLM_Poisson_Forward\\weights_Poisson_forward" ^
    --min_full_devexpl 0.1 ^
    --forward_modulated_only ^
    --features "Position,Speed,roll,yaw,pitch"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from contribution_utils import (
    collect_dropone_plot_data as collect_dropone_plot_data_devexpl,
    load_forward_selected_neurons as load_forward_selected_neurons_devexpl,
    load_dropone_session_stats as load_dropone_session_stats_devexpl,
    plot_dropone_suite as plot_dropone_suite_devexpl,
)
from contribution_rllr_utils import (
    collect_dropone_plot_data as collect_dropone_plot_data_rllr,
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
        default="devexpl",
        choices=["devexpl", "rllr"],
        help="Which contribution metric to plot: devexpl (DROPONE_STATS) or rllr (RLLR_STATS).",
    )
    ap.add_argument("--min_full_devexpl", type=float, default=0.1,
                    help="Only neurons with full DevExpl >= this threshold enter downstream analyses.")
    ap.add_argument("--min_full_ll_gain", type=float, default=0.0,
                    help="Only neurons with full LL gain >= this threshold enter downstream analyses (metric=rllr).")
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

    weights_base = Path(args.weights_base)
    out_dir = weights_base / ("RLLR_SUMMARY" if args.metric == "rllr" else "DROPONE_SUMMARY")
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
            if args.metric == "rllr":
                whitelist = load_forward_selected_neurons_rllr(sess_dir, features=features)
            else:
                whitelist = load_forward_selected_neurons_devexpl(sess_dir, features=features)
        if args.metric == "rllr":
            st = load_dropone_session_stats_rllr(
                sess_dir,
                features=features,
                feature_neuron_whitelist=whitelist,
                use_zscore=use_zscore,
            )
        else:
            st = load_dropone_session_stats_devexpl(
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
            if args.metric == "rllr":
                st = st.__class__(
                    session=st.session,
                    group=st.group,
                    full_ll_gain=st.full_ll_gain,
                    frac_by_feature=filtered_frac,
                    shuf_mean_by_feature=filtered_shuf_mean,
                    shuf_std_by_feature=filtered_shuf_std,
                )
            else:
                st = st.__class__(
                    session=st.session,
                    group=st.group,
                    full_devexpl=st.full_devexpl,
                    frac_by_feature=filtered_frac,
                    shuf_mean_by_feature=filtered_shuf_mean,
                    shuf_std_by_feature=filtered_shuf_std,
                )
        if st is not None:
            sess_stats.append(st)
        elif args.forward_modulated_only:
            print(f"[SKIP] {sess_dir.name}: no neurons matched forward-search filter for requested features.")
    if not sess_stats:
        raise SystemExit(f"[FATAL] No sessions found with valid drop-one stats under: {weights_base}")

    if args.metric == "rllr":
        plot_data = collect_dropone_plot_data_rllr(
            sess_stats,
            features=features,
            min_full_ll_gain=args.min_full_ll_gain,
            compute_delta=not use_zscore,
        )
        summary_csv = plot_dropone_suite_rllr(
            out_dir,
            features=features,
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
    else:
        plot_data = collect_dropone_plot_data_devexpl(
            sess_stats,
            features=features,
            min_full_devexpl=args.min_full_devexpl,
            compute_delta=not use_zscore,
        )
        summary_csv = plot_dropone_suite_devexpl(
            out_dir,
            features=features,
            plot_data=plot_data,
            min_full_devexpl=args.min_full_devexpl,
            seed=args.seed,
            max_scatter_points=args.max_scatter_points,
            ylim_pad_frac=args.ylim_pad_frac,
            use_zscore=use_zscore,
        )

    print(f"[OK] Sessions loaded: {len(sess_stats)}")
    print(f"[OK] Features used: {features}")
    if args.metric == "rllr":
        print(f"[OK] Filter: full LL gain >= {args.min_full_ll_gain:g}")
    else:
        print(f"[OK] Filter: full DevExpl >= {args.min_full_devexpl:g}")
    print(f"[OK] Contribution metric: {'z-score' if use_zscore else 'raw fraction'}")
    print(
        f"[OK] Kept neurons: indoor {plot_data.kept_counts['indoor']}/{plot_data.total_counts['indoor']}, "
        f"outdoor {plot_data.kept_counts['outdoor']}/{plot_data.total_counts['outdoor']}",
    )
    print(f"[OK] Summary CSV: {summary_csv}")
    print(f"[OK] Wrote plots/summary to: {out_dir}")


if __name__ == "__main__":
    main()
