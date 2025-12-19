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
  - No bootstrap CI bars. Use boxplot whiskers/caps; y-lims auto from whiskers/caps.

Inputs per session:
  <WEIGHTS_BASE>/<session>/DROPONE_STATS/
      full_devexpl_pyr.csv
      dropone_contrib_pyr.csv  columns include: feature, neuron_idx, frac_full_dev

Outputs:
  <WEIGHTS_BASE>/DROPONE_SUMMARY/
      BOX_dropone_frac_indoor_vs_outdoor_*.png
      BOX_dropone_frac_indoor_only_*.png
      BOX_dropone_frac_outdoor_only_*.png
      (Optional if full exists) BOX_full_devexpl_*.png, BOX_dropone_delta_*.png
      boxplot_dropone_summary_*.csv

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

from contribution_utils import (
    DroponeSessionStats,
    collect_dropone_plot_data,
    load_forward_selected_neurons,
    load_dropone_session_stats,
    plot_dropone_suite,
)


DEFAULT_FEATURES = ["Speed", "roll", "yaw", "pitch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_base", type=str,
                    default=r"D:\Jiaqi\Projects\GLM_File\GLM_Poisson_Forward\weights_Poisson_forward")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_scatter_points", type=int, default=0, help="0 = no subsampling")
    ap.add_argument("--ylim_pad_frac", type=float, default=0.08)
    ap.add_argument("--min_full_devexpl", type=float, default=0.1,
                    help="Only neurons with full DevExpl >= this threshold enter downstream analyses.")
    ap.add_argument("--features", type=str, default=",".join(DEFAULT_FEATURES),
                    help="Comma-separated feature names to plot; others are ignored.")
    ap.add_argument(
        "--forward_modulated_only",
        action="store_true",
        help="If set, per-feature neurons are restricted to those whose forward-search final model includes that feature.",
    )
    args = ap.parse_args()

    features = [s.strip() for s in args.features.split(",") if s.strip()]
    if not features:
        raise SystemExit("[FATAL] --features parsed to empty list.")

    weights_base = Path(args.weights_base)
    out_dir = weights_base / "DROPONE_SUMMARY"
    out_dir.mkdir(parents=True, exist_ok=True)

    sess_stats: List[DroponeSessionStats] = []
    for sess_dir in sorted([p for p in weights_base.iterdir() if p.is_dir()]):
        whitelist = None
        if args.forward_modulated_only:
            whitelist = load_forward_selected_neurons(sess_dir, features=features)
        st = load_dropone_session_stats(
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

    plot_data = collect_dropone_plot_data(
        sess_stats,
        features=features,
        min_full_devexpl=args.min_full_devexpl,
    )
    summary_csv = plot_dropone_suite(
        out_dir,
        features=features,
        plot_data=plot_data,
        min_full_devexpl=args.min_full_devexpl,
        seed=args.seed,
        max_scatter_points=args.max_scatter_points,
        ylim_pad_frac=args.ylim_pad_frac,
    )

    print(f"[OK] Sessions loaded: {len(sess_stats)}")
    print(f"[OK] Features used: {features}")
    print(f"[OK] Filter: full DevExpl >= {args.min_full_devexpl:g}")
    print(
        f"[OK] Kept neurons: indoor {plot_data.kept_counts['indoor']}/{plot_data.total_counts['indoor']}, "
        f"outdoor {plot_data.kept_counts['outdoor']}/{plot_data.total_counts['outdoor']}",
    )
    print(f"[OK] Summary CSV: {summary_csv}")
    print(f"[OK] Wrote plots/summary to: {out_dir}")


if __name__ == "__main__":
    main()
