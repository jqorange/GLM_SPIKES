#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-model DevExpl plotting (pyramidal-only) with:
  - indoor vs outdoor combined plot (boxplots + jitter)
  - Session filtering matches plot_contribution.py
  - Filter: only neurons with full DevExpl >= --min_full_devexpl enter analysis

Inputs per session:
  <WEIGHTS_BASE>/<session>/DROPONE_STATS/
      full_devexpl_pyr.csv
      dropone_contrib_pyr.csv

Outputs:
  <WEIGHTS_BASE>/DROPONE_SUMMARY/
      BOX_full_devexpl_indoor_vs_outdoor_*.png
      boxplot_full_devexpl_summary_*.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np

from contribution_utils import (
    DroponeSessionStats,
    collect_dropone_plot_data,
    load_dropone_session_stats,
)
from contribution_utils.plotting import plot_combined_indoor_outdoor, suffix_for_threshold


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
                    help="Comma-separated feature names to load; others are ignored.")
    args = ap.parse_args()

    features = [s.strip() for s in args.features.split(",") if s.strip()]
    if not features:
        raise SystemExit("[FATAL] --features parsed to empty list.")

    weights_base = Path(args.weights_base)
    out_dir = weights_base / "DROPONE_SUMMARY"
    out_dir.mkdir(parents=True, exist_ok=True)

    sess_stats: List[DroponeSessionStats] = []
    allowed_session_tokens = ["F5D10", "F5D2", "F5D3", "F6D10", "F6D2", "F6D3"]
    session_dirs = [p for p in weights_base.iterdir() if p.is_dir()]
    session_dirs = [
        p for p in session_dirs
        if any(token in p.name for token in allowed_session_tokens)
    ]
    for sess_dir in sorted(session_dirs):
        st = load_dropone_session_stats(
            sess_dir,
            features=features,
            use_zscore=True,
        )
        if st is not None:
            sess_stats.append(st)

    if not sess_stats:
        raise SystemExit(f"[FATAL] No sessions found with valid drop-one stats under: {weights_base}")

    plot_data = collect_dropone_plot_data(
        sess_stats,
        features=features,
        min_full_devexpl=args.min_full_devexpl,
        compute_delta=False,
    )

    if not plot_data.has_full:
        raise SystemExit("[FATAL] No valid full DevExpl values after filtering.")

    suffix = suffix_for_threshold(args.min_full_devexpl)
    out_png = out_dir / f"BOX_full_devexpl_indoor_vs_outdoor{suffix}.png"
    plot_combined_indoor_outdoor(
        out_png,
        title=f"Full DevExpl (pyramidal; ≥ {args.min_full_devexpl:g}) | whiskers/caps + jitter",
        ylabel="DevExpl (full model)",
        features=["FULL"],
        data_in={"FULL": plot_data.full_pooled["indoor"]["FULL"]},
        data_out={"FULL": plot_data.full_pooled["outdoor"]["FULL"]},
        seed=args.seed,
        max_scatter_points=args.max_scatter_points,
        ylim_pad_frac=args.ylim_pad_frac,
    )

    summary_csv = out_dir / f"boxplot_full_devexpl_summary{suffix}.csv"
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["group", "n", "mean", "median", "min_full_devexpl", "features_used"],
        )
        w.writeheader()
        for grp in ["indoor", "outdoor"]:
            arr = np.asarray(plot_data.full_pooled[grp]["FULL"], dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            w.writerow({
                "group": grp,
                "n": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "min_full_devexpl": args.min_full_devexpl,
                "features_used": ",".join(features),
            })

    print(f"[OK] Sessions loaded: {len(sess_stats)}")
    print(f"[OK] Filter: full DevExpl >= {args.min_full_devexpl:g}")
    print(
        f"[OK] Kept neurons: indoor {plot_data.kept_counts['indoor']}/{plot_data.total_counts['indoor']}, "
        f"outdoor {plot_data.kept_counts['outdoor']}/{plot_data.total_counts['outdoor']}",
    )
    print(f"[OK] Plot: {out_png}")
    print(f"[OK] Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
