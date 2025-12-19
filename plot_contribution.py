#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drop-one contribution plotting (pyramidal-only) with:
  - indoor vs outdoor combined plot (per-feature, side-by-side boxplots + jitter)
  - PLUS indoor-only and outdoor-only plots (per-feature), where feature order is
    sorted by group mean (high -> low)
  - Feature selection via --features (comma-separated). Features not listed are ignored.
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
  python glm_dropone_plots_devthr_features_sorted.py ^
    --weights_base "D:\Jiaqi\Projects\GLM_File\GLM_Poisson_Forward\weights_Poisson_forward" ^
    --min_full_devexpl 0.1 ^
    --features "Position,Speed,roll,yaw,pitch"
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DEFAULT_FEATURES = ["Speed", "roll", "yaw", "pitch"]


# -------------------------------
# Safe CSV reading (no pandas)
# -------------------------------
def _safe_float(x) -> float:
    if x is None:
        return float("nan")
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def read_csv_dicts_safe(path: Path) -> List[dict]:
    path = Path(path)
    if (not path.exists()) or path.stat().st_size < 1:
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        raw = f.read()
    if "\x00" in raw:
        raw = raw.replace("\x00", "")
    import io
    buf = io.StringIO(raw)
    reader = csv.DictReader(buf)
    rows = []
    for r in reader:
        if r is None:
            continue
        if all((v is None or str(v).strip() == "") for v in r.values()):
            continue
        rows.append(r)
    return rows


# -------------------------------
# Data structures
# -------------------------------
@dataclass
class SessionStats:
    session: str
    group: str  # indoor/outdoor
    full_devexpl: Dict[int, float]                 # neuron_idx -> full DevExpl
    frac_by_feature: Dict[str, Dict[int, float]]   # feature -> neuron_idx -> frac


def infer_group(session_name: str) -> Optional[str]:
    s = session_name.lower()
    if "indoor" in s:
        return "indoor"
    if "outdoor" in s:
        return "outdoor"
    return None


def load_one_session_stats(session_dir: Path, features: List[str]) -> Optional[SessionStats]:
    session = session_dir.name
    group = infer_group(session)
    if group is None:
        return None

    stats_dir = session_dir / "DROPONE_STATS"
    full_csv = stats_dir / "full_devexpl_pyr.csv"
    contrib_csv = stats_dir / "dropone_contrib_pyr.csv"

    contrib_rows = read_csv_dicts_safe(contrib_csv)
    if not contrib_rows:
        return None
    full_rows = read_csv_dicts_safe(full_csv)

    full_dev: Dict[int, float] = {}
    for r in full_rows:
        ni = r.get("neuron_idx", r.get("neuron", None))
        dv = r.get("full_devexpl", r.get("devexpl_full", r.get("devexpl", None)))
        if ni is None or dv is None:
            continue
        try:
            idx = int(float(str(ni).strip()))
        except Exception:
            continue
        val = _safe_float(dv)
        if np.isfinite(val):
            full_dev[idx] = float(val)

    frac: Dict[str, Dict[int, float]] = {f: {} for f in features}
    for r in contrib_rows:
        feat = str(r.get("feature", "")).strip()
        if feat not in frac:
            continue
        ni = r.get("neuron_idx", None)
        fv = r.get("frac_full_dev", r.get("frac", None))
        if ni is None or fv is None:
            continue
        try:
            idx = int(float(str(ni).strip()))
        except Exception:
            continue
        val = _safe_float(fv)
        if np.isfinite(val):
            frac[feat][idx] = float(val)

    if all(len(frac[f]) == 0 for f in features):
        return None

    return SessionStats(session=session, group=group, full_devexpl=full_dev, frac_by_feature=frac)


# -------------------------------
# Pooling helpers
# -------------------------------
def pooled_all(values_by_session: Dict[str, np.ndarray]) -> np.ndarray:
    pooled_list = []
    for v in values_by_session.values():
        if v is None:
            continue
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        if a.size:
            pooled_list.append(a)
    if not pooled_list:
        return np.array([], dtype=float)
    return np.concatenate(pooled_list, axis=0)


def jitter_points(rng: np.random.Generator, n: int, *, scale: float) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)
    return rng.normal(0.0, scale, size=n)


def set_ylim_from_boxplot(bp: dict, ax: plt.Axes, *, pad_frac: float):
    ys = []

    def _collect(lines):
        for ln in lines:
            try:
                y = np.asarray(ln.get_ydata(), dtype=float)
                y = y[np.isfinite(y)]
                if y.size:
                    ys.extend(y.tolist())
            except Exception:
                pass

    _collect(bp.get("whiskers", []))
    _collect(bp.get("caps", []))
    _collect(bp.get("medians", []))

    if not ys:
        return

    y0 = float(np.min(ys))
    y1 = float(np.max(ys))
    span = y1 - y0
    if span <= 0:
        span = 1.0
    pad = pad_frac * span
    ax.set_ylim(y0 - pad, y1 + pad)


# -------------------------------
# Plotting
# -------------------------------
def plot_combined_indoor_outdoor(
    out_png: Path,
    *,
    title: str,
    ylabel: str,
    features: List[str],
    data_in: Dict[str, np.ndarray],
    data_out: Dict[str, np.ndarray],
    seed: int,
    max_scatter_points: int,
    ylim_pad_frac: float,
):
    rng = np.random.default_rng(seed)

    x = np.arange(len(features), dtype=float)
    offset = 0.20
    pos_in = x - offset
    pos_out = x + offset
    width = 0.28

    indoor_fc = "#b0b0b0"   # gray
    outdoor_fc = "#8fd19e"  # green
    edge_c = "k"

    fig, ax = plt.subplots(figsize=(14, 5))

    bp_data, bp_pos, bp_group = [], [], []
    for i, f in enumerate(features):
        yin = np.asarray(data_in.get(f, np.array([], dtype=float)), dtype=float)
        yout = np.asarray(data_out.get(f, np.array([], dtype=float)), dtype=float)
        yin = yin[np.isfinite(yin)]
        yout = yout[np.isfinite(yout)]

        bp_data.append(yin if yin.size else np.array([np.nan]))
        bp_pos.append(pos_in[i])
        bp_group.append("indoor")

        bp_data.append(yout if yout.size else np.array([np.nan]))
        bp_pos.append(pos_out[i])
        bp_group.append("outdoor")

    bp = ax.boxplot(
        bp_data,
        positions=bp_pos,
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": "k", "linewidth": 1.5},
        boxprops={"linewidth": 1.2, "edgecolor": edge_c},
        whiskerprops={"linewidth": 1.2, "color": edge_c},
        capprops={"linewidth": 1.2, "color": edge_c},
    )

    for j, box in enumerate(bp["boxes"]):
        grp = bp_group[j]
        box.set_facecolor(indoor_fc if grp == "indoor" else outdoor_fc)
        box.set_alpha(0.55)

    for i, f in enumerate(features):
        yin = np.asarray(data_in.get(f, np.array([], dtype=float)), dtype=float)
        yout = np.asarray(data_out.get(f, np.array([], dtype=float)), dtype=float)
        yin = yin[np.isfinite(yin)]
        yout = yout[np.isfinite(yout)]

        if max_scatter_points and yin.size > max_scatter_points:
            yin = rng.choice(yin, size=max_scatter_points, replace=False)
        if max_scatter_points and yout.size > max_scatter_points:
            yout = rng.choice(yout, size=max_scatter_points, replace=False)

        ax.scatter(pos_in[i] + jitter_points(rng, yin.size, scale=0), yin, s=12, alpha=0.70, color="k", zorder=3)
        ax.scatter(pos_out[i] + jitter_points(rng, yout.size, scale=0), yout, s=12, alpha=0.70, color="k", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, linewidth=1.0, alpha=0.35)

    ax.legend(
        handles=[
            Patch(facecolor=indoor_fc, edgecolor=edge_c, alpha=0.55, label="indoor"),
            Patch(facecolor=outdoor_fc, edgecolor=edge_c, alpha=0.55, label="outdoor"),
        ],
        frameon=False,
        loc="upper right",
    )

    set_ylim_from_boxplot(bp, ax, pad_frac=ylim_pad_frac)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def plot_single_group_sorted(
    out_png: Path,
    *,
    title: str,
    ylabel: str,
    group_label: str,
    features_sorted: List[str],
    data: Dict[str, np.ndarray],
    seed: int,
    max_scatter_points: int,
    ylim_pad_frac: float,
):
    rng = np.random.default_rng(seed)

    x = np.arange(len(features_sorted), dtype=float)
    width = 0.50

    # distinguish by group
    if group_label == "indoor":
        fc = "#b0b0b0"
    else:
        fc = "#8fd19e"
    edge_c = "k"

    fig, ax = plt.subplots(figsize=(12, 5))

    bp_data = []
    for f in features_sorted:
        y = np.asarray(data.get(f, np.array([], dtype=float)), dtype=float)
        y = y[np.isfinite(y)]
        bp_data.append(y if y.size else np.array([np.nan]))

    bp = ax.boxplot(
        bp_data,
        positions=x,
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        medianprops={"color": "k", "linewidth": 1.5},
        boxprops={"linewidth": 1.2, "edgecolor": edge_c},
        whiskerprops={"linewidth": 1.2, "color": edge_c},
        capprops={"linewidth": 1.2, "color": edge_c},
    )

    for box in bp["boxes"]:
        box.set_facecolor(fc)
        box.set_alpha(0.55)

    # jitter scatter
    for i, f in enumerate(features_sorted):
        y = np.asarray(data.get(f, np.array([], dtype=float)), dtype=float)
        y = y[np.isfinite(y)]
        if max_scatter_points and y.size > max_scatter_points:
            y = rng.choice(y, size=max_scatter_points, replace=False)
        ax.scatter(x[i] + jitter_points(rng, y.size, scale=0), y, s=12, alpha=0.70, color="k", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(features_sorted, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, linewidth=1.0, alpha=0.35)

    ax.legend(
        handles=[Patch(facecolor=fc, edgecolor=edge_c, alpha=0.55, label=group_label)],
        frameon=False,
        loc="upper right",
    )

    set_ylim_from_boxplot(bp, ax, pad_frac=ylim_pad_frac)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def group_feature_means(data: Dict[str, np.ndarray], features: List[str]) -> Dict[str, float]:
    out = {}
    for f in features:
        a = np.asarray(data.get(f, np.array([], dtype=float)), dtype=float)
        a = a[np.isfinite(a)]
        out[f] = float(np.mean(a)) if a.size else float("-inf")
    return out


# -------------------------------
# Main
# -------------------------------
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
    args = ap.parse_args()

    # Parse features list
    features = [s.strip() for s in args.features.split(",") if s.strip()]
    if not features:
        raise SystemExit("[FATAL] --features parsed to empty list.")

    weights_base = Path(args.weights_base)
    out_dir = weights_base / "DROPONE_SUMMARY"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load sessions
    sess_stats: List[SessionStats] = []
    for sess_dir in sorted([p for p in weights_base.iterdir() if p.is_dir()]):
        st = load_one_session_stats(sess_dir, features=features)
        if st is not None:
            sess_stats.append(st)
    if not sess_stats:
        raise SystemExit(f"[FATAL] No sessions found under: {weights_base}")

    # Per-session arrays
    frac_by_session: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"indoor": {f: {} for f in features}, "outdoor": {f: {} for f in features}}
    full_by_session: Dict[str, Dict[str, np.ndarray]] = {"indoor": {}, "outdoor": {}}
    delta_by_session: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"indoor": {f: {} for f in features}, "outdoor": {f: {} for f in features}}

    kept_counts = {"indoor": 0, "outdoor": 0}
    total_counts = {"indoor": 0, "outdoor": 0}

    for st in sess_stats:
        g = st.group

        eligible: Set[int] = set()
        for ni, dv in st.full_devexpl.items():
            if np.isfinite(dv):
                total_counts[g] += 1
                if dv >= args.min_full_devexpl:
                    eligible.add(int(ni))
                    kept_counts[g] += 1

        # Full DevExpl distribution (filtered)
        if eligible:
            arr_full = np.asarray([st.full_devexpl[ni] for ni in eligible if np.isfinite(st.full_devexpl.get(ni, np.nan))], dtype=float)
            arr_full = arr_full[np.isfinite(arr_full)]
            if arr_full.size:
                full_by_session[g][st.session] = arr_full

        # Drop-one frac / delta (filtered to eligible)
        for f in features:
            m = st.frac_by_feature.get(f, {})
            if not m or not eligible:
                continue

            vals_frac = []
            vals_delta = []
            for ni, fracv in m.items():
                if ni not in eligible:
                    continue
                dv = st.full_devexpl.get(ni, np.nan)
                if np.isfinite(fracv) and np.isfinite(dv):
                    vals_frac.append(float(fracv))
                    vals_delta.append(float(fracv) * float(dv))

            arr_frac = np.asarray(vals_frac, dtype=float)
            arr_frac = arr_frac[np.isfinite(arr_frac)]
            if arr_frac.size:
                frac_by_session[g][f][st.session] = arr_frac

            arr_delta = np.asarray(vals_delta, dtype=float)
            arr_delta = arr_delta[np.isfinite(arr_delta)]
            if arr_delta.size:
                delta_by_session[g][f][st.session] = arr_delta

    # pooled distributions
    frac_pooled = {
        "indoor": {f: pooled_all(frac_by_session["indoor"][f]) for f in features},
        "outdoor": {f: pooled_all(frac_by_session["outdoor"][f]) for f in features},
    }
    delta_pooled = {
        "indoor": {f: pooled_all(delta_by_session["indoor"][f]) for f in features},
        "outdoor": {f: pooled_all(delta_by_session["outdoor"][f]) for f in features},
    }
    full_pooled = {
        "indoor": {"FULL": pooled_all(full_by_session["indoor"])},
        "outdoor": {"FULL": pooled_all(full_by_session["outdoor"])},
    }

    have_full = (full_pooled["indoor"]["FULL"].size > 0) or (full_pooled["outdoor"]["FULL"].size > 0)

    suffix = f"_FULLGE{args.min_full_devexpl:g}".replace(".", "p")

    # (A) Combined plot
    plot_combined_indoor_outdoor(
        out_dir / f"BOX_dropone_frac_indoor_vs_outdoor{suffix}.png",
        title=f"Drop-one fraction (pyramidal; full DevExpl ≥ {args.min_full_devexpl:g}) | whiskers/caps + jitter",
        ylabel="frac(full DevExpl)",
        features=features,
        data_in=frac_pooled["indoor"],
        data_out=frac_pooled["outdoor"],
        seed=args.seed,
        max_scatter_points=args.max_scatter_points,
        ylim_pad_frac=args.ylim_pad_frac,
    )

    # (B) Indoor-only and Outdoor-only plots with feature sorting by group mean
    indoor_means = group_feature_means(frac_pooled["indoor"], features)
    outdoor_means = group_feature_means(frac_pooled["outdoor"], features)

    features_indoor_sorted = sorted(features, key=lambda f: indoor_means.get(f, float("-inf")), reverse=True)
    features_outdoor_sorted = sorted(features, key=lambda f: outdoor_means.get(f, float("-inf")), reverse=True)

    plot_single_group_sorted(
        out_dir / f"BOX_dropone_frac_indoor_only_sorted{suffix}.png",
        title=f"Indoor only (sorted by mean) | drop-one fraction | full DevExpl ≥ {args.min_full_devexpl:g}",
        ylabel="frac(full DevExpl)",
        group_label="indoor",
        features_sorted=features_indoor_sorted,
        data=frac_pooled["indoor"],
        seed=args.seed,
        max_scatter_points=args.max_scatter_points,
        ylim_pad_frac=args.ylim_pad_frac,
    )

    plot_single_group_sorted(
        out_dir / f"BOX_dropone_frac_outdoor_only_sorted{suffix}.png",
        title=f"Outdoor only (sorted by mean) | drop-one fraction | full DevExpl ≥ {args.min_full_devexpl:g}",
        ylabel="frac(full DevExpl)",
        group_label="outdoor",
        features_sorted=features_outdoor_sorted,
        data=frac_pooled["outdoor"],
        seed=args.seed,
        max_scatter_points=args.max_scatter_points,
        ylim_pad_frac=args.ylim_pad_frac,
    )

    # Optional: full DevExpl and delta DevExpl combined plot (kept consistent with earlier script)
    if have_full:
        plot_combined_indoor_outdoor(
            out_dir / f"BOX_full_devexpl_indoor_vs_outdoor{suffix}.png",
            title=f"Full DevExpl (pyramidal; ≥ {args.min_full_devexpl:g}) | whiskers/caps + jitter",
            ylabel="DevExpl (full model)",
            features=["FULL"],
            data_in={"FULL": full_pooled["indoor"]["FULL"]},
            data_out={"FULL": full_pooled["outdoor"]["FULL"]},
            seed=args.seed,
            max_scatter_points=args.max_scatter_points,
            ylim_pad_frac=args.ylim_pad_frac,
        )

        plot_combined_indoor_outdoor(
            out_dir / f"BOX_dropone_delta_indoor_vs_outdoor{suffix}.png",
            title=f"Drop-one ΔDevExpl (= frac×full) (pyramidal; ≥ {args.min_full_devexpl:g}) | whiskers/caps + jitter",
            ylabel="ΔDevExpl",
            features=features,
            data_in=delta_pooled["indoor"],
            data_out=delta_pooled["outdoor"],
            seed=args.seed,
            max_scatter_points=args.max_scatter_points,
            ylim_pad_frac=args.ylim_pad_frac,
        )

    # Summary CSV (means/medians for reference, for the plotted metric)
    out_csv = out_dir / f"boxplot_dropone_summary{suffix}.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["metric", "group", "feature", "n", "mean", "median", "min_full_devexpl", "features_used"],
        )
        w.writeheader()
        for feat in features:
            for grp in ["indoor", "outdoor"]:
                arr = np.asarray(frac_pooled[grp][feat], dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                w.writerow({
                    "metric": "dropone_frac",
                    "group": grp,
                    "feature": feat,
                    "n": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "min_full_devexpl": args.min_full_devexpl,
                    "features_used": ",".join(features),
                })
        if have_full:
            for grp in ["indoor", "outdoor"]:
                arr = np.asarray(full_pooled[grp]["FULL"], dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                w.writerow({
                    "metric": "full_devexpl",
                    "group": grp,
                    "feature": "FULL",
                    "n": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "min_full_devexpl": args.min_full_devexpl,
                    "features_used": ",".join(features),
                })

    print(f"[OK] Sessions loaded: {len(sess_stats)}")
    print(f"[OK] Features used: {features}")
    print(f"[OK] Filter: full DevExpl >= {args.min_full_devexpl:g}")
    print(f"[OK] Kept neurons: indoor {kept_counts['indoor']}/{total_counts['indoor']}, outdoor {kept_counts['outdoor']}/{total_counts['outdoor']}")
    print(f"[OK] Wrote plots/summary to: {out_dir}")


if __name__ == "__main__":
    main()
