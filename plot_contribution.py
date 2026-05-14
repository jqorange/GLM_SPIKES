#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drop-one contribution plotting (pyramidal-only) with:
  - indoor vs outdoor combined plot (per-feature, side-by-side boxplots + jitter)
  - PLUS indoor-only and outdoor-only plots (per-feature), where feature order is
    sorted by group mean (high -> low)
  - Feature selection via --features (comma-separated). Features not listed are ignored.
  - Optional: use --no-forward_modulated_only to disable the default filter that limits each
    feature to neurons whose forward-search final model includes that feature.
  - Optional: use --paired_fit_only to keep only cells that were fit for a feature in both indoor and outdoor.
  - Filter: only neurons with full LL gain/LLHI >= threshold enter downstream analysis
  - By default, plots rLLR, ΔLLHI, rLLHI, and rSCC in one run.
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

  - rscc:
      <WEIGHTS_BASE>/RSCC_SUMMARY/
          BOX_rscc_indoor_outdoor_*.png
          BOX_rscc_indoor_sorted_*.png
          BOX_rscc_outdoor_sorted_*.png
          (Optional if full exists) BOX_full_ll_gain_*.png
          boxplot_dropone_rscc_summary_*.csv

Run example:
  python plot_contribution.py ^
    --weights_base "D:\\Jiaqi\\Projects\\GLM_File\\GLM_Poisson_Forward\\weights_Poisson_forward" ^
    --forward_modulated_only ^
    --features "Position,Speed,roll,yaw,pitch"
"""

from __future__ import annotations

import argparse
import base64
import struct
from pathlib import Path
import matplotlib as mpl
from contribution_rllr_utils import (
    HEAD_POSE_COMPONENTS,
    HEAD_POSE_FEATURE,
    build_dayid_to_cellinfo,
    collect_dropone_plot_data as collect_dropone_plot_data_rllr,
    load_dropone_llhi_session_stats as load_dropone_llhi_session_stats_rllr,
    load_dropone_rllhi_session_stats as load_dropone_rllhi_session_stats_rllr,
    load_dropone_rscc_session_stats as load_dropone_rscc_session_stats_rllr,
    load_forward_selected_neurons as load_forward_selected_neurons_rllr,
    load_dropone_session_stats as load_dropone_session_stats_rllr,
    plot_dropone_suite as plot_dropone_suite_rllr,
    pyramidal_indices_for_session,
)
from glm_poisson_forward.config import FS_HZ, INCLUDE_TIME_VARIABLE, VARIABLE_COMPOSITES, WEIGHTS_BASE
from glm_poisson_forward.io_utils import load_spikes_50hz_counts, session_paths

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "DejaVu Sans"]


DEFAULT_FEATURES = ["Position", "Speed", "roll", "yaw", "pitch"] + (["Time"] if INCLUDE_TIME_VARIABLE else [])
DEFAULT_MIN_FIRING_RATE_HZ = 0.02
DEFAULT_H_DIST_N_BINS = 100
DEFAULT_METRICS = ["rllr", "delta_llhi", "rllhi", "rscc"]


def _parse_composite_spec(spec: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        name, members = item.split("=", 1)
        comp = [s.strip() for s in members.split("+") if s.strip()]
        if name.strip() and comp:
            out[name.strip()] = comp
    return out


def _parse_metrics(spec: str) -> list[str]:
    allowed = {m: m for m in DEFAULT_METRICS}
    aliases = {
        "llhi": "delta_llhi",
        "delta-llhi": "delta_llhi",
    }
    metrics = []
    seen = set()
    for item in str(spec).split(","):
        name = item.strip().lower()
        if not name:
            continue
        canonical = aliases.get(name, name)
        if canonical not in allowed:
            valid = ", ".join(DEFAULT_METRICS)
            raise SystemExit(f"[FATAL] Unknown metric '{item}'. Valid choices: {valid}")
        if canonical not in seen:
            metrics.append(canonical)
            seen.add(canonical)
    if not metrics:
        raise SystemExit("[FATAL] --metrics parsed to empty list.")
    return metrics


def _save_current_figure_png_svg(png_path: Path, *, dpi: int = 300) -> None:
    import matplotlib.pyplot as plt

    png_path = Path(png_path)
    svg_path = png_path.with_suffix(".svg")
    plt.savefig(png_path, dpi=dpi)
    plt.savefig(svg_path, format="svg")


def _png_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            return (1000, 700)
        _chunk_len = f.read(4)
        if f.read(4) != b"IHDR":
            return (1000, 700)
        data = f.read(13)
        if len(data) != 13:
            return (1000, 700)
        width, height = struct.unpack(">II", data[:8])
    return int(width), int(height)


def _write_svg_sidecar_from_png(png_path: Path) -> Path:
    png_path = Path(png_path)
    svg_path = png_path.with_suffix(".svg")
    if svg_path.exists():
        return svg_path

    width, height = _png_size(png_path)
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    svg_text = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'<image href="data:image/png;base64,{b64}" width="{width}" height="{height}"/>\n'
        "</svg>\n"
    )
    svg_path.write_text(svg_text, encoding="utf-8")
    return svg_path


def _ensure_svg_for_all_pngs(out_dir: Path) -> None:
    for png in Path(out_dir).rglob("*.png"):
        _write_svg_sidecar_from_png(png)


def _active_cell_count_for_session(
    session: str,
    *,
    min_firing_rate_hz: float,
    pyramidal_only: bool,
    dayid2cellinfo,
) -> int | None:
    import numpy as np

    spike_path = session_paths(session)["spike"]
    if not spike_path.exists():
        print(f"[WARN] {session}: spike file missing for H distribution denominator: {spike_path}")
        return None
    try:
        y50 = load_spikes_50hz_counts(spike_path)
    except Exception as exc:
        print(f"[WARN] {session}: failed loading spike file for H distribution denominator: {exc}")
        return None
    if y50.ndim != 2 or y50.shape[0] == 0:
        return None

    n_cells = int(y50.shape[1])
    fr_hz = y50.mean(axis=0).astype(float) * float(FS_HZ)
    scope_mask = np.ones(n_cells, dtype=bool)
    if pyramidal_only:
        pyr_idx = pyramidal_indices_for_session(session, dayid2cellinfo, n_cells)
        if pyr_idx is None:
            print(f"[WARN] {session}: pyramidal indices unavailable for H distribution denominator.")
            return None
        scope_mask[:] = False
        scope_mask[pyr_idx] = True
    active_mask = fr_hz >= float(min_firing_rate_hz)
    return int(np.sum(scope_mask & active_mask))


def _collect_h_by_session(
    session_stats,
    *,
    min_full_ll_gain: float,
    include_missing_cells: bool,
    min_firing_rate_hz: float,
    pyramidal_only: bool,
    dayid2cellinfo,
) -> dict[str, dict[str, dict[str, object]]]:
    import numpy as np

    out = {"indoor": {}, "outdoor": {}}
    for st in session_stats:
        g = st.group
        if g not in ("indoor", "outdoor"):
            continue

        denom = _active_cell_count_for_session(
            st.session,
            min_firing_rate_hz=min_firing_rate_hz,
            pyramidal_only=pyramidal_only,
            dayid2cellinfo=dayid2cellinfo,
        )
        if denom is None or denom <= 0:
            continue

        h_map = st.frac_by_feature.get("H", {})
        if not h_map and not include_missing_cells:
            continue

        unfit_neurons = set(st.unfit_neuron_ids or [])
        if include_missing_cells:
            all_neurons = set(st.all_neuron_ids or [])
            if not all_neurons:
                all_neurons = set(st.full_ll_gain.keys()) | unfit_neurons
        else:
            all_neurons = set(st.full_ll_gain.keys())

        eligible = set()
        for ni in all_neurons:
            if ni in unfit_neurons:
                if include_missing_cells:
                    eligible.add(int(ni))
                continue
            ll_gain = st.full_ll_gain.get(ni, 0.0)
            if not np.isfinite(ll_gain):
                ll_gain = 0.0
            if ll_gain >= min_full_ll_gain:
                eligible.add(int(ni))

        vals = []
        for ni in eligible:
            fracv = h_map.get(ni, 0.0 if include_missing_cells else np.nan)
            if np.isfinite(fracv):
                vals.append(float(fracv))
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        out[g][st.session] = {"values": arr, "denom": int(denom)}
    return out


def _h_bins_from_sessions(by_session: dict[str, dict[str, dict[str, object]]], n_bins: int) -> tuple[object, object] | tuple[None, None]:
    import numpy as np

    combined = []
    for group_map in by_session.values():
        for session_info in group_map.values():
            arr = np.asarray(session_info.get("values", []), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                combined.append(arr)
    if not combined:
        return None, None

    combined_all = np.concatenate(combined)
    vmin = float(np.min(combined_all))
    vmax = float(np.max(combined_all))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None
    if np.isclose(vmin, vmax):
        pad = max(1e-3, abs(vmin) * 0.05 + 1e-3)
        vmin -= pad
        vmax += pad
    bins = np.linspace(vmin, vmax, int(max(5, n_bins)) + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return bins, centers


def _gaussian_smooth_1d(arr, sigma_bins: float = 1.5):
    import numpy as np

    x = np.asarray(arr, dtype=float)
    if x.ndim != 1 or x.size <= 1 or sigma_bins <= 0:
        return x

    radius = max(1, int(np.ceil(3.0 * float(sigma_bins))))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / float(sigma_bins)) ** 2)
    kernel /= np.sum(kernel)

    x_pad = np.pad(x, (radius, radius), mode="edge")
    y = np.convolve(x_pad, kernel, mode="same")
    return y[radius:-radius]


def _write_h_distribution_plot(
    out_dir: Path,
    metric_tag: str,
    session_stats,
    *,
    min_full_ll_gain: float,
    include_missing_cells: bool,
    min_firing_rate_hz: float,
    pyramidal_only: bool,
    dayid2cellinfo,
    n_bins: int,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    by_session = _collect_h_by_session(
        session_stats,
        min_full_ll_gain=min_full_ll_gain,
        include_missing_cells=include_missing_cells,
        min_firing_rate_hz=min_firing_rate_hz,
        pyramidal_only=pyramidal_only,
        dayid2cellinfo=dayid2cellinfo,
    )
    bins, _ = _h_bins_from_sessions(by_session, n_bins=n_bins)
    if bins is None:
        print(f"[WARN] Skip H distribution for metric={metric_tag}: missing indoor/outdoor H values.")
        return

    plt.figure(figsize=(7.2, 4.8))
    for group, color in [("indoor", "#2B6CB0"), ("outdoor", "#DD6B20")]:
        values = []
        weights = []
        n_sessions = 0
        for session_info in by_session[group].values():
            arr = np.asarray(session_info.get("values", []), dtype=float)
            arr = arr[np.isfinite(arr)]
            denom = int(session_info.get("denom", 0))
            if arr.size == 0 or denom <= 0:
                continue
            values.append(arr)
            weights.append(np.full(arr.size, 100.0 / float(denom), dtype=float))
            n_sessions += 1
        if not values:
            continue
        vals = np.concatenate(values)
        wts = np.concatenate(weights)
        plt.hist(
            vals,
            bins=bins,
            weights=wts,
            alpha=0.45,
            color=color,
            edgecolor="white",
            label=f"{group} (n={n_sessions} sessions)",
        )

    plt.xlabel(metric_tag)
    plt.ylabel("Percent of active cells in current session (%)")
    plt.title(f"H contribution distribution: indoor vs outdoor ({metric_tag})")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_png = Path(out_dir) / f"H_DIST_{metric_tag}_indoor_vs_outdoor.png"
    _save_current_figure_png_svg(out_png, dpi=300)
    plt.close()
    print(f"[OK] Wrote H distribution plot: {out_png}")


def _write_h_distribution_curve_plot(
    out_dir: Path,
    metric_tag: str,
    session_stats,
    *,
    min_full_ll_gain: float,
    include_missing_cells: bool,
    min_firing_rate_hz: float,
    pyramidal_only: bool,
    dayid2cellinfo,
    n_bins: int,
) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    by_session = _collect_h_by_session(
        session_stats,
        min_full_ll_gain=min_full_ll_gain,
        include_missing_cells=include_missing_cells,
        min_firing_rate_hz=min_firing_rate_hz,
        pyramidal_only=pyramidal_only,
        dayid2cellinfo=dayid2cellinfo,
    )
    in_sessions = by_session["indoor"]
    out_sessions = by_session["outdoor"]
    if not in_sessions or not out_sessions:
        print(f"[WARN] Skip H curve plot for metric={metric_tag}: missing indoor/outdoor session data.")
        return

    bins, centers = _h_bins_from_sessions(by_session, n_bins=n_bins)
    if bins is None:
        print(f"[WARN] Skip H curve plot for metric={metric_tag}: no finite H values.")
        return

    def _session_prop_curves(group_map):
        curves = []
        for session_info in group_map.values():
            arr = np.asarray(session_info.get("values", []), dtype=float)
            arr = arr[np.isfinite(arr)]
            denom = int(session_info.get("denom", 0))
            if denom <= 0:
                continue
            hist, _ = np.histogram(arr, bins=bins)
            curves.append(hist.astype(float) * 100.0 / float(denom))
        if not curves:
            return None
        mat = np.vstack(curves)
        return {
            "mean": np.nanmean(mat, axis=0),
            "lo": np.nanpercentile(mat, 25, axis=0),
            "hi": np.nanpercentile(mat, 75, axis=0),
            "n_sessions": int(mat.shape[0]),
        }

    curve_in = _session_prop_curves(in_sessions)
    curve_out = _session_prop_curves(out_sessions)
    if curve_in is None or curve_out is None:
        print(f"[WARN] Skip H curve plot for metric={metric_tag}: insufficient session curves.")
        return

    for curve in (curve_in, curve_out):
        curve["mean"] = _gaussian_smooth_1d(curve["mean"])
        curve["lo"] = _gaussian_smooth_1d(curve["lo"])
        curve["hi"] = _gaussian_smooth_1d(curve["hi"])
        lo = np.minimum(curve["lo"], curve["hi"])
        hi = np.maximum(curve["lo"], curve["hi"])
        curve["lo"] = lo
        curve["hi"] = hi

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(centers, curve_in["mean"], color="#2B6CB0", linewidth=2.0, label=f"indoor mean (n={curve_in['n_sessions']} sessions)")
    plt.fill_between(centers, curve_in["lo"], curve_in["hi"], color="#2B6CB0", alpha=0.20, linewidth=0)
    plt.plot(centers, curve_out["mean"], color="#DD6B20", linewidth=2.0, label=f"outdoor mean (n={curve_out['n_sessions']} sessions)")
    plt.fill_between(centers, curve_out["lo"], curve_out["hi"], color="#DD6B20", alpha=0.20, linewidth=0)
    plt.xlabel(metric_tag)
    plt.ylabel("Percent of active cells in current session (%)")
    plt.title(f"H contribution distribution by session: indoor vs outdoor ({metric_tag})")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_png = Path(out_dir) / f"H_DIST_CURVE_{metric_tag}_indoor_vs_outdoor.png"
    _save_current_figure_png_svg(out_png, dpi=300)
    plt.close()
    print(f"[OK] Wrote H distribution curve plot: {out_png}")


def _write_forest_plot_per_metric(
    out_dir: Path,
    metric_tag: str,
    features: list[str],
    plot_data,
    resamples: int = 5000,
    file_suffix: str = "",
) -> None:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    try:
        import dabest
    except Exception:
        print("[WARN] dabest not available; skip forest/cliffs outputs.")
        return

    def _cliffs_delta(control, test):
        if hasattr(dabest, "cliffs_delta"):
            return float(dabest.cliffs_delta(control, test))
        c = np.asarray(control, dtype=float)
        t = np.asarray(test, dtype=float)
        c = c[np.isfinite(c)]
        t = t[np.isfinite(t)]
        if c.size == 0 or t.size == 0:
            return np.nan
        gt = (t[:, None] > c[None, :]).sum()
        lt = (t[:, None] < c[None, :]).sum()
        return float((gt - lt) / (t.size * c.size))

    contrasts = []
    labels = []
    effect_rows = []
    for name in features:
        arr_in = np.asarray(plot_data.frac_pooled.get("indoor", {}).get(name, []), dtype=float)
        arr_out = np.asarray(plot_data.frac_pooled.get("outdoor", {}).get(name, []), dtype=float)
        arr_in = arr_in[np.isfinite(arr_in)]
        arr_out = arr_out[np.isfinite(arr_out)]
        if arr_in.size == 0 or arr_out.size == 0:
            print(f"[WARN] Forest skip feature={name}: missing indoor or outdoor data.")
            continue

        indoor_name = f"{name}_indoor"
        outdoor_name = f"{name}_outdoor"
        df_feature = pd.DataFrame(
            {
                "value": np.concatenate([arr_in, arr_out]),
                "group": [indoor_name] * arr_in.size + [outdoor_name] * arr_out.size,
            }
        )

        try:
            d = dabest.load(
                data=df_feature,
                x="group",
                y="value",
                idx=(indoor_name, outdoor_name),
                resamples=resamples,
            )
        except Exception as e:
            print(f"[WARN] Forest skip feature={name}: {e}")
            continue

        effect_rows.append(
            {
                "feature": name,
                "n_indoor": int(arr_in.size),
                "n_outdoor": int(arr_out.size),
                "cliffs_delta": _cliffs_delta(arr_in, arr_out),
            }
        )
        contrasts.append(d)
        labels.append(name)

    suffix = f"_{file_suffix}" if file_suffix else ""
    if not contrasts:
        print(f"[WARN] No valid feature data for forest plot: metric={metric_tag}")
        return

    dabest.forest_plot(
        data=contrasts,
        labels=labels,
        effect_size="median_diff",
        ci_type="bca",
        horizontal=False,
    )
    plt.axhline(0.0, color="k", linewidth=1.0, alpha=0.8)
    plt.title("Outdoor − Indoor")
    plt.tight_layout()
    mean_forest_path = out_dir / f"FOREST_{metric_tag}{suffix}_by_feature.png"
    _save_current_figure_png_svg(mean_forest_path, dpi=300)
    plt.close()
    print(f"[OK] Wrote forest plot: {mean_forest_path}")

    dabest.forest_plot(
        data=contrasts,
        labels=labels,
        effect_size="cliffs_delta",
        ci_type="bca",
        horizontal=False,
    )
    plt.axhline(0.0, color="k", linewidth=1.0, alpha=0.8)
    plt.title("Outdoor − Indoor (Cliff's delta)")
    plt.tight_layout()
    cliffs_forest_path = out_dir / f"FOREST_{metric_tag}{suffix}_cliffs_delta.png"
    _save_current_figure_png_svg(cliffs_forest_path, dpi=300)
    plt.close()
    print(f"[OK] Wrote forest plot: {cliffs_forest_path}")

    effect_csv = out_dir / f"EFFECTS_{metric_tag}{suffix}_by_feature.csv"
    pd.DataFrame(effect_rows).to_csv(effect_csv, index=False)
    print(f"[OK] Wrote effect summary: {effect_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_base", type=str,
                    default=str(WEIGHTS_BASE))
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
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics to plot. Choices: rllr, delta_llhi, rllhi, rscc.",
    )
    ap.add_argument(
        "--composite_features",
        type=str,
        default=",".join(f"{k}={'+'.join(v)}" for k, v in VARIABLE_COMPOSITES.items()),
        help="Composite features, e.g. H=roll+yaw+pitch,Body=Position+Speed",
    )
    ap.add_argument(
        "--use_raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot raw ΔLLHI instead of shuffle-normalized z-scores. Default: z-scores.",
    )
    ap.add_argument(
        "--forward_modulated_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict each feature to neurons whose forward-search final model includes that feature. Default: on.",
    )
    ap.add_argument(
        "--include_unfit_cells",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include unfit cells as zeros in aggregation. Default: off.",
    )
    ap.add_argument(
        "--paired_fit_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict each feature to cells fit in both indoor and outdoor. Default: off.",
    )
    ap.add_argument(
        "--pyramidal_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use pyramidal-cell statistics only. Default: on.",
    )
    ap.add_argument(
        "--draw_suite_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw the main boxplot suite. Use --no-draw_suite_plots to skip figure rendering but keep summary CSVs.",
    )
    ap.add_argument(
        "--draw_forest_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw forest plots and compute their effect summaries.",
    )
    ap.add_argument(
        "--draw_h_dist_plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw the H histogram-style distribution plot.",
    )
    ap.add_argument(
        "--draw_h_dist_curve_plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw the H session-curve distribution plot.",
    )
    ap.add_argument(
        "--min_firing_rate_hz",
        type=float,
        default=DEFAULT_MIN_FIRING_RATE_HZ,
        help="Active-cell threshold used as H-distribution denominator (Hz).",
    )
    ap.add_argument(
        "--h_dist_n_bins",
        type=int,
        default=DEFAULT_H_DIST_N_BINS,
        help="Number of bins for H distribution histogram/curve plots.",
    )
    args = ap.parse_args()

    features_raw = [s.strip() for s in args.features.split(",") if s.strip()]
    # Accept common aliases while preserving canonical feature names used in CSV/config.
    feature_aliases = {"time": "Time", "headpose": "H", "h": "H"}
    features = []
    seen = set()
    for name in features_raw:
        canonical = feature_aliases.get(name.lower(), name)
        if canonical == "Time" and not INCLUDE_TIME_VARIABLE:
            continue
        if canonical not in seen:
            features.append(canonical)
            seen.add(canonical)
    composite_features = _parse_composite_spec(args.composite_features)
    composite_features = {k: v for k, v in composite_features.items() if k != "H"}
    selected_metrics = _parse_metrics(args.metrics)
    if not features:
        raise SystemExit("[FATAL] --features parsed to empty list.")

    extra_forest_features = ["Position", "Speed", "H"] + (["Time"] if INCLUDE_TIME_VARIABLE else [])
    all_covariate_forest_features = ["Position", "Speed", "roll", "yaw", "pitch"] + (["Time"] if INCLUDE_TIME_VARIABLE else [])

    plot_features = features[:]
    for comp_name in composite_features:
        if comp_name not in plot_features:
            plot_features.append(comp_name)

    load_features = plot_features[:]
    if "H" not in load_features:
        load_features.append("H")

    weights_base = Path(args.weights_base)
    dayid2cellinfo = build_dayid_to_cellinfo()
    sess_stats_by_metric = {}
    allowed_session_tokens = ["F5D10", "F5D3", "F5D2", "F5D7", "F6D7", "F6D10", "F6D3", "F6D8"]
    session_dirs = [p for p in weights_base.iterdir() if p.is_dir()]
    session_dirs = [
        p for p in session_dirs
        if any(token in p.name for token in allowed_session_tokens)
    ]

    for metric in selected_metrics:
        sess_stats = []
        for sess_dir in sorted(session_dirs):
            whitelist = None
            if args.forward_modulated_only:
                whitelist = load_forward_selected_neurons_rllr(sess_dir, features=features)
            if metric == "rllr":
                st = load_dropone_session_stats_rllr(
                    sess_dir,
                    features=load_features,
                    feature_neuron_whitelist=whitelist,
                    pyramidal_only=args.pyramidal_only,
                    use_zscore=True,
                )
            elif metric == "delta_llhi":
                st = load_dropone_llhi_session_stats_rllr(
                    sess_dir,
                    features=load_features,
                    feature_neuron_whitelist=whitelist,
                    pyramidal_only=args.pyramidal_only,
                    use_zscore=not args.use_raw,
                )
            elif metric == "rllhi":
                st = load_dropone_rllhi_session_stats_rllr(
                    sess_dir,
                    features=load_features,
                    feature_neuron_whitelist=whitelist,
                    pyramidal_only=args.pyramidal_only,
                )
            else:
                st = load_dropone_rscc_session_stats_rllr(
                    sess_dir,
                    features=load_features,
                    feature_neuron_whitelist=whitelist,
                    pyramidal_only=args.pyramidal_only,
                )
            if st is not None:
                sess_stats.append(st)
            elif args.forward_modulated_only:
                has_forward_matches = bool(whitelist) and any(len(v) > 0 for v in whitelist.values())
                if has_forward_matches:
                    print(
                        f"[SKIP] {sess_dir.name}: no valid drop-one rows remained after filtering "
                        "(possibly missing z-score columns)."
                    )
                else:
                    print(f"[SKIP] {sess_dir.name}: no neurons matched forward-search filter for requested features.")
        if not sess_stats:
            raise SystemExit(f"[FATAL] No sessions found with valid drop-one stats under: {weights_base}")
        h_rows = sum(len(st.frac_by_feature.get("H", {})) for st in sess_stats)
        if h_rows == 0:
            raise SystemExit(
                f"[FATAL] metric={metric}: no raw 'H' entries found in drop-one stats. "
                "H must be precomputed as a joint drop (roll+yaw+pitch+heading) in CSV."
            )
        sess_stats_by_metric[metric] = sess_stats

    summaries = []

    for metric in selected_metrics:
        sess_stats = sess_stats_by_metric[metric]
        if metric == "rllr":
            out_dir = weights_base / "RLLR_SUMMARY"
            plot_data = collect_dropone_plot_data_rllr(
                sess_stats,
                features=load_features,
                min_full_ll_gain=args.min_full_ll_gain,
                compute_delta=True,
                include_missing_cells=args.include_unfit_cells,
                paired_fit_only=args.paired_fit_only,
                include_paired_points=args.paired_fit_only,
                include_head_pose=HEAD_POSE_FEATURE in plot_features,
                head_pose_components=HEAD_POSE_COMPONENTS,
                composite_features=composite_features,
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
                paired_points=plot_data.paired_points if args.paired_fit_only else None,
                draw_plots=args.draw_suite_plots,
            )
            filter_label = f"full LL gain ≥ {args.min_full_ll_gain:g}"
            metric_threshold = float(args.min_full_ll_gain)
        elif metric == "delta_llhi":
            out_dir = weights_base / "LLHI_SUMMARY"
            plot_data = collect_dropone_plot_data_rllr(
                sess_stats,
                features=load_features,
                min_full_ll_gain=args.min_full_llhi,
                compute_delta=False,
                include_missing_cells=args.include_unfit_cells,
                paired_fit_only=args.paired_fit_only,
                include_paired_points=args.paired_fit_only,
                include_head_pose=HEAD_POSE_FEATURE in plot_features,
                head_pose_components=HEAD_POSE_COMPONENTS,
                composite_features=composite_features,
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
                paired_points=plot_data.paired_points if args.paired_fit_only else None,
                draw_plots=args.draw_suite_plots,
            )
            filter_label = f"full LLHI ≥ {args.min_full_llhi:g}"
            metric_threshold = float(args.min_full_llhi)
        elif metric == "rllhi":
            out_dir = weights_base / "RLLHI_SUMMARY"
            plot_data = collect_dropone_plot_data_rllr(
                sess_stats,
                features=load_features,
                min_full_ll_gain=args.min_full_llhi,
                compute_delta=False,
                include_missing_cells=args.include_unfit_cells,
                paired_fit_only=args.paired_fit_only,
                include_paired_points=args.paired_fit_only,
                include_head_pose=HEAD_POSE_FEATURE in plot_features,
                head_pose_components=HEAD_POSE_COMPONENTS,
                composite_features=composite_features,
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
                paired_points=plot_data.paired_points if args.paired_fit_only else None,
                draw_plots=args.draw_suite_plots,
            )
            filter_label = f"full LLHI ≥ {args.min_full_llhi:g}"
            metric_threshold = float(args.min_full_llhi)
        else:
            out_dir = weights_base / "RSCC_SUMMARY"
            plot_data = collect_dropone_plot_data_rllr(
                sess_stats,
                features=load_features,
                min_full_ll_gain=args.min_full_llhi,
                compute_delta=False,
                include_missing_cells=args.include_unfit_cells,
                paired_fit_only=args.paired_fit_only,
                include_paired_points=args.paired_fit_only,
                include_head_pose=HEAD_POSE_FEATURE in plot_features,
                head_pose_components=HEAD_POSE_COMPONENTS,
                composite_features=composite_features,
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
                metric_tag="rscc",
                ylabel="rSCC (ΔLLHI / ||Δ||₂)",
                title_metric="drop-one rSCC",
                summary_metric="dropone_rscc",
                include_delta_plot=False,
                use_shuffle_line=False,
                paired_points=plot_data.paired_points if args.paired_fit_only else None,
                draw_plots=args.draw_suite_plots,
            )
            filter_label = f"full LLHI ≥ {args.min_full_llhi:g}"
            metric_threshold = float(args.min_full_llhi)

        summaries.append((metric, out_dir, plot_data, summary_csv, filter_label, metric_threshold))

    for metric, out_dir, plot_data, summary_csv, filter_label, metric_threshold in summaries:
        full_feature_set = [f for f in all_covariate_forest_features if f in load_features]
        if full_feature_set and args.draw_forest_plots:
            _write_forest_plot_per_metric(
                out_dir=out_dir,
                metric_tag=metric,
                features=full_feature_set,
                plot_data=plot_data,
                file_suffix="Position_Speed_roll_yaw_pitch",
            )
        forest_features = [f for f in extra_forest_features if f in load_features]
        if forest_features and args.draw_forest_plots:
            _write_forest_plot_per_metric(
                out_dir=out_dir,
                metric_tag=metric,
                features=forest_features,
                plot_data=plot_data,
                file_suffix="Position_Speed_Headpose",
            )
        if args.draw_h_dist_plot:
            _write_h_distribution_plot(
                out_dir=out_dir,
                metric_tag=metric,
                session_stats=sess_stats_by_metric[metric],
                min_full_ll_gain=metric_threshold,
                include_missing_cells=bool(args.include_unfit_cells),
                min_firing_rate_hz=args.min_firing_rate_hz,
                pyramidal_only=bool(args.pyramidal_only),
                dayid2cellinfo=dayid2cellinfo,
                n_bins=args.h_dist_n_bins,
            )
        if args.draw_h_dist_curve_plot:
            _write_h_distribution_curve_plot(
                out_dir=out_dir,
                metric_tag=metric,
                session_stats=sess_stats_by_metric[metric],
                min_full_ll_gain=metric_threshold,
                include_missing_cells=bool(args.include_unfit_cells),
                min_firing_rate_hz=args.min_firing_rate_hz,
                pyramidal_only=bool(args.pyramidal_only),
                dayid2cellinfo=dayid2cellinfo,
                n_bins=args.h_dist_n_bins,
            )
        if args.draw_suite_plots or args.draw_forest_plots or args.draw_h_dist_plot or args.draw_h_dist_curve_plot:
            _ensure_svg_for_all_pngs(out_dir)
        print(f"[OK] Metric: {metric}")
        print(f"[OK] Sessions loaded: {len(sess_stats_by_metric[metric])}")
        print(f"[OK] Features used: {plot_features}")
        print(f"[OK] Cell scope: {'pyramidal only' if args.pyramidal_only else 'all cells'}")
        print(f"[OK] Filter: {filter_label}")
        print(
            f"[OK] Kept neurons: indoor {plot_data.kept_counts['indoor']}/{plot_data.total_counts['indoor']}, "
            f"outdoor {plot_data.kept_counts['outdoor']}/{plot_data.total_counts['outdoor']}",
        )
        print(f"[OK] Summary CSV: {summary_csv}")
        print(f"[OK] Wrote plots/summary to: {out_dir}")


if __name__ == "__main__":
    main()
