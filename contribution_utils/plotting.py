from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from glm_poisson_forward.config import VARS_ALL


# ----------------------------------------------------------------------
# Shared data containers
# ----------------------------------------------------------------------
@dataclass
class DroponeSessionStats:
    session: str
    group: str  # indoor/outdoor
    full_devexpl: Dict[int, float]  # neuron_idx -> full DevExpl
    frac_by_feature: Dict[str, Dict[int, float]]  # feature -> neuron_idx -> frac


@dataclass
class DroponePlotData:
    frac_pooled: Dict[str, Dict[str, np.ndarray]]
    delta_pooled: Dict[str, Dict[str, np.ndarray]]
    full_pooled: Dict[str, Dict[str, np.ndarray]]
    kept_counts: Dict[str, int]
    total_counts: Dict[str, int]

    @property
    def has_full(self) -> bool:
        return any(
            self.full_pooled.get(g, {}).get("FULL", np.array([], dtype=float)).size > 0 for g in ("indoor", "outdoor")
        )


# ----------------------------------------------------------------------
# Safe CSV helpers
# ----------------------------------------------------------------------
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
    """
    Read a CSV file into a list of dicts while being tolerant to BOM/null bytes and empty rows.
    """
    path = Path(path)
    if (not path.exists()) or path.stat().st_size < 1:
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        raw = f.read()
    if "\x00" in raw:
        raw = raw.replace("\x00", "")
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


# ----------------------------------------------------------------------
# Loading + aggregation
# ----------------------------------------------------------------------
def infer_group(session_name: str) -> Optional[str]:
    s = session_name.lower()
    if "indoor" in s:
        return "indoor"
    if "outdoor" in s:
        return "outdoor"
    return None


def load_dropone_session_stats(session_dir: Path, features: Iterable[str]) -> Optional[DroponeSessionStats]:
    session = Path(session_dir).name
    group = infer_group(session)
    if group is None:
        return None

    stats_dir = Path(session_dir) / "DROPONE_STATS"
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

    feat_list = [str(f).strip() for f in features]
    frac: Dict[str, Dict[int, float]] = {f: {} for f in feat_list}
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

    if all(len(frac[f]) == 0 for f in feat_list):
        return None

    return DroponeSessionStats(session=session, group=group, full_devexpl=full_dev, frac_by_feature=frac)


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


def collect_dropone_plot_data(
    session_stats: Iterable[DroponeSessionStats],
    *,
    features: Sequence[str],
    min_full_devexpl: float,
) -> DroponePlotData:
    feat_list = [str(f).strip() for f in features]

    frac_by_session: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"indoor": {f: {} for f in feat_list}, "outdoor": {f: {} for f in feat_list}}
    full_by_session: Dict[str, Dict[str, np.ndarray]] = {"indoor": {}, "outdoor": {}}
    delta_by_session: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"indoor": {f: {} for f in feat_list}, "outdoor": {f: {} for f in feat_list}}

    kept_counts = {"indoor": 0, "outdoor": 0}
    total_counts = {"indoor": 0, "outdoor": 0}

    for st in session_stats:
        g = st.group
        if g not in ("indoor", "outdoor"):
            continue

        eligible = set()
        for ni, dv in st.full_devexpl.items():
            if np.isfinite(dv):
                total_counts[g] += 1
                if dv >= min_full_devexpl:
                    eligible.add(int(ni))
                    kept_counts[g] += 1

        if eligible:
            arr_full = np.asarray(
                [st.full_devexpl[ni] for ni in eligible if np.isfinite(st.full_devexpl.get(ni, np.nan))],
                dtype=float,
            )
            arr_full = arr_full[np.isfinite(arr_full)]
            if arr_full.size:
                full_by_session[g][st.session] = arr_full

        for f in feat_list:
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

    frac_pooled = {
        "indoor": {f: pooled_all(frac_by_session["indoor"][f]) for f in feat_list},
        "outdoor": {f: pooled_all(frac_by_session["outdoor"][f]) for f in feat_list},
    }
    delta_pooled = {
        "indoor": {f: pooled_all(delta_by_session["indoor"][f]) for f in feat_list},
        "outdoor": {f: pooled_all(delta_by_session["outdoor"][f]) for f in feat_list},
    }
    full_pooled = {
        "indoor": {"FULL": pooled_all(full_by_session["indoor"])},
        "outdoor": {"FULL": pooled_all(full_by_session["outdoor"])},
    }

    return DroponePlotData(
        frac_pooled=frac_pooled,
        delta_pooled=delta_pooled,
        full_pooled=full_pooled,
        kept_counts=kept_counts,
        total_counts=total_counts,
    )


# ----------------------------------------------------------------------
# Plotting primitives
# ----------------------------------------------------------------------
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


def plot_combined_indoor_outdoor(
    out_png: Path,
    *,
    title: str,
    ylabel: str,
    features: Sequence[str],
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

    indoor_fc = "#b0b0b0"  # gray
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
    features_sorted: Sequence[str],
    data: Dict[str, np.ndarray],
    seed: int,
    max_scatter_points: int,
    ylim_pad_frac: float,
):
    rng = np.random.default_rng(seed)

    x = np.arange(len(features_sorted), dtype=float)
    width = 0.50

    fc = "#b0b0b0" if group_label == "indoor" else "#8fd19e"
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


def group_feature_means(data: Dict[str, np.ndarray], features: Sequence[str]) -> Dict[str, float]:
    out = {}
    for f in features:
        a = np.asarray(data.get(f, np.array([], dtype=float)), dtype=float)
        a = a[np.isfinite(a)]
        out[f] = float(np.mean(a)) if a.size else float("-inf")
    return out


def suffix_for_threshold(min_full_devexpl: float) -> str:
    return f"_FULLGE{min_full_devexpl:g}".replace(".", "p")


def plot_dropone_suite(
    out_dir: Path,
    *,
    features: Sequence[str],
    plot_data: DroponePlotData,
    min_full_devexpl: float,
    seed: int,
    max_scatter_points: int,
    ylim_pad_frac: float,
) -> Path:
    suffix = suffix_for_threshold(min_full_devexpl)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_combined_indoor_outdoor(
        out_dir / f"BOX_dropone_frac_indoor_vs_outdoor{suffix}.png",
        title=f"Drop-one fraction (pyramidal; full DevExpl ≥ {min_full_devexpl:g}) | whiskers/caps + jitter",
        ylabel="frac(full DevExpl)",
        features=features,
        data_in=plot_data.frac_pooled["indoor"],
        data_out=plot_data.frac_pooled["outdoor"],
        seed=seed,
        max_scatter_points=max_scatter_points,
        ylim_pad_frac=ylim_pad_frac,
    )

    indoor_means = group_feature_means(plot_data.frac_pooled["indoor"], features)
    outdoor_means = group_feature_means(plot_data.frac_pooled["outdoor"], features)

    features_indoor_sorted = sorted(features, key=lambda f: indoor_means.get(f, float("-inf")), reverse=True)
    features_outdoor_sorted = sorted(features, key=lambda f: outdoor_means.get(f, float("-inf")), reverse=True)

    plot_single_group_sorted(
        out_dir / f"BOX_dropone_frac_indoor_only_sorted{suffix}.png",
        title=f"Indoor only (sorted by mean) | drop-one fraction | full DevExpl ≥ {min_full_devexpl:g}",
        ylabel="frac(full DevExpl)",
        group_label="indoor",
        features_sorted=features_indoor_sorted,
        data=plot_data.frac_pooled["indoor"],
        seed=seed,
        max_scatter_points=max_scatter_points,
        ylim_pad_frac=ylim_pad_frac,
    )

    plot_single_group_sorted(
        out_dir / f"BOX_dropone_frac_outdoor_only_sorted{suffix}.png",
        title=f"Outdoor only (sorted by mean) | drop-one fraction | full DevExpl ≥ {min_full_devexpl:g}",
        ylabel="frac(full DevExpl)",
        group_label="outdoor",
        features_sorted=features_outdoor_sorted,
        data=plot_data.frac_pooled["outdoor"],
        seed=seed,
        max_scatter_points=max_scatter_points,
        ylim_pad_frac=ylim_pad_frac,
    )

    if plot_data.has_full:
        plot_combined_indoor_outdoor(
            out_dir / f"BOX_full_devexpl_indoor_vs_outdoor{suffix}.png",
            title=f"Full DevExpl (pyramidal; ≥ {min_full_devexpl:g}) | whiskers/caps + jitter",
            ylabel="DevExpl (full model)",
            features=["FULL"],
            data_in={"FULL": plot_data.full_pooled["indoor"]["FULL"]},
            data_out={"FULL": plot_data.full_pooled["outdoor"]["FULL"]},
            seed=seed,
            max_scatter_points=max_scatter_points,
            ylim_pad_frac=ylim_pad_frac,
        )

        plot_combined_indoor_outdoor(
            out_dir / f"BOX_dropone_delta_indoor_vs_outdoor{suffix}.png",
            title=f"Drop-one ΔDevExpl (= frac×full) (pyramidal; ≥ {min_full_devexpl:g}) | whiskers/caps + jitter",
            ylabel="ΔDevExpl",
            features=features,
            data_in=plot_data.delta_pooled["indoor"],
            data_out=plot_data.delta_pooled["outdoor"],
            seed=seed,
            max_scatter_points=max_scatter_points,
            ylim_pad_frac=ylim_pad_frac,
        )

    out_csv = out_dir / f"boxplot_dropone_summary{suffix}.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["metric", "group", "feature", "n", "mean", "median", "min_full_devexpl", "features_used"],
        )
        w.writeheader()
        for feat in features:
            for grp in ["indoor", "outdoor"]:
                arr = np.asarray(plot_data.frac_pooled[grp][feat], dtype=float)
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
                    "min_full_devexpl": min_full_devexpl,
                    "features_used": ",".join(features),
                })
        if plot_data.has_full:
            for grp in ["indoor", "outdoor"]:
                arr = np.asarray(plot_data.full_pooled[grp]["FULL"], dtype=float)
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
                    "min_full_devexpl": min_full_devexpl,
                    "features_used": ",".join(features),
                })

    return out_csv


# ----------------------------------------------------------------------
# Legacy summary figure (hierarchical bootstrap)
# ----------------------------------------------------------------------
def plot_summary_figure(
    out_png: Path,
    title: str,
    full_stat: Tuple[float, float, float],
    feature_stats: Dict[str, Tuple[float, float, float]],
):
    """
    Two-panel figure:
      left: full DevExpl
      right: per-feature contribution fraction
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    features = VARS_ALL[:]  # keep order
    means = [feature_stats[f][0] for f in features]
    los = [feature_stats[f][1] for f in features]
    his = [feature_stats[f][2] for f in features]

    yerr_low = np.array(means) - np.array(los)
    yerr_high = np.array(his) - np.array(means)
    yerr = np.vstack([yerr_low, yerr_high])

    fig = plt.figure(figsize=(10, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    m, lo, hi = full_stat
    ax0.bar([0], [m], width=0.6, edgecolor="black", linewidth=0.8)
    ax0.errorbar([0], [m], yerr=[[m - lo], [hi - m]], fmt="none", capsize=4, linewidth=1.2)
    ax0.set_xticks([0])
    ax0.set_xticklabels(["Full\nmodel"])
    ax0.set_ylabel("Deviance explained")
    ax0.set_title("Full model")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(features))
    ax1.bar(x, means, width=0.65, edgecolor="black", linewidth=0.8)
    ax1.errorbar(x, means, yerr=yerr, fmt="none", capsize=4, linewidth=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=0)
    ax1.set_ylabel("Fraction of full-model dev.")
    ax1.set_title("Drop-one contribution (pyramidal only)")
    ax1.axhline(0.0, linewidth=0.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=250)
    plt.close(fig)
