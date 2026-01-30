from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from glm_poisson_forward.config import VARS_ALL

HEAD_POSE_FEATURE = "head_pose"
HEAD_POSE_COMPONENTS = ("roll", "yaw", "pitch")
INDOOR_COLOR = "#1f77b4"
OUTDOOR_COLOR = "#ff7f0e"
PAIRED_LINE_COLOR = (1.0, 0.4, 0.4)

@dataclass
class DroponeSessionStats:
    session: str
    group: str
    full_ll_gain: Dict[int, float]
    frac_by_feature: Dict[str, Dict[int, float]]
    shuf_mean_by_feature: Dict[str, Dict[int, float]]
    shuf_std_by_feature: Dict[str, Dict[int, float]]
    all_neuron_ids: Optional[Sequence[int]] = None
    unfit_neuron_ids: Optional[Sequence[int]] = None


@dataclass
class DroponePlotData:
    frac_pooled: Dict[str, Dict[str, np.ndarray]]
    delta_pooled: Dict[str, Dict[str, np.ndarray]]
    full_pooled: Dict[str, Dict[str, np.ndarray]]
    shuf95_pooled: Dict[str, Dict[str, np.ndarray]]
    paired_points: Dict[str, List[Tuple[float, float]]]
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


def load_forward_selected_neurons(session_dir: Path, features: Iterable[str]) -> Dict[str, set[int]]:
    feat_list = [str(f).strip() for f in features]
    out: Dict[str, set[int]] = {f: set() for f in feat_list}

    selected_csv = Path(session_dir) / "selected_models.csv"
    rows = read_csv_dicts_safe(selected_csv)
    if not rows:
        return out

    for r in rows:
        neuron_name = str(r.get("neuron", "")).strip()
        final_model = str(r.get("final_model", "")).strip()
        if not neuron_name or not final_model:
            continue

        m = re.search(r"(\d+)$", neuron_name)
        if not m:
            continue
        try:
            ni = int(m.group(1)) - 1
        except Exception:
            continue
        if ni < 0:
            continue

        model_feats = [s.strip() for s in final_model.replace(",", "_").split("_") if s.strip()]
        for mf in model_feats:
            if mf in out:
                out[mf].add(ni)

    return out


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


def base_session_id(session_name: str) -> str:
    base = re.sub(r"[_-]?(indoor|outdoor)$", "", session_name, flags=re.IGNORECASE)
    return base if base else session_name


def compute_head_pose_map(
    frac_by_feature: Dict[str, Dict[int, float]],
    *,
    components: Sequence[str] = HEAD_POSE_COMPONENTS,
    include_missing: bool = False,
    neuron_ids: Optional[Iterable[int]] = None,
) -> Dict[int, float]:
    component_maps = [frac_by_feature.get(c, {}) for c in components]
    if include_missing:
        target_neurons = set(neuron_ids or [])
        if not target_neurons:
            for m in component_maps:
                target_neurons.update(m.keys())
    else:
        target_neurons = set()
        for m in component_maps:
            target_neurons.update(m.keys())

    head_pose: Dict[int, float] = {}
    for ni in target_neurons:
        vals = []
        for m in component_maps:
            v = m.get(ni, np.nan)
            if np.isfinite(v):
                vals.append(float(v))
            elif include_missing:
                vals.append(0.0)
        if not vals:
            if include_missing:
                head_pose[ni] = 0.0
            continue
        head_pose[ni] = float(np.mean(vals))
    return head_pose


def load_dropone_session_stats(
    session_dir: Path,
    features: Iterable[str],
    *,
    feature_neuron_whitelist: Optional[Dict[str, set[int]]] = None,
    use_zscore: bool = True,
) -> Optional[DroponeSessionStats]:
    session = Path(session_dir).name
    group = infer_group(session)
    if group is None:
        return None

    stats_dir = Path(session_dir) / "RLLR_STATS"
    full_csv = stats_dir / "full_rllr_pyr.csv"
    contrib_csv = stats_dir / "dropone_rllr_pyr.csv"

    contrib_rows = read_csv_dicts_safe(contrib_csv)
    if not contrib_rows:
        return None
    full_rows = read_csv_dicts_safe(full_csv)

    feat_list = [str(f).strip() for f in features]
    whitelist = None
    if feature_neuron_whitelist is not None:
        whitelist = {f: set(feature_neuron_whitelist.get(f, set())) for f in feat_list}

    full_ll: Dict[int, float] = {}
    unfit_neurons: List[int] = []
    for r in full_rows:
        ni = r.get("neuron_idx", r.get("neuron", None))
        dv = r.get("ll_gain", r.get("full_ll_gain", None))
        if ni is None or dv is None:
            continue
        try:
            idx = int(float(str(ni).strip()))
        except Exception:
            continue
        val = _safe_float(dv)
        if np.isfinite(val):
            full_ll[idx] = float(val)
            if val < 0:
                unfit_neurons.append(idx)

    frac: Dict[str, Dict[int, float]] = {f: {} for f in feat_list}
    shuf_mean: Dict[str, Dict[int, float]] = {f: {} for f in feat_list}
    shuf_std: Dict[str, Dict[int, float]] = {f: {} for f in feat_list}
    for r in contrib_rows:
        feat = str(r.get("feature", "")).strip()
        if feat not in frac:
            continue
        allowed = whitelist.get(feat, None) if whitelist is not None else None
        ni = r.get("neuron_idx", None)
        mu = r.get("rllr_shuf_mean", None)
        std = r.get("rllr_shuf_std", None)
        if use_zscore:
            fv = r.get("rllr_z", None)
            if fv is None:
                fv = r.get("rllr", None)
        else:
            fv = r.get("rllr", None)
        if ni is None or fv is None:
            continue
        try:
            idx = int(float(str(ni).strip()))
        except Exception:
            continue
        if allowed is not None and idx not in allowed:
            continue
        val = _safe_float(fv)
        if np.isfinite(val):
            frac[feat][idx] = float(val)
        mu_val = _safe_float(mu)
        if np.isfinite(mu_val):
            shuf_mean[feat][idx] = float(mu_val)
        std_val = _safe_float(std)
        if np.isfinite(std_val):
            shuf_std[feat][idx] = float(std_val)

    if all(len(frac[f]) == 0 for f in feat_list):
        return None

    return DroponeSessionStats(
        session=session,
        group=group,
        full_ll_gain={ni: v for ni, v in full_ll.items() if ni not in unfit_neurons},
        frac_by_feature=frac,
        shuf_mean_by_feature=shuf_mean,
        shuf_std_by_feature=shuf_std,
        all_neuron_ids=sorted(set(full_ll.keys()) | set(unfit_neurons)),
        unfit_neuron_ids=sorted(set(unfit_neurons)),
    )


def load_dropone_llhi_session_stats(
    session_dir: Path,
    features: Iterable[str],
    *,
    feature_neuron_whitelist: Optional[Dict[str, set[int]]] = None,
    use_zscore: bool = False,
) -> Optional[DroponeSessionStats]:
    session = Path(session_dir).name
    group = infer_group(session)
    if group is None:
        return None

    stats_dir = Path(session_dir) / "RLLR_STATS"
    full_csv = stats_dir / "full_llhi_pyr.csv"
    contrib_csv = stats_dir / "dropone_llhi_pyr.csv"

    contrib_rows = read_csv_dicts_safe(contrib_csv)
    if not contrib_rows:
        return None
    full_rows = read_csv_dicts_safe(full_csv)

    feat_list = [str(f).strip() for f in features]
    whitelist = None
    if feature_neuron_whitelist is not None:
        whitelist = {f: set(feature_neuron_whitelist.get(f, set())) for f in feat_list}

    unfit_neurons: List[int] = []
    full_llhi: Dict[int, float] = {}
    full_rllr_csv = stats_dir / "full_rllr_pyr.csv"
    full_rllr_rows = read_csv_dicts_safe(full_rllr_csv)
    for r in full_rllr_rows:
        ni = r.get("neuron_idx", r.get("neuron", None))
        dv = r.get("ll_gain", r.get("full_ll_gain", None))
        if ni is None or dv is None:
            continue
        try:
            idx = int(float(str(ni).strip()))
        except Exception:
            continue
        val = _safe_float(dv)
        if np.isfinite(val) and val < 0:
            unfit_neurons.append(idx)
    for r in full_rows:
        ni = r.get("neuron_idx", r.get("neuron", None))
        dv = r.get("llhi_full", r.get("full_llhi", None))
        if ni is None or dv is None:
            continue
        try:
            idx = int(float(str(ni).strip()))
        except Exception:
            continue
        val = _safe_float(dv)
        if np.isfinite(val):
            full_llhi[idx] = float(val)

    frac: Dict[str, Dict[int, float]] = {f: {} for f in feat_list}
    shuf_mean: Dict[str, Dict[int, float]] = {f: {} for f in feat_list}
    shuf_std: Dict[str, Dict[int, float]] = {f: {} for f in feat_list}
    for r in contrib_rows:
        feat = str(r.get("feature", "")).strip()
        if feat not in frac:
            continue
        allowed = whitelist.get(feat, None) if whitelist is not None else None
        ni = r.get("neuron_idx", None)
        if use_zscore:
            fv = r.get("delta_llhi_z", None)
            if fv is None:
                fv = r.get("delta_llhi", None)
        else:
            fv = r.get("delta_llhi", None)
        mu = r.get("delta_llhi_shuf_mean", None)
        std = r.get("delta_llhi_shuf_std", None)
        if ni is None or fv is None:
            continue
        try:
            idx = int(float(str(ni).strip()))
        except Exception:
            continue
        if allowed is not None and idx not in allowed:
            continue
        if idx in unfit_neurons:
            continue
        val = _safe_float(fv)
        if np.isfinite(val):
            frac[feat][idx] = float(val)
        mu_val = _safe_float(mu)
        if np.isfinite(mu_val):
            shuf_mean[feat][idx] = float(mu_val)
        std_val = _safe_float(std)
        if np.isfinite(std_val):
            shuf_std[feat][idx] = float(std_val)

    if all(len(frac[f]) == 0 for f in feat_list):
        return None

    return DroponeSessionStats(
        session=session,
        group=group,
        full_ll_gain={ni: v for ni, v in full_llhi.items() if ni not in unfit_neurons},
        frac_by_feature=frac,
        shuf_mean_by_feature=shuf_mean,
        shuf_std_by_feature=shuf_std,
        all_neuron_ids=sorted(set(full_llhi.keys()) | set(unfit_neurons)),
        unfit_neuron_ids=sorted(set(unfit_neurons)),
    )


def load_dropone_rllhi_session_stats(
    session_dir: Path,
    features: Iterable[str],
    *,
    feature_neuron_whitelist: Optional[Dict[str, set[int]]] = None,
) -> Optional[DroponeSessionStats]:
    llhi_stats = load_dropone_llhi_session_stats(
        session_dir,
        features,
        feature_neuron_whitelist=feature_neuron_whitelist,
        use_zscore=False,
    )
    if llhi_stats is None:
        return None

    feat_list = [str(f).strip() for f in features]
    rllhi_by_feature: Dict[str, Dict[int, float]] = {f: {} for f in feat_list}
    rllhi_shuf_mean: Dict[str, Dict[int, float]] = {f: {} for f in feat_list}
    rllhi_shuf_std: Dict[str, Dict[int, float]] = {f: {} for f in feat_list}
    for feat in feat_list:
        for ni, delta_val in llhi_stats.frac_by_feature.get(feat, {}).items():
            full_val = llhi_stats.full_ll_gain.get(ni, np.nan)
            if not np.isfinite(full_val) or full_val == 0 or not np.isfinite(delta_val):
                continue
            rllhi_val = float(delta_val) / float(full_val)
            rllhi_by_feature[feat][ni] = rllhi_val

    if all(len(rllhi_by_feature[f]) == 0 for f in feat_list):
        return None

    return DroponeSessionStats(
        session=llhi_stats.session,
        group=llhi_stats.group,
        full_ll_gain=llhi_stats.full_ll_gain,
        frac_by_feature=rllhi_by_feature,
        shuf_mean_by_feature=rllhi_shuf_mean,
        shuf_std_by_feature=rllhi_shuf_std,
        unfit_neuron_ids=llhi_stats.unfit_neuron_ids,
    )


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
    min_full_ll_gain: float,
    compute_delta: bool = True,
    include_missing_cells: bool = False,
    include_head_pose: bool = False,
    include_paired_points: bool = False,
    paired_fit_only: bool = False,
    head_pose_components: Sequence[str] = HEAD_POSE_COMPONENTS,
) -> DroponePlotData:
    feat_list = [str(f).strip() for f in features]
    if include_head_pose and HEAD_POSE_FEATURE not in feat_list:
        feat_list.append(HEAD_POSE_FEATURE)

    paired_fit_map: Dict[str, Dict[str, set[int]]] = {}
    if paired_fit_only:
        per_feature_group: Dict[str, Dict[str, Dict[str, set[int]]]] = {f: {} for f in feat_list}
        for st in session_stats:
            g = st.group
            if g not in ("indoor", "outdoor"):
                continue
            base_id = base_session_id(st.session)
            head_pose_map: Dict[int, float] = {}
            if include_head_pose:
                head_pose_map = compute_head_pose_map(
                    st.frac_by_feature,
                    components=head_pose_components,
                    include_missing=False,
                    neuron_ids=None,
                )
            for f in feat_list:
                if f == HEAD_POSE_FEATURE:
                    m = head_pose_map
                else:
                    m = st.frac_by_feature.get(f, {})
                if not m:
                    continue
                ids = {int(ni) for ni, val in m.items() if np.isfinite(val)}
                if not ids:
                    continue
                per_feature_group.setdefault(f, {}).setdefault(base_id, {}).setdefault(g, set()).update(ids)

        paired_fit_map = {f: {} for f in feat_list}
        for feat, base_map in per_feature_group.items():
            for base_id, group_map in base_map.items():
                indoor_ids = group_map.get("indoor", set())
                outdoor_ids = group_map.get("outdoor", set())
                paired_ids = indoor_ids & outdoor_ids
                if paired_ids:
                    paired_fit_map[feat][base_id] = paired_ids

    frac_by_session: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"indoor": {f: {} for f in feat_list}, "outdoor": {f: {} for f in feat_list}}
    full_by_session: Dict[str, Dict[str, np.ndarray]] = {"indoor": {}, "outdoor": {}}
    delta_by_session: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"indoor": {f: {} for f in feat_list}, "outdoor": {f: {} for f in feat_list}}
    shuf95_by_session: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"indoor": {f: {} for f in feat_list}, "outdoor": {f: {} for f in feat_list}}
    paired_points_raw: Dict[str, Dict[Tuple[str, int], Dict[str, float]]] = {f: {} for f in feat_list}

    kept_counts = {"indoor": 0, "outdoor": 0}
    total_counts = {"indoor": 0, "outdoor": 0}

    for st in session_stats:
        g = st.group
        if g not in ("indoor", "outdoor"):
            continue
        base_id = base_session_id(st.session)

        unfit_neurons = set(st.unfit_neuron_ids or [])
        if include_missing_cells:
            all_neurons = set(st.all_neuron_ids or [])
            if not all_neurons:
                all_neurons = set(st.full_ll_gain.keys()) | unfit_neurons
            total_counts[g] += len(all_neurons)
        else:
            all_neurons = set(st.full_ll_gain.keys())
            for ni, ll_gain in st.full_ll_gain.items():
                if ni in unfit_neurons:
                    continue
                if np.isfinite(ll_gain):
                    total_counts[g] += 1

        eligible = set()
        for ni in all_neurons:
            if ni in unfit_neurons:
                if include_missing_cells:
                    eligible.add(int(ni))
                    kept_counts[g] += 1
                continue
            ll_gain = st.full_ll_gain.get(ni, 0.0)
            if not np.isfinite(ll_gain):
                ll_gain = 0.0
            if ll_gain >= min_full_ll_gain:
                eligible.add(int(ni))
                kept_counts[g] += 1

        if eligible:
            arr_full = np.asarray(
                [
                    0.0 if ni in unfit_neurons else st.full_ll_gain.get(ni, 0.0)
                    for ni in eligible
                    if np.isfinite(st.full_ll_gain.get(ni, 0.0)) or ni in unfit_neurons
                ],
                dtype=float,
            )
            arr_full = arr_full[np.isfinite(arr_full)]
            if arr_full.size:
                full_by_session[g][st.session] = arr_full

        head_pose_map: Dict[int, float] = {}
        if include_head_pose:
            head_pose_map = compute_head_pose_map(
                st.frac_by_feature,
                components=head_pose_components,
                include_missing=include_missing_cells,
                neuron_ids=eligible,
            )

        for f in feat_list:
            if f == HEAD_POSE_FEATURE:
                m = head_pose_map
            else:
                m = st.frac_by_feature.get(f, {})
            if not eligible:
                continue
            eligible_feature = eligible
            if paired_fit_only:
                paired_ids = paired_fit_map.get(f, {}).get(base_id, set())
                eligible_feature = eligible & paired_ids
                if not eligible_feature:
                    continue

            vals_frac = []
            vals_delta = []
            vals_shuf95 = []
            for ni in eligible_feature:
                fracv = m.get(ni, 0.0 if include_missing_cells else np.nan)
                ll_gain = 0.0 if ni in unfit_neurons else st.full_ll_gain.get(ni, 0.0)
                if np.isfinite(fracv) and np.isfinite(ll_gain):
                    vals_frac.append(float(fracv))
                    if compute_delta:
                        vals_delta.append(float(fracv) * float(ll_gain))
                if f != HEAD_POSE_FEATURE:
                    mu = st.shuf_mean_by_feature.get(f, {}).get(ni, np.nan)
                    std = st.shuf_std_by_feature.get(f, {}).get(ni, np.nan)
                    if np.isfinite(mu) and np.isfinite(std) and std > 0:
                        vals_shuf95.append(float(mu + 1.644854 * std))

                if include_paired_points:
                    base_id = base_session_id(st.session)
                    key = (base_id, int(ni))
                    if key not in paired_points_raw[f]:
                        paired_points_raw[f][key] = {}
                    paired_points_raw[f][key][g] = float(fracv) if np.isfinite(fracv) else 0.0

            arr_frac = np.asarray(vals_frac, dtype=float)
            arr_frac = arr_frac[np.isfinite(arr_frac)]
            if arr_frac.size:
                frac_by_session[g][f][st.session] = arr_frac

            if compute_delta:
                arr_delta = np.asarray(vals_delta, dtype=float)
                arr_delta = arr_delta[np.isfinite(arr_delta)]
                if arr_delta.size:
                    delta_by_session[g][f][st.session] = arr_delta
            arr_shuf95 = np.asarray(vals_shuf95, dtype=float)
            arr_shuf95 = arr_shuf95[np.isfinite(arr_shuf95)]
            if arr_shuf95.size:
                shuf95_by_session[g][f][st.session] = arr_shuf95

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
    shuf95_pooled = {
        "indoor": {f: pooled_all(shuf95_by_session["indoor"][f]) for f in feat_list},
        "outdoor": {f: pooled_all(shuf95_by_session["outdoor"][f]) for f in feat_list},
    }

    paired_points: Dict[str, List[Tuple[float, float]]] = {f: [] for f in feat_list}
    if include_paired_points:
        for feat, key_map in paired_points_raw.items():
            pairs = []
            for values in key_map.values():
                if "indoor" in values and "outdoor" in values:
                    pairs.append((values["indoor"], values["outdoor"]))
            paired_points[feat] = pairs

    return DroponePlotData(
        frac_pooled=frac_pooled,
        delta_pooled=delta_pooled,
        full_pooled=full_pooled,
        shuf95_pooled=shuf95_pooled,
        paired_points=paired_points,
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
                continue

    for key in ["whiskers", "caps", "medians", "boxes"]:
        if key not in bp:
            continue
        _collect(bp[key])

    if not ys:
        return

    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    pad = (y_max - y_min) * pad_frac if y_max > y_min else 1.0
    ax.set_ylim(y_min - pad, y_max + pad)


def plot_combined_indoor_outdoor(
    out_png: Path,
    *,
    title: str,
    ylabel: str,
    features: Sequence[str],
    data_in: Dict[str, np.ndarray],
    data_out: Dict[str, np.ndarray],
    paired_points: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    seed: int,
    max_scatter_points: int,
    ylim_pad_frac: float,
    shuffle95_line: float | None = None,
):
    rng = np.random.default_rng(seed)
    features_sorted = list(features)

    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(features_sorted))
    vals_in = [data_in.get(f, np.array([], dtype=float)) for f in features_sorted]
    vals_out = [data_out.get(f, np.array([], dtype=float)) for f in features_sorted]

    bp_in = ax.boxplot(vals_in, positions=x - 0.18, widths=0.3, patch_artist=True, showfliers=False)
    bp_out = ax.boxplot(vals_out, positions=x + 0.18, widths=0.3, patch_artist=True, showfliers=False)

    for box in bp_in["boxes"]:
        box.set_facecolor(INDOOR_COLOR)
        box.set_alpha(0.55)
    for box in bp_out["boxes"]:
        box.set_facecolor(OUTDOOR_COLOR)
        box.set_alpha(0.55)

    for i, f in enumerate(features_sorted):
        y = np.asarray(data_in.get(f, np.array([], dtype=float)), dtype=float)
        y = y[np.isfinite(y)]
        if max_scatter_points and y.size > max_scatter_points:
            y = rng.choice(y, size=max_scatter_points, replace=False)
        ax.scatter(x[i] - 0.18 + jitter_points(rng, y.size, scale=0), y, s=12, alpha=0.70, color="k", zorder=3)

        y = np.asarray(data_out.get(f, np.array([], dtype=float)), dtype=float)
        y = y[np.isfinite(y)]
        if max_scatter_points and y.size > max_scatter_points:
            y = rng.choice(y, size=max_scatter_points, replace=False)
        ax.scatter(x[i] + 0.18 + jitter_points(rng, y.size, scale=0), y, s=12, alpha=0.70, color="k", zorder=3)

        if paired_points is not None:
            pairs = paired_points.get(f, [])
            for yin, yout in pairs:
                if not (np.isfinite(yin) and np.isfinite(yout)):
                    continue
                ax.plot(
                    [x[i] - 0.18, x[i] + 0.18],
                    [yin, yout],
                    color=PAIRED_LINE_COLOR,
                    alpha=0.35,
                    linewidth=0.6,
                    zorder=2,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(features_sorted, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0.0, linewidth=1.0, alpha=0.35)

    ax.legend(
        handles=[
            Patch(facecolor=INDOOR_COLOR, edgecolor=INDOOR_COLOR, alpha=0.55, label="indoor"),
            Patch(facecolor=OUTDOOR_COLOR, edgecolor=OUTDOOR_COLOR, alpha=0.55, label="outdoor"),
        ],
        frameon=False,
        loc="upper right",
    )

    if shuffle95_line is not None and np.isfinite(shuffle95_line):
        ax.axhline(shuffle95_line, linestyle="--", linewidth=1.2, color="red", alpha=0.8)

    set_ylim_from_boxplot({"whiskers": bp_in["whiskers"] + bp_out["whiskers"], "caps": bp_in["caps"] + bp_out["caps"], "medians": bp_in["medians"] + bp_out["medians"], "boxes": bp_in["boxes"] + bp_out["boxes"]}, ax, pad_frac=ylim_pad_frac)

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
    shuffle95_line: float | None = None,
):
    rng = np.random.default_rng(seed)
    features_sorted = list(features_sorted)

    fig, ax = plt.subplots(figsize=(9, 4))

    x = np.arange(len(features_sorted))
    vals = [data.get(f, np.array([], dtype=float)) for f in features_sorted]

    bp = ax.boxplot(vals, positions=x, widths=0.5, patch_artist=True, showfliers=False)

    fc = INDOOR_COLOR if group_label == "indoor" else OUTDOOR_COLOR
    edge_c = INDOOR_COLOR if group_label == "indoor" else OUTDOOR_COLOR

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

    if shuffle95_line is not None and np.isfinite(shuffle95_line):
        ax.axhline(shuffle95_line, linestyle="--", linewidth=1.2, color="red", alpha=0.8)

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


def suffix_for_threshold(min_full_ll_gain: float) -> str:
    return f"_FULLGE{min_full_ll_gain:g}".replace(".", "p")


def plot_dropone_suite(
    out_dir: Path,
    *,
    features: Sequence[str],
    plot_data: DroponePlotData,
    min_full_ll_gain: float,
    seed: int,
    max_scatter_points: int,
    ylim_pad_frac: float,
    use_zscore: bool = False,
    metric_tag: str = "rllr",
    ylabel: str = "rLLR",
    title_metric: str = "drop-one contribution",
    summary_metric: str = "dropone_rllr",
    include_delta_plot: bool = True,
    use_shuffle_line: bool = True,
    paired_points: Optional[Dict[str, List[Tuple[float, float]]]] = None,
) -> Path:
    suffix = suffix_for_threshold(min_full_ll_gain)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_zscore:
        ylabel = f"{ylabel} z-score"
        title_metric = f"{title_metric} (z)"

    z95 = 1.644854
    if use_zscore and use_shuffle_line:
        shuffle95_combined = z95
        shuffle95_indoor = z95
        shuffle95_outdoor = z95
    elif use_shuffle_line:
        combined_vals = []
        indoor_vals = []
        outdoor_vals = []
        for f in features:
            if plot_data.shuf95_pooled["indoor"][f].size:
                indoor_vals.append(plot_data.shuf95_pooled["indoor"][f])
                combined_vals.append(plot_data.shuf95_pooled["indoor"][f])
            if plot_data.shuf95_pooled["outdoor"][f].size:
                outdoor_vals.append(plot_data.shuf95_pooled["outdoor"][f])
                combined_vals.append(plot_data.shuf95_pooled["outdoor"][f])
        combined = np.concatenate(combined_vals, axis=0) if combined_vals else np.array([], dtype=float)
        indoor = np.concatenate(indoor_vals, axis=0) if indoor_vals else np.array([], dtype=float)
        outdoor = np.concatenate(outdoor_vals, axis=0) if outdoor_vals else np.array([], dtype=float)
        shuffle95_combined = float(np.nanmedian(combined)) if combined.size else None
        shuffle95_indoor = float(np.nanmedian(indoor)) if indoor.size else None
        shuffle95_outdoor = float(np.nanmedian(outdoor)) if outdoor.size else None
    else:
        shuffle95_combined = None
        shuffle95_indoor = None
        shuffle95_outdoor = None

    plot_combined_indoor_outdoor(
        out_dir / f"BOX_{metric_tag}_indoor_outdoor{suffix}.png",
        title=f"{title_metric} | full ≥ {min_full_ll_gain:g}",
        ylabel=ylabel,
        features=features,
        data_in=plot_data.frac_pooled["indoor"],
        data_out=plot_data.frac_pooled["outdoor"],
        paired_points=paired_points,
        seed=seed,
        max_scatter_points=max_scatter_points,
        ylim_pad_frac=ylim_pad_frac,
        shuffle95_line=shuffle95_combined,
    )

    indoor_means = group_feature_means(plot_data.frac_pooled["indoor"], features)
    outdoor_means = group_feature_means(plot_data.frac_pooled["outdoor"], features)

    features_indoor_sorted = sorted(features, key=lambda f: indoor_means.get(f, float("-inf")), reverse=True)
    features_outdoor_sorted = sorted(features, key=lambda f: outdoor_means.get(f, float("-inf")), reverse=True)

    plot_single_group_sorted(
        out_dir / f"BOX_{metric_tag}_indoor_sorted{suffix}.png",
        title=f"Indoor | {title_metric}",
        ylabel=ylabel,
        group_label="indoor",
        features_sorted=features_indoor_sorted,
        data=plot_data.frac_pooled["indoor"],
        seed=seed,
        max_scatter_points=max_scatter_points,
        ylim_pad_frac=ylim_pad_frac,
        shuffle95_line=shuffle95_indoor,
    )

    plot_single_group_sorted(
        out_dir / f"BOX_{metric_tag}_outdoor_sorted{suffix}.png",
        title=f"Outdoor | {title_metric}",
        ylabel=ylabel,
        group_label="outdoor",
        features_sorted=features_outdoor_sorted,
        data=plot_data.frac_pooled["outdoor"],
        seed=seed,
        max_scatter_points=max_scatter_points,
        ylim_pad_frac=ylim_pad_frac,
        shuffle95_line=shuffle95_outdoor,
    )

    if plot_data.has_full:
        plot_combined_indoor_outdoor(
            out_dir / f"BOX_full_ll_gain_indoor_outdoor{suffix}.png",
            title=f"Full LL gain | ≥ {min_full_ll_gain:g}",
            ylabel="LL gain (full - intercept)",
            features=["FULL"],
            data_in={"FULL": plot_data.full_pooled["indoor"]["FULL"]},
            data_out={"FULL": plot_data.full_pooled["outdoor"]["FULL"]},
            seed=seed,
            max_scatter_points=max_scatter_points,
            ylim_pad_frac=ylim_pad_frac,
        )

        if include_delta_plot and not use_zscore:
            plot_combined_indoor_outdoor(
                out_dir / f"BOX_delta_ll_indoor_outdoor{suffix}.png",
                title="Drop-one ΔLL",
                ylabel="ΔLL",
                features=features,
                data_in=plot_data.delta_pooled["indoor"],
                data_out=plot_data.delta_pooled["outdoor"],
                seed=seed,
                max_scatter_points=max_scatter_points,
                ylim_pad_frac=ylim_pad_frac,
            )

    out_csv = out_dir / f"boxplot_dropone_{metric_tag}_summary{suffix}.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["metric", "group", "feature", "n", "mean", "median", "min_full_ll_gain", "features_used"],
        )
        w.writeheader()
        for feat in features:
            for grp in ["indoor", "outdoor"]:
                arr = np.asarray(plot_data.frac_pooled[grp][feat], dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                w.writerow({
                    "metric": summary_metric,
                    "group": grp,
                    "feature": feat,
                    "n": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "min_full_ll_gain": min_full_ll_gain,
                    "features_used": ",".join(features),
                })
        if plot_data.has_full:
            for grp in ["indoor", "outdoor"]:
                arr = np.asarray(plot_data.full_pooled[grp]["FULL"], dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                w.writerow({
                    "metric": "full_ll_gain",
                    "group": grp,
                    "feature": "FULL",
                    "n": int(arr.size),
                    "mean": float(np.mean(arr)),
                    "median": float(np.median(arr)),
                    "min_full_ll_gain": min_full_ll_gain,
                    "features_used": ",".join(features),
                })

    return out_csv


# ----------------------------------------------------------------------
# Summary figure (hierarchical bootstrap)
# ----------------------------------------------------------------------

def plot_summary_figure(
    out_png: Path,
    title: str,
    full_stat: Tuple[float, float, float],
    feature_stats: Dict[str, Tuple[float, float, float]],
    *,
    full_ylabel: str = "LL gain (full - intercept)",
    feature_ylabel: str = "rLLR",
    features: Optional[Sequence[str]] = None,
):
    out_png.parent.mkdir(parents=True, exist_ok=True)

    features = list(features) if features is not None else VARS_ALL[:]
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
    ax0.set_ylabel(full_ylabel)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.bar(np.arange(len(features)), means, yerr=yerr, capsize=4, linewidth=1.2, edgecolor="black")
    ax1.set_xticks(np.arange(len(features)))
    ax1.set_xticklabels(features, rotation=0)
    ax1.set_ylabel(feature_ylabel)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=250)
    plt.close(fig)
