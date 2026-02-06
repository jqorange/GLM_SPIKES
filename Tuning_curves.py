# -*- coding: utf-8 -*-
"""
plot_feature_tuning_curves_FULL_FIT_coef_only.py

Goal
----
Only read FULL_FIT GLM coefficients (10-fold averaged) and plot model-based tuning curves.
No IMU/DLC/Position raw inputs are used.

Model-based tuning (Poisson GLM, log link)
------------------------------------------
eta = intercept + sum_v effect_v[bin_v]
lambda(count per bin) = exp(eta)
rate(spikes/s) = exp(eta) / BIN_SEC

Baseline (no raw data available)
-------------------------------
We cannot compute E_t[effect_j[bin_j(t)]] from empirical bins (would require input data).
Instead we use a coefficient-only baseline:

BASELINE_MODE = "uniform_other":
  baseline_eta(v) = intercept + sum_{j!=v} mean(effect_j over its bins)

This yields a stable, coefficient-only tuning curve shape and level.

Weights layout (your pipeline)
------------------------------
WEIGHTS_ROOT/<session>/<FULL_FIT_DIR>/<FULL_MODEL_DIR>/<neuron>/
  preferred:
    weights_mean.csv   (or weights_avg.csv / coef_mean.csv)
  fallback:
    fold1/weights.csv ... fold10/weights.csv   (we will align by column names and average)

We assume OneHotEncoder(drop="first") feature naming like:
  position_1, position_2, ...
  head_v_1, head_v_2, ...
  roll_1, ...
  yaw_1, ...
  pitch_1, ...
and an intercept column named "intercept" (case-insensitive) in the wide CSV.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glm_poisson_forward.angle_utils import angle_bin_centers
from glm_poisson_forward.config import ANGLE_BINS_BY_VAR, ANGLE_RANGES_BY_VAR, BIN_MS, SPEED_N_BINS

# =========================
# CONFIG (EDIT AS NEEDED)
# =========================

CSV_DIFF_PATH = Path(
    r"D:\Jiaqi\Projects\GLM_File\GLM_Poisson_Forward\weights_Poisson_forward\indoor_outdoor_position_vs_multi.csv"
)

WEIGHTS_ROOT = Path(
    r"D:\Jiaqi\Projects\GLM_File\GLM_Poisson_Forward\weights_Poisson_forward"
)

# Where the "full model fit" lives (as you requested)
FULL_FIT_DIRNAME = "FULL_FIT"

# Canonical full-model directory name in FULL_FIT
FULL_MODEL_KEY = "Position_Speed_roll_yaw_pitch"
FULL_MODEL_TOKENS = {"Position", "Speed", "roll", "yaw", "pitch"}

# Session naming: we will try exact "<session_id>_indoor" and "<session_id>_outdoor",
# and fall back to glob "<session_id>_indoor*" / "<session_id>_outdoor*"
INDOOR_SUFFIX = "_indoor"
OUTDOOR_SUFFIX = "_outdoor"

# Output directory
OUT_DIR = WEIGHTS_ROOT / "plots_tuning_FULL_FIT_coef_only"

# Bin settings (match forward-search/training)
BIN_SEC = BIN_MS / 1000.0

SPEED_RANGE = (0.0, 1.5)

# Smoothing (in bins). Set None/0 to disable.
SMOOTH_SIGMA_BINS: float | None = 1.5

# Baseline mode (coef-only)
BASELINE_MODE = "uniform_other"  # "intercept_only" or "uniform_other"


# =========================
# Diff token mapping
# =========================

# diff_outdoor_minus_indoor column may contain tokens like: Speed;pitch;roll;yaw
_DIFF_TOKEN_TO_VAR = {
    "speed": "Speed",
    "head_v": "Speed",
    "v": "Speed",
    "velocity": "Speed",

    "roll": "roll",
    "yaw": "yaw",
    "pitch": "pitch",
}

_CANON_PLOT_ORDER = ["Speed", "roll", "yaw", "pitch"]


def _diff_to_plot_vars(diff_str: str) -> List[str]:
    if diff_str is None:
        return []
    s = str(diff_str).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return []
    toks = [t.strip() for t in s.replace(",", ";").split(";") if t.strip()]
    out = []
    for t in toks:
        k = t.strip().lower()
        if k in _DIFF_TOKEN_TO_VAR:
            out.append(_DIFF_TOKEN_TO_VAR[k])
    # de-dup while keeping canonical order
    out_set = set(out)
    return [v for v in _CANON_PLOT_ORDER if v in out_set]


# =========================
# Small utilities
# =========================

def _gaussian_smooth_1d(y: np.ndarray, sigma_bins: float, circular: bool) -> np.ndarray:
    if sigma_bins <= 0:
        return y
    sigma = float(sigma_bins)
    radius = int(max(2, math.ceil(4 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    k /= np.sum(k)

    y0 = np.asarray(y, dtype=np.float64)
    if circular:
        pad = radius
        ypad = np.concatenate([y0[-pad:], y0, y0[:pad]])
        ys = np.convolve(ypad, k, mode="same")[pad:-pad]
        return ys
    else:
        pad = radius
        ypad = np.pad(y0, (pad, pad), mode="edge")
        ys = np.convolve(ypad, k, mode="same")[pad:-pad]
        return ys


def _x_axis_for_var(var: str, K: int) -> Tuple[np.ndarray, str]:
    """
    We do NOT have per-session speed edges anymore (no raw data).
    So:
      - Speed: linear bins on [0, 1.5] like training (centers)
      - roll/yaw: degrees centers in [0..360)
      - pitch: degrees centers in [0..360)
    """
    if var == "Speed":
        edges = np.linspace(SPEED_RANGE[0], SPEED_RANGE[1], K + 1, dtype=np.float64)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, "Speed bin center (m/s)"
    if var in {"roll", "yaw", "pitch"}:
        centers = angle_bin_centers(ANGLE_RANGES_BY_VAR[var], K)
        return np.rad2deg(centers), f"{var} bin center (deg)"
    raise ValueError(f"Unsupported var for x-axis: {var}")


# =========================
# Locate sessions / FULL_FIT dirs
# =========================

def _pick_session_dir(session_id: str, suffix: str) -> Path:
    """
    Prefer exact match: <session_id><suffix>
    Else fallback to glob: <session_id><suffix>*
    """
    exact = WEIGHTS_ROOT / f"{session_id}{suffix}"
    if exact.exists():
        return exact

    cands = sorted(WEIGHTS_ROOT.glob(f"{session_id}{suffix}*"))
    if len(cands) == 0:
        raise FileNotFoundError(f"No session dir found for '{session_id}{suffix}' under {WEIGHTS_ROOT}")
    return cands[0]


def _find_full_fit_model_dir(session_dir: Path) -> Path:
    """
    Prefer: session/FULL_FIT/FULL_MODEL_KEY
    Else: search session/FULL_FIT/* and match by token set.
    """
    ff = session_dir / FULL_FIT_DIRNAME
    if not ff.exists():
        raise FileNotFoundError(f"Missing FULL_FIT dir: {ff}")

    p = ff / FULL_MODEL_KEY
    if p.exists():
        return p

    # token-set fallback
    for d in sorted(ff.iterdir()):
        if not d.is_dir():
            continue
        toks = set([t for t in d.name.split("_") if t])
        if toks == FULL_MODEL_TOKENS:
            return d

    raise FileNotFoundError(f"FULL_FIT model dir not found under {ff} (expected {FULL_MODEL_KEY} or token match)")


# =========================
# Read weights (prefer mean; fallback fold averaging)
# =========================

def _read_wide_weights_csv(csv_path: Path) -> Tuple[List[str], np.ndarray, float]:
    """
    Read a 'wide' weights csv where intercept is a COLUMN, not a ROW.

    Expected typical shape:
      one row, many columns:
        [Unnamed: 0] position_1 ... head_v_1 ... pitch_14 intercept

    Returns:
      coef_names (excluding intercept), coef_vals, intercept
    """
    df = pd.read_csv(csv_path)
    if df.shape[0] < 1 or df.shape[1] < 2:
        raise ValueError(f"Bad weights csv shape: {csv_path}, shape={df.shape}")

    # pick intercept column (case-insensitive)
    cols = list(df.columns)
    lower = {str(c).strip().lower(): c for c in cols}
    int_col = None
    for key in ["intercept", "(intercept)", "bias", "b0"]:
        if key in lower:
            int_col = lower[key]
            break
    if int_col is None:
        raise ValueError(f"No intercept COLUMN found in {csv_path}. Columns={cols[:10]}...")

    row = df.iloc[0]

    # drop obvious index columns
    coef_series = row.drop(labels=[int_col], errors="ignore")
    coef_series = coef_series[~coef_series.index.to_series().astype(str).str.startswith("Unnamed")]

    # parse numbers
    coef_names = [str(c) for c in coef_series.index.tolist()]
    coef_vals = pd.to_numeric(coef_series, errors="coerce").to_numpy(dtype=np.float64)

    if np.any(np.isnan(coef_vals)):
        bad = np.where(np.isnan(coef_vals))[0][:10]
        raise ValueError(f"NaN in coef columns of {csv_path} at indices {bad.tolist()} (first 10).")

    intercept = float(pd.to_numeric(pd.Series([row[int_col]]), errors="coerce").iloc[0])
    if not np.isfinite(intercept):
        raise ValueError(f"Intercept is not finite in {csv_path}: {row[int_col]}")

    return coef_names, coef_vals, intercept


def _load_full_fit_avg_weights(neuron_dir: Path) -> Tuple[List[str], np.ndarray, float]:
    """
    Prefer a pre-averaged csv in neuron_dir (weights_mean/avg).
    Else average fold*/weights.csv (wide format) aligned by column names.
    """
    if not neuron_dir.exists():
        raise FileNotFoundError(f"Neuron dir not found: {neuron_dir}")

    # 1) prefer averaged files
    for fn in ["weights_mean.csv", "weights_avg.csv", "coef_mean.csv", "coef_avg.csv", "weights.csv"]:
        p = neuron_dir / fn
        if p.exists() and (neuron_dir / "fold1").exists() is False:
            # neuron-level weights.csv might exist (some pipelines save directly)
            return _read_wide_weights_csv(p)
        if p.exists() and fn in {"weights_mean.csv", "weights_avg.csv", "coef_mean.csv", "coef_avg.csv"}:
            return _read_wide_weights_csv(p)

    # 2) fallback: fold averaging
    fold_dirs = sorted([d for d in neuron_dir.iterdir() if d.is_dir() and re.fullmatch(r"fold\d+", d.name)])
    if len(fold_dirs) == 0:
        raise FileNotFoundError(f"No averaged weights csv and no fold* dirs under {neuron_dir}")

    ref_names, ref_coef, ref_int = _read_wide_weights_csv(fold_dirs[0] / "weights.csv")
    ref_index = {n: i for i, n in enumerate(ref_names)}
    P = len(ref_names)

    coef_stack = []
    ints = []

    for fd in fold_dirs:
        names, coef, intercept = _read_wide_weights_csv(fd / "weights.csv")
        if len(names) != P:
            raise ValueError(f"Coef column count mismatch in {fd}/weights.csv: got {len(names)} expected {P}")

        idx_map = {n: i for i, n in enumerate(names)}
        if set(idx_map.keys()) != set(ref_index.keys()):
            missing = sorted(list(set(ref_index.keys()) - set(idx_map.keys())))[:10]
            extra = sorted(list(set(idx_map.keys()) - set(ref_index.keys())))[:10]
            raise ValueError(
                f"Coef column name set mismatch in {fd}/weights.csv.\n"
                f"Missing(first10)={missing}\nExtra(first10)={extra}"
            )

        aligned = np.empty(P, dtype=np.float64)
        for n, j in ref_index.items():
            aligned[j] = float(coef[idx_map[n]])
        coef_stack.append(aligned)
        ints.append(float(intercept))

    coef_mean = np.mean(np.stack(coef_stack, axis=0), axis=0)
    int_mean = float(np.mean(np.asarray(ints, dtype=np.float64)))
    return ref_names, coef_mean, int_mean


# =========================
# Parse coefficient vector -> per-variable bin effects
# =========================

_PREFIX_TO_VAR = {
    "position": "Position",
    "head_v": "Speed",
    "roll": "roll",
    "yaw": "yaw",
    "pitch": "pitch",
}


def _infer_bin_id(colname: str) -> Optional[int]:
    """
    Try parse trailing _<int> as bin id (1..K-1 typically).
    """
    m = re.search(r"_(\d+)$", colname)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _split_effects_from_names(coef_names: List[str], coef_vals: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Build effect vectors per variable:
      effect[0]=0 (dropped reference bin)
      effect[1:] = coefficients in ascending bin-id order if parsable, else file order.

    Returns dict var -> effect (length K)
    """
    coef_vals = np.asarray(coef_vals, dtype=np.float64).ravel()
    if len(coef_names) != coef_vals.size:
        raise ValueError("coef_names length != coef_vals size")

    # collect per-var columns
    per_var: Dict[str, List[Tuple[int, str, float]]] = {}  # (order_key, name, value)
    for name, val in zip(coef_names, coef_vals):
        n = str(name)
        v = None
        for pfx, vv in _PREFIX_TO_VAR.items():
            if n == pfx or n.startswith(pfx + "_"):
                v = vv
                break
        if v is None:
            continue

        bid = _infer_bin_id(n)
        # if we can parse bin id, use it; else stable order key by appearance
        order_key = bid if bid is not None else (10**9)  # push non-parsable to the end
        per_var.setdefault(v, []).append((order_key, n, float(val)))

    effects: Dict[str, np.ndarray] = {}
    for v, items in per_var.items():
        # if at least one parsable id exists, sort primarily by bin id, else preserve insertion order
        has_id = any((_infer_bin_id(n) is not None) for _, n, _ in items)
        if has_id:
            items_sorted = sorted(items, key=lambda t: (t[0], t[1]))
        else:
            items_sorted = items  # original order

        vals = np.array([vv for _, _, vv in items_sorted], dtype=np.float64)
        K = vals.size + 1
        eff = np.zeros(K, dtype=np.float64)
        if vals.size > 0:
            eff[1:] = vals
        effects[v] = eff

    return effects


def _baseline_eta(intercept: float, effects: Dict[str, np.ndarray], target_var: str) -> float:
    eta = float(intercept)
    if BASELINE_MODE == "intercept_only":
        return eta
    if BASELINE_MODE != "uniform_other":
        raise ValueError(f"Unknown BASELINE_MODE: {BASELINE_MODE}")

    for v, eff in effects.items():
        if v == target_var:
            continue
        # uniform mean over bins, purely coef-based
        eta += float(np.mean(eff))
    return eta


def _compute_curve_rate(var: str, intercept: float, effects: Dict[str, np.ndarray]) -> np.ndarray:
    eff = effects.get(var, None)
    if eff is None:
        # var absent -> flat curve
        if var == "Speed":
            eff = np.zeros(SPEED_N_BINS, dtype=np.float64)
        elif var in {"roll", "yaw", "pitch"}:
            eff = np.zeros(ANGLE_BINS_BY_VAR[var], dtype=np.float64)
        else:
            eff = np.zeros(1, dtype=np.float64)

    base_eta = _baseline_eta(intercept, effects, target_var=var)
    eta = base_eta + eff
    rate = np.exp(eta) / BIN_SEC  # spikes/s

    if SMOOTH_SIGMA_BINS is not None and float(SMOOTH_SIGMA_BINS) > 0:
        circular = var in {"roll", "yaw"}
        rate = _gaussian_smooth_1d(rate, float(SMOOTH_SIGMA_BINS), circular=circular)
    return rate


# =========================
# Plotting
# =========================

def _plot_one_var(session_id: str, neuron: str, var: str,
                  indoor: Tuple[List[str], np.ndarray, float],
                  outdoor: Tuple[List[str], np.ndarray, float],
                  paired_output_session_ids: Sequence[str] | None = None) -> None:
    names_i, coef_i, int_i = indoor
    names_o, coef_o, int_o = outdoor

    eff_i = _split_effects_from_names(names_i, coef_i)
    eff_o = _split_effects_from_names(names_o, coef_o)

    curve_i = _compute_curve_rate(var, int_i, eff_i)
    curve_o = _compute_curve_rate(var, int_o, eff_o)

    Ki = int(curve_i.size)
    Ko = int(curve_o.size)
    if Ki != Ko:
        # keep going: plot on their own x axes
        xi, xlabel = _x_axis_for_var(var, Ki)
        xo, _ = _x_axis_for_var(var, Ko)
    else:
        xi, xlabel = _x_axis_for_var(var, Ki)
        xo = xi

    plt.figure(figsize=(9.5, 4.2))
    plt.plot(xi, curve_i, label="Indoor (FULL_FIT)", linewidth=2)
    plt.plot(xo, curve_o, label="Outdoor (FULL_FIT)", linewidth=2, alpha=0.9)

    plt.xlabel(xlabel)
    plt.ylabel("Predicted firing rate (spikes/s)")
    plt.title(f"{session_id} | {neuron} | {var} tuning (coef-only baseline={BASELINE_MODE})")
    plt.legend()
    plt.tight_layout()

    out_png = OUT_DIR / session_id / neuron / f"tuning_FULL_FIT_{var}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

    if var in {"roll", "yaw", "pitch"}:
        def _close_curve(theta_deg: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            theta_rad = np.deg2rad(theta_deg)
            return np.concatenate([theta_rad, theta_rad[:1]]), np.concatenate([r, r[:1]])

        def _polar_plot(theta_deg: np.ndarray, r: np.ndarray, label: str, out_path: Path,
                        color: str | None = None) -> None:
            theta_c, r_c = _close_curve(theta_deg, r)
            plt.figure(figsize=(5.5, 5.5))
            ax = plt.subplot(111, projection="polar")
            line_kwargs = {"linewidth": 2}
            if color:
                line_kwargs["color"] = color
            ax.plot(theta_c, r_c, label=label, **line_kwargs)
            ax.fill(theta_c, r_c, alpha=0.25, color=line_kwargs.get("color", None))
            ax.set_title(f"{session_id} | {neuron} | {var} ({label})")
            ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
            plt.tight_layout()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_path, dpi=200)
            plt.close()

        theta_i = xi
        theta_o = xo

        _polar_plot(theta_i, curve_i, "Indoor (FULL_FIT)",
                    OUT_DIR / session_id / neuron / f"tuning_FULL_FIT_{var}_polar_indoor.png",
                    color="#1f77b4")
        _polar_plot(theta_o, curve_o, "Outdoor (FULL_FIT)",
                    OUT_DIR / session_id / neuron / f"tuning_FULL_FIT_{var}_polar_outdoor.png",
                    color="#ff7f0e")

        theta_i_c, r_i_c = _close_curve(theta_i, curve_i)
        theta_o_c, r_o_c = _close_curve(theta_o, curve_o)

        plt.figure(figsize=(6.0, 6.0))
        ax = plt.subplot(111, projection="polar")
        ax.plot(theta_i_c, r_i_c, label="Indoor (FULL_FIT)", linewidth=2, color="#1f77b4")
        ax.fill(theta_i_c, r_i_c, alpha=0.22, color="#1f77b4")
        ax.plot(theta_o_c, r_o_c, label="Outdoor (FULL_FIT)", linewidth=2, color="#ff7f0e", alpha=0.95)
        ax.fill(theta_o_c, r_o_c, alpha=0.2, color="#ff7f0e")
        ax.set_title(f"{session_id} | {neuron} | {var} (Indoor vs Outdoor)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        plt.tight_layout()
        out_png_polar = OUT_DIR / session_id / neuron / f"tuning_FULL_FIT_{var}_polar_indoor_vs_outdoor.png"
        out_png_polar.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png_polar, dpi=200)

        if paired_output_session_ids:
            for paired_id in dict.fromkeys(paired_output_session_ids):
                if paired_id == session_id:
                    continue
                paired_out = OUT_DIR / paired_id / neuron / f"tuning_FULL_FIT_{var}_polar_indoor_vs_outdoor.png"
                paired_out.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(paired_out, dpi=200)
        plt.close()


def _process_one_row(session_id: str, neuron: str, diff_str: str) -> Optional[str]:
    try:
        plot_vars = _diff_to_plot_vars(diff_str)
        if not plot_vars:
            return None

        indoor_sess_dir = _pick_session_dir(session_id, INDOOR_SUFFIX)
        outdoor_sess_dir = _pick_session_dir(session_id, OUTDOOR_SUFFIX)

        indoor_model_dir = _find_full_fit_model_dir(indoor_sess_dir)
        outdoor_model_dir = _find_full_fit_model_dir(outdoor_sess_dir)

        neuron_dir_i = indoor_model_dir / neuron
        neuron_dir_o = outdoor_model_dir / neuron

        indoor_pack = _load_full_fit_avg_weights(neuron_dir_i)   # (names, coef, intercept)
        outdoor_pack = _load_full_fit_avg_weights(neuron_dir_o)

        paired_output_session_ids = [indoor_sess_dir.name, outdoor_sess_dir.name]
        for v in plot_vars:
            _plot_one_var(session_id, neuron, v, indoor_pack, outdoor_pack,
                          paired_output_session_ids=paired_output_session_ids)

        return None

    except Exception as e:
        return str(e)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_DIFF_PATH.exists():
        raise FileNotFoundError(f"Missing diff CSV: {CSV_DIFF_PATH}")

    df = pd.read_csv(CSV_DIFF_PATH, sep=None, engine="python")

    required = {"session_id", "neuron", "diff_outdoor_minus_indoor"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Diff CSV missing columns: {sorted(list(missing))}. Got: {list(df.columns)}")

    fails = []
    for _, r in df.iterrows():
        sid = str(r["session_id"])
        neu = str(r["neuron"])
        diff = r["diff_outdoor_minus_indoor"]
        err = _process_one_row(sid, neu, diff)
        if err:
            fails.append((sid, neu, err))
            print(f"[FAIL] {sid} {neu}: {err}")
        else:
            print(f"[OK] {sid} {neu}")

    if fails:
        fail_csv = OUT_DIR / "failures.csv"
        pd.DataFrame(fails, columns=["session_id", "neuron", "error"]).to_csv(fail_csv, index=False, encoding="utf-8-sig")
        print(f"\nDone with failures: {len(fails)}. See {fail_csv}")
    else:
        print("\nDone. No failures.")
    print(f"Plots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
