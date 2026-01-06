from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage, signal

from glm_poisson_forward.config import ANGLE_N_BINS, POSITION_CELL_CM, SPEED_N_BINS
from glm_poisson_forward.design_matrix import bin_col
from .config import (
    ADAPTIVE_SMOOTH_ALPHA,
    BIN_SEC,
    HD_SMOOTH_DEG,
    SHUFFLE_MIN_SEC,
    SPEED_MAX_M_S,
    SPEED_MIN_M_S,
    SPEED_SMOOTH_MS,
)


@dataclass
class TuningInputs:
    head_x: np.ndarray
    head_y: np.ndarray
    heading_rad: np.ndarray
    head_v: np.ndarray
    roll: np.ndarray
    pitch: np.ndarray
    spikes: np.ndarray


@dataclass
class SessionBinning:
    x_bin: np.ndarray
    y_bin: np.ndarray
    speed_bin: np.ndarray
    hd_bin: np.ndarray
    roll_bin: np.ndarray
    pitch_bin: np.ndarray
    x_min: int
    y_min: int
    x_size: int
    y_size: int


@dataclass
class ScoreResult:
    grid_score: float
    border_score: float
    hd_score: float
    roll_score: float
    pitch_score: float
    speed_score: float
    speed_stability: float
    spatial_stability: float
    angular_stability: float
    roll_stability: float
    pitch_stability: float


def build_bins(inputs: TuningInputs) -> SessionBinning:
    cell = float(POSITION_CELL_CM)
    x_bin = (inputs.head_x.astype(np.float32) // cell).astype(int)
    y_bin = (inputs.head_y.astype(np.float32) // cell).astype(int)

    x_min = int(np.min(x_bin))
    y_min = int(np.min(y_bin))
    x_size = int(np.max(x_bin) - x_min + 1)
    y_size = int(np.max(y_bin) - y_min + 1)

    speed_bin = bin_col(inputs.head_v, n_bins=SPEED_N_BINS, vmin=SPEED_MIN_M_S, vmax=SPEED_MAX_M_S)
    hd_bin = bin_col(inputs.heading_rad, n_bins=ANGLE_N_BINS, vmin=0.0, vmax=2.0 * np.pi)
    roll_bin = bin_col(inputs.roll, n_bins=ANGLE_N_BINS, vmin=0.0, vmax=2.0 * np.pi)
    pitch_bin = bin_col(inputs.pitch, n_bins=ANGLE_N_BINS, vmin=0.0, vmax=2.0 * np.pi)

    return SessionBinning(
        x_bin=x_bin,
        y_bin=y_bin,
        speed_bin=speed_bin,
        hd_bin=hd_bin,
        roll_bin=roll_bin,
        pitch_bin=pitch_bin,
        x_min=x_min,
        y_min=y_min,
        x_size=x_size,
        y_size=y_size,
    )


def valid_speed_mask(head_v: np.ndarray) -> np.ndarray:
    return (head_v >= SPEED_MIN_M_S) & (head_v <= SPEED_MAX_M_S)


def _bincount_2d(x_bin, y_bin, x_min, y_min, x_size, y_size, weights=None) -> np.ndarray:
    idx = (y_bin - y_min) * x_size + (x_bin - x_min)
    flat = np.bincount(idx, weights=weights, minlength=x_size * y_size)
    return flat.reshape(y_size, x_size)


def rate_map_2d(
    x_bin: np.ndarray,
    y_bin: np.ndarray,
    spikes: np.ndarray,
    mask: np.ndarray,
    x_min: int,
    y_min: int,
    x_size: int,
    y_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x_sel = x_bin[mask]
    y_sel = y_bin[mask]
    spikes_sel = spikes[mask]

    occupancy = _bincount_2d(x_sel, y_sel, x_min, y_min, x_size, y_size)
    spike_map = _bincount_2d(x_sel, y_sel, x_min, y_min, x_size, y_size, weights=spikes_sel)

    occupancy_sec = occupancy * BIN_SEC
    rate_map = np.full_like(occupancy_sec, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_map[occupancy_sec > 0] = spike_map[occupancy_sec > 0] / occupancy_sec[occupancy_sec > 0]
    return rate_map, occupancy_sec


def adaptive_smooth(rate_map: np.ndarray, occupancy_sec: np.ndarray, alpha: float = ADAPTIVE_SMOOTH_ALPHA) -> np.ndarray:
    smoothed = np.full_like(rate_map, np.nan, dtype=np.float64)
    spikes_map = np.nan_to_num(rate_map * occupancy_sec, nan=0.0)
    max_radius = int(max(rate_map.shape))

    ys, xs = np.indices(rate_map.shape)

    for i in range(rate_map.shape[0]):
        for j in range(rate_map.shape[1]):
            best_val = np.nan
            for r in range(1, max_radius + 1):
                dist = np.sqrt((ys - i) ** 2 + (xs - j) ** 2)
                mask = dist <= r
                occ = np.sum(occupancy_sec[mask])
                spk = np.sum(spikes_map[mask])
                if occ <= 0:
                    continue
                if spk > 0 and (spk / occ) >= (alpha / (occ * np.sqrt(spk))):
                    best_val = spk / occ
                    break
            if np.isnan(best_val) and occupancy_sec[i, j] > 0:
                best_val = spikes_map[i, j] / occupancy_sec[i, j]
            smoothed[i, j] = best_val
    return smoothed


def _autocorr_2d(rate_map: np.ndarray) -> np.ndarray:
    rm = np.array(rate_map, dtype=np.float64)
    mask = np.isfinite(rm)
    if not np.any(mask):
        return np.full((1, 1), np.nan)
    rm0 = rm.copy()
    rm0[~mask] = 0.0
    mean = np.mean(rm0[mask])
    rm0 = rm0 - mean
    rm0[~mask] = 0.0

    numerator = signal.correlate2d(rm0, rm0, mode="full", boundary="fill", fillvalue=0)
    overlap = signal.correlate2d(mask.astype(float), mask.astype(float), mode="full", boundary="fill", fillvalue=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = numerator / np.maximum(overlap, 1.0)
    return out


def _rotate_corr(base: np.ndarray, angle: float, ring_mask: np.ndarray) -> float:
    rotated = ndimage.rotate(base, angle, reshape=False, order=1, mode="constant", cval=np.nan)
    vals1 = base[ring_mask]
    vals2 = rotated[ring_mask]
    ok = np.isfinite(vals1) & np.isfinite(vals2)
    if np.sum(ok) < 5:
        return np.nan
    v1 = vals1[ok]
    v2 = vals2[ok]
    if np.std(v1) == 0 or np.std(v2) == 0:
        return np.nan
    return float(np.corrcoef(v1, v2)[0, 1])


def grid_score(rate_map: np.ndarray) -> Tuple[float, np.ndarray]:
    autocorr = _autocorr_2d(rate_map)
    if autocorr.size == 1:
        return float("nan"), autocorr

    cy, cx = np.array(autocorr.shape) // 2
    ys, xs = np.indices(autocorr.shape)
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    r_max = min(cy, cx) - 1
    if r_max <= 2:
        return float("nan"), autocorr

    ring_mask = (dist >= 2) & (dist <= r_max)

    rot_angles = [30, 60, 90, 120, 150]
    corrs = {a: _rotate_corr(autocorr, a, ring_mask) for a in rot_angles}

    hi = np.nanmin([corrs[60], corrs[120]])
    lo = np.nanmax([corrs[30], corrs[90], corrs[150]])
    if np.isnan(hi) or np.isnan(lo):
        return float("nan"), autocorr
    return float(hi - lo), autocorr


def border_score(rate_map: np.ndarray, occupancy_sec: np.ndarray) -> float:
    if rate_map.size == 0 or np.all(~np.isfinite(rate_map)):
        return float("nan")

    smoothed = adaptive_smooth(rate_map, occupancy_sec)
    if not np.any(np.isfinite(smoothed)):
        return float("nan")

    peak = np.nanmax(smoothed)
    if peak <= 0 or np.isnan(peak):
        return float("nan")

    high = smoothed >= (0.2 * peak)
    if not np.any(high):
        return float("nan")

    y_size, x_size = smoothed.shape
    wall_lengths = {
        "north": x_size,
        "south": x_size,
        "west": y_size,
        "east": y_size,
    }

    coverages = []
    for wall in ["north", "south", "west", "east"]:
        if wall == "north":
            bins = high[0, :]
        elif wall == "south":
            bins = high[-1, :]
        elif wall == "west":
            bins = high[:, 0]
        else:
            bins = high[:, -1]
        coverages.append(np.sum(bins) / max(wall_lengths[wall], 1))

    cm = float(np.max(coverages))

    ys, xs = np.indices(smoothed.shape)
    dist_to_wall = np.minimum.reduce([ys, xs, (y_size - 1) - ys, (x_size - 1) - xs]).astype(np.float64)
    max_dist = float(np.nanmax(dist_to_wall)) if np.isfinite(dist_to_wall).any() else 1.0

    weights = np.nan_to_num(smoothed, nan=0.0)
    dm = float(np.sum(weights * dist_to_wall) / max(np.sum(weights), 1e-6))
    dm /= max(max_dist, 1e-6)

    return float((cm - dm) / (cm + dm + 1e-6))


def angular_score(angle_bin: np.ndarray, spikes: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray]:
    spikes_sel = spikes[mask]
    bins_sel = angle_bin[mask]

    occ = np.bincount(bins_sel, minlength=ANGLE_N_BINS).astype(np.float64)
    spk = np.bincount(bins_sel, weights=spikes_sel, minlength=ANGLE_N_BINS).astype(np.float64)

    occ_sec = occ * BIN_SEC
    rate = np.full(ANGLE_N_BINS, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate[occ_sec > 0] = spk[occ_sec > 0] / occ_sec[occ_sec > 0]

    rate = np.nan_to_num(rate, nan=0.0)
    bin_deg = 360.0 / ANGLE_N_BINS
    sigma_bins = HD_SMOOTH_DEG / bin_deg
    if sigma_bins > 0:
        rate_smooth = ndimage.gaussian_filter1d(rate, sigma=sigma_bins, mode="wrap")
    else:
        rate_smooth = rate

    angles = np.linspace(0.0, 2 * np.pi, ANGLE_N_BINS, endpoint=False)
    weights = np.nan_to_num(rate_smooth, nan=0.0)
    if np.sum(weights) <= 0:
        return float("nan"), rate_smooth

    vect = np.sum(weights * np.exp(1j * angles)) / np.sum(weights)
    return float(np.abs(vect)), rate_smooth


def speed_score(head_v: np.ndarray, spikes: np.ndarray, mask: np.ndarray) -> float:
    if spikes.ndim != 1:
        raise ValueError("speed_score expects 1D spikes array")

    rate = spikes.astype(np.float64) / BIN_SEC
    sigma_bins = SPEED_SMOOTH_MS / (BIN_SEC * 1000.0)
    if sigma_bins > 0:
        rate = ndimage.gaussian_filter1d(rate, sigma=sigma_bins, mode="nearest")

    v = head_v.astype(np.float64)
    ok = mask & np.isfinite(v)
    if np.sum(ok) < 5:
        return float("nan")
    if np.std(rate[ok]) == 0 or np.std(v[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(rate[ok], v[ok])[0, 1])


def speed_tuning(head_v: np.ndarray, spikes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bins = bin_col(head_v, n_bins=SPEED_N_BINS, vmin=SPEED_MIN_M_S, vmax=SPEED_MAX_M_S)
    bins_sel = bins[mask]
    spikes_sel = spikes[mask]

    occ = np.bincount(bins_sel, minlength=SPEED_N_BINS).astype(np.float64)
    spk = np.bincount(bins_sel, weights=spikes_sel, minlength=SPEED_N_BINS).astype(np.float64)

    occ_sec = occ * BIN_SEC
    rate = np.full(SPEED_N_BINS, np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate[occ_sec > 0] = spk[occ_sec > 0] / occ_sec[occ_sec > 0]
    rate = np.nan_to_num(rate, nan=0.0)
    sigma_bins = SPEED_SMOOTH_MS / (BIN_SEC * 1000.0)
    if sigma_bins > 0:
        rate = ndimage.gaussian_filter1d(rate, sigma=sigma_bins, mode="nearest")
    return rate


def speed_stability(head_v: np.ndarray, spikes: np.ndarray, mask: np.ndarray) -> float:
    n = len(head_v)
    quarters = np.array_split(np.arange(n), 4)
    curves = []
    for idx in quarters:
        qmask = mask.copy()
        qmask[:] = False
        qmask[idx] = mask[idx]
        curves.append(speed_tuning(head_v, spikes, qmask))

    corrs = []
    for i in range(len(curves)):
        for j in range(i + 1, len(curves)):
            a = curves[i]
            b = curves[j]
            ok = np.isfinite(a) & np.isfinite(b)
            if np.sum(ok) < 2:
                continue
            if np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
                continue
            corrs.append(np.corrcoef(a[ok], b[ok])[0, 1])

    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def spatial_stability(
    x_bin: np.ndarray,
    y_bin: np.ndarray,
    spikes: np.ndarray,
    mask: np.ndarray,
    x_min: int,
    y_min: int,
    x_size: int,
    y_size: int,
) -> float:
    n = len(spikes)
    mid = n // 2
    mask1 = mask.copy()
    mask2 = mask.copy()
    mask1[mid:] = False
    mask2[:mid] = False

    rm1, _ = rate_map_2d(x_bin, y_bin, spikes, mask1, x_min, y_min, x_size, y_size)
    rm2, _ = rate_map_2d(x_bin, y_bin, spikes, mask2, x_min, y_min, x_size, y_size)

    ok = np.isfinite(rm1) & np.isfinite(rm2)
    if np.sum(ok) < 5:
        return float("nan")
    if np.std(rm1[ok]) == 0 or np.std(rm2[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(rm1[ok], rm2[ok])[0, 1])


def angular_stability(angle_bin: np.ndarray, spikes: np.ndarray, mask: np.ndarray) -> float:
    n = len(spikes)
    mid = n // 2
    mask1 = mask.copy()
    mask2 = mask.copy()
    mask1[mid:] = False
    mask2[:mid] = False

    _, curve1 = angular_score(angle_bin, spikes, mask1)
    _, curve2 = angular_score(angle_bin, spikes, mask2)

    ok = np.isfinite(curve1) & np.isfinite(curve2)
    if np.sum(ok) < 2:
        return float("nan")
    if np.std(curve1[ok]) == 0 or np.std(curve2[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(curve1[ok], curve2[ok])[0, 1])


def compute_scores_for_neuron(inputs: TuningInputs, bins: SessionBinning, neuron_idx: int) -> Tuple[ScoreResult, Dict[str, np.ndarray]]:
    spikes = inputs.spikes[:, neuron_idx].astype(np.float64)
    mask = valid_speed_mask(inputs.head_v)

    rate_map, occ = rate_map_2d(
        bins.x_bin,
        bins.y_bin,
        spikes,
        mask,
        bins.x_min,
        bins.y_min,
        bins.x_size,
        bins.y_size,
    )

    g_score, autocorr = grid_score(rate_map)
    b_score = border_score(rate_map, occ)
    hd_score, hd_curve = angular_score(bins.hd_bin, spikes, mask)
    roll_score, roll_curve = angular_score(bins.roll_bin, spikes, mask)
    pitch_score, pitch_curve = angular_score(bins.pitch_bin, spikes, mask)
    s_score = speed_score(inputs.head_v, spikes, mask)
    s_stability = speed_stability(inputs.head_v, spikes, mask)
    spat_stab = spatial_stability(
        bins.x_bin,
        bins.y_bin,
        spikes,
        mask,
        bins.x_min,
        bins.y_min,
        bins.x_size,
        bins.y_size,
    )
    ang_stab = angular_stability(bins.hd_bin, spikes, mask)
    roll_stab = angular_stability(bins.roll_bin, spikes, mask)
    pitch_stab = angular_stability(bins.pitch_bin, spikes, mask)

    aux = {
        "rate_map": rate_map,
        "autocorr": autocorr,
        "hd_curve": hd_curve,
        "roll_curve": roll_curve,
        "pitch_curve": pitch_curve,
        "speed_curve": speed_tuning(inputs.head_v, spikes, mask),
    }

    return (
        ScoreResult(
            grid_score=g_score,
            border_score=b_score,
            hd_score=hd_score,
            roll_score=roll_score,
            pitch_score=pitch_score,
            speed_score=s_score,
            speed_stability=s_stability,
            spatial_stability=spat_stab,
            angular_stability=ang_stab,
            roll_stability=roll_stab,
            pitch_stability=pitch_stab,
        ),
        aux,
    )


def time_shift_spikes(spikes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(spikes)
    min_shift = int(round(SHUFFLE_MIN_SEC / BIN_SEC))
    max_shift = n - min_shift
    if max_shift <= min_shift:
        return spikes
    shift = int(rng.integers(min_shift, max_shift))
    return np.roll(spikes, shift)


def compute_shuffle_scores(
    inputs: TuningInputs,
    bins: SessionBinning,
    neuron_idx: int,
    n_shuffle: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    base_spikes = inputs.spikes[:, neuron_idx].astype(np.float64)
    mask = valid_speed_mask(inputs.head_v)

    scores = {
        "grid_score": [],
        "border_score": [],
        "hd_score": [],
        "roll_score": [],
        "pitch_score": [],
        "speed_score": [],
        "speed_stability": [],
    }

    for _ in range(n_shuffle):
        sh_spikes = time_shift_spikes(base_spikes, rng)
        rate_map, occ = rate_map_2d(
            bins.x_bin,
            bins.y_bin,
            sh_spikes,
            mask,
            bins.x_min,
            bins.y_min,
            bins.x_size,
            bins.y_size,
        )
        g_score, _ = grid_score(rate_map)
        scores["grid_score"].append(g_score)
        scores["border_score"].append(border_score(rate_map, occ))
        scores["hd_score"].append(angular_score(bins.hd_bin, sh_spikes, mask)[0])
        scores["roll_score"].append(angular_score(bins.roll_bin, sh_spikes, mask)[0])
        scores["pitch_score"].append(angular_score(bins.pitch_bin, sh_spikes, mask)[0])
        scores["speed_score"].append(speed_score(inputs.head_v, sh_spikes, mask))
        scores["speed_stability"].append(speed_stability(inputs.head_v, sh_spikes, mask))

    return {k: np.array(v, dtype=np.float64) for k, v in scores.items()}
