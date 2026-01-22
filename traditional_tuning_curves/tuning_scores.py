from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

from glm_poisson_forward.config import ANGLE_N_BINS, POSITION_CELL_CM, SPEED_N_BINS
from glm_poisson_forward.design_matrix import bin_col
from .config import (
    ANGULAR_K_MAX,
    BIN_SEC,
    BIN_SMOOTH_SIGMA_BINS,
    MIN_BIN_OCCUPANCY_SEC,
    SHUFFLE_MIN_SEC,
    SPEED_MAX_M_S,
    SPEED_MIN_M_S,
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
    valid = occupancy_sec >= MIN_BIN_OCCUPANCY_SEC
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_map[valid] = spike_map[valid] / occupancy_sec[valid]
    return rate_map, occupancy_sec


def _vector_length_k(weights: np.ndarray, angles: np.ndarray, k: int) -> float:
    vect = np.sum(weights * np.exp(1j * k * angles)) / np.sum(weights)
    return float(np.abs(vect))


def angular_score(angle_bin: np.ndarray, spikes: np.ndarray, mask: np.ndarray) -> Tuple[float, np.ndarray]:
    spikes_sel = spikes[mask]
    bins_sel = angle_bin[mask]

    occ = np.bincount(bins_sel, minlength=ANGLE_N_BINS).astype(np.float64)
    spk = np.bincount(bins_sel, weights=spikes_sel, minlength=ANGLE_N_BINS).astype(np.float64)

    occ_sec = occ * BIN_SEC
    rate = np.full(ANGLE_N_BINS, np.nan, dtype=np.float64)
    valid = occ_sec >= MIN_BIN_OCCUPANCY_SEC
    with np.errstate(invalid="ignore", divide="ignore"):
        rate[valid] = spk[valid] / occ_sec[valid]

    rate = np.nan_to_num(rate, nan=0.0)
    bin_deg = 360.0 / ANGLE_N_BINS
    sigma_bins = float(BIN_SMOOTH_SIGMA_BINS)
    if sigma_bins > 0:
        rate_smooth = ndimage.gaussian_filter1d(rate, sigma=sigma_bins, mode="wrap")
    else:
        rate_smooth = rate

    angles = np.linspace(0.0, 2 * np.pi, ANGLE_N_BINS, endpoint=False)
    weights = np.nan_to_num(rate_smooth, nan=0.0)
    if np.sum(weights) <= 0:
        return float("nan"), rate_smooth
    k_max = max(int(ANGULAR_K_MAX), 1)
    k_scores = [_vector_length_k(weights, angles, k) for k in range(1, k_max + 1)]
    return float(np.nanmax(k_scores)), rate_smooth


def speed_score(head_v: np.ndarray, spikes: np.ndarray, mask: np.ndarray) -> float:
    if spikes.ndim != 1:
        raise ValueError("speed_score expects 1D spikes array")

    bins = bin_col(head_v, n_bins=SPEED_N_BINS, vmin=SPEED_MIN_M_S, vmax=SPEED_MAX_M_S)
    bins_sel = bins[mask]
    spikes_sel = spikes[mask]

    occ = np.bincount(bins_sel, minlength=SPEED_N_BINS).astype(np.float64)
    spk = np.bincount(bins_sel, weights=spikes_sel, minlength=SPEED_N_BINS).astype(np.float64)

    occ_sec = occ * BIN_SEC
    valid = occ_sec >= MIN_BIN_OCCUPANCY_SEC
    if np.sum(valid) < 2:
        return float("nan")

    p_b = occ_sec[valid] / np.sum(occ_sec[valid])
    with np.errstate(invalid="ignore", divide="ignore"):
        r_b = spk[valid] / occ_sec[valid]
    r_bar = float(np.sum(p_b * r_b))
    if not np.isfinite(r_bar) or r_bar <= 0:
        return float("nan")

    ratio = r_b / r_bar
    info_mask = ratio > 0
    if not np.any(info_mask):
        return float("nan")
    info = np.sum(p_b[info_mask] * ratio[info_mask] * np.log2(ratio[info_mask]))
    return float(info)


def speed_tuning(head_v: np.ndarray, spikes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bins = bin_col(head_v, n_bins=SPEED_N_BINS, vmin=SPEED_MIN_M_S, vmax=SPEED_MAX_M_S)
    bins_sel = bins[mask]
    spikes_sel = spikes[mask]

    occ = np.bincount(bins_sel, minlength=SPEED_N_BINS).astype(np.float64)
    spk = np.bincount(bins_sel, weights=spikes_sel, minlength=SPEED_N_BINS).astype(np.float64)

    occ_sec = occ * BIN_SEC
    rate = np.full(SPEED_N_BINS, np.nan, dtype=np.float64)
    valid = occ_sec >= MIN_BIN_OCCUPANCY_SEC
    with np.errstate(invalid="ignore", divide="ignore"):
        rate[valid] = spk[valid] / occ_sec[valid]
    rate = np.nan_to_num(rate, nan=0.0)
    sigma_bins = float(BIN_SMOOTH_SIGMA_BINS)
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

    rate_map, _ = rate_map_2d(
        bins.x_bin,
        bins.y_bin,
        spikes,
        mask,
        bins.x_min,
        bins.y_min,
        bins.x_size,
        bins.y_size,
    )

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
        "hd_curve": hd_curve,
        "roll_curve": roll_curve,
        "pitch_curve": pitch_curve,
        "speed_curve": speed_tuning(inputs.head_v, spikes, mask),
    }

    return (
        ScoreResult(
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
        "hd_score": [],
        "roll_score": [],
        "pitch_score": [],
        "speed_score": [],
        "speed_stability": [],
    }

    for _ in range(n_shuffle):
        sh_spikes = time_shift_spikes(base_spikes, rng)
        scores["hd_score"].append(angular_score(bins.hd_bin, sh_spikes, mask)[0])
        scores["roll_score"].append(angular_score(bins.roll_bin, sh_spikes, mask)[0])
        scores["pitch_score"].append(angular_score(bins.pitch_bin, sh_spikes, mask)[0])
        scores["speed_score"].append(speed_score(inputs.head_v, sh_spikes, mask))
        scores["speed_stability"].append(speed_stability(inputs.head_v, sh_spikes, mask))

    return {k: np.array(v, dtype=np.float64) for k, v in scores.items()}
