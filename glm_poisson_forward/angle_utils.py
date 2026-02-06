from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

from .design_matrix import bin_col


AngleRange = Tuple[float, float]


def normalize_angles(vals: np.ndarray, wrap: float | None = 2 * np.pi) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float32)
    if wrap is None:
        return arr
    return np.mod(arr, wrap)


def clip_to_ranges(vals: np.ndarray, ranges: Iterable[AngleRange]) -> np.ndarray:
    arr = np.asarray(vals, dtype=np.float32)
    ranges_list: List[AngleRange] = list(ranges)
    if not ranges_list:
        raise ValueError("ranges must be non-empty")

    inside = np.zeros(arr.shape, dtype=bool)
    for start, end in ranges_list:
        inside |= (arr >= start) & (arr <= end)
    if np.all(inside):
        return arr

    best_dist = np.full(arr.shape, np.inf, dtype=np.float32)
    best_val = arr.copy()
    for start, end in ranges_list:
        below = arr < start
        above = arr > end
        dist = np.where(below, start - arr, np.where(above, arr - end, 0.0))
        cand = np.where(below, start, np.where(above, end, arr))
        update = dist < best_dist
        best_dist = np.where(update, dist, best_dist)
        best_val = np.where(update, cand, best_val)

    return np.where(inside, arr, best_val)


def _linearize_angles(vals: np.ndarray, ranges: List[AngleRange]) -> tuple[np.ndarray, float]:
    offsets: list[float] = []
    total = 0.0
    for start, end in ranges:
        offsets.append(total)
        total += float(end - start)

    pos = np.zeros_like(vals, dtype=np.float32)
    assigned = np.zeros(vals.shape, dtype=bool)
    for (start, end), offset in zip(ranges, offsets):
        mask = (vals >= start) & (vals <= end)
        if np.any(mask):
            pos[mask] = offset + (vals[mask] - start)
            assigned |= mask

    if not np.all(assigned):
        pos = np.where(assigned, pos, np.nan)
    return pos, total


def bin_angle(
    vals: np.ndarray,
    ranges: Iterable[AngleRange],
    n_bins: int,
    *,
    wrap: float | None = 2 * np.pi,
) -> np.ndarray:
    vals_norm = normalize_angles(vals, wrap=wrap)
    ranges_list = list(ranges)
    if len(ranges_list) == 1:
        start, end = ranges_list[0]
        return bin_col(vals_norm, n_bins=n_bins, vmin=start, vmax=end)

    clipped = clip_to_ranges(vals_norm, ranges_list)
    pos, total = _linearize_angles(clipped, ranges_list)
    edges = np.linspace(0.0, total, n_bins + 1, dtype=np.float32)
    out = np.digitize(pos, edges) - 1
    out = np.clip(out, 0, n_bins - 1)
    return out.astype(np.int32)


def angle_bin_centers(ranges: Iterable[AngleRange], n_bins: int) -> np.ndarray:
    ranges_list = list(ranges)
    if not ranges_list:
        raise ValueError("ranges must be non-empty")

    total = sum(end - start for start, end in ranges_list)
    edges = np.linspace(0.0, total, n_bins + 1, dtype=np.float64)
    centers_lin = 0.5 * (edges[:-1] + edges[1:])
    centers = np.zeros_like(centers_lin)

    offset = 0.0
    for idx, (start, end) in enumerate(ranges_list):
        length = end - start
        if idx == len(ranges_list) - 1:
            mask = (centers_lin >= offset) & (centers_lin <= offset + length)
        else:
            mask = (centers_lin >= offset) & (centers_lin < offset + length)
        centers[mask] = start + (centers_lin[mask] - offset)
        offset += length

    return centers
