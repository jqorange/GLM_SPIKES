import numpy as np


def circular_trim_range(
    angles: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0
) -> tuple[float, float]:
    angles = np.asarray(angles, dtype=np.float64)
    angles = angles[np.isfinite(angles)]
    if angles.size == 0:
        return 0.0, 2.0 * np.pi
    angles = np.mod(angles, 2.0 * np.pi)
    if angles.size == 1:
        return float(angles[0]), 2.0 * np.pi

    sorted_angles = np.sort(angles)
    n = sorted_angles.size
    keep = int(np.ceil((upper_pct - lower_pct) / 100.0 * n))
    keep = max(1, min(keep, n))

    extended = np.concatenate([sorted_angles, sorted_angles + 2.0 * np.pi])
    min_width = np.inf
    start_idx = 0
    for i in range(n):
        width = extended[i + keep - 1] - extended[i]
        if width < min_width:
            min_width = width
            start_idx = i

    start = float(extended[start_idx] % (2.0 * np.pi))
    width = float(min_width)
    if not np.isfinite(width) or width <= 0:
        width = 2.0 * np.pi
    return start, width


def shift_angles(angles: np.ndarray, start: float) -> np.ndarray:
    angles = np.asarray(angles, dtype=np.float32)
    return np.mod(angles - float(start), 2.0 * np.pi).astype(np.float32)
