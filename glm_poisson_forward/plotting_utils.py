from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.ndimage import gaussian_filter1d

from .config import BIN_MS, PLOT_END_SEC, PLOT_SMOOTH_MS, PLOT_START_SEC, PLOT_ZSCORE


def plot_fitting_curve(
    out_png: Path,
    title: str,
    y_cnt: np.ndarray,
    mu_cnt: np.ndarray,
    *,
    bin_ms: int = BIN_MS,
    smooth_ms: float = PLOT_SMOOTH_MS,
    start_sec: float = PLOT_START_SEC,
    end_sec: float = PLOT_END_SEC,
    do_zscore: bool = PLOT_ZSCORE,
    zscore_eps: float = 1e-8,
):
    y_cnt = np.asarray(y_cnt, dtype=np.float64).ravel()
    mu_cnt = np.asarray(mu_cnt, dtype=np.float64).ravel()
    assert y_cnt.shape == mu_cnt.shape

    bin_sec = bin_ms / 1000.0
    y_rate = y_cnt / bin_sec
    mu_rate = mu_cnt / bin_sec

    sigma_bins = float(smooth_ms) / float(bin_ms)
    if sigma_bins > 0:
        y_s = gaussian_filter1d(y_rate.astype(np.float32), sigma=sigma_bins)
        mu_s = gaussian_filter1d(mu_rate.astype(np.float32), sigma=sigma_bins)
    else:
        y_s = y_rate.astype(np.float32)
        mu_s = mu_rate.astype(np.float32)

    t = np.arange(len(y_s), dtype=np.float64) * bin_sec
    s0 = max(0, int(np.floor(start_sec / bin_sec)))
    s1 = min(len(y_s), int(np.ceil(end_sec / bin_sec))) if end_sec is not None else len(y_s)
    if s1 <= s0:
        return

    y_w = y_s[s0:s1]
    mu_w = mu_s[s0:s1]
    t_w = t[s0:s1]

    if do_zscore:
        def _z(x: np.ndarray) -> np.ndarray:
            m = float(np.mean(x))
            sd = float(np.std(x))
            return (x - m) / max(sd, zscore_eps)

        y_plot = _z(y_w)
        mu_plot = _z(mu_w)
        ylab = "Z-score (smoothed rate)"
        lab_true = "True rate (z-scored)"
        lab_pred = "Pred rate (z-scored)"
    else:
        y_plot = y_w
        mu_plot = mu_w
        ylab = "Spikes/s (smoothed)"
        lab_true = "True rate (smoothed)"
        lab_pred = "Pred rate (smoothed)"

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 4))
    plt.plot(t_w, y_plot, label=lab_true, linewidth=2)
    plt.plot(t_w, mu_plot, label=lab_pred, linewidth=1.6, alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def load_oof_from_neuron_dir(neuron_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    fold_dirs = sorted([p for p in neuron_dir.glob("fold*") if p.is_dir()])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold dirs under {neuron_dir}")

    max_idx = -1
    parts = []
    for fd in fold_dirs:
        h5p = fd / "pred.h5"
        with h5py.File(h5p, "r") as hf:
            va_idx = hf["va_idx"][:].astype(np.int64)
            pred_mu = hf["pred_mu"][:].astype(np.float64)
            true_cnt = hf["true_cnt"][:].astype(np.float64)
        max_idx = max(max_idx, int(np.max(va_idx)))
        parts.append((va_idx, pred_mu, true_cnt))

    T = max_idx + 1
    mu_oof = np.full(T, np.nan, dtype=np.float64)
    y_oof = np.full(T, np.nan, dtype=np.float64)

    for va_idx, pred_mu, true_cnt in parts:
        mu_oof[va_idx] = pred_mu
        y_oof[va_idx] = true_cnt

    if np.any(~np.isfinite(mu_oof)) or np.any(~np.isfinite(y_oof)):
        m = np.nanmean(y_oof)
        mu_oof = np.where(np.isfinite(mu_oof), mu_oof, m)
        y_oof = np.where(np.isfinite(y_oof), y_oof, m)

    return y_oof, mu_oof

