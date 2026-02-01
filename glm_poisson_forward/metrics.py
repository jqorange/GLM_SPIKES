from typing import Tuple

import numpy as np
from scipy.stats import wilcoxon

EPS = 1e-12


def bernoulli_loglik(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def compute_llr_per_spike_bernoulli(
    y_cnt: np.ndarray,
    p_pred: np.ndarray,
    p_base: np.ndarray | float,
) -> float:
    y = np.asarray(y_cnt, dtype=np.float64).ravel()
    p = np.asarray(p_pred, dtype=np.float64).ravel()
    if y.size == 0:
        return float("nan")
    if np.isscalar(p_base):
        p0 = np.full_like(y, float(p_base), dtype=np.float64)
    else:
        p0 = np.asarray(p_base, dtype=np.float64).ravel()

    ll_m = bernoulli_loglik(y, p)
    ll_b = bernoulli_loglik(y, p0)

    nsp = float(np.sum(y))
    if nsp <= 0:
        return float("nan")
    return (ll_m - ll_b) / nsp


def build_oof_intercept_prob(y_cnt: np.ndarray, folds_idx) -> np.ndarray:
    y = np.asarray(y_cnt, dtype=np.float64).ravel()
    p_oof = np.full_like(y, fill_value=EPS, dtype=np.float64)
    for tr, va in folds_idx:
        mean_tr = float(np.mean(y[tr]))
        mean_tr = float(np.clip(mean_tr, EPS, 1.0 - EPS))
        p_oof[va] = mean_tr
    return p_oof


def wilcoxon_greater(a: np.ndarray, b: np.ndarray = None) -> Tuple[float, float, int]:
    if b is None:
        x = np.asarray(a, dtype=np.float64)
    else:
        x = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or np.allclose(x, 0):
        return 0.0, 1.0, 0
    stat, p = wilcoxon(
        x,
        alternative="greater",
        zero_method="wilcox",
        correction=False,
        mode="auto",
    )
    return float(stat), float(p), int(x.size)
