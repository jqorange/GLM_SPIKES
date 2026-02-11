from typing import Tuple

import numpy as np
from scipy.stats import wilcoxon


def poisson_ll_noconst(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    mu = np.clip(mu, 1e-12, None)
    return float(np.sum(y * np.log(mu) - mu))


def compute_llhi_bps_poisson(y_cnt: np.ndarray, mu_pred: np.ndarray) -> float:
    y = np.asarray(y_cnt, dtype=np.float64).ravel()
    mu = np.asarray(mu_pred, dtype=np.float64).ravel()
    if y.size == 0:
        return float("nan")
    mu0 = np.full_like(y, fill_value=np.mean(y), dtype=np.float64)

    ll_m = poisson_ll_noconst(y, mu)
    ll_b = poisson_ll_noconst(y, mu0)

    nsp = float(np.sum(y))
    if nsp <= 0:
        return float("nan")
    return (ll_m - ll_b) / (nsp * np.log(2))


def compute_llhi_bps_poisson_vs_baseline(
    y_cnt: np.ndarray,
    mu_pred: np.ndarray,
    mu_base: np.ndarray,
) -> float:
    y = np.asarray(y_cnt, dtype=np.float64).ravel()
    mu = np.asarray(mu_pred, dtype=np.float64).ravel()
    mu0 = np.asarray(mu_base, dtype=np.float64).ravel()
    if y.size == 0:
        return float("nan")

    ll_m = poisson_ll_noconst(y, mu)
    ll_b = poisson_ll_noconst(y, mu0)

    nsp = float(np.sum(y))
    if nsp <= 0:
        return float("nan")
    return (ll_m - ll_b) / (nsp * np.log(2))


def compute_deviance_explained_poisson_vs_baseline(
    y_cnt: np.ndarray,
    mu_pred: np.ndarray,
    mu_base: np.ndarray,
) -> float:
    """Compute Poisson deviance explained relative to a baseline model.

    Returns 1 - D_model / D_base where D is Poisson deviance against the
    saturated model. Values may be negative when the model is worse than the
    baseline and approach 1.0 for near-perfect predictions.
    """
    y = np.asarray(y_cnt, dtype=np.float64).ravel()
    mu = np.asarray(mu_pred, dtype=np.float64).ravel()
    mu0 = np.asarray(mu_base, dtype=np.float64).ravel()
    if y.size == 0:
        return float("nan")

    mu = np.clip(mu, 1e-12, None)
    mu0 = np.clip(mu0, 1e-12, None)

    # Saturated-model log-likelihood (up to additive constants) for Poisson.
    # Use a mask instead of np.where to avoid evaluating log(0) on zero-count bins.
    ll_sat = np.zeros_like(y, dtype=np.float64)
    pos = y > 0
    ll_sat[pos] = y[pos] * np.log(y[pos]) - y[pos]
    ll_m = y * np.log(mu) - mu
    ll_b = y * np.log(mu0) - mu0

    dev_model = 2.0 * np.sum(ll_sat - ll_m)
    dev_base = 2.0 * np.sum(ll_sat - ll_b)

    if not np.isfinite(dev_base) or dev_base <= 0:
        return float("nan")
    return float(1.0 - (dev_model / dev_base))


def dll_bits_series_poisson(y_cnt: np.ndarray, mu_pred: np.ndarray) -> np.ndarray:
    y = np.asarray(y_cnt, dtype=np.float64).ravel()
    mu = np.asarray(mu_pred, dtype=np.float64).ravel()
    if y.size == 0:
        return np.array([], dtype=np.float32)

    mu = np.clip(mu, 1e-12, None)
    mean_rate = float(np.mean(y))
    mean_rate = max(mean_rate, 1e-12)

    ll_m = y * np.log(mu) - mu
    ll_b = y * np.log(mean_rate) - mean_rate
    dll = ll_m - ll_b
    return (dll / np.log(2)).astype(np.float32)


def build_oof_constant_mu(y_cnt: np.ndarray, folds_idx) -> np.ndarray:
    y = np.asarray(y_cnt, dtype=np.float64).ravel()
    mu_oof = np.full_like(y, fill_value=1e-12, dtype=np.float64)
    for tr, va in folds_idx:
        mean_tr = float(np.mean(y[tr]))
        mu_oof[va] = max(mean_tr, 1e-12)
    return mu_oof


def dll_bits_series_poisson_vs_baseline(
    y_cnt: np.ndarray,
    mu_pred: np.ndarray,
    mu_base: np.ndarray,
) -> np.ndarray:
    y = np.asarray(y_cnt, dtype=np.float64).ravel()
    mu = np.asarray(mu_pred, dtype=np.float64).ravel()
    mu0 = np.asarray(mu_base, dtype=np.float64).ravel()
    if y.size == 0:
        return np.array([], dtype=np.float32)

    mu = np.clip(mu, 1e-12, None)
    mu0 = np.clip(mu0, 1e-12, None)

    ll_m = y * np.log(mu) - mu
    ll_b = y * np.log(mu0) - mu0
    dll = ll_m - ll_b
    return (dll / np.log(2)).astype(np.float32)


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
