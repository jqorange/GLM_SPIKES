from typing import Dict, Tuple

import numpy as np
from scipy.special import gammaln

from .constants import MU_EPS, CI_HI, CI_LO, N_BOOT


def poisson_loglik(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Full Poisson log-likelihood including -log(y!), needed for deviance.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    mu = np.clip(mu, MU_EPS, None)
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1.0)))


def poisson_loglik_saturated(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).ravel()
    term = np.zeros_like(y, dtype=float)
    mask = y > 0
    term[mask] = y[mask] * np.log(y[mask])
    return float(np.sum(term - y - gammaln(y + 1.0)))


def deviance_from_ll(ll_sat: float, ll_model: float) -> float:
    return float(2.0 * (ll_sat - ll_model))


def devexpl_from_deviances(D_model: float, D_null: float) -> float:
    if not np.isfinite(D_model) or not np.isfinite(D_null) or D_null <= 0:
        return float("nan")
    return float(1.0 - (D_model / D_null))


def hierarchical_bootstrap_mean(
    session_to_values: Dict[str, np.ndarray],
    n_boot: int = N_BOOT,
    ci_lo: float = CI_LO,
    ci_hi: float = CI_HI,
    seed: int | None = None,
) -> Tuple[float, float, float]:
    """
    session_to_values: session -> 1D array of neuron-level values
    Bootstrap:
      sample sessions with replacement
      within each sampled session sample neurons with replacement
      compute grand mean over pooled sampled neurons
    Returns: (mean, lo, hi)
    """
    rng = np.random.default_rng(seed)
    sessions = list(session_to_values.keys())
    if len(sessions) == 0:
        return float("nan"), float("nan"), float("nan")

    pooled = np.concatenate([session_to_values[s] for s in sessions], axis=0)
    pooled = pooled[np.isfinite(pooled)]
    point = float(np.mean(pooled)) if pooled.size else float("nan")

    boots = []
    for _ in range(int(n_boot)):
        ss = rng.choice(sessions, size=len(sessions), replace=True)
        vals = []
        for s in ss:
            arr = session_to_values[s]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            draw = rng.choice(arr, size=arr.size, replace=True)
            vals.append(draw)
        if not vals:
            boots.append(np.nan)
        else:
            vv = np.concatenate(vals, axis=0)
            boots.append(float(np.mean(vv)))
    boots = np.asarray(boots, dtype=np.float64)
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(boots, ci_lo))
    hi = float(np.percentile(boots, ci_hi))
    return point, lo, hi
