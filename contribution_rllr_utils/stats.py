from typing import Dict, Tuple

import numpy as np
from .constants import CI_HI, CI_LO, MU_EPS, N_BOOT


def bernoulli_loglik(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(p, dtype=np.float64).ravel()
    p = np.clip(p, MU_EPS, 1.0 - MU_EPS)
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def build_oof_intercept_prob(y: np.ndarray, folds_idx) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    p_oof = np.full_like(y, MU_EPS, dtype=np.float64)
    for tr, va in folds_idx:
        mean_tr = float(np.mean(y[tr]))
        mean_tr = float(np.clip(mean_tr, MU_EPS, 1.0 - MU_EPS))
        p_oof[va] = mean_tr
    return p_oof


def hierarchical_bootstrap_mean(
    session_to_values: Dict[str, np.ndarray],
    n_boot: int = N_BOOT,
    ci_lo: float = CI_LO,
    ci_hi: float = CI_HI,
    seed: int | None = None,
) -> Tuple[float, float, float]:
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
