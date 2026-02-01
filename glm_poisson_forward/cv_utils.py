from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import KFold


def _blocks_to_indices(block_ids: np.ndarray, block_size: int, total_len: int) -> np.ndarray:
    parts = []
    for b in block_ids:
        start = int(b) * block_size
        end = min(start + block_size, total_len)
        if start < end:
            parts.append(np.arange(start, end, dtype=np.int64))
    if not parts:
        return np.array([], dtype=np.int64)
    return np.concatenate(parts)


def build_block_folds(
    total_len: int,
    block_size: int,
    n_splits: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if total_len <= 0:
        raise ValueError("total_len must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    n_blocks = int(np.ceil(total_len / block_size))
    n_splits = min(n_splits, n_blocks)
    if n_splits < 2:
        raise ValueError("need at least 2 blocks to build CV splits")

    block_ids = np.arange(n_blocks, dtype=np.int64)
    kf = KFold(n_splits=n_splits, shuffle=False)
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for tr_blocks, va_blocks in kf.split(block_ids):
        tr_idx = _blocks_to_indices(block_ids[tr_blocks], block_size, total_len)
        va_idx = _blocks_to_indices(block_ids[va_blocks], block_size, total_len)
        folds.append((tr_idx, va_idx))
    return folds
