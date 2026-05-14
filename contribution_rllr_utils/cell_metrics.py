import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import scipy.io

from .constants import DAY_SEARCH_DIRS


def parse_day_id_from_session(session_name: str) -> Optional[str]:
    mF = re.search(r"F(\d+)", session_name, flags=re.IGNORECASE)
    mD = re.search(r"D(\d+)", session_name, flags=re.IGNORECASE)
    if mF and mD:
        return f"F{int(mF.group(1))}D{int(mD.group(1))}"
    return None


def parse_day_id_from_path(day_dir: Path) -> Optional[str]:
    s = str(day_dir)
    mF = re.search(r"F(\d+)", s, flags=re.IGNORECASE)
    mD = re.search(r"day\\s*([0-9]+)", s, flags=re.IGNORECASE)
    if mD is None:
        mD = re.search(r"day\s*([0-9]+)", s, flags=re.IGNORECASE)
    if mF and mD:
        return f"F{int(mF.group(1))}D{int(mD.group(1))}"
    return None


def find_cellinfo_mat(day_dir: Path) -> Optional[Path]:
    for pat in ["*cell_metrics.cellinfo.mat"]:
        hits = list(day_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def build_dayid_to_cellinfo(day_search_dirs: Optional[List[Path]] = None) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    search_dirs = day_search_dirs or DAY_SEARCH_DIRS
    for dd in search_dirs:
        day_id = parse_day_id_from_path(dd)
        if not day_id:
            continue
        ci = find_cellinfo_mat(dd)
        if ci:
            mapping[day_id] = ci
    return mapping


def load_cell_types(cellinfo_mat: Path) -> List[str]:
    md = scipy.io.loadmat(str(cellinfo_mat), squeeze_me=True, struct_as_record=False)
    if "cell_metrics" not in md:
        raise KeyError(f"{cellinfo_mat} missing cell_metrics")
    cm = md["cell_metrics"]
    if hasattr(cm, "putativeCellType"):
        raw = cm.putativeCellType
    elif isinstance(cm, dict) and "putativeCellType" in cm:
        raw = cm["putativeCellType"]
    else:
        raise KeyError(f"{cellinfo_mat} missing putativeCellType")
    if isinstance(raw, np.ndarray):
        return [str(x).strip() for x in raw.tolist()]
    return [str(raw).strip()]


def pyramidal_indices_for_session(session: str, dayid2cellinfo: Dict[str, Path], n_neurons: int) -> Optional[np.ndarray]:
    day_id = parse_day_id_from_session(session)
    if not day_id or day_id not in dayid2cellinfo:
        return None
    try:
        types = load_cell_types(dayid2cellinfo[day_id])
    except Exception:
        return None
    mask = [(t.lower() == "pyramidal cell") for t in types]
    if len(mask) < n_neurons:
        n_use = len(mask)
    else:
        n_use = n_neurons
    idx = np.array([i for i in range(n_use) if mask[i]], dtype=np.int32)
    return idx
