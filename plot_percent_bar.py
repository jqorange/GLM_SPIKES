# -*- coding: utf-8 -*-
import argparse
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import scipy.io
import scipy.stats
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from glm_poisson_forward.config import FS_HZ, INCLUDE_TIME_VARIABLE, SPIKE_INPUT_MODE, SPIKE_INPUT_ROOT
from glm_poisson_forward.io_utils import load_spikes_50hz_counts

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Liberation Sans", "Arial", "DejaVu Sans"]

INDOOR_BASE_COLOR = "#838383"
OUTDOOR_BASE_COLOR = "#61B861"
POINT_COLOR = "#838383"
PAIR_LINE_COLOR = "#B5B5B5"
YLABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 15
AXIS_LINEWIDTH = 1.8
BOX_LINEWIDTH = 1.8
MEDIAN_LINEWIDTH = 2.2
WHISKER_LINEWIDTH = 1.8
CAP_LINEWIDTH = 1.8
ANNOTATION_LINEWIDTH = 1.8


def _shift_color(color: str, amount: float):
    rgb = np.array(mpl.colors.to_rgb(color), dtype=float)
    amount = float(np.clip(amount, -0.22, 0.42))
    if amount >= 0.0:
        mixed = rgb * (1.0 - amount) + amount
    else:
        mixed = rgb * (1.0 + amount)
    return tuple(np.clip(mixed, 0.0, 1.0))


def _label_shade_map(labels_sorted, base_color: str) -> Dict[str, tuple]:
    labels = list(labels_sorted)
    if not labels:
        return {}
    if len(labels) == 1:
        return {labels[0]: _shift_color(base_color, 0.0)}
    shade_levels = np.linspace(-0.14, 0.34, len(labels))
    return {lb: _shift_color(base_color, shade) for lb, shade in zip(labels, shade_levels)}

# ===================== 配置区 =====================

WEIGHTS_BASE = Path(r"/home/js3785/Codes/GLM_Git/GLM_SPIKES/weights_Poisson_forward")  # 修改成你的路径
 
# DAY_SEARCH_DIRS = [
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F4/day1",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F4/day4",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F5/Merged/day2/121_day2",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F5/Merged/day3/121_day3",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F5/Merged/day4/121_day4",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F5/Merged/day5/121_day5",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F5/Merged/day6/3E6_day6",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F5/Merged/day7/121_day7",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F5/Merged/day10/121_day10",
#     # r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F6/Merged/day2/3E6_day2",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F6/Merged/day3/3E6_day3",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F6/Merged/day5/3E6_day5",
#     #r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F6/Merged/day7/3E6_day7",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F6/Merged/day8/3E6_day8",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F6/Merged/day9/3E6_day9",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F6/Merged/day10/3E6_day10",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F6/Merged/day4/3E6_day4",
#     r"/local/storage/backup/ayadataA/current/ayadata4/data/FieldRat/2024/F6/Merged/day6/121_day6",
# ]
DAY_SEARCH_DIRS = [
    r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F5/Merged/day2/121_day2",
    r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F5/Merged/day3/121_day3",
    r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F5/Merged/day7/121_day7",
    r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F5/Merged/day10/121_day10",


    r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day3/3E6_day3",
    r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day7/3E6_day7",
    r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day8/3E6_day8",
    r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day10/3E6_day10",

    # r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F8/Merged/day2/3E6_day2", # WIP
]

DAY_SEARCH_DIRS = [Path(p) for p in DAY_SEARCH_DIRS]

BASE_LETTER_ORDER = "PSRYI"
LETTER_ORDER = f"{BASE_LETTER_ORDER}{'T' if INCLUDE_TIME_VARIABLE else ''}H"
NAME2LETTER = {
    "position": "P", "p": "P",
    "speed":    "S", "s": "S",
    "roll":     "R", "r": "R",
    "yaw":      "Y", "y": "Y",
    "pitch":    "I", "i": "I",
    "time":     "T", "t": "T",

}
DEFAULT_PRESENCE_LETTERS = f"{BASE_LETTER_ORDER}{'T' if INCLUDE_TIME_VARIABLE else ''}"
DEFAULT_PRESENCE_SPECS = [
    DEFAULT_PRESENCE_LETTERS,
    f"PSH{'T' if INCLUDE_TIME_VARIABLE else ''}:H=RYI",
]
FITTED_DENOMINATOR_PRESENCE_SPECS = [
    (f"PSH{'T' if INCLUDE_TIME_VARIABLE else ''}", {"H": {"R", "Y", "I"}}),
    (DEFAULT_PRESENCE_LETTERS, {}),
]
DEFAULT_MIN_FIRING_RATE_HZ = 0.01
# ===================== 工具函数 =====================


def _parse_composite_map(expr: str) -> Dict[str, set]:
    """
    解析形如 "H=RYI,V=ZXY" 的合成规则，返回 {"H":{"R","Y","I"}, "V":{"Z","X","Y"}}
    也支持 "H=R+Y+I" 这种写法。
    """
    out: Dict[str, set] = {}
    expr = (expr or "").strip()
    if not expr:
        return out
    for item in expr.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        lhs, rhs = item.split("=", 1)
        lhs = lhs.strip().upper()
        rhs = rhs.strip().upper().replace("+", "")
        if not lhs or len(lhs) != 1:
            continue
        members = {ch for ch in rhs if ch.isalpha()}
        if members:
            out[lhs] = members
    return out


def _parse_presence_spec(spec: str) -> Tuple[str, Dict[str, set]]:
    """
    解析一个统计规格：
      - "PSRYIT"
      - "PSHT:H=RYI"
    返回 (letters, composite_map)
    """
    s = (spec or "").strip()
    if not s:
        return DEFAULT_PRESENCE_LETTERS, {}
    if ":" not in s:
        letters = "".join([ch for ch in s.upper() if ch.isalpha()])
        return (_sanitize_presence_letters(letters) or DEFAULT_PRESENCE_LETTERS), {}
    letters_part, comp_part = s.split(":", 1)
    letters = _sanitize_presence_letters("".join([ch for ch in letters_part.upper() if ch.isalpha()]))
    comp = _parse_composite_map(comp_part)
    return (letters if letters else DEFAULT_PRESENCE_LETTERS), comp


def _presence_tag(letters: str) -> str:
    letters = _sanitize_presence_letters("".join([ch for ch in (letters or "").upper() if ch.isalpha()]))
    if not letters:
        letters = DEFAULT_PRESENCE_LETTERS
    return letters


def _should_exclude_combo_label(label: str) -> bool:
    s = str(label).upper()
    return s == "N"


def _drop_excluded_combo_labels(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty or "label" not in long_df.columns:
        return long_df.copy()
    out = long_df.copy()
    out["label"] = out["label"].astype(str)
    return out.loc[~out["label"].map(_should_exclude_combo_label)].copy()


def _sanitize_presence_letters(letters: str) -> str:
    clean = []
    seen = set()
    for ch in str(letters).upper():
        if (not ch.isalpha()) or ch == "N" or ch in seen:
            continue
        if ch == "T" and not INCLUDE_TIME_VARIABLE:
            continue
        clean.append(ch)
        seen.add(ch)
    return "".join(clean)


def _selected_model_map(df: pd.DataFrame) -> Dict[int, object]:
    """
    从 selected_models.csv 构造 1-based neuron_id -> final_model 的映射。
    不再假设 CSV 行顺序与 cell index 一一对应。
    """
    if "final_model" not in df.columns:
        return {}

    if "neuron" not in df.columns:
        # 兼容旧格式：如果没有 neuron 列，只能退回到行序假设。
        return {i + 1: m for i, m in enumerate(df["final_model"].tolist())}

    out: Dict[int, object] = {}
    for _, row in df.iterrows():
        neuron_name = str(row.get("neuron", "")).strip()
        m = re.search(r"(\d+)$", neuron_name)
        if not m:
            continue
        neuron_id = int(m.group(1))
        if neuron_id <= 0:
            continue
        out[neuron_id] = row.get("final_model", None)
    return out

def _canonical_label(model, letter_order=LETTER_ORDER):
    """
    把模型名/token归一化为字母组合(例如 'position+speed' -> 'PS')，并按 LETTER_ORDER 排序。
    改动：缺失/NaN/空/不可解析 -> 直接返回 'N'
    """
    # 缺失：None / NaN
    try:
        if model is None or pd.isna(model):
            return "N"
    except Exception:
        if model is None:
            return "N"

    # 字符串规范化
    if isinstance(model, str):
        s0 = model.strip()
        if s0 == "":
            return "N"
        low = s0.lower()
        if low in {"nan", "none", "null", "na"}:
            return "N"

        def parts_from_str(s: str):
            ps = re.split(r"[ _\-+&/,]+", s.strip())
            # 若是像 "PS" 这种纯大写字母缩写
            if len(ps) == 1 and ps[0] and ps[0].isalpha() and ps[0].upper() == ps[0]:
                return list(ps[0])
            return ps

        tokens = [t.lower() for t in parts_from_str(s0) if t]
        candidates = set(tokens)
        # Recover multi-token variable names split by "_" in model keys,
        # e.g., ego + heading -> ego_heading, imu + dlc + d -> imu_dlc_d.
        n = len(tokens)
        for i in range(n):
            if i + 1 < n:
                candidates.add(f"{tokens[i]}_{tokens[i+1]}")
            if i + 2 < n:
                candidates.add(f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}")
            if i + 3 < n:
                candidates.add(f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}_{tokens[i+3]}")
        letters = set(NAME2LETTER.get(tok, None) for tok in candidates)

    else:
        # 非字符串（比如数字等）也视为不可解析
        return "N"

    letters.discard(None)
    if not letters:
        return "N"

    ordered = [c for c in letter_order if c in letters]
    if not ordered:
        return "N"
    return "".join(ordered)

def _sort_labels(labels, letter_order=LETTER_ORDER):
    pos = {c: i for i, c in enumerate(letter_order)}

    def key(lb):
        if lb == "N":
            return (10**9, [10**9])  # 永远最后
        return (len(lb), [pos[c] for c in lb])

    return sorted(labels, key=key)


def _pvalue_to_stars(pvalue: float) -> str:
    if pd.isna(pvalue):
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def _save_figure_png_svg(fig: plt.Figure, save_path: Path) -> None:
    save_path = Path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".svg"), format="svg", bbox_inches="tight")
    print(f"图已保存到: {save_path}")

def parse_day_id_from_session(session_name: str) -> Optional[str]:
    mF = re.search(r'F(\d+)', session_name, flags=re.IGNORECASE)
    mD = re.search(r'D(\d+)', session_name, flags=re.IGNORECASE)
    if mF and mD:
        return f"F{int(mF.group(1))}D{int(mD.group(1))}"
    return None

def parse_day_id_from_path(day_dir: Path) -> Optional[str]:
    s = str(day_dir)
    mF = re.search(r'F(\d+)', s, flags=re.IGNORECASE)
    mD = re.search(r'day\s*([0-9]+)', s, flags=re.IGNORECASE)
    if mF and mD:
        return f"F{int(mF.group(1))}D{int(mD.group(1))}"
    return None

def find_cellinfo_mat(day_dir: Path) -> Optional[Path]:
    for pat in ["*cell_metrics*.mat", "*cellinfo*.mat"]:
        hits = list(day_dir.glob(pat))
        if hits:
            return hits[0]
    return None

def build_dayid_to_cellinfo() -> Dict[str, Path]:
    mapping = {}
    for dd in DAY_SEARCH_DIRS:
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
        raise KeyError(f"{cellinfo_mat} 没有 cell_metrics")

    cellm = md["cell_metrics"]
    if hasattr(cellm, "putativeCellType"):
        raw = cellm.putativeCellType
    elif isinstance(cellm, dict) and "putativeCellType" in cellm:
        raw = cellm["putativeCellType"]
    else:
        raise KeyError(f"{cellinfo_mat} 没有 putativeCellType")

    if isinstance(raw, np.ndarray):
        return [str(x).strip() for x in raw.tolist()]
    return [str(raw).strip()]

def _infer_group_from_session_name(session_name: str) -> Optional[str]:
    s = session_name.lower()
    if "indoor" in s:
        return "indoor"
    if "outdoor" in s:
        return "outdoor"
    return None

def _pair_id_from_session_name(session_name: str) -> str:
    """
    生成配对 ID：从 session 名里去掉 indoor/outdoor，尽量把同一条记录的 indoor/outdoor 对齐。
    你如果有更严格的命名规则，也可以在这里改成更精准的解析。
    """
    s = session_name.lower()
    s = s.replace("indoor", "").replace("outdoor", "")
    s = re.sub(r"__+", "_", s)
    s = re.sub(r"[-\s]+", "_", s)
    s = s.strip("_")
    return s


def _cell_type_mask(cell_types: List[str], pyramidal_only: bool) -> np.ndarray:
    if pyramidal_only:
        return np.array([t.lower() == "pyramidal cell" for t in cell_types], dtype=bool)
    return np.ones(len(cell_types), dtype=bool)


def _firing_rate_mask_from_1000hz(
    session_name: str,
    n_cells: int,
    min_firing_rate_hz: float,
) -> Optional[np.ndarray]:
    """
    返回按平均发放率筛选的细胞掩码（长度 = n_cells）。
    平均发放率基于当前配置的 spike 输入：
    - binary: *_1000Hz.h5，经 load_spikes_50hz_counts 聚合到 50 Hz
    - count:  *_50Hz_count.h5，直接读取 spike_count
    """
    if min_firing_rate_hz <= 0:
        return np.ones(n_cells, dtype=bool)

    if SPIKE_INPUT_MODE == "binary":
        spike_path = SPIKE_INPUT_ROOT / f"{session_name}_1000Hz.h5"
    else:
        fs_tag = str(int(FS_HZ)) if float(FS_HZ).is_integer() else f"{FS_HZ:g}"
        spike_path = SPIKE_INPUT_ROOT / f"{session_name}_{fs_tag}Hz_count.h5"
    if not spike_path.exists():
        print(f"[SKIP] {session_name}: spike file not found: {spike_path}")
        return None

    try:
        y50 = load_spikes_50hz_counts(spike_path)  # (T50, N)
    except Exception as exc:
        print(f"[SKIP] {session_name}: failed to load spike file ({spike_path}): {exc}")
        return None

    if y50.ndim != 2 or y50.shape[0] == 0:
        print(f"[SKIP] {session_name}: invalid spike shape {y50.shape}")
        return None

    n_spk_cells = int(y50.shape[1])
    n_common = min(n_cells, n_spk_cells)
    if n_common <= 0:
        print(f"[SKIP] {session_name}: no overlap cells between cell_metrics and spike file")
        return None

    fr_hz = y50[:, :n_common].mean(axis=0).astype(float) * float(FS_HZ)
    mask = np.zeros(n_cells, dtype=bool)
    mask[:n_common] = fr_hz >= float(min_firing_rate_hz)

    if n_cells != n_spk_cells:
        print(
            f"[WARN] {session_name}: cell count mismatch cell_metrics={n_cells}, "
            f"spike={n_spk_cells}; using first {n_common} cells."
        )
    return mask


def collect_filtered_neuron_records(
    weights_base: Path,
    dayid2cellinfo: Dict[str, Path],
    *,
    pyramidal_only: bool = True,
    min_firing_rate_hz: float = DEFAULT_MIN_FIRING_RATE_HZ,
) -> pd.DataFrame:
    """
    导出被筛掉的 neuron 编号。
    筛选规则来自最终统计实际使用的 cell_mask = type_mask & fr_mask。
    仅输出 excluded 的 neuron，并标明是被 cell type、firing rate 或两者共同筛掉。
    """
    records = []

    for sess_dir in weights_base.iterdir():
        if not sess_dir.is_dir():
            continue

        session_name = sess_dir.name
        group = _infer_group_from_session_name(session_name)
        if group is None:
            continue

        csv_path = sess_dir / "selected_models.csv"
        if not csv_path.exists():
            continue

        day_id = parse_day_id_from_session(session_name)
        if (not day_id) or (day_id not in dayid2cellinfo):
            continue

        try:
            cell_types = load_cell_types(dayid2cellinfo[day_id])
        except Exception:
            continue

        type_mask = _cell_type_mask(cell_types, pyramidal_only=pyramidal_only)
        fr_mask = _firing_rate_mask_from_1000hz(session_name, len(cell_types), min_firing_rate_hz)
        if fr_mask is None:
            continue

        n_cells = len(cell_types)
        n_spk_cells = int(np.sum(fr_mask)) if min_firing_rate_hz <= 0 else None
        if min_firing_rate_hz <= 0:
            fr_hz = np.full(n_cells, np.nan, dtype=float)
        else:
            if SPIKE_INPUT_MODE == "binary":
                spike_path = SPIKE_INPUT_ROOT / f"{session_name}_1000Hz.h5"
            else:
                fs_tag = str(int(FS_HZ)) if float(FS_HZ).is_integer() else f"{FS_HZ:g}"
                spike_path = SPIKE_INPUT_ROOT / f"{session_name}_{fs_tag}Hz_count.h5"
            try:
                y50 = load_spikes_50hz_counts(spike_path)
                n_spk_cells = int(y50.shape[1]) if y50.ndim == 2 else 0
                n_common = min(n_cells, n_spk_cells)
                fr_hz = np.full(n_cells, np.nan, dtype=float)
                if n_common > 0:
                    fr_hz[:n_common] = y50[:, :n_common].mean(axis=0).astype(float) * float(FS_HZ)
            except Exception:
                fr_hz = np.full(n_cells, np.nan, dtype=float)

        cell_mask = type_mask & fr_mask
        for idx0 in range(n_cells):
            if cell_mask[idx0]:
                continue
            excluded_by_type = not bool(type_mask[idx0])
            excluded_by_fr = not bool(fr_mask[idx0])
            reasons = []
            if excluded_by_type:
                reasons.append("cell_type")
            if excluded_by_fr:
                reasons.append("firing_rate")

            records.append(
                {
                    "session": session_name,
                    "group": group,
                    "day_id": day_id,
                    "neuron_idx": int(idx0 + 1),
                    "neuron_idx_0based": int(idx0),
                    "cell_type": str(cell_types[idx0]).strip(),
                    "firing_rate_hz": float(fr_hz[idx0]) if np.isfinite(fr_hz[idx0]) else np.nan,
                    "min_firing_rate_hz": float(min_firing_rate_hz),
                    "excluded_by_cell_type": bool(excluded_by_type),
                    "excluded_by_firing_rate": bool(excluded_by_fr),
                    "exclude_reason": "+".join(reasons),
                    "in_spike_file_range": bool(np.isfinite(fr_hz[idx0])),
                    "pyramidal_only": bool(pyramidal_only),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=[
                "session",
                "group",
                "day_id",
                "neuron_idx",
                "neuron_idx_0based",
                "cell_type",
                "firing_rate_hz",
                "min_firing_rate_hz",
                "excluded_by_cell_type",
                "excluded_by_firing_rate",
                "exclude_reason",
                "in_spike_file_range",
                "pyramidal_only",
            ]
        )

    return pd.DataFrame(records)

# ===================== 核心：按 session 计算百分比（组合标签版） =====================

def gather_session_percentages(weights_base: Path, dayid2cellinfo: Dict[str, Path],
                               letter_order=LETTER_ORDER,
                               pyramidal_only: bool = True,
                               min_firing_rate_hz: float = DEFAULT_MIN_FIRING_RATE_HZ) -> Tuple[pd.DataFrame, List[str]]:
    """
    返回：
      long_df: 每行 = (session, group, label, percent, count, total_cells)
      labels_sorted: 全部出现过的 label 的排序列表（用于作图顺序一致）
    """
    records = []
    all_labels = set()

    for sess_dir in weights_base.iterdir():
        if not sess_dir.is_dir():
            continue

        session_name = sess_dir.name
        group = _infer_group_from_session_name(session_name)
        if group is None:
            continue

        csv_path = sess_dir / "selected_models.csv"
        if not csv_path.exists():
            continue

        day_id = parse_day_id_from_session(session_name)
        if (not day_id) or (day_id not in dayid2cellinfo):
            continue

        try:
            cell_types = load_cell_types(dayid2cellinfo[day_id])
        except Exception:
            continue

        type_mask = _cell_type_mask(cell_types, pyramidal_only=pyramidal_only)
        fr_mask = _firing_rate_mask_from_1000hz(session_name, len(cell_types), min_firing_rate_hz)
        if fr_mask is None:
            continue
        cell_mask = type_mask & fr_mask

        df = pd.read_csv(csv_path)
        if "final_model" not in df.columns:
            continue
        model_map = _selected_model_map(df)

        # 假设 CSV 行顺序与细胞顺序一致：只保留 pyramidal 的 model
        # === 分母用“全部 pyramidal cell”（以 cell_metrics 为准） ===
        total_cells = int(np.sum(cell_mask))
        if total_cells <= 0:
            continue

        labels_for_pyr = []

        # 对每个 pyramidal cell，都产生一个 label；缺失/不可解析 -> 'N'
        for i, use_cell in enumerate(cell_mask):
            if not use_cell:
                continue
            m = model_map.get(i + 1, None)  # neuron_1 表示第一个细胞
            labels_for_pyr.append(_canonical_label(m, letter_order=letter_order))

        # 该 session 内计数（此时 sum(counts) 必须 == total_cells）
        c = Counter(labels_for_pyr)
        if "N" not in c:
            c["N"] = 0

        all_labels.update(c.keys())

        total = total_cells
        for lb, cnt in c.items():
            pct = (cnt / total) * 100.0
            records.append({
                "session": session_name,
                "group": group,
                "label": lb,
                "percent": pct,
                "count": int(cnt),
                "total_cells": int(total),
            })

    if not records:
        return pd.DataFrame(columns=["session", "group", "label", "percent", "count", "total_cells"]), []

    long_df = pd.DataFrame(records)
    labels_sorted = _sort_labels(list(all_labels), letter_order=letter_order)

    # 补齐每个 session 缺失 label = 0%
    sessions = long_df[["session", "group"]].drop_duplicates()

    pivot = long_df.pivot_table(index="session", columns="label", values="percent", aggfunc="first")
    pivot = pivot.reindex(columns=labels_sorted).fillna(0.0)

    pivot_long = pivot.reset_index().melt(id_vars="session", var_name="label", value_name="percent")

    sess2group = dict(zip(sessions["session"], sessions["group"]))
    pivot_long["group"] = pivot_long["session"].map(sess2group)

    # count/total_cells 对 boxplot 不是必须；保留 percent/group/session/label 即可
    return pivot_long, labels_sorted

# ===================== 新增：按 session 计算“单字母覆盖率”百分比（PSRYITN 七类） =====================

def gather_session_letter_presence(weights_base: Path, dayid2cellinfo: Dict[str, Path],
                                   letters: str = DEFAULT_PRESENCE_LETTERS,
                                   composite_map: Optional[Dict[str, set]] = None,
                                   letter_order: str = LETTER_ORDER,
                                   pyramidal_only: bool = True,
                                   min_firing_rate_hz: float = DEFAULT_MIN_FIRING_RATE_HZ,
                                   denominator_mode: str = "all_filtered") -> Tuple[pd.DataFrame, List[str]]:
    """
    画第二张图：只统计 P/S/R/Y/I/T/N 七类的百分比（按 pyramidal cell 分母）。
    规则：只要 final_model 的 canonical label 里包含该字母，就计入该字母（可重叠计数）。
          缺失/不可解析 -> 'N'，并且只计入 N（不计入其它字母）。
    返回：
      long_df2: 每行 = (session, group, label in {P,S,R,Y,I,T,N}, percent)
      labels_sorted2: 固定顺序 ['P','S','R','Y','I','T','N']（可按你想要的顺序改）
    """
    letters = letters.upper()
    composite_map = composite_map or {}
    letters = _sanitize_presence_letters(letters)
    labels_sorted2 = list(letters)
    if denominator_mode not in {"all_filtered", "fitted_filtered"}:
        raise ValueError("denominator_mode must be 'all_filtered' or 'fitted_filtered'")

    records = []
    for sess_dir in weights_base.iterdir():
        if not sess_dir.is_dir():
            continue

        session_name = sess_dir.name
        group = _infer_group_from_session_name(session_name)
        if group is None:
            continue

        csv_path = sess_dir / "selected_models.csv"
        if not csv_path.exists():
            continue

        day_id = parse_day_id_from_session(session_name)
        if (not day_id) or (day_id not in dayid2cellinfo):
            continue

        try:
            cell_types = load_cell_types(dayid2cellinfo[day_id])
        except Exception:
            continue

        type_mask = _cell_type_mask(cell_types, pyramidal_only=pyramidal_only)
        fr_mask = _firing_rate_mask_from_1000hz(session_name, len(cell_types), min_firing_rate_hz)
        if fr_mask is None:
            continue
        cell_mask = type_mask & fr_mask

        df = pd.read_csv(csv_path)
        if "final_model" not in df.columns:
            continue
        model_map = _selected_model_map(df)

        # 统计：每个 letter 出现于多少通过基础筛选的 cells（允许重叠）。
        # 分母固定为 total_cells；其中 N 保留在分母里，但不会计入任何 letter。
        counts = {lb: 0 for lb in labels_sorted2}
        denom_cells = 0

        for i, use_cell in enumerate(cell_mask):
            if not use_cell:
                continue
            m = model_map.get(i + 1, None)
            lab = _canonical_label(m, letter_order=letter_order)  # e.g., "PSR" or "N"

            if lab == "N":
                if denominator_mode == "all_filtered":
                    denom_cells += 1
                continue
            denom_cells += 1

            # 只要有就算：比如 "PS" 同时计入 P 和 S
            lab_set = set(lab)
            for ch in letters:
                members = composite_map.get(ch, {ch})
                if lab_set & members:
                    counts[ch] += 1

        if denom_cells <= 0:
            continue

        # 写 records（分母固定为 total_cells）
        for lb in labels_sorted2:
            pct = (counts[lb] / denom_cells) * 100.0
            records.append({
                "session": session_name,
                "group": group,
                "label": lb,
                "percent": float(pct),
                "count": int(counts[lb]),
                "total_cells": int(denom_cells),
            })

    if not records:
        return pd.DataFrame(columns=["session", "group", "label", "percent", "count", "total_cells"]), labels_sorted2

    long_df2 = pd.DataFrame(records)

    # 补齐每个 session 缺失 label = 0%（理论上不会缺，但稳妥）
    sessions = long_df2[["session", "group"]].drop_duplicates()

    pivot = long_df2.pivot_table(index="session", columns="label", values="percent", aggfunc="first")
    pivot = pivot.reindex(columns=labels_sorted2).fillna(0.0)

    pivot_long = pivot.reset_index().melt(id_vars="session", var_name="label", value_name="percent")
    sess2group = dict(zip(sessions["session"], sessions["group"]))
    pivot_long["group"] = pivot_long["session"].map(sess2group)
    totals = long_df2.groupby(["session", "group"], as_index=False).agg(total_cells=("total_cells", "max"))
    pivot_long = pivot_long.merge(totals, on=["session", "group"], how="left")

    return pivot_long, labels_sorted2

# ===================== 作图：box plot + session 点 =====================

def plot_percent_box_by_session(long_df, labels_sorted,
                                title=None,
                                save_path=None,
                                show_points=True,
                                box_alpha=0.35,
                                point_alpha=0.65,
                                connect_pairs=True,
                                pair_line_alpha=0.35,
                                pair_line_width=0.7,
                                stats_df: Optional[pd.DataFrame] = None,
                                significance_pvalue_col: str = "paired_t_pvalue",
                                cell_scope: str = "pyramidal"):

    """
    对每个 label：画 indoor/outdoor 两个箱线图（按 session 分布），叠加每个 session 的散点（灰色、无 jitter），
    并可选把 paired 的 outdoor/indoor 点之间用浅红细线连接。
    """
    if long_df.empty or not labels_sorted:
        raise ValueError("没有可用数据：long_df 为空或 labels_sorted 为空。")

    groups = ["indoor", "outdoor"]  # 固定顺序：左 indoor，右 outdoor
    present_groups = [g for g in groups if g in set(long_df["group"].unique())]
    if not present_groups:
        raise ValueError("long_df 中没有 indoor/outdoor 组数据。")

    group_color = {
        "outdoor": OUTDOOR_BASE_COLOR,
        "indoor": INDOOR_BASE_COLOR,
    }
    label_color_map = {
        g: _label_shade_map(labels_sorted, base_color)
        for g, base_color in group_color.items()
    }

    df = long_df.copy()
    df["pair_id"] = df["session"].astype(str).map(_pair_id_from_session_name)

    fig_w = max(4.2, 0.34 * len(labels_sorted) + 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, 5.25))

    x = np.arange(len(labels_sorted))
    offsets = {"indoor": -0.2, "outdoor": +0.2}
    box_width = 0.32

    # Debug prints（保留你原有的检查逻辑）
    print("long_df rows:", df.shape[0])
    print("unique sessions:", df["session"].nunique())
    print("unique labels:", df["label"].nunique())
    print("unique pair_id:", df["pair_id"].nunique())

    # 看 session-level 的 indoor/outdoor 是否真的成对
    sess = df[["session", "group", "pair_id"]].drop_duplicates()
    pairs = sess.pivot_table(index="pair_id", columns="group", values="session", aggfunc="first")
    print("paired rows (both indoor & outdoor present):", pairs.dropna().shape[0])
    print("unpaired rows:", pairs[pairs.isna().any(axis=1)].shape[0])

    # 先画箱线图（两组）
    for g in present_groups:
        data = []
        for lb in labels_sorted:
            vals = df.loc[(df["group"] == g) & (df["label"] == lb), "percent"].astype(float).values
            data.append(vals)

        positions = x + offsets[g]

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(linewidth=BOX_LINEWIDTH),
            medianprops=dict(linewidth=MEDIAN_LINEWIDTH),
            whiskerprops=dict(linewidth=WHISKER_LINEWIDTH, alpha=0.9),
            capprops=dict(linewidth=CAP_LINEWIDTH, alpha=0.9),
        )
        for i, lb in enumerate(labels_sorted):
            color = label_color_map[g][lb]
            bp["boxes"][i].set_facecolor(color)
            bp["boxes"][i].set_edgecolor(color)
            bp["boxes"][i].set_alpha(box_alpha)
            bp["medians"][i].set_color(color)
            for whisker in bp["whiskers"][2 * i: 2 * i + 2]:
                whisker.set_color(color)
            for cap in bp["caps"][2 * i: 2 * i + 2]:
                cap.set_color(color)

    # 再画 paired 连线（放在点下面）
    if connect_pairs and ("outdoor" in present_groups) and ("indoor" in present_groups):
        # 以 pair_id + label 为单位，找到 indoor/outdoor 都存在的配对
        piv = df.pivot_table(index=["pair_id", "label"], columns="group", values="percent", aggfunc="first")
        if "outdoor" in piv.columns and "indoor" in piv.columns:
            for i, lb in enumerate(labels_sorted):
                sub = piv.loc[piv.index.get_level_values("label") == lb]
                sub = sub.dropna(subset=["outdoor", "indoor"], how="any")
                if sub.empty:
                    continue

                x_out = x[i] + offsets["outdoor"]
                x_in  = x[i] + offsets["indoor"]

                y_out = sub["outdoor"].astype(float).values
                y_in  = sub["indoor"].astype(float).values

                for yo, yi in zip(y_out, y_in):
                    ax.plot([x_out, x_in], [yo, yi],
                            color=PAIR_LINE_COLOR,
                            alpha=pair_line_alpha,
                            linewidth=pair_line_width,
                            zorder=2)

    # 最后画灰色小点（放在最上层）
    if show_points:
        for g in present_groups:
            positions = x + offsets[g]
            for i, lb in enumerate(labels_sorted):
                vals = df.loc[(df["group"] == g) & (df["label"] == lb), "percent"].astype(float).values
                if vals.size == 0:
                    continue
                ax.scatter(
                    np.full(vals.size, positions[i]),
                    vals,
                    s=10,
                    c=[label_color_map[g][lb]],
                    alpha=point_alpha,
                    linewidths=0,
                    zorder=3,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted)
    ax.set_ylabel(f"Percentage of {cell_scope} cells per session (%)", fontsize=YLABEL_FONTSIZE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(axis="both", width=AXIS_LINEWIDTH, labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlim(-0.6, len(labels_sorted) - 0.4)

    ymax = max(5.0, float(np.nanmax(df["percent"].values)) * 1.15)

    stats_lookup = {}
    pvalue_col = significance_pvalue_col
    if stats_df is not None and (not stats_df.empty) and ("label" in stats_df.columns):
        if pvalue_col not in stats_df.columns and pvalue_col == "paired_t_test_pvalue" and "paired_t_pvalue" in stats_df.columns:
            pvalue_col = "paired_t_pvalue"
        if pvalue_col in stats_df.columns:
            stats_lookup = (
                stats_df[["label", pvalue_col]]
                .dropna(subset=["label"])
                .drop_duplicates(subset=["label"], keep="first")
                .set_index("label")[pvalue_col]
                .to_dict()
            )

    annotations = []
    if ("outdoor" in present_groups) and ("indoor" in present_groups) and stats_lookup:
        y_pad = max(0.8, ymax * 0.03)
        text_pad = max(0.5, ymax * 0.02)
        for i, lb in enumerate(labels_sorted):
            pvalue = stats_lookup.get(lb, np.nan)
            stars = _pvalue_to_stars(pvalue)
            if not stars:
                continue

            vals = df.loc[df["label"] == lb, "percent"].astype(float).values
            local_max = float(np.nanmax(vals)) if vals.size else 0.0
            line_y = local_max + y_pad
            text_y = line_y + text_pad
            annotations.append((i, line_y, text_y, stars))

        if annotations:
            ymax = max(ymax, max(text_y for _, _, text_y, _ in annotations) + text_pad)

    ax.set_ylim(0, ymax)

    for i, line_y, text_y, stars in annotations:
        x_out = x[i] + offsets["outdoor"]
        x_in = x[i] + offsets["indoor"]
        ax.plot([x_out, x_out, x_in, x_in],
                [line_y - 0.2, line_y, line_y, line_y - 0.2],
                color="0.2",
                linewidth=ANNOTATION_LINEWIDTH,
                zorder=4)
        ax.text(x[i], text_y, stars, ha="center", va="bottom", color="0.15", fontsize=12, zorder=5)

    plt.tight_layout()
    if save_path:
        _save_figure_png_svg(fig, save_path)

    return ax


def _collect_group_label_counts(weights_base: Path, dayid2cellinfo: Dict[str, Path],
                                *,
                                mode: str,
                                letters: str = "PSRYI",
                                composite_map: Optional[Dict[str, set]] = None,
                                letter_order: str = LETTER_ORDER,
                                pyramidal_only: bool = True,
                                min_firing_rate_hz: float = DEFAULT_MIN_FIRING_RATE_HZ,
                                denominator_mode: str = "all_filtered") -> Tuple[pd.DataFrame, List[str]]:
    """汇总所有 session 的细胞计数，再按组(outdoor/indoor)计算百分比。"""
    if mode not in {"combo", "letter"}:
        raise ValueError("mode must be 'combo' or 'letter'")
    if denominator_mode not in {"all_filtered", "fitted_filtered"}:
        raise ValueError("denominator_mode must be 'all_filtered' or 'fitted_filtered'")

    records = []
    all_labels = set()
    letters = _sanitize_presence_letters(letters)
    composite_map = composite_map or {}
    labels_fixed = list(letters)

    for sess_dir in weights_base.iterdir():
        if not sess_dir.is_dir():
            continue
        session_name = sess_dir.name
        group = _infer_group_from_session_name(session_name)
        if group is None:
            continue
        csv_path = sess_dir / "selected_models.csv"
        if not csv_path.exists():
            continue

        day_id = parse_day_id_from_session(session_name)
        if (not day_id) or (day_id not in dayid2cellinfo):
            continue

        try:
            cell_types = load_cell_types(dayid2cellinfo[day_id])
        except Exception:
            continue

        type_mask = _cell_type_mask(cell_types, pyramidal_only=pyramidal_only)
        fr_mask = _firing_rate_mask_from_1000hz(session_name, len(cell_types), min_firing_rate_hz)
        if fr_mask is None:
            continue
        cell_mask = type_mask & fr_mask

        df = pd.read_csv(csv_path)
        if "final_model" not in df.columns:
            continue
        model_map = _selected_model_map(df)

        if mode == "combo":
            counts = Counter()
            for i, use_cell in enumerate(cell_mask):
                if not use_cell:
                    continue
                m = model_map.get(i + 1, None)
                lb = _canonical_label(m, letter_order=letter_order)
                counts[lb] += 1
            if "N" not in counts:
                counts["N"] = 0
            all_labels.update(counts.keys())
            for lb, cnt in counts.items():
                records.append({
                    "group": group,
                    "label": lb,
                    "count": int(cnt),
                    "total_cells": int(np.sum(cell_mask)),
                })
        else:
            counts = {lb: 0 for lb in labels_fixed}
            denom_cells = 0
            for i, use_cell in enumerate(cell_mask):
                if not use_cell:
                    continue
                m = model_map.get(i + 1, None)
                lab = _canonical_label(m, letter_order=letter_order)
                if lab == "N":
                    if denominator_mode == "all_filtered":
                        denom_cells += 1
                    continue
                denom_cells += 1
                lab_set = set(lab)
                for ch in letters:
                    members = composite_map.get(ch, {ch})
                    if lab_set & members:
                        counts[ch] += 1
            if denom_cells <= 0:
                continue
            all_labels.update(labels_fixed)
            for lb in labels_fixed:
                records.append({
                    "group": group,
                    "label": lb,
                    "count": int(counts[lb]),
                    "total_cells": int(denom_cells),
                })

    if not records:
        return pd.DataFrame(columns=["group", "label", "count", "total_cells", "percent"]), []

    raw = pd.DataFrame(records)
    grouped = raw.groupby(["group", "label"], as_index=False).agg(
        count=("count", "sum"),
        total_cells=("total_cells", "sum"),
    )
    grouped["percent"] = np.where(grouped["total_cells"] > 0, grouped["count"] / grouped["total_cells"] * 100.0, 0.0)

    if mode == "combo":
        labels_sorted = _sort_labels(list(all_labels), letter_order=letter_order)
    else:
        labels_sorted = labels_fixed

    full = pd.MultiIndex.from_product([
        ["outdoor", "indoor"], labels_sorted
    ], names=["group", "label"]).to_frame(index=False)
    out = full.merge(grouped, on=["group", "label"], how="left").fillna({"count": 0, "total_cells": 0, "percent": 0.0})
    out["count"] = out["count"].astype(int)
    out["total_cells"] = out["total_cells"].astype(int)
    return out, labels_sorted


def plot_percent_bar_aggregated(long_df, labels_sorted,
                                title=None,
                                save_path=None,
                                bar_alpha=0.55,
                                cell_scope: str = "pyramidal"):

    """聚合图：不画 session 点，使用每组总 pyramidal 细胞作为分母。"""
    if long_df.empty or not labels_sorted:
        raise ValueError("没有可用数据：long_df 为空或 labels_sorted 为空。")

    groups = ["indoor", "outdoor"]
    group_color = {"outdoor": OUTDOOR_BASE_COLOR, "indoor": INDOOR_BASE_COLOR}
    label_color_map = {
        g: _label_shade_map(labels_sorted, base_color)
        for g, base_color in group_color.items()
    }

    fig_w = max(4.8, 0.56 * len(labels_sorted) + 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    x = np.arange(len(labels_sorted))
    offsets = {"indoor": -0.18, "outdoor": +0.18}
    bar_width = 0.30

    for g in groups:
        sub = long_df[long_df["group"] == g].set_index("label")
        vals = [float(sub.at[lb, "percent"]) if lb in sub.index else 0.0 for lb in labels_sorted]
        colors = [label_color_map[g][lb] for lb in labels_sorted]
        ax.bar(x + offsets[g], vals, width=bar_width, color=colors, alpha=bar_alpha, label=g)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted)
    ax.set_ylabel(f"Percentage of {cell_scope} cells across all sessions (%)", fontsize=YLABEL_FONTSIZE)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LINEWIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINEWIDTH)
    ax.tick_params(axis="both", width=AXIS_LINEWIDTH, labelsize=TICK_LABEL_FONTSIZE)
    ax.set_xlim(-0.6, len(labels_sorted) - 0.4)
    ymax = max(5.0, float(np.nanmax(long_df["percent"].values)) * 1.20)
    ax.set_ylim(0, ymax)

    plt.tight_layout()
    if save_path:
        _save_figure_png_svg(fig, save_path)
    return ax


def recode_position_behavior_n_per_session(
    long_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    把组合标签重编码为三类：
      - position only: 仅 'P'
      - speed only: 仅 'S'
      - Head_pose: 任何包含 roll/yaw/pitch 的组合
    """
    if long_df.empty:
        return pd.DataFrame(columns=["session", "group", "label", "percent"]), ["position only", "speed only", "Head_pose"]

    src = _drop_excluded_combo_labels(long_df)
    if src.empty:
        return pd.DataFrame(columns=["session", "group", "label", "percent"]), ["position only", "speed only", "Head_pose"]

    labels_sorted = ["position only", "speed only", "Head_pose"]
    rows = []
    for (session, group), sub in src.groupby(["session", "group"], sort=False):
        label2pct = dict(zip(sub["label"].astype(str), sub["percent"].astype(float)))
        head_pose_pct = float(
            sub.loc[sub["label"].astype(str).map(lambda s: any(ch in s for ch in "RYI")), "percent"].sum()
        )
        rows.extend([
            {"session": session, "group": group, "label": "position only", "percent": float(label2pct.get("P", 0.0))},
            {"session": session, "group": group, "label": "speed only", "percent": float(label2pct.get("S", 0.0))},
            {"session": session, "group": group, "label": "Head_pose", "percent": head_pose_pct},
        ])

    out = pd.DataFrame(rows)
    sessions = out[["session", "group"]].drop_duplicates()
    pivot = out.pivot_table(index="session", columns="label", values="percent", aggfunc="first")
    pivot = pivot.reindex(columns=labels_sorted).fillna(0.0)
    out = pivot.reset_index().melt(id_vars="session", var_name="label", value_name="percent")
    sess2group = dict(zip(sessions["session"], sessions["group"]))
    out["group"] = out["session"].map(sess2group)
    return out, labels_sorted


def recode_position_behavior_n_aggregated(
    agg_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """把聚合计数表重编码为 position only / speed only / Head_pose 三类并重算百分比。"""
    labels_sorted = ["position only", "speed only", "Head_pose"]
    if agg_df.empty:
        return pd.DataFrame(columns=["group", "label", "count", "total_cells", "percent"]), labels_sorted
    src = _drop_excluded_combo_labels(agg_df)
    if src.empty:
        return pd.DataFrame(columns=["group", "label", "count", "total_cells", "percent"]), labels_sorted

    total_by_group = src.groupby("group", as_index=False)["total_cells"].max()
    rows = []
    for group, sub in src.groupby("group", sort=False):
        label2count = dict(zip(sub["label"].astype(str), sub["count"].astype(int)))
        total_cells = int(sub["total_cells"].max()) if not sub.empty else 0
        head_pose_count = int(
            sub.loc[sub["label"].astype(str).map(lambda s: any(ch in s for ch in "RYI")), "count"].sum()
        )
        rows.extend([
            {"group": group, "label": "position only", "count": int(label2count.get("P", 0)), "total_cells": total_cells},
            {"group": group, "label": "speed only", "count": int(label2count.get("S", 0)), "total_cells": total_cells},
            {"group": group, "label": "Head_pose", "count": head_pose_count, "total_cells": total_cells},
        ])

    out = pd.DataFrame(rows).merge(total_by_group, on="group", how="left", suffixes=("", "_group"))
    if "total_cells_group" in out.columns:
        out["total_cells"] = out["total_cells_group"].fillna(out["total_cells"]).astype(int)
        out = out.drop(columns=["total_cells_group"])
    out["percent"] = np.where(out["total_cells"] > 0, out["count"] / out["total_cells"] * 100.0, 0.0)

    full = pd.MultiIndex.from_product(
        [["outdoor", "indoor"], labels_sorted], names=["group", "label"]
    ).to_frame(index=False)
    out = full.merge(out, on=["group", "label"], how="left").fillna({"count": 0, "total_cells": 0, "percent": 0.0})
    out["count"] = out["count"].astype(int)
    out["total_cells"] = out["total_cells"].astype(int)
    return out, labels_sorted


def paired_stats_summary(
    long_df: pd.DataFrame,
    labels_sorted: List[str],
) -> pd.DataFrame:
    """
    对每个 label 只做 indoor/outdoor 的 paired t-test。
    差值定义为: outdoor - indoor。
    """
    if long_df.empty:
        return pd.DataFrame()

    df = long_df.copy()
    df["pair_id"] = df["session"].astype(str).map(_pair_id_from_session_name)
    piv = df.pivot_table(index=["pair_id", "label"], columns="group", values="percent", aggfunc="first")
    if ("outdoor" not in piv.columns) or ("indoor" not in piv.columns):
        return pd.DataFrame()

    rows = []
    for lb in labels_sorted:
        sub = piv.loc[piv.index.get_level_values("label") == lb].dropna(subset=["outdoor", "indoor"], how="any")
        if sub.empty:
            continue

        outdoor = sub["outdoor"].astype(float).to_numpy()
        indoor = sub["indoor"].astype(float).to_numpy()
        diffs = outdoor - indoor
        n_pairs = int(diffs.size)

        mean_diff = float(np.mean(diffs))
        mean_out = float(np.mean(outdoor))
        mean_in = float(np.mean(indoor))

        # Paired t-test
        if n_pairs < 2:
            t_stat = np.nan
            t_p = np.nan
        elif np.allclose(diffs, 0.0, atol=1e-12, rtol=0.0):
            t_stat = 0.0
            t_p = 1.0
        else:
            try:
                t = scipy.stats.ttest_rel(outdoor, indoor, nan_policy="omit", alternative="two-sided")
                t_stat = float(t.statistic)
                t_p = float(t.pvalue)
            except Exception:
                t_stat = np.nan
                t_p = np.nan

        rows.append(
            {
                "label": lb,
                "n_pairs": n_pairs,
                "mean_outdoor_percent": mean_out,
                "mean_indoor_percent": mean_in,
                "mean_diff_percent": mean_diff,
                "difference_definition": "outdoor-indoor",
                "paired_t_statistic": t_stat,
                "paired_t_pvalue": t_p,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["paired_t_pvalue_rank"] = out["paired_t_pvalue"].rank(method="min", na_option="bottom")
    return out

# ===================== main =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyramidal_only", dest="pyramidal_only", default=True, 
                    help="Compute statistics using pyramidal cells only (default).")
    ap.add_argument("--all_cells", dest="pyramidal_only", action="store_false",
                    help="Use all cells instead of pyramidal-only.")
    ap.add_argument(
        "--presence_spec",
        action="append",
        default=None,
        help=(
            "Letter-presence spec, can be passed multiple times. "
            "Format: 'LETTERS' or 'LETTERS:NEW=OLD1OLD2,...'. "
            "Examples: 'PSRYIT' , 'PSHT:H=RYI'."
        ),
    )
    ap.add_argument(
        "--min_firing_rate_hz",
        type=float,
        default=DEFAULT_MIN_FIRING_RATE_HZ,
        help="Exclude cells with average firing rate below this threshold (Hz). Set <=0 to disable.",
    )
    args = ap.parse_args()

    dayid2cellinfo = build_dayid_to_cellinfo()
    cell_scope = "pyramidal" if args.pyramidal_only else "all"
    scope_tag = "PYR" if args.pyramidal_only else "ALL"

    filtered_neurons_df = collect_filtered_neuron_records(
        WEIGHTS_BASE,
        dayid2cellinfo,
        pyramidal_only=args.pyramidal_only,
        min_firing_rate_hz=args.min_firing_rate_hz,
    )
    out_filtered_csv = WEIGHTS_BASE / f"filtered_neuron_ids_{scope_tag}.csv"
    filtered_neurons_df.to_csv(out_filtered_csv, index=False)
    print(f"被筛选掉的 neuron 编号已保存: {out_filtered_csv}")
    if not filtered_neurons_df.empty:
        print(
            filtered_neurons_df.groupby("exclude_reason")["neuron_idx"]
            .count()
            .rename("n_neurons")
            .sort_values(ascending=False)
        )

    # 1) 按 session 计算每个“组合标签”的百分比（pyramidal only）
    long_df, labels_sorted = gather_session_percentages(
        WEIGHTS_BASE,
        dayid2cellinfo,
        pyramidal_only=args.pyramidal_only,
        min_firing_rate_hz=args.min_firing_rate_hz,
    )
    long_df = _drop_excluded_combo_labels(long_df)
    labels_sorted = [lb for lb in labels_sorted if not _should_exclude_combo_label(lb)]

    # 1) long_df 里实际用到了哪些 day_id？
    sess = long_df[["session", "group"]].drop_duplicates().copy()
    sess["day_id"] = sess["session"].map(parse_day_id_from_session)
    print("\n[Used day_id from WEIGHTS_BASE sessions]")
    print(sess.groupby("day_id")["session"].nunique().sort_index())

    # 2) 你 DAY_SEARCH_DIRS 期望的 day_id 有哪些？dayid2cellinfo 实际找到了哪些？
    expected = []
    for dd in DAY_SEARCH_DIRS:
        expected.append(parse_day_id_from_path(dd))
    expected = [x for x in expected if x is not None]
    print("\n[Expected day_id from DAY_SEARCH_DIRS]")
    print(sorted(expected))

    dayid2cellinfo = build_dayid_to_cellinfo()
    print("\n[Found day_id with cellinfo]")
    print(sorted(dayid2cellinfo.keys()))

    # 3) 找“期望有但没用到”的 day_id
    used = set(sess["day_id"].dropna().unique())
    missing = [d for d in expected if d not in used]
    print("\n[Missing day_id not used in stats]")
    print(missing)

    if long_df.empty:
        print("没有找到可用 session（可能是路径、cellinfo、selected_models.csv 或命名规则未匹配）。")
        return

    # 保存按 session 的 long 表（组合标签版）
    out_csv = WEIGHTS_BASE / f"model_type_percentages_{scope_tag}_per_session_long.csv"
    long_df.to_csv(out_csv, index=False)
    print(f"按 session 的 long 表已保存: {out_csv}")

    stats_combo = paired_stats_summary(long_df, labels_sorted)
    if not stats_combo.empty:
        out_w = WEIGHTS_BASE / f"model_type_percentages_{scope_tag}_paired_t_summary.csv"
        stats_combo.to_csv(out_w, index=False)
        print(f"组合标签 paired t-test 汇总已保存: {out_w}")

    # 图1：组合标签的 per-session boxplot
    save_path1 = WEIGHTS_BASE / f"model_type_percentages_{scope_tag}_boxplot_by_session.png"
    plot_percent_box_by_session(
        long_df,
        labels_sorted,
        title=f"Session-wise percentage of {cell_scope} cells modulated by variables",
        save_path=save_path1,
        show_points=True,
        stats_df=stats_combo,
        significance_pvalue_col="paired_t_pvalue",
        cell_scope=cell_scope,
    )

    # 2) 字母覆盖率图：支持自定义统计类和变量合成
    presence_specs_raw = args.presence_spec if args.presence_spec else DEFAULT_PRESENCE_SPECS
    presence_specs = [_parse_presence_spec(s) for s in presence_specs_raw]

    for letters, composite_map in presence_specs:
        presence_tag = _presence_tag(letters)
        comp_txt = ", ".join([f"{k}={''.join(sorted(v))}" for k, v in composite_map.items()])
        if comp_txt:
            print(f"[presence] classes={letters}, composites: {comp_txt}")
        else:
            print(f"[presence] classes={letters}")

        long_df_p, labels_sorted_p = gather_session_letter_presence(
            WEIGHTS_BASE,
            dayid2cellinfo,
            letters=letters,
            composite_map=composite_map,
            letter_order=LETTER_ORDER,
            pyramidal_only=args.pyramidal_only,
            min_firing_rate_hz=args.min_firing_rate_hz,
        )
        if long_df_p.empty:
            print(f"字母覆盖率图无可用数据：{presence_tag}")
            continue

        out_csv_p = WEIGHTS_BASE / f"model_letter_presence_{presence_tag}_{scope_tag}_per_session_long.csv"
        long_df_p.to_csv(out_csv_p, index=False)
        print(f"{presence_tag} 字母覆盖率 long 表已保存: {out_csv_p}")

        stats_presence = paired_stats_summary(long_df_p, labels_sorted_p)
        if not stats_presence.empty:
            out_w_p = WEIGHTS_BASE / f"model_letter_presence_{presence_tag}_{scope_tag}_paired_t_summary.csv"
            stats_presence.to_csv(out_w_p, index=False)
            print(f"{presence_tag} paired t-test 汇总已保存: {out_w_p}")

        title_suffix = f"; {comp_txt}" if comp_txt else ""
        save_path_p = WEIGHTS_BASE / f"model_letter_presence_{presence_tag}_{scope_tag}_boxplot_by_session.png"
        plot_percent_box_by_session(
            long_df_p,
            labels_sorted_p,
            title=f"Session-wise percentage of {cell_scope} cells modulated by variable groups ({presence_tag}{title_suffix})",
            save_path=save_path_p,
            show_points=True,
            stats_df=stats_presence,
            significance_pvalue_col="paired_t_pvalue",
            cell_scope=cell_scope,
        )

    # 4) 新增聚合图：不是每个 session 一个点，而是先把所有 session 的 cell 计数相加
    agg1, agg_labels1 = _collect_group_label_counts(
        WEIGHTS_BASE,
        dayid2cellinfo,
        mode="combo",
        letter_order=LETTER_ORDER,
        pyramidal_only=args.pyramidal_only,
        min_firing_rate_hz=args.min_firing_rate_hz,
    )
    agg1 = _drop_excluded_combo_labels(agg1)
    agg_labels1 = [lb for lb in agg_labels1 if not _should_exclude_combo_label(lb)]
    if not agg1.empty:
        out_csv4 = WEIGHTS_BASE / f"model_type_percentages_{scope_tag}_all_sessions_aggregated_long.csv"
        agg1.to_csv(out_csv4, index=False)
        print(f"组合标签聚合 long 表已保存: {out_csv4}")
        save_path4 = WEIGHTS_BASE / f"model_type_percentages_{scope_tag}_aggregated_bar.png"
        plot_percent_bar_aggregated(
            agg1,
            agg_labels1,
            title=f"Percentage of {cell_scope} cells modulated by variables across all sessions",
            save_path=save_path4,
            cell_scope=cell_scope,
        )

    for letters, composite_map in presence_specs:
        presence_tag = _presence_tag(letters)
        comp_txt = ", ".join([f"{k}={''.join(sorted(v))}" for k, v in composite_map.items()])
        agg_p, agg_labels_p = _collect_group_label_counts(
            WEIGHTS_BASE,
            dayid2cellinfo,
            mode="letter",
            letters=letters,
            composite_map=composite_map,
            letter_order=LETTER_ORDER,
            pyramidal_only=args.pyramidal_only,
            min_firing_rate_hz=args.min_firing_rate_hz,
        )
        if agg_p.empty:
            continue
        out_csv_p = WEIGHTS_BASE / f"model_letter_presence_{presence_tag}_{scope_tag}_all_sessions_aggregated_long.csv"
        agg_p.to_csv(out_csv_p, index=False)
        print(f"{presence_tag} 聚合 long 表已保存: {out_csv_p}")
        title_suffix = f"; {comp_txt}" if comp_txt else ""
        save_path_p = WEIGHTS_BASE / f"model_letter_presence_{presence_tag}_{scope_tag}_aggregated_bar.png"
        plot_percent_bar_aggregated(
            agg_p,
            agg_labels_p,
            title=f"Percentage of {cell_scope} cells modulated by variable groups ({presence_tag}{title_suffix}) across all sessions",
            save_path=save_path_p,
            cell_scope=cell_scope,
        )

    fitted_scope_tag = f"{scope_tag}_FITTEDDENOM"
    for letters, composite_map in FITTED_DENOMINATOR_PRESENCE_SPECS:
        presence_tag = _presence_tag(letters)
        comp_txt = ", ".join([f"{k}={''.join(sorted(v))}" for k, v in composite_map.items()])

        long_df_fit, labels_sorted_fit = gather_session_letter_presence(
            WEIGHTS_BASE,
            dayid2cellinfo,
            letters=letters,
            composite_map=composite_map,
            letter_order=LETTER_ORDER,
            pyramidal_only=args.pyramidal_only,
            min_firing_rate_hz=args.min_firing_rate_hz,
            denominator_mode="fitted_filtered",
        )
        if not long_df_fit.empty:
            out_csv_fit = WEIGHTS_BASE / f"model_letter_presence_{presence_tag}_{fitted_scope_tag}_per_session_long.csv"
            long_df_fit.to_csv(out_csv_fit, index=False)
            print(f"{presence_tag} fitted-denominator long 表已保存: {out_csv_fit}")

            stats_fit = paired_stats_summary(long_df_fit, labels_sorted_fit)
            if not stats_fit.empty:
                out_stats_fit = WEIGHTS_BASE / f"model_letter_presence_{presence_tag}_{fitted_scope_tag}_paired_t_summary.csv"
                stats_fit.to_csv(out_stats_fit, index=False)
                print(f"{presence_tag} fitted-denominator paired t-test 汇总已保存: {out_stats_fit}")

            title_suffix = f"; {comp_txt}" if comp_txt else ""
            out_png_fit = WEIGHTS_BASE / f"model_letter_presence_{presence_tag}_{fitted_scope_tag}_boxplot_by_session.png"
            plot_percent_box_by_session(
                long_df_fit,
                labels_sorted_fit,
                title=f"Session-wise percentage of fitted {cell_scope} cells modulated by variable groups ({presence_tag}{title_suffix})",
                save_path=out_png_fit,
                show_points=True,
                stats_df=stats_fit,
                significance_pvalue_col="paired_t_pvalue",
                cell_scope=f"fitted {cell_scope}",
            )

        agg_fit, agg_labels_fit = _collect_group_label_counts(
            WEIGHTS_BASE,
            dayid2cellinfo,
            mode="letter",
            letters=letters,
            composite_map=composite_map,
            letter_order=LETTER_ORDER,
            pyramidal_only=args.pyramidal_only,
            min_firing_rate_hz=args.min_firing_rate_hz,
            denominator_mode="fitted_filtered",
        )
        if agg_fit.empty:
            continue

        out_csv_fit_agg = WEIGHTS_BASE / f"model_letter_presence_{presence_tag}_{fitted_scope_tag}_all_sessions_aggregated_long.csv"
        agg_fit.to_csv(out_csv_fit_agg, index=False)
        print(f"{presence_tag} fitted-denominator 聚合 long 表已保存: {out_csv_fit_agg}")

        title_suffix = f"; {comp_txt}" if comp_txt else ""
        out_png_fit_agg = WEIGHTS_BASE / f"model_letter_presence_{presence_tag}_{fitted_scope_tag}_aggregated_bar.png"
        plot_percent_bar_aggregated(
            agg_fit,
            agg_labels_fit,
            title=f"Percentage of fitted {cell_scope} cells modulated by variable groups ({presence_tag}{title_suffix}) across all sessions",
            save_path=out_png_fit_agg,
            cell_scope=f"fitted {cell_scope}",
        )

if __name__ == "__main__":
    main()
