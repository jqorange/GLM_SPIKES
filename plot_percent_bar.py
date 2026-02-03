# -*- coding: utf-8 -*-
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from matplotlib.patches import Patch

# ===================== 配置区 =====================

WEIGHTS_BASE = Path(r"D:\Jiaqi\Projects\GLM_test\GLM_SPIKES\weights_Poisson_forward")  # 修改成你的路径

DAY_SEARCH_DIRS = [
    r"I:\data\FieldRat\2024\F4\day1",
    r"I:\data\FieldRat\2024\F4\day4",
    r"I:\data\FieldRat\2024\F5\Merged\day2\121_day2",
    r"I:\data\FieldRat\2024\F5\Merged\day3\121_day3",
    r"I:\data\FieldRat\2024\F5\Merged\day5\121_day5",
    r"I:\data\FieldRat\2024\F5\Merged\day6\3E6_day6",
    r"I:\data\FieldRat\2024\F5\Merged\day10\121_day10",
    r"I:\data\FieldRat\2024\F6\Merged\day3\3E6_day3",
    r"I:\data\FieldRat\2024\F6\Merged\day5\3E6_day5",
    r"I:\data\FieldRat\2024\F6\Merged\day8\3E6_day8",
    r"I:\data\FieldRat\2024\F6\Merged\day9\3E6_day9",
    r"I:\data\FieldRat\2024\F6\Merged\day10\3E6_day10",
    r"I:\data\FieldRat\2024\F6\Merged\day2\3E6_day2",
    r"I:\data\FieldRat\2024\F6\Merged\day4\3E6_day4",
    r"I:\data\FieldRat\2024\F6\Merged\day6\121_day6",
    r"I:\data\FieldRat\2024\F5\Merged\day2\121_day2",
    r"I:\data\FieldRat\2024\F5\Merged\day3\121_day3",
    r"I:\data\FieldRat\2024\F5\Merged\day7\121_day7",
    r"I:\data\FieldRat\2024\F5\Merged\day10\121_day10",

    r"I:\data\FieldRat\2024\F6\Merged\day2\3E6_day2",
    r"I:\data\FieldRat\2024\F6\Merged\day3\3E6_day3",
    #r"I:\data\FieldRat\2024\F6\Merged\day7\3E6_day7",
    r"I:\data\FieldRat\2024\F6\Merged\day8\3E6_day8",
    r"I:\data\FieldRat\2024\F6\Merged\day10\3E6_day10",
]
# DAY_SEARCH_DIRS = [
#     r"I:\data\FieldRat\2024\F5\Merged\day2\121_day2",
#     r"I:\data\FieldRat\2024\F5\Merged\day3\121_day3",
#     r"I:\data\FieldRat\2024\F5\Merged\day7\121_day7",
#     r"I:\data\FieldRat\2024\F5\Merged\day10\121_day10",
#
#     r"I:\data\FieldRat\2024\F6\Merged\day2\3E6_day2",
#     r"I:\data\FieldRat\2024\F6\Merged\day3\3E6_day3",
#    # r"I:\data\FieldRat\2024\F6\Merged\day7\3E6_day7",
#     r"I:\data\FieldRat\2024\F6\Merged\day8\3E6_day8",
#     r"I:\data\FieldRat\2024\F6\Merged\day10\3E6_day10",
#
#     # r"I:\data\FieldRat\2024\F8\Merged\day2\3E6_day2", # WIP
# ]
DAY_SEARCH_DIRS = [Path(p) for p in DAY_SEARCH_DIRS]

LETTER_ORDER = "PSIRY"
NAME2LETTER = {
    "position": "P", "p": "P",
    "speed":    "S", "s": "S",
    "pitch":    "I", "i": "I",
    "roll":     "R", "r": "R",
    "yaw":      "Y", "y": "Y",
}

# ===================== 工具函数 =====================

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

        tokens = parts_from_str(s0)
        letters = set(NAME2LETTER.get(tok.lower(), None) for tok in tokens)

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

# ===================== 核心：按 session 计算百分比（组合标签版） =====================

def gather_session_percentages(weights_base: Path, dayid2cellinfo: Dict[str, Path],
                               letter_order=LETTER_ORDER) -> Tuple[pd.DataFrame, List[str]]:
    """
    返回：
      long_df: 每行 = (session, group, label, percent, count, total_pyr)
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

        pyr_mask = np.array([t.lower() == "pyramidal cell" for t in cell_types], dtype=bool)

        df = pd.read_csv(csv_path)
        if "final_model" not in df.columns:
            continue

        # 假设 CSV 行顺序与细胞顺序一致：只保留 pyramidal 的 model
        # === 分母用“全部 pyramidal cell”（以 cell_metrics 为准） ===
        total_pyr_all = int(np.sum(pyr_mask))
        if total_pyr_all <= 0:
            continue

        final_models = df["final_model"].tolist()  # 保留 NaN（不要 astype(str)）
        labels_for_pyr = []

        # 对每个 pyramidal cell，都产生一个 label；缺失/不可解析 -> 'N'
        for i, is_pyr in enumerate(pyr_mask):
            if not is_pyr:
                continue
            m = final_models[i] if i < len(final_models) else None  # CSV 缺行也算缺失 -> N
            labels_for_pyr.append(_canonical_label(m, letter_order=letter_order))

        # 该 session 内计数（此时 sum(counts) 必须 == total_pyr_all）
        c = Counter(labels_for_pyr)
        if "N" not in c:
            c["N"] = 0

        all_labels.update(c.keys())

        total = total_pyr_all
        for lb, cnt in c.items():
            pct = (cnt / total) * 100.0
            records.append({
                "session": session_name,
                "group": group,
                "label": lb,
                "percent": pct,
                "count": int(cnt),
                "total_pyr": int(total),
            })

    if not records:
        return pd.DataFrame(columns=["session", "group", "label", "percent", "count", "total_pyr"]), []

    long_df = pd.DataFrame(records)
    labels_sorted = _sort_labels(list(all_labels), letter_order=letter_order)

    # 补齐每个 session 缺失 label = 0%
    sessions = long_df[["session", "group"]].drop_duplicates()

    pivot = long_df.pivot_table(index="session", columns="label", values="percent", aggfunc="first")
    pivot = pivot.reindex(columns=labels_sorted).fillna(0.0)

    pivot_long = pivot.reset_index().melt(id_vars="session", var_name="label", value_name="percent")

    sess2group = dict(zip(sessions["session"], sessions["group"]))
    pivot_long["group"] = pivot_long["session"].map(sess2group)

    # count/total_pyr 对 boxplot 不是必须；保留 percent/group/session/label 即可
    return pivot_long, labels_sorted

# ===================== 新增：按 session 计算“单字母覆盖率”百分比（PSRYIN 六类） =====================

def gather_session_letter_presence(weights_base: Path, dayid2cellinfo: Dict[str, Path],
                                   letters: str = "PSRYI",
                                   head_pose: bool = False,
                                   letter_order: str = LETTER_ORDER) -> Tuple[pd.DataFrame, List[str]]:
    """
    画第二张图：只统计 P/S/R/Y/I/N 六类的百分比（按 pyramidal cell 分母）。
    规则：只要 final_model 的 canonical label 里包含该字母，就计入该字母（可重叠计数）。
          缺失/不可解析 -> 'N'，并且只计入 N（不计入其它字母）。
    返回：
      long_df2: 每行 = (session, group, label in {P,S,R,Y,I,N}, percent)
      labels_sorted2: 固定顺序 ['P','S','R','Y','I','N']（可按你想要的顺序改）
    """
    letters = letters.upper()
    labels_sorted2 = list(letters) + ["N"]  # 你要求 PSRYIN 六类：把 N 放最后

    records = []
    head_pose_list = ["R", "Y", "I"]
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

        pyr_mask = np.array([t.lower() == "pyramidal cell" for t in cell_types], dtype=bool)
        total_pyr_all = int(np.sum(pyr_mask))
        if total_pyr_all <= 0:
            continue

        df = pd.read_csv(csv_path)
        if "final_model" not in df.columns:
            continue
        final_models = df["final_model"].tolist()

        # 统计：每个 letter 出现于多少 pyramidal cells（允许重叠）
        counts = {lb: 0 for lb in labels_sorted2}

        for i, is_pyr in enumerate(pyr_mask):
            if not is_pyr:
                continue
            m = final_models[i] if i < len(final_models) else None
            lab = _canonical_label(m, letter_order=letter_order)  # e.g., "PSR" or "N"

            if lab == "N":
                counts["N"] += 1
                continue

            if head_pose == True:
                for ch in letters:
                    if ch == "H" and (set(head_pose_list) & set(lab)):
                        counts [ch] += 1
                    elif ch in lab:
                        counts[ch] += 1

            else:

                # 只要有就算：比如 "PS" 同时计入 P 和 S
                for ch in letters:
                    if ch in lab:
                        counts[ch] += 1

        # 写 records（分母固定为 total_pyr_all）
        for lb in labels_sorted2:
            pct = (counts[lb] / total_pyr_all) * 100.0
            records.append({
                "session": session_name,
                "group": group,
                "label": lb,
                "percent": float(pct),
            })

    if not records:
        return pd.DataFrame(columns=["session", "group", "label", "percent"]), labels_sorted2

    long_df2 = pd.DataFrame(records)

    # 补齐每个 session 缺失 label = 0%（理论上不会缺，但稳妥）
    sessions = long_df2[["session", "group"]].drop_duplicates()

    pivot = long_df2.pivot_table(index="session", columns="label", values="percent", aggfunc="first")
    pivot = pivot.reindex(columns=labels_sorted2).fillna(0.0)

    pivot_long = pivot.reset_index().melt(id_vars="session", var_name="label", value_name="percent")
    sess2group = dict(zip(sessions["session"], sessions["group"]))
    pivot_long["group"] = pivot_long["session"].map(sess2group)

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
                                pair_line_width=0.7):
    """
    对每个 label：画 indoor/outdoor 两个箱线图（按 session 分布），叠加每个 session 的散点（灰色、无 jitter），
    并可选把 paired 的 outdoor/indoor 点之间用浅红细线连接。
    """
    if long_df.empty or not labels_sorted:
        raise ValueError("没有可用数据：long_df 为空或 labels_sorted 为空。")

    groups = ["outdoor", "indoor"]  # 固定顺序：左 outdoor，右 indoor
    present_groups = [g for g in groups if g in set(long_df["group"].unique())]
    if not present_groups:
        raise ValueError("long_df 中没有 indoor/outdoor 组数据。")

    # 颜色：复用蓝/橙风格（与你之前一致）
    blue_cmap = mpl_cm.get_cmap("Blues")
    orange_cmap = mpl_cm.get_cmap("Oranges")
    OUTDOOR_COLOR = blue_cmap(0.75)
    INDOOR_COLOR  = orange_cmap(0.75)

    group_color = {
        "outdoor": OUTDOOR_COLOR,
        "indoor": INDOOR_COLOR,
    }

    df = long_df.copy()
    df["pair_id"] = df["session"].astype(str).map(_pair_id_from_session_name)

    fig_w = max(7, 0.85 * len(labels_sorted))
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    x = np.arange(len(labels_sorted))
    offsets = {"outdoor": -0.18, "indoor": +0.18}
    box_width = 0.28

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
        color = group_color[g]

        data = []
        for lb in labels_sorted:
            vals = df.loc[(df["group"] == g) & (df["label"] == lb), "percent"].astype(float).values
            data.append(vals)

        positions = x + offsets[g]

        ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=color, edgecolor=color, linewidth=1.1, alpha=box_alpha),
            medianprops=dict(color=color, linewidth=1.3),
            whiskerprops=dict(color=color, linewidth=1.0, alpha=0.9),
            capprops=dict(color=color, linewidth=1.0, alpha=0.9),
        )

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
                            color=(1.0, 0.4, 0.4),  # 浅红
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
                    c="0.45",
                    alpha=point_alpha,
                    linewidths=0,
                    zorder=3,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted)
    ax.set_ylabel("Percentage of pyramidal cells per session (%)")
    if title:
        ax.set_title(title)

    # 图例
    handles = []
    if "outdoor" in present_groups:
        handles.append(Patch(facecolor=OUTDOOR_COLOR, edgecolor=OUTDOOR_COLOR, alpha=box_alpha, label="outdoor"))
    if "indoor" in present_groups:
        handles.append(Patch(facecolor=INDOOR_COLOR, edgecolor=INDOOR_COLOR, alpha=box_alpha, label="indoor"))
    if handles:
        ax.legend(handles=handles, frameon=False, loc="upper right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.6, len(labels_sorted) - 0.4)

    ymax = max(5.0, float(np.nanmax(df["percent"].values)) * 1.15)
    ax.set_ylim(0, ymax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"图已保存到: {save_path}")

    return ax

# ===================== main =====================

def main():
    dayid2cellinfo = build_dayid_to_cellinfo()

    # 1) 按 session 计算每个“组合标签”的百分比（pyramidal only）
    long_df, labels_sorted = gather_session_percentages(WEIGHTS_BASE, dayid2cellinfo)

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
    out_csv = WEIGHTS_BASE / "model_type_percentages_PYR_per_session_long.csv"
    long_df.to_csv(out_csv, index=False)
    print(f"按 session 的 long 表已保存: {out_csv}")

    # 图1：组合标签的 per-session boxplot
    save_path1 = WEIGHTS_BASE / "model_type_percentages_PYR_boxplot_by_session.png"
    plot_percent_box_by_session(
        long_df,
        labels_sorted,
        title="Model type composition (Pyramidal only) — per-session distribution (indoor vs outdoor)",
        save_path=save_path1,
        show_points=True,
    )
    plt.show()

    # 2) 新图：只统计 PSRYIN 六类（字母覆盖率，允许重叠计数）
    long_df2, labels_sorted2 = gather_session_letter_presence(
        WEIGHTS_BASE,
        dayid2cellinfo,
        letters="PSRYI",
        head_pose=False,# 统计 P/S/R/Y/I
        letter_order=LETTER_ORDER # canonical label 仍按 PSIRY 解析
    )

    if long_df2.empty:
        print("第二张图没有可用数据（letter presence long_df2 为空）。")
        return

    out_csv2 = WEIGHTS_BASE / "model_letter_presence_percentages_PYR_per_session_long.csv"
    long_df2.to_csv(out_csv2, index=False)
    print(f"PSRYIN 六类（字母覆盖率）long 表已保存: {out_csv2}")

    save_path2 = WEIGHTS_BASE / "model_letter_presence_percentages_PYR_boxplot_by_session_PSRYIN.png"
    plot_percent_box_by_session(
        long_df2,
        labels_sorted2,
        title="Letter presence (P/S/R/Y/I/N; overlaps allowed) — per-session distribution (Pyramidal only)",
        save_path=save_path2,
        show_points=True,
    )
    plt.show()





    # 3) 新图：只统计 PSHN 4类（字母覆盖率，允许重叠计数）
    long_df3, labels_sorted3 = gather_session_letter_presence(
        WEIGHTS_BASE,
        dayid2cellinfo,
        letters="PSH",
        head_pose=True,# 统计 P/S/R/Y/I
        letter_order=LETTER_ORDER # canonical label 仍按 PSIRY 解析
    )

    if long_df3.empty:
        print("第3张图没有可用数据（letter presence long_df3为空）。")
        return

    out_csv3 = WEIGHTS_BASE / "model_letter_PSH.csv"
    long_df3.to_csv(out_csv3, index=False)
    print(f"PSH 4类（字母覆盖率）long 表已保存: {out_csv3}")

    save_path3 = WEIGHTS_BASE / "model_letter_PSH.png"
    plot_percent_box_by_session(
        long_df3,
        labels_sorted3,
        title="Letter presence (P/S/H/N; overlaps allowed)",
        save_path=save_path3,
        show_points=True,
    )
    plt.show()

if __name__ == "__main__":
    main()
