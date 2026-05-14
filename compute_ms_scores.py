#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_BASE = REPO_ROOT / "weights_Poisson_forward"
DEFAULT_OUTPUT_DIR = DEFAULT_WEIGHTS_BASE / "MS_SCORE"
FILTERED_NEURON_CSV = "filtered_neuron_ids_ALL.csv"
EPS = 1e-12
CANONICAL_FEATURES = ("Position", "Speed", "H")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-neuron MS scores from session-level "
            "weights_Poisson_forward/*/RLLR_STATS/dropone_llhi.csv files "
            "using canonical features Position, Speed, and H."
        )
    )
    parser.add_argument(
        "--weights-base",
        type=Path,
        default=DEFAULT_WEIGHTS_BASE,
        help=f"Base weights directory (default: {DEFAULT_WEIGHTS_BASE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for per-session MS-score CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--session",
        action="append",
        default=None,
        help="Optional session name. Pass multiple times to restrict processing.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=EPS,
        help=f"Small epsilon used in the MS-score formula (default: {EPS})",
    )
    return parser.parse_args()


def discover_sessions(weights_base: Path, requested_sessions: list[str] | None) -> list[str]:
    if requested_sessions:
        sessions = requested_sessions
    else:
        sessions = []
        for session_dir in sorted(weights_base.iterdir()):
            if not session_dir.is_dir():
                continue
            if (session_dir / "RLLR_STATS" / "dropone_llhi.csv").exists():
                sessions.append(session_dir.name)
    return sorted(set(sessions))


def split_model_string(model_str: str) -> list[str]:
    model_str = str(model_str).strip()
    if not model_str:
        return []
    return [token for token in model_str.replace(",", "_").split("_") if token]


def load_selected_model_neurons(session_dir: Path) -> set[int]:
    selected_csv = session_dir / "selected_models.csv"
    if not selected_csv.exists():
        return set()

    neuron_ids: set[int] = set()
    with open(selected_csv, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            neuron_name = str(row.get("neuron", "")).strip()
            model_str = str(row.get("final_model", "")).strip()
            if not neuron_name or not model_str:
                continue
            if not neuron_name.lower().startswith("neuron_"):
                continue
            if not split_model_string(model_str):
                continue
            try:
                neuron_idx = int(neuron_name.split("_", maxsplit=1)[1]) - 1
            except Exception:
                continue
            if neuron_idx >= 0:
                neuron_ids.add(neuron_idx)
    return neuron_ids


def load_unclassified_neurons(session_dir: Path) -> set[int]:
    txt_path = session_dir / "unclassified_neurons.txt"
    if not txt_path.exists():
        return set()

    neuron_ids: set[int] = set()
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("neuron_"):
            continue
        try:
            neuron_idx = int(line.split("_", maxsplit=1)[1]) - 1
        except Exception:
            continue
        if neuron_idx >= 0:
            neuron_ids.add(neuron_idx)
    return neuron_ids


def load_filtered_neurons(weights_base: Path, session: str) -> set[int]:
    filtered_csv = weights_base / FILTERED_NEURON_CSV
    if not filtered_csv.exists():
        return set()

    df = pd.read_csv(filtered_csv)
    if "session" not in df.columns:
        return set()

    df = df[df["session"].astype(str) == str(session)]
    if df.empty:
        return set()

    idx_col = "neuron_idx_0based" if "neuron_idx_0based" in df.columns else "neuron_idx"
    ids = pd.to_numeric(df[idx_col], errors="coerce").dropna().astype(int)
    return set(ids.tolist())


def load_valid_fitted_neurons(session_dir: Path) -> tuple[set[int], set[int]]:
    full_rllr_csv = session_dir / "RLLR_STATS" / "full_rllr.csv"
    if not full_rllr_csv.exists():
        return set(), set()

    df = pd.read_csv(full_rllr_csv)
    if "neuron_idx" not in df.columns:
        return set(), set()

    if "ll_gain" in df.columns:
        ll_gain = pd.to_numeric(df["ll_gain"], errors="coerce")
        valid_mask = ll_gain.notna() & (ll_gain >= 0)
        invalid_mask = ~valid_mask
    else:
        valid_mask = pd.Series(True, index=df.index)
        invalid_mask = pd.Series(False, index=df.index)

    valid_ids = set(pd.to_numeric(df.loc[valid_mask, "neuron_idx"], errors="coerce").dropna().astype(int).tolist())
    invalid_ids = set(pd.to_numeric(df.loc[invalid_mask, "neuron_idx"], errors="coerce").dropna().astype(int).tolist())
    return valid_ids, invalid_ids


def infer_all_neuron_ids(weights_base: Path, session: str, session_dir: Path) -> list[int]:
    known_ids: set[int] = set()
    known_ids |= load_selected_model_neurons(session_dir)
    known_ids |= load_filtered_neurons(weights_base, session)
    known_ids |= load_unclassified_neurons(session_dir)

    full_rllr_csv = session_dir / "RLLR_STATS" / "full_rllr.csv"
    if full_rllr_csv.exists():
        df = pd.read_csv(full_rllr_csv)
        if "neuron_idx" in df.columns:
            ids = pd.to_numeric(df["neuron_idx"], errors="coerce").dropna().astype(int)
            known_ids |= set(ids.tolist())

    if not known_ids:
        raise ValueError(f"{session}: could not infer neuron ids from selected/filtered/unclassified metadata.")

    max_idx = max(known_ids)
    return list(range(max_idx + 1))


def ordered_unique(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        label = str(value).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        ordered.append(label)
    return ordered


def infer_group(session: str, df: pd.DataFrame) -> str:
    if "group" in df.columns:
        groups = ordered_unique(df["group"].tolist())
        if groups:
            return groups[0]
    if "_" in session:
        return session.rsplit("_", maxsplit=1)[-1]
    return ""


def compute_ms_for_session(
    weights_base: Path,
    session: str,
    *,
    output_dir: Path,
    eps: float,
) -> Path:
    session_dir = weights_base / session
    dropone_csv = session_dir / "RLLR_STATS" / "dropone_llhi.csv"
    if not dropone_csv.exists():
        raise FileNotFoundError(f"{session}: missing {dropone_csv}")

    df = pd.read_csv(dropone_csv)
    if df.empty:
        raise ValueError(f"{session}: dropone_llhi.csv is empty.")
    if "feature" not in df.columns or "neuron_idx" not in df.columns or "delta_llhi" not in df.columns:
        raise ValueError(f"{session}: dropone_llhi.csv missing required columns.")

    group = infer_group(session, df)
    features_in_csv = set(ordered_unique(df["feature"].tolist()))
    features = [feat for feat in CANONICAL_FEATURES if feat in features_in_csv]
    if not features:
        raise ValueError(
            f"{session}: none of the canonical features {CANONICAL_FEATURES} were found in dropone_llhi.csv."
        )

    all_neuron_ids = infer_all_neuron_ids(weights_base, session, session_dir)
    valid_fitted_ids, invalid_fitted_ids = load_valid_fitted_neurons(session_dir)

    wide = (
        df.pivot_table(index="neuron_idx", columns="feature", values="delta_llhi", aggfunc="first")
        .apply(pd.to_numeric, errors="coerce")
        .reindex(index=all_neuron_ids, columns=features)
    )

    # Any neuron that was not successfully fit receives MS = 0 by construction.
    positive_delta = wide.fillna(0.0).clip(lower=0.0)
    is_valid_fit = pd.Index(all_neuron_ids).isin(sorted(valid_fitted_ids))
    positive_delta.loc[~is_valid_fit, :] = 0.0

    t_i = positive_delta.sum(axis=1)
    positive_feature_count = (positive_delta > 0).sum(axis=1)
    k = len(features)
    if k > 1:
        p_i = positive_delta.div(t_i + float(eps), axis=0)
        d_i = -(p_i * np.log(p_i + float(eps))).sum(axis=1) / math.log(k)
        d_i = d_i.where(t_i > 0, 0.0)
    else:
        p_i = positive_delta.copy()
        d_i = pd.Series(0.0, index=positive_delta.index, dtype=float)

    d_i = d_i.clip(lower=0.0)
    d_i = d_i.where(positive_feature_count > 1, 0.0)
    ms_i = (t_i * d_i).clip(lower=0.0)

    out_df = pd.DataFrame(
        {
            "session": session,
            "group": group,
            "neuron_idx": all_neuron_ids,
            "fitted": is_valid_fit.astype(int),
            "ms_score": ms_i.to_numpy(dtype=float),
            "T_i": t_i.to_numpy(dtype=float),
            "D_i": d_i.to_numpy(dtype=float),
            "K": int(k),
            "n_positive_features": positive_feature_count.to_numpy(dtype=int),
        }
    )

    for feat in features:
        out_df[f"d_{feat}"] = positive_delta[feat].to_numpy(dtype=float)
        out_df[f"p_{feat}"] = p_i[feat].to_numpy(dtype=float)

    if invalid_fitted_ids:
        out_df["invalid_fit_zeroed"] = out_df["neuron_idx"].isin(sorted(invalid_fitted_ids)).astype(int)
    else:
        out_df["invalid_fit_zeroed"] = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{session}_ms_score.csv"
    out_df.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    args = parse_args()
    weights_base = args.weights_base.resolve()
    output_dir = args.output_dir.resolve()

    if not weights_base.exists():
        raise FileNotFoundError(f"Weights directory not found: {weights_base}")

    sessions = discover_sessions(weights_base, args.session)
    if not sessions:
        raise RuntimeError(f"No sessions found under {weights_base}")

    print(f"[INFO] weights base: {weights_base}")
    print(f"[INFO] output dir:   {output_dir}")
    print(f"[INFO] sessions:     {len(sessions)}")

    written: list[Path] = []
    for session in sessions:
        out_csv = compute_ms_for_session(
            weights_base,
            session,
            output_dir=output_dir,
            eps=float(args.eps),
        )
        written.append(out_csv)
        print(f"[OK] {session} -> {out_csv}")

    print(f"[DONE] Wrote {len(written)} session CSV file(s).")


if __name__ == "__main__":
    main()
