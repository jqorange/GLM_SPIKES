from __future__ import annotations

from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .config import OUT_ROOT, PERCENTILE, PLOT_MAX_NEURONS, SHUFFLE_N
from .io_utils import list_sessions_all, load_session_raw
from .plotting import binning_note, plot_neuron_summary
from .tuning_scores import SessionBinning, TuningInputs, compute_scores_for_neuron, compute_shuffle_scores, build_bins


def _write_lines(path: Path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + ("\n" if not str(s).endswith("\n") else ""))


def _process_neuron(args):
    inputs, bins, n_idx, n_shuffle, seed, include_aux = args
    scores, aux = compute_scores_for_neuron(inputs, bins, n_idx)
    rng = np.random.default_rng(seed=seed)
    shuffle_scores = compute_shuffle_scores(inputs, bins, n_idx, n_shuffle, rng)
    if not include_aux:
        aux = {}
    return n_idx, scores, aux, shuffle_scores


def _default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def process_session(session: str, n_shuffle: int = SHUFFLE_N) -> Path:
    data = load_session_raw(session)
    inputs = TuningInputs(
        head_x=data["head_x"],
        head_y=data["head_y"],
        heading_rad=data["heading_rad"],
        head_v=data["head_v"],
        roll=data["roll"],
        pitch=data["pitch"],
        spikes=data["spikes"],
    )
    bins = build_bins(inputs)

    session_dir = OUT_ROOT / session
    session_dir.mkdir(parents=True, exist_ok=True)
    binning_note(session_dir / "binning_notes.txt")

    n_neurons = inputs.spikes.shape[1]
    rows = []
    worker_count = _default_workers()
    tasks = [
        (inputs, bins, n_idx, n_shuffle, 1000 + n_idx, n_idx < PLOT_MAX_NEURONS)
        for n_idx in range(n_neurons)
    ]

    if worker_count > 1:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = executor.map(_process_neuron, tasks)
    else:
        results = map(_process_neuron, tasks)

    for n_idx, scores, aux, shuffle_scores in results:
        thresholds = {k: np.nanpercentile(v, PERCENTILE) for k, v in shuffle_scores.items()}

        row = {
            "neuron": n_idx,
            "grid_score": scores.grid_score,
            "border_score": scores.border_score,
            "hd_score": scores.hd_score,
            "roll_score": scores.roll_score,
            "pitch_score": scores.pitch_score,
            "speed_score": scores.speed_score,
            "speed_stability": scores.speed_stability,
            "spatial_stability": scores.spatial_stability,
            "angular_stability": scores.angular_stability,
            "roll_stability": scores.roll_stability,
            "pitch_stability": scores.pitch_stability,
            "grid_thresh": thresholds.get("grid_score"),
            "border_thresh": thresholds.get("border_score"),
            "hd_thresh": thresholds.get("hd_score"),
            "roll_thresh": thresholds.get("roll_score"),
            "pitch_thresh": thresholds.get("pitch_score"),
            "speed_thresh": thresholds.get("speed_score"),
            "speed_stab_thresh": thresholds.get("speed_stability"),
        }

        row["is_grid"] = row["grid_score"] > row["grid_thresh"]
        row["is_border"] = row["border_score"] > row["border_thresh"]
        row["is_hd"] = row["hd_score"] > row["hd_thresh"]
        row["is_roll"] = row["roll_score"] > row["roll_thresh"]
        row["is_pitch"] = row["pitch_score"] > row["pitch_thresh"]
        row["is_speed"] = (row["speed_score"] > row["speed_thresh"]) and (
            row["speed_stability"] > row["speed_stab_thresh"]
        )

        rows.append(row)

        if n_idx < PLOT_MAX_NEURONS and aux:
            plot_neuron_summary(
                session_dir / "plots" / f"neuron_{n_idx:03d}.png",
                n_idx,
                row,
                aux,
            )

    df = pd.DataFrame(rows)
    out_csv = session_dir / "tuning_scores.csv"
    df.to_csv(out_csv, index=False)
    return out_csv


def main():
    sessions = list_sessions_all()
    if not sessions:
        print("[FATAL] No sessions with required inputs found.")
        return

    _write_lines(OUT_ROOT / "sessions_all_present.txt", sessions)
    print(f"[INFO] Found {len(sessions)} sessions with all required inputs present.")

    processed = []
    for session in sessions:
        try:
            out_csv = process_session(session)
        except Exception as exc:  # pragma: no cover - runtime logging
            print(f"[SKIP] {session}: {exc}")
            continue
        processed.append(session)
        print(f"[DONE] {session}: {out_csv}")

    _write_lines(OUT_ROOT / "sessions_processed.txt", processed)


if __name__ == "__main__":
    main()
