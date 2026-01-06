from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import N_WORKERS, OUT_ROOT, PERCENTILE, SHUFFLE_N
from .io_utils import list_sessions_all, load_session_raw
from .plotting import binning_note, plot_neuron_summary
from .tuning_scores import (
    ScoreResult,
    SessionBinning,
    TuningInputs,
    compute_scores_for_neuron,
    compute_shuffle_scores,
    build_bins,
)
from contribution_utils.cell_metrics import build_dayid_to_cellinfo, pyramidal_indices_for_session


def _write_lines(path: Path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + ("\n" if not str(s).endswith("\n") else ""))


_WORKER_INPUTS: TuningInputs | None = None
_WORKER_BINS: SessionBinning | None = None
_WORKER_N_SHUFFLE: int = 0


def _init_worker(inputs: TuningInputs, bins: SessionBinning, n_shuffle: int) -> None:
    global _WORKER_INPUTS, _WORKER_BINS, _WORKER_N_SHUFFLE
    _WORKER_INPUTS = inputs
    _WORKER_BINS = bins
    _WORKER_N_SHUFFLE = n_shuffle


def _process_neuron(args: tuple[int, int]) -> tuple[int, ScoreResult, dict[str, np.ndarray], dict[str, np.ndarray]]:
    neuron_idx, seed = args
    if _WORKER_INPUTS is None or _WORKER_BINS is None:
        raise RuntimeError("Worker not initialized with inputs and bins.")
    rng = np.random.default_rng(seed=seed)
    scores, aux = compute_scores_for_neuron(_WORKER_INPUTS, _WORKER_BINS, neuron_idx)
    shuffle_scores = compute_shuffle_scores(_WORKER_INPUTS, _WORKER_BINS, neuron_idx, _WORKER_N_SHUFFLE, rng)
    return neuron_idx, scores, aux, shuffle_scores


def process_session(session: str, dayid2cellinfo: dict[str, Path], n_shuffle: int = SHUFFLE_N) -> Path:
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
    pyr_idx = pyramidal_indices_for_session(session, dayid2cellinfo, n_neurons)
    if pyr_idx is None or pyr_idx.size == 0:
        raise RuntimeError("pyramidal cell info not found or empty")
    rows = []
    seed_seq = np.random.SeedSequence(0)
    seeds = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(n_neurons)]

    if N_WORKERS > 1 and pyr_idx.size > 1:
        tasks = [(int(idx), seeds[int(idx)]) for idx in pyr_idx]
        with ProcessPoolExecutor(
            max_workers=N_WORKERS,
            initializer=_init_worker,
            initargs=(inputs, bins, n_shuffle),
        ) as executor:
            results = executor.map(_process_neuron, tasks)
            iterable = results
            for n_idx, scores, aux, shuffle_scores in iterable:
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

                plot_neuron_summary(
                    session_dir / f"neuron_{n_idx:03d}",
                    n_idx,
                    row,
                    aux,
                )
    else:
        rng = np.random.default_rng(seed=0)
        for n_idx in pyr_idx.tolist():
            scores, aux = compute_scores_for_neuron(inputs, bins, n_idx)
            shuffle_scores = compute_shuffle_scores(inputs, bins, n_idx, n_shuffle, rng)

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

            plot_neuron_summary(
                session_dir / f"neuron_{n_idx:03d}",
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

    dayid2cellinfo = build_dayid_to_cellinfo()
    _write_lines(OUT_ROOT / "sessions_all_present.txt", sessions)
    print(f"[INFO] Found {len(sessions)} sessions with all required inputs present.")

    processed = []
    for session in sessions:
        try:
            out_csv = process_session(session, dayid2cellinfo)
        except Exception as exc:  # pragma: no cover - runtime logging
            print(f"[SKIP] {session}: {exc}")
            continue
        processed.append(session)
        print(f"[DONE] {session}: {out_csv}")

    _write_lines(OUT_ROOT / "sessions_processed.txt", processed)


if __name__ == "__main__":
    main()
