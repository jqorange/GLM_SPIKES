from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import N_WORKERS, OUT_ROOT, REBUILD_PAIRED_POLAR_PLOTS, SCORE_PERCENTILES, SHUFFLE_N
from .io_utils import list_sessions_all, load_session_raw
from .plotting import binning_note, plot_neuron_summary, plot_paired_polar_curve, plot_paired_speed_curve
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


def _score_thresholds(shuffle_scores: dict[str, np.ndarray]) -> dict[str, float]:
    thresholds = {}
    for key, percentile in SCORE_PERCENTILES.items():
        values = shuffle_scores.get(key)
        if values is None or values.size == 0:
            thresholds[key] = float("nan")
        else:
            thresholds[key] = float(np.nanpercentile(values, percentile))
    return thresholds


def _save_polar_curves(out_dir: Path, aux: dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "polar_curves.npz",
        hd_curve=aux["hd_curve"],
        roll_curve=aux["roll_curve"],
        pitch_curve=aux["pitch_curve"],
        speed_curve=aux["speed_curve"],
    )


def _load_polar_curves(out_dir: Path) -> dict[str, np.ndarray] | None:
    path = out_dir / "polar_curves.npz"
    if not path.exists():
        return None
    data = np.load(path)
    curves = {
        "hd_curve": data["hd_curve"],
        "roll_curve": data["roll_curve"],
        "pitch_curve": data["pitch_curve"],
    }
    if "speed_curve" in data:
        curves["speed_curve"] = data["speed_curve"]
    return curves


def _find_session_pairs(sessions: list[str]) -> list[tuple[str, str]]:
    sessions_set = set(sessions)
    pairs = []
    for session in sessions:
        if session.endswith("_indoor"):
            base = session[: -len("_indoor")]
            outdoor = f"{base}_outdoor"
            if outdoor in sessions_set:
                pairs.append((session, outdoor))
    return pairs


def _session_complete(session: str) -> bool:
    session_dir = OUT_ROOT / session
    out_csv = session_dir / "tuning_scores.csv"
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return False
    neuron_dirs = list(session_dir.glob("neuron_*"))
    if not neuron_dirs:
        return False
    return True


def _plot_paired_polar(neuron_dir_a: Path, neuron_dir_b: Path) -> None:
    curves_a = _load_polar_curves(neuron_dir_a)
    curves_b = _load_polar_curves(neuron_dir_b)
    if curves_a is None or curves_b is None:
        return
    theta_deg = np.linspace(0.0, 360.0, len(curves_a["hd_curve"]), endpoint=False)
    plot_paired_polar_curve(
        neuron_dir_a / "yaw_indoor_outdoor.png",
        theta_deg,
        curves_a["hd_curve"],
        curves_b["hd_curve"],
        "Yaw tuning (indoor vs outdoor)",
    )
    plot_paired_polar_curve(
        neuron_dir_a / "roll_indoor_outdoor.png",
        theta_deg,
        curves_a["roll_curve"],
        curves_b["roll_curve"],
        "Roll tuning (indoor vs outdoor)",
    )
    plot_paired_polar_curve(
        neuron_dir_a / "pitch_indoor_outdoor.png",
        theta_deg,
        curves_a["pitch_curve"],
        curves_b["pitch_curve"],
        "Pitch tuning (indoor vs outdoor)",
    )
    if "speed_curve" in curves_a and "speed_curve" in curves_b:
        plot_paired_speed_curve(
            neuron_dir_a / "speed_indoor_outdoor.png",
            curves_a["speed_curve"],
            curves_b["speed_curve"],
            "Speed tuning (indoor vs outdoor)",
        )
    plot_paired_polar_curve(
        neuron_dir_b / "yaw_indoor_outdoor.png",
        theta_deg,
        curves_a["hd_curve"],
        curves_b["hd_curve"],
        "Yaw tuning (indoor vs outdoor)",
    )
    plot_paired_polar_curve(
        neuron_dir_b / "roll_indoor_outdoor.png",
        theta_deg,
        curves_a["roll_curve"],
        curves_b["roll_curve"],
        "Roll tuning (indoor vs outdoor)",
    )
    plot_paired_polar_curve(
        neuron_dir_b / "pitch_indoor_outdoor.png",
        theta_deg,
        curves_a["pitch_curve"],
        curves_b["pitch_curve"],
        "Pitch tuning (indoor vs outdoor)",
    )
    if "speed_curve" in curves_a and "speed_curve" in curves_b:
        plot_paired_speed_curve(
            neuron_dir_b / "speed_indoor_outdoor.png",
            curves_a["speed_curve"],
            curves_b["speed_curve"],
            "Speed tuning (indoor vs outdoor)",
        )


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
                thresholds = _score_thresholds(shuffle_scores)

                row = {
                    "neuron": n_idx,
                    "hd_score": scores.hd_score,
                    "roll_score": scores.roll_score,
                    "pitch_score": scores.pitch_score,
                    "speed_score": scores.speed_score,
                    "speed_stability": scores.speed_stability,
                    "spatial_stability": scores.spatial_stability,
                    "angular_stability": scores.angular_stability,
                    "roll_stability": scores.roll_stability,
                    "pitch_stability": scores.pitch_stability,
                    "hd_thresh": thresholds.get("hd_score"),
                    "roll_thresh": thresholds.get("roll_score"),
                    "pitch_thresh": thresholds.get("pitch_score"),
                    "speed_thresh": thresholds.get("speed_score"),
                    "speed_stab_thresh": thresholds.get("speed_stability"),
                }

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
                    aux,
                )
                _save_polar_curves(session_dir / f"neuron_{n_idx:03d}", aux)
    else:
        rng = np.random.default_rng(seed=0)
        for n_idx in pyr_idx.tolist():
            scores, aux = compute_scores_for_neuron(inputs, bins, n_idx)
            shuffle_scores = compute_shuffle_scores(inputs, bins, n_idx, n_shuffle, rng)

            thresholds = _score_thresholds(shuffle_scores)

            row = {
                "neuron": n_idx,
                "hd_score": scores.hd_score,
                "roll_score": scores.roll_score,
                "pitch_score": scores.pitch_score,
                "speed_score": scores.speed_score,
                "speed_stability": scores.speed_stability,
                "spatial_stability": scores.spatial_stability,
                "angular_stability": scores.angular_stability,
                "roll_stability": scores.roll_stability,
                "pitch_stability": scores.pitch_stability,
                "hd_thresh": thresholds.get("hd_score"),
                "roll_thresh": thresholds.get("roll_score"),
                "pitch_thresh": thresholds.get("pitch_score"),
                "speed_thresh": thresholds.get("speed_score"),
                "speed_stab_thresh": thresholds.get("speed_stability"),
            }

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
                aux,
            )
            _save_polar_curves(session_dir / f"neuron_{n_idx:03d}", aux)

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

    completed = []
    pending = []
    for session in sessions:
        if _session_complete(session):
            completed.append(session)
        else:
            pending.append(session)
    if completed:
        print(f"[INFO] Skipping {len(completed)} completed sessions.")

    processed = []
    for session in pending:
        try:
            out_csv = process_session(session, dayid2cellinfo)
        except Exception as exc:  # pragma: no cover - runtime logging
            print(f"[SKIP] {session}: {exc}")
            continue
        processed.append(session)
        print(f"[DONE] {session}: {out_csv}")

    all_processed = completed + processed
    _write_lines(OUT_ROOT / "sessions_processed.txt", all_processed)

    pair_sessions = all_processed if REBUILD_PAIRED_POLAR_PLOTS else processed
    pairs = _find_session_pairs(pair_sessions)
    for indoor_session, outdoor_session in pairs:
        indoor_dir = OUT_ROOT / indoor_session
        outdoor_dir = OUT_ROOT / outdoor_session
        indoor_neurons = sorted(indoor_dir.glob("neuron_*"))
        if not indoor_neurons:
            continue
        for neuron_dir in indoor_neurons:
            outdoor_neuron = outdoor_dir / neuron_dir.name
            if outdoor_neuron.exists():
                _plot_paired_polar(neuron_dir, outdoor_neuron)


if __name__ == "__main__":
    main()
