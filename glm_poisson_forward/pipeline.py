from pathlib import Path

from .config import DLC_ROOT, IMU_ROOT, POSITION_ROOT, SPIKE_ROOT, WEIGHTS_BASE
from .forward_search import run_one_session
from .io_utils import (
    is_session_done,
    list_sessions_dlc_final,
    list_sessions_imu,
    list_sessions_position,
    list_sessions_spike,
)


def _write_lines(path: Path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + ("\n" if not str(s).endswith("\n") else ""))


def main():
    set_imu = list_sessions_imu(IMU_ROOT)
    set_spk = list_sessions_spike(SPIKE_ROOT)
    set_dlc = list_sessions_dlc_final(DLC_ROOT)
    set_pos = list_sessions_position(POSITION_ROOT)

    all_present = sorted(list(set_imu & set_spk & set_dlc & set_pos))
    if not all_present:
        print("[FATAL] No sessions found with all required inputs present.")
        return

    _write_lines(WEIGHTS_BASE / "sessions_all_present.txt", all_present)
    print(f"[INFO] Found {len(all_present)} sessions with all required inputs present.")

    already_done = [s for s in all_present if is_session_done(s, WEIGHTS_BASE)]
    todo = [s for s in all_present if s not in already_done]

    _write_lines(WEIGHTS_BASE / "sessions_already_done.txt", already_done)
    _write_lines(WEIGHTS_BASE / "sessions_todo.txt", todo)

    print(f"[INFO] Already done: {len(already_done)} (see sessions_already_done.txt)")
    print(f"[INFO] To compute:   {len(todo)} (see sessions_todo.txt)")

    if not todo:
        print("[INFO] No sessions left to compute. Exiting.")
        return

    processed, skipped = [], []
    for session in todo:
        try:
            ok, msg = run_one_session(session)
        except Exception as e:  # pragma: no cover - runtime logging
            ok, msg = False, str(e)

        if ok:
            processed.append(session)
            print(f"[DONE] {session}: {msg}")
        else:
            skipped.append((session, msg))
            print(f"[SKIP] {session}: {msg}")

    _write_lines(WEIGHTS_BASE / "sessions_processed.txt", processed)
    with open(WEIGHTS_BASE / "sessions_skipped.txt", "w", encoding="utf-8") as f:
        for s, reason in skipped:
            f.write(f"{s}\t{reason}\n")

    print("\n=== Batch complete ===")
    print(f"All-present list: {WEIGHTS_BASE / 'sessions_all_present.txt'}")
    print(f"Already done:     {WEIGHTS_BASE / 'sessions_already_done.txt'}")
    print(f"To compute:       {WEIGHTS_BASE / 'sessions_todo.txt'}")
    print(f"Processed:        {WEIGHTS_BASE / 'sessions_processed.txt'}")
    print(f"Skipped:          {WEIGHTS_BASE / 'sessions_skipped.txt'}")


if __name__ == "__main__":
    main()
