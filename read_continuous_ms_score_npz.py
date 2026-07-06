#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a continuous MS-score NPZ file produced by "
            "compute_continuous_ms_scores.py."
        )
    )
    parser.add_argument(
        "npz_path",
        type=Path,
        help="Path to a *_continuous_ms_score.npz file.",
    )
    parser.add_argument(
        "--neuron-idx",
        type=int,
        default=None,
        help="Optional 0-based neuron index to inspect/export.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional CSV path for exporting one neuron's full time series.",
    )
    parser.add_argument(
        "--show-keys",
        action="store_true",
        help="Print all keys stored in the NPZ.",
    )
    return parser.parse_args()


def _as_scalar_str(value) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return str(arr.item())
    if arr.size == 1:
        return str(arr.reshape(-1)[0])
    return str(arr.tolist())


def load_npz(npz_path: Path) -> dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def print_summary(payload: dict[str, np.ndarray]) -> None:
    feature_names = [str(x) for x in np.asarray(payload["feature_names"]).tolist()]
    neuron_idx = np.asarray(payload["neuron_idx"], dtype=int)
    time_bin = np.asarray(payload["time_bin"], dtype=int)
    fitted = np.asarray(payload["fitted"], dtype=int)
    status = np.asarray(payload["status"]).astype(str)

    unique_status, counts = np.unique(status, return_counts=True)
    status_text = ", ".join(f"{s}:{int(c)}" for s, c in zip(unique_status, counts))

    print(f"session:           {_as_scalar_str(payload['session'])}")
    print(f"group:             {_as_scalar_str(payload['group'])}")
    if "full_model_source" in payload:
        print(f"full_model_source: {_as_scalar_str(payload['full_model_source'])}")
    print(f"features:          {feature_names}")
    print(f"n_neurons:         {neuron_idx.size}")
    print(f"n_time_bins:       {time_bin.size}")
    print(f"n_fitted:          {int(np.sum(fitted > 0))}")
    print(f"status_counts:     {status_text}")


def build_neuron_dataframe(payload: dict[str, np.ndarray], neuron_idx: int) -> pd.DataFrame:
    neuron_ids = np.asarray(payload["neuron_idx"], dtype=int)
    matches = np.where(neuron_ids == int(neuron_idx))[0]
    if matches.size == 0:
        raise KeyError(f"neuron_idx={neuron_idx} not found in NPZ.")
    row = int(matches[0])

    feature_names = [str(x) for x in np.asarray(payload["feature_names"]).tolist()]
    time_bin = np.asarray(payload["time_bin"], dtype=int)
    time_sec = np.asarray(payload["time_sec"], dtype=float)

    df = pd.DataFrame(
        {
            "session": _as_scalar_str(payload["session"]),
            "group": _as_scalar_str(payload["group"]),
            "neuron_idx": int(neuron_idx),
            "fitted": int(np.asarray(payload["fitted"], dtype=int)[row]),
            "status": str(np.asarray(payload["status"]).astype(str)[row]),
            "full_model": str(np.asarray(payload["full_model"]).astype(str)[row]),
            "full_ll_gain": float(np.asarray(payload["full_ll_gain"], dtype=float)[row]),
            "time_bin": time_bin,
            "time_sec": time_sec,
            "T_i": np.asarray(payload["T_i"], dtype=float)[row],
            "D_i": np.asarray(payload["D_i"], dtype=float)[row],
            "ms_score": np.asarray(payload["ms_score"], dtype=float)[row],
        }
    )

    for feat in feature_names:
        delta_key = f"delta_ll_{feat}"
        d_key = f"d_{feat}"
        p_key = f"p_{feat}"
        if delta_key in payload:
            df[delta_key] = np.asarray(payload[delta_key], dtype=float)[row]
        if d_key in payload:
            df[d_key] = np.asarray(payload[d_key], dtype=float)[row]
        if p_key in payload:
            df[p_key] = np.asarray(payload[p_key], dtype=float)[row]
    return df


def main() -> None:
    args = parse_args()
    npz_path = args.npz_path.resolve()
    payload = load_npz(npz_path)

    if args.show_keys:
        print("keys:")
        for key in sorted(payload.keys()):
            print(f"  {key}")
        print("")

    print_summary(payload)

    if args.neuron_idx is not None:
        df = build_neuron_dataframe(payload, neuron_idx=int(args.neuron_idx))
        print("")
        print(f"neuron_idx:        {int(args.neuron_idx)}")
        print(f"rows:              {df.shape[0]}")
        print(f"mean_ms_score:     {float(df['ms_score'].mean()):.6g}")
        print(f"sum_ms_score:      {float(df['ms_score'].sum()):.6g}")
        print(f"max_ms_score:      {float(df['ms_score'].max()):.6g}")
        print(df.head(10).to_string(index=False))

        if args.out_csv is not None:
            out_csv = args.out_csv.resolve()
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv, index=False)
            print("")
            print(f"wrote_csv:         {out_csv}")


if __name__ == "__main__":
    main()
