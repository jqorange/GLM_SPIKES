from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from threading import Lock
import tempfile

import h5py
import numpy as np
import pandas as pd

from .angle_utils import circular_trim_range, shift_angles
from .config import (
    AGG_FACTOR,
    FS_HZ,
    IMU_ROOT,
    INPUT_FILES,
    MATCHED_SESSION_ALIGN,
    MATCH_INDOOR_OUTDOOR_LENGTHS,
    SPIKE_COUNT_ROOT,
    SPIKE_INPUT_MODE,
    VARS_ALL,
    VARIABLE_SPECS,
    SPIKE_ROOT,
)
from .design_matrix import bin_col, build_position_index

_CIRCULAR_TRIM_RANGE: dict[str, tuple[float, float]] = {}
_CIRCULAR_TRIM_LOCK = Lock()


def _spike_fs_tag() -> str:
    fs_hz = float(FS_HZ)
    return str(int(fs_hz)) if fs_hz.is_integer() else f"{fs_hz:g}"


def _spike_file_name(session: str) -> str:
    if SPIKE_INPUT_MODE == "binary":
        return f"{session}_1000Hz.h5"
    return f"{session}_{_spike_fs_tag()}Hz_count.h5"


def _spike_root() -> object:
    return SPIKE_ROOT if SPIKE_INPUT_MODE == "binary" else SPIKE_COUNT_ROOT


def _source_keys(spec: dict) -> list[str]:
    src = spec.get("source")
    if isinstance(src, str):
        return [src]
    if isinstance(src, (list, tuple)):
        out = [str(s) for s in src]
        if not out:
            raise ValueError("source list is empty")
        return out
    raise TypeError(f"source must be str/list/tuple, got {type(src)}")


def _resolve_column_source(
    paths: Dict[str, object],
    source_keys: list[str],
    csv_col: str,
) -> str:
    for src_key in source_keys:
        path = paths[src_key]
        try:
            cols = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if csv_col in cols:
            return src_key
    src_txt = ",".join(source_keys)
    raise KeyError(f"column '{csv_col}' not found in any source [{src_txt}]")


def _continuous_channel_specs(var_name: str) -> list[dict]:
    spec = VARIABLE_SPECS[var_name]
    col_spec = spec.get("column")
    if col_spec is None:
        return []
    source_keys = _source_keys(spec)

    n_bins_spec = spec.get("n_bins")
    design_base = spec.get("design_key", var_name)
    raw_base = spec.get("raw_key", design_base)
    bin_base = spec.get("bin_key", f"{var_name}_bin")

    def _pick_n_bins(i: int, key: str) -> int:
        if isinstance(n_bins_spec, dict):
            if key in n_bins_spec:
                return int(n_bins_spec[key])
            if i in n_bins_spec:
                return int(n_bins_spec[i])
            return int(spec["n_bins"])
        if isinstance(n_bins_spec, (list, tuple)):
            if i < len(n_bins_spec):
                return int(n_bins_spec[i])
            return int(spec["n_bins"])
        return int(spec["n_bins"])

    if isinstance(col_spec, str):
        src_key = source_keys[0]
        return [{
            "source_key": src_key,
            "csv_col": col_spec,
            "design_key": design_base,
            "raw_key": raw_base,
            "bin_key": bin_base,
            "n_bins": _pick_n_bins(0, col_spec),
        }]

    if isinstance(col_spec, dict):
        items = list(col_spec.items())
    elif isinstance(col_spec, (list, tuple)):
        items = [(str(c), str(c)) for c in col_spec]
    elif isinstance(col_spec, set):
        items = [(str(c), str(c)) for c in sorted(col_spec)]
    else:
        raise TypeError(f"{var_name}.column must be str/list/tuple/set/dict, got {type(col_spec)}")

    out = []
    for i, (name, csv_col) in enumerate(items):
        chan = str(name)
        out.append({
            "source_key": source_keys[0],
            "csv_col": str(csv_col),
            "design_key": chan,
            "raw_key": chan,
            "bin_key": f"{chan}_bin",
            "n_bins": _pick_n_bins(i, chan),
        })
    return out


def list_sessions_imu(root):
    if not root.exists():
        return set()
    spec = INPUT_FILES["imu"]
    out = set()
    for sess_dir in root.iterdir():
        if not sess_dir.is_dir():
            continue
        stem = sess_dir.name
        f = sess_dir / spec["filename"].format(session=stem)
        if f.exists():
            out.add(stem)
    return out


def _load_global_circular_trim_range(
    var_name: str,
    csv_col: str | None = None,
    source_key: str | None = None,
) -> tuple[float, float]:
    cache_key = f"{var_name}:{source_key}:{csv_col}" if csv_col is not None else var_name
    with _CIRCULAR_TRIM_LOCK:
        if cache_key in _CIRCULAR_TRIM_RANGE:
            return _CIRCULAR_TRIM_RANGE[cache_key]

    var_spec = VARIABLE_SPECS[var_name]
    src_key = source_key if source_key is not None else _source_keys(var_spec)[0]
    col = csv_col if csv_col is not None else var_spec["column"]
    lower_pct, upper_pct = var_spec.get("trim_percentiles", (1.0, 99.0))

    sessions = sorted(list_sessions_imu(IMU_ROOT))
    val_all = []

    def _load_one_session_to_tmp(session: str, tmp_dir: Path) -> tuple[str, Path] | None:
        paths = session_paths(session)
        path = paths[src_key]
        if not path.exists():
            return None
        vals = pd.read_csv(path, usecols=[col]).astype(np.float32)[col].to_numpy(dtype=np.float32)
        if var_spec.get("unit") == "deg":
            vals = np.deg2rad(vals).astype(np.float32)
        if var_spec.get("wrap_2pi", False):
            vals = np.mod(vals, 2.0 * np.pi).astype(np.float32)

        tmp_path = tmp_dir / f"{session}.npy"
        np.save(tmp_path, vals, allow_pickle=False)
        return session, tmp_path

    with tempfile.TemporaryDirectory(prefix="trim_range_") as td:
        tmp_dir = Path(td)
        per_session_tmp: dict[str, Path] = {}
        if sessions:
            max_workers = max(1, min(len(sessions), 32))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for rec in ex.map(lambda s: _load_one_session_to_tmp(s, tmp_dir), sessions):
                    if rec is None:
                        continue
                    sess, npy_path = rec
                    per_session_tmp[sess] = npy_path

        for session in sessions:
            npy_path = per_session_tmp.get(session)
            if npy_path is None:
                continue
            vals = np.load(npy_path, allow_pickle=False).astype(np.float32, copy=False)
            val_all.append(vals)

    if val_all:
        vals = np.concatenate(val_all)
        trim_start, trim_width = circular_trim_range(vals, lower_pct, upper_pct)
    else:
        trim_start, trim_width = 0.0, 2.0 * np.pi

    # Print actual range used after percentile trim (once per variable/channel via cache_key).
    if np.isclose(lower_pct, 1.0) and np.isclose(upper_pct, 99.0):
        range_end = trim_start + trim_width
        col_txt = f", source={src_key}, column={col}" if csv_col is not None else ""
        print(
            f"[trim_range] var={var_name}{col_txt}, "
            f"percentiles=({lower_pct:.1f},{upper_pct:.1f}), "
            f"actual_range=({trim_start:.6f}, {range_end:.6f}), width={trim_width:.6f}"
        )

    with _CIRCULAR_TRIM_LOCK:
        _CIRCULAR_TRIM_RANGE[cache_key] = (trim_start, trim_width)
        return _CIRCULAR_TRIM_RANGE[cache_key]


def warmup_global_circular_trim_ranges(max_workers: int = 8) -> None:
    """Precompute circular trim ranges once before running per-session jobs."""
    tasks: list[tuple[str, str, str | None]] = []
    for var_name in VARS_ALL:
        spec = VARIABLE_SPECS.get(var_name, {})
        if "trim_percentiles" not in spec:
            continue
        source_keys = _source_keys(spec)
        source_key = source_keys[0] if source_keys else None
        for ch in _continuous_channel_specs(var_name):
            tasks.append((var_name, ch["csv_col"], source_key))

    if not tasks:
        return

    n_workers = max(1, min(int(max_workers), len(tasks)))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [
            ex.submit(_load_global_circular_trim_range, var_name, csv_col, source_key)
            for var_name, csv_col, source_key in tasks
        ]
        for fut in futures:
            try:
                fut.result()
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[WARN] trim warmup failed: {exc}")


def list_sessions_spike(root):
    if not root.exists():
        return set()
    if SPIKE_INPUT_MODE == "binary":
        return {f.stem.replace("_1000Hz", "") for f in root.glob("*_1000Hz.h5")}

    suffix = f"_{_spike_fs_tag()}Hz_count"
    out = set()
    for f in root.glob(f"*{suffix}.h5"):
        stem = f.stem
        if stem.endswith(suffix):
            out.add(stem[:-len(suffix)])
    return out


def list_sessions_dlc_final(root):
    if not root.exists():
        return set()
    spec = INPUT_FILES["dlc_final"]
    out = set()
    for sess_dir in root.iterdir():
        if not sess_dir.is_dir():
            continue
        stem = sess_dir.name
        f1 = sess_dir / spec["filename"].format(session=stem)
        if f1.exists():
            out.add(stem)
    return out


def list_sessions_position(root):
    if not root.exists():
        return set()
    spec = INPUT_FILES["position"]
    prefix, suffix = spec["filename"].split("{session}")
    out = set()
    for f in root.glob(f"{prefix}*{suffix}"):
        stem = f.name
        if stem.startswith(prefix) and stem.endswith(suffix):
            out.add(stem[len(prefix): len(stem) - len(suffix)])
    return out


def session_paths(session: str) -> Dict[str, object]:
    paths = {"spike": _spike_root() / _spike_file_name(session)}
    for key, spec in INPUT_FILES.items():
        filename = spec["filename"].format(session=session)
        if spec.get("parent_dir_is_session", False):
            paths[key] = spec["root"] / session / filename
        else:
            paths[key] = spec["root"] / filename
    return paths


def paired_session_name(session: str) -> str | None:
    if session.endswith("_indoor"):
        return f"{session[:-7]}_outdoor"
    if session.endswith("_outdoor"):
        return f"{session[:-8]}_indoor"
    return None


def _slice_time_axis(arr: np.ndarray, target_len: int, align: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim < 1 or arr.shape[0] <= target_len:
        return arr
    if align == "end":
        return arr[-target_len:]
    return arr[:target_len]


def _spike_length_50hz(h5_path: Path) -> int:
    with h5py.File(h5_path, "r") as hf:
        if SPIKE_INPUT_MODE == "binary":
            t1000 = int(hf["spike_binary"].shape[0])
            return (t1000 // AGG_FACTOR)
        return int(hf["spike_count"].shape[0])


@lru_cache(maxsize=None)
def session_effective_length_50hz(session: str) -> int | None:
    paths = session_paths(session)
    required_inputs = ["spike"] + list(INPUT_FILES.keys())
    for key in required_inputs:
        path = paths[key]
        if not Path(path).exists():
            return None

    try:
        loaded = {var: _load_variable_series(paths, var) for var in VARS_ALL}
    except Exception:
        return None

    lengths = []
    for var in VARS_ALL:
        for arr in loaded[var].values():
            lengths.append(len(arr))
    t_cov = min(lengths) if lengths else 0
    if t_cov <= 0:
        return None

    try:
        t_spk = _spike_length_50hz(Path(paths["spike"]))
    except Exception:
        return None

    return int(min(t_cov, t_spk))


def matched_pair_target_length_50hz(session: str) -> int | None:
    if not MATCH_INDOOR_OUTDOOR_LENGTHS:
        return None
    pair_session = paired_session_name(session)
    if not pair_session:
        return None
    this_len = session_effective_length_50hz(session)
    pair_len = session_effective_length_50hz(pair_session)
    if this_len is None or pair_len is None:
        return None
    return int(min(this_len, pair_len))


def truncate_to_matched_pair_length(
    session: str,
    data_dict: Dict[str, np.ndarray],
    y50: np.ndarray,
) -> tuple[Dict[str, np.ndarray], np.ndarray, int | None]:
    target_len = matched_pair_target_length_50hz(session)
    if target_len is None:
        return data_dict, y50, None

    t_cov = int(data_dict.get("T", 0))
    t_spk = int(y50.shape[0]) if getattr(y50, "ndim", 0) >= 1 else 0
    local_len = min(t_cov, t_spk)
    if target_len <= 0 or local_len <= 0:
        return data_dict, y50, None
    if target_len >= local_len:
        return data_dict, y50, target_len

    out = {}
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == t_cov:
            out[key] = _slice_time_axis(value, target_len, MATCHED_SESSION_ALIGN)
        else:
            out[key] = value
    out["T"] = int(target_len)
    y_out = _slice_time_axis(y50, target_len, MATCHED_SESSION_ALIGN)
    return out, y_out, target_len


def _resample_series(values: np.ndarray, src_fs_hz: float, dst_fs_hz: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0:
        return values
    if np.isclose(src_fs_hz, dst_fs_hz):
        return values

    t_src = np.arange(len(values), dtype=np.float32) / float(src_fs_hz)
    duration = t_src[-1]
    n_dst = int(np.floor(duration * float(dst_fs_hz))) + 1
    n_dst = max(n_dst, 1)
    t_dst = np.arange(n_dst, dtype=np.float32) / float(dst_fs_hz)
    t_dst = np.clip(t_dst, 0.0, duration)
    return np.interp(t_dst, t_src, values).astype(np.float32)


def _load_variable_series(paths: Dict[str, object], variable_name: str) -> dict[str, np.ndarray]:
    spec = VARIABLE_SPECS[variable_name]
    if spec.get("kind") == "time":
        return {}
    source_keys = _source_keys(spec)
    src_file_specs = {k: INPUT_FILES[k] for k in source_keys}
    fs_by_src = {k: float(v["fs_hz"]) for k, v in src_file_specs.items()}

    if "columns" in spec:
        src_key = source_keys[0]
        fs_src = fs_by_src[src_key]
        columns = list(spec["columns"].values())
        raw_df = pd.read_csv(paths[src_key], usecols=columns).astype(np.float32)
        out = {}
        for target_name, csv_name in spec["columns"].items():
            raw = raw_df[csv_name].to_numpy(dtype=np.float32)
            out[target_name] = _resample_series(raw, fs_src, FS_HZ)
        return out

    channels = _continuous_channel_specs(variable_name)
    for c in channels:
        if len(source_keys) > 1:
            c["source_key"] = _resolve_column_source(paths, source_keys, c["csv_col"])
        else:
            c["source_key"] = source_keys[0]

    by_source: Dict[str, list[str]] = {}
    for c in channels:
        by_source.setdefault(c["source_key"], []).append(c["csv_col"])
    raw_by_source = {}
    for src_key, cols in by_source.items():
        # remove duplicates, keep order
        cols_u = list(dict.fromkeys(cols))
        raw_by_source[src_key] = pd.read_csv(paths[src_key], usecols=cols_u).astype(np.float32)

    out = {}
    for c in channels:
        src_key = c["source_key"]
        raw = raw_by_source[src_key][c["csv_col"]].to_numpy(dtype=np.float32)
        series = _resample_series(raw, fs_by_src[src_key], FS_HZ)
        if spec.get("unit") == "deg":
            series = np.deg2rad(series).astype(np.float32)
        if spec.get("wrap_2pi", False):
            series = np.mod(series, 2.0 * np.pi).astype(np.float32)
        out[c["raw_key"]] = series
    return out


def is_session_done(session: str, weights_base) -> bool:
    out_dir = weights_base / session
    if not out_dir.exists():
        return False
    if (out_dir / "_SUCCESS").exists():
        return True
    sel = out_dir / "selected_models.csv"
    if sel.exists():
        try:
            df = pd.read_csv(sel)
            if df.shape[0] > 0:
                return True
        except Exception:
            pass
    return False


def load_spikes_50hz_counts(h5_path) -> np.ndarray:
    with h5py.File(h5_path, "r") as hf:
        if SPIKE_INPUT_MODE == "binary":
            Y1000 = hf["spike_binary"][:].astype(np.int16)  # (T1000, N)

            T1000, N = Y1000.shape
            T1000_trim = (T1000 // AGG_FACTOR) * AGG_FACTOR
            if T1000_trim <= 0:
                raise ValueError("Spike length too short after trimming.")

            Y1000 = Y1000[:T1000_trim]
            Y50 = Y1000.reshape(-1, AGG_FACTOR, N).sum(axis=1)  # (T50, N)
            return Y50.astype(np.int32)

        Y50 = hf["spike_count"][:]
        rate_hz = hf.attrs.get("rate_hz")

    if rate_hz is not None and not np.isclose(float(rate_hz), float(FS_HZ)):
        raise ValueError(f"Spike count file rate_hz={rate_hz} does not match FS_HZ={FS_HZ}.")
    return np.asarray(Y50, dtype=np.int32)


def _build_time_continuous_series(n_frames: int, time_bin_sec: float) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    frames_per_bin = max(int(round(float(time_bin_sec) * float(FS_HZ))), 1)
    bin_idx = (np.arange(n_frames, dtype=np.int32) // frames_per_bin).astype(np.int32)
    n_bins = int(bin_idx[-1]) + 1
    if n_bins <= 1:
        return np.full((n_frames,), 1.0, dtype=np.float32)
    return (bin_idx.astype(np.float32) / float(n_bins - 1)).astype(np.float32)


def rebuild_inputs_50hz(session: str, paths: Dict[str, object]) -> Dict[str, np.ndarray]:
    loaded = {var: _load_variable_series(paths, var) for var in VARS_ALL}
    lengths = []
    for var in VARS_ALL:
        for arr in loaded[var].values():
            lengths.append(len(arr))
    L = min(lengths) if lengths else 0

    out: Dict[str, np.ndarray] = {"T": int(L)}

    for var in VARS_ALL:
        spec = VARIABLE_SPECS[var]
        kind = spec.get("kind", "continuous")
        if kind == "position2d":
            x_name, y_name = list(spec["columns"].keys())[:2]
            head_x = loaded[var][x_name][:L]
            head_y = loaded[var][y_name][:L]
            pos_idx, n_pos, pos_xy_by_idx = build_position_index(head_x, head_y)
            out["position"] = pos_idx.astype(np.int32)
            out["n_pos"] = int(n_pos)
            out["position_xy_by_idx"] = pos_xy_by_idx.astype(np.int32)
            continue

        if kind == "time":
            value_key = spec.get("value_key", spec.get("design_key", var))
            time_bin_sec = float(spec.get("time_bin_sec", 1.0))
            out[value_key] = _build_time_continuous_series(L, time_bin_sec)
            continue

        source_keys = _source_keys(spec)
        channels = _continuous_channel_specs(var)
        for c in channels:
            if len(source_keys) > 1:
                c["source_key"] = _resolve_column_source(paths, source_keys, c["csv_col"])
            else:
                c["source_key"] = source_keys[0]

        for c in channels:
            series = loaded[var][c["raw_key"]][:L].astype(np.float32)
            out[c["raw_key"]] = series

            n_bins = int(c["n_bins"])
            vmin = vmax = None
            if "bin_range" in spec:
                vmin, vmax = spec["bin_range"]
            if "trim_percentiles" in spec:
                trim_start, trim_width = _load_global_circular_trim_range(
                    var,
                    c["csv_col"],
                    source_key=c.get("source_key"),
                )
                series_to_bin = shift_angles(series, trim_start)
                vmin, vmax = 0.0, trim_width
            else:
                series_to_bin = series
            out[c["bin_key"]] = bin_col(series_to_bin, n_bins=n_bins, vmin=vmin, vmax=vmax).astype(np.int32)

    return out


def filter_by_min_speed(
    data_dict: Dict[str, np.ndarray],
    Y_all: np.ndarray,
    min_speed_cm_s: float,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray | None]:
    if min_speed_cm_s <= 0:
        return data_dict, Y_all, None
    head_v = data_dict.get("head_v")
    if head_v is None:
        return data_dict, Y_all, None
    mask = head_v >= min_speed_cm_s
    if mask.ndim != 1:
        mask = mask.reshape(-1)
    filtered = {}
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray) and v.shape[0] == mask.shape[0]:
            filtered[k] = v[mask]
        else:
            filtered[k] = v
    filtered["T"] = int(np.sum(mask))
    Y_all = Y_all[mask]
    return filtered, Y_all, mask


def apply_residual_speed(data_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    speed_spec = VARIABLE_SPECS.get("Speed", {})
    speed_channels = _continuous_channel_specs("Speed") if speed_spec else []
    speed_raw_key = "head_v"
    speed_bin_key = "head_v_bin"
    if speed_channels:
        if any(c["raw_key"] == "head_v" for c in speed_channels):
            speed_raw_key = "head_v"
            speed_bin_key = "head_v_bin"
        else:
            speed_raw_key = speed_channels[0]["raw_key"]
            speed_bin_key = speed_channels[0]["bin_key"]

    head_v = data_dict.get(speed_raw_key)
    pos_idx = data_dict.get("position")
    n_pos = data_dict.get("n_pos")
    if head_v is None or pos_idx is None or n_pos is None:
        return data_dict

    n_pos = int(n_pos)
    sums = np.bincount(pos_idx, weights=head_v, minlength=n_pos)
    counts = np.bincount(pos_idx, minlength=n_pos)
    mean_speed = np.divide(sums, counts, out=np.zeros_like(sums, dtype=np.float32), where=counts > 0)
    speed_hat = mean_speed[pos_idx]
    speed_res = head_v - speed_hat

    updated = dict(data_dict)
    updated[f"{speed_raw_key}_raw"] = head_v.astype(np.float32)
    updated[speed_raw_key] = speed_res.astype(np.float32)
    updated["speed_hat"] = speed_hat.astype(np.float32)
    n_bins = 15
    for c in speed_channels:
        if c["raw_key"] == speed_raw_key:
            n_bins = int(c["n_bins"])
            break
    updated[speed_bin_key] = bin_col(speed_res, n_bins=n_bins)
    return updated
