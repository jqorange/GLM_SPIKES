from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from scipy import optimize, sparse
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm

from glm_poisson_forward.config import (
    DLC_ROOT,
    FS_HZ,
    L1_LAMBDA,
    L1_PROX_LR,
    L1_PROX_STEPS,
    MAX_MISMATCH_FRAMES_50HZ,
    MAX_ITER,
    MIN_SPEED_CM_S,
    N_JOBS,
    POISSON_ALPHA,
    POSITION_ROOT,
    SPIKE_INPUT_MODE,
    SPIKE_INPUT_ROOT,
    VARIABLE_SPECS,
)
from glm_poisson_forward.design_matrix import (
    bin_col,
    build_design_matrix,
    build_position_index,
    build_smoothness_rows,
)
from glm_poisson_forward.io_utils import (
    _continuous_channel_specs,
    _load_variable_series,
    _resolve_speed_channel_keys,
    list_sessions_dlc_final,
    list_sessions_position,
    list_sessions_spike,
    load_spikes_50hz_counts,
    session_paths,
)


MODEL_VARS = ["Position", "Speed"]
RESIDUAL_SPIKE_COUNT_ROOT = Path(
    "/home/js3785/Dataset/GLM_Data/spike_count_residual"
)

# Set to a list such as ["F5D2", "F5D3"] if you want to restrict sessions.
KEEP_DAYS: list[str] | None = None


def _write_lines(path: Path, lines) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + ("\n" if not str(s).endswith("\n") else ""))


def _trim_time_axis(data_dict: Dict[str, np.ndarray], target_len: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    src_len = int(data_dict.get("T", 0))
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == src_len:
            out[key] = value[:target_len]
        else:
            out[key] = value
    out["T"] = int(target_len)
    return out


def _soft_threshold(weights: np.ndarray, thr: float) -> np.ndarray:
    return np.sign(weights) * np.maximum(np.abs(weights) - thr, 0.0)


def _prox_refine_poisson_l1(
    x_data: sparse.csr_matrix,
    y_data: np.ndarray,
    smooth_rows: sparse.csr_matrix,
    w_init: np.ndarray,
    b_init: float,
) -> Tuple[np.ndarray, float]:
    if L1_PROX_STEPS <= 0 or L1_LAMBDA <= 0:
        return w_init.astype(np.float64, copy=False), float(b_init)

    weights = w_init.astype(np.float64, copy=True)
    bias = float(b_init)
    n_samples = float(max(x_data.shape[0] + smooth_rows.shape[0], 1))
    step = float(L1_PROX_LR)

    for _ in range(int(L1_PROX_STEPS)):
        eta_data = np.asarray(x_data.dot(weights)).ravel() + bias
        np.clip(eta_data, -20.0, 20.0, out=eta_data)
        mu_data = np.exp(eta_data)
        residual_data = mu_data - y_data

        grad_w = np.asarray(x_data.T.dot(residual_data)).ravel()
        if smooth_rows.shape[0] > 0:
            eta_smooth = np.asarray(smooth_rows.dot(weights)).ravel()
            np.clip(eta_smooth, -20.0, 20.0, out=eta_smooth)
            mu_smooth = np.exp(eta_smooth)
            grad_w += np.asarray(smooth_rows.T.dot(mu_smooth)).ravel()

        grad_w = grad_w / n_samples + float(POISSON_ALPHA) * weights
        grad_b = float(np.sum(residual_data) / n_samples)

        weights = _soft_threshold(
            weights - step * grad_w,
            step * float(L1_LAMBDA),
        )
        bias = bias - step * grad_b

    return weights, bias


def _fit_poisson_with_prox_l1(
    x_data: sparse.csr_matrix,
    y_data: np.ndarray,
    smooth_rows: sparse.csr_matrix,
) -> Tuple[np.ndarray, float]:
    mdl = PoissonRegressor(
        alpha=POISSON_ALPHA,
        max_iter=MAX_ITER,
        fit_intercept=True,
    )
    mdl.fit(x_data, y_data)
    w0 = mdl.coef_.ravel().astype(np.float64, copy=False)
    b0 = float(mdl.intercept_)

    n_total = float(max(x_data.shape[0] + smooth_rows.shape[0], 1))
    alpha = float(POISSON_ALPHA)

    def _objective_and_grad(params: np.ndarray) -> Tuple[float, np.ndarray]:
        weights = params[:-1]
        bias = float(params[-1])

        eta_data = np.asarray(x_data.dot(weights)).ravel() + bias
        np.clip(eta_data, -20.0, 20.0, out=eta_data)
        mu_data = np.exp(eta_data)

        loss = float(np.sum(mu_data - y_data * eta_data))
        grad_w = np.asarray(x_data.T.dot(mu_data - y_data)).ravel()
        grad_b = float(np.sum(mu_data - y_data))

        if smooth_rows.shape[0] > 0:
            eta_smooth = np.asarray(smooth_rows.dot(weights)).ravel()
            np.clip(eta_smooth, -20.0, 20.0, out=eta_smooth)
            mu_smooth = np.exp(eta_smooth)
            loss += float(np.sum(mu_smooth))
            grad_w += np.asarray(smooth_rows.T.dot(mu_smooth)).ravel()

        loss = loss / n_total + 0.5 * alpha * float(np.dot(weights, weights))
        grad_w = grad_w / n_total + alpha * weights
        grad_b = grad_b / n_total
        grad = np.concatenate([grad_w, np.array([grad_b], dtype=np.float64)])
        return loss, grad

    x0 = np.concatenate([w0, np.array([b0], dtype=np.float64)])
    opt = optimize.minimize(
        fun=_objective_and_grad,
        x0=x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(MAX_ITER)},
    )
    if not opt.success and not np.isfinite(opt.fun):
        raise RuntimeError(f"Poisson fit failed: {opt.message}")

    w_opt = np.asarray(opt.x[:-1], dtype=np.float64)
    b_opt = float(opt.x[-1])
    return _prox_refine_poisson_l1(
        x_data,
        y_data.astype(np.float64, copy=False),
        smooth_rows,
        w_opt,
        b_opt,
    )


def _build_position_speed_inputs_50hz(paths: Dict[str, object]) -> Dict[str, np.ndarray]:
    loaded_position = _load_variable_series(paths, "Position")
    loaded_speed = _load_variable_series(paths, "Speed")

    lengths = [len(arr) for arr in loaded_position.values()]
    lengths.extend(len(arr) for arr in loaded_speed.values())
    if not lengths:
        raise ValueError("No Position/Speed covariates were loaded.")

    t_cov = int(min(lengths))
    out: Dict[str, np.ndarray] = {"T": t_cov}

    pos_spec = VARIABLE_SPECS["Position"]
    x_name, y_name = list(pos_spec["columns"].keys())[:2]
    head_x = loaded_position[x_name][:t_cov].astype(np.float32)
    head_y = loaded_position[y_name][:t_cov].astype(np.float32)
    pos_idx, n_pos, pos_xy_by_idx = build_position_index(head_x, head_y)
    out["position"] = pos_idx.astype(np.int32)
    out["n_pos"] = int(n_pos)
    out["position_xy_by_idx"] = pos_xy_by_idx.astype(np.int32)

    speed_spec = VARIABLE_SPECS["Speed"]
    speed_channels = _continuous_channel_specs("Speed")
    if not speed_channels:
        raise ValueError("Speed channel config is empty.")

    for channel in speed_channels:
        raw_key = channel["raw_key"]
        series = loaded_speed[raw_key][:t_cov].astype(np.float32)
        out[raw_key] = series

        vmin = vmax = None
        if "bin_range" in speed_spec:
            vmin, vmax = speed_spec["bin_range"]
        out[channel["bin_key"]] = bin_col(
            series,
            n_bins=int(channel["n_bins"]),
            vmin=vmin,
            vmax=vmax,
        ).astype(np.int32)

    return out


def prepare_session_position_speed(
    session: str,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    paths = session_paths(session)
    required_keys = ("spike", "position", "dlc_final")
    for key in required_keys:
        path = Path(paths[key])
        if not path.exists():
            raise FileNotFoundError(f"Missing input {key}: {path}")

    data_dict = _build_position_speed_inputs_50hz(paths)
    y_spike = load_spikes_50hz_counts(paths["spike"]).astype(np.float64)
    t_spk, n_neurons = y_spike.shape
    t_cov = int(data_dict["T"])

    if abs(t_cov - t_spk) > MAX_MISMATCH_FRAMES_50HZ:
        raise ValueError(
            f"Length mismatch @50Hz (> {MAX_MISMATCH_FRAMES_50HZ}): cov={t_cov}, spk={t_spk}"
        )

    t_common = min(t_cov, t_spk)
    data_common = _trim_time_axis(data_dict, t_common)
    y_common = y_spike[:t_common]

    speed_raw_key, _ = _resolve_speed_channel_keys()
    speed = data_common.get(speed_raw_key)
    if speed is None:
        speed = data_common.get("head_v")
    if speed is None:
        fit_mask = np.ones(t_common, dtype=bool)
    elif MIN_SPEED_CM_S <= 0:
        fit_mask = np.ones(speed.shape[0], dtype=bool)
    else:
        fit_mask = np.asarray(speed >= MIN_SPEED_CM_S, dtype=bool).reshape(-1)

    if not fit_mask.any():
        raise ValueError(f"No samples >= min speed {MIN_SPEED_CM_S:g} cm/s")

    meta: Dict[str, object] = {
        "session": session,
        "paths": {k: str(v) for k, v in paths.items()},
        "t_cov": int(t_cov),
        "t_spk": int(t_spk),
        "t_common": int(t_common),
        "n_neurons": int(n_neurons),
        "fit_frames": int(np.sum(fit_mask)),
        "excluded_low_speed_frames": int(t_common - np.sum(fit_mask)),
        "nan_tail_frames": int(t_spk - t_common),
        "matched_indoor_outdoor_lengths": False,
        "min_speed_cm_s": float(MIN_SPEED_CM_S),
        "model_vars": MODEL_VARS,
        "residual_transform": "clip_min_0",
        "tail_fill_value": 0.0,
    }
    return data_common, y_common, y_spike, fit_mask, meta


def fit_position_speed_residuals(
    data_common: Dict[str, np.ndarray],
    y_common: np.ndarray,
    y_spike: np.ndarray,
    fit_mask: np.ndarray,
) -> np.ndarray:
    x_all, feature_names = build_design_matrix(MODEL_VARS, data_common)
    pos_xy = data_common.get("position_xy_by_idx")
    smooth_rows = build_smoothness_rows(feature_names, position_xy_by_idx=pos_xy)
    fit_idx = np.flatnonzero(fit_mask)
    x_fit = x_all[fit_idx]
    n_common, n_neurons = y_common.shape

    def _fit_one_neuron(neuron_idx: int) -> np.ndarray:
        y_fit = y_common[fit_mask, neuron_idx].astype(np.float64, copy=False)
        if float(np.mean(y_fit)) <= 0:
            mu_all = np.full(n_common, 1e-12, dtype=np.float64)
        else:
            coef, intercept = _fit_poisson_with_prox_l1(x_fit, y_fit, smooth_rows)
            eta_all = np.asarray(x_all.dot(coef)).ravel() + float(intercept)
            np.clip(eta_all, -20.0, 20.0, out=eta_all)
            mu_all = np.clip(np.exp(eta_all), 1e-12, None)
        residual = y_common[:, neuron_idx].astype(np.float64, copy=False) - mu_all
        return np.clip(residual, 0.0, None).astype(np.float32)

    n_jobs = max(1, min(int(N_JOBS), int(n_neurons)))
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        residual_cols = list(
            tqdm(
                executor.map(_fit_one_neuron, range(n_neurons)),
                total=n_neurons,
                desc="Fit Position+Speed residuals",
                leave=False,
            )
        )

    residual_common = np.stack(residual_cols, axis=1).astype(np.float32, copy=False)
    residual_full = np.zeros(y_spike.shape, dtype=np.float32)
    residual_full[:n_common, :] = residual_common
    return residual_full


def save_residual_spike_count(
    session: str,
    residual_spike_count: np.ndarray,
    output_root: Path,
    meta: Dict[str, object],
) -> Path:
    src_path = Path(meta["paths"]["spike"])
    out_path = output_root / src_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src_attrs = {}
    with h5py.File(src_path, "r") as src_hf:
        for key, value in src_hf.attrs.items():
            src_attrs[key] = value

    with h5py.File(out_path, "w") as out_hf:
        out_hf.create_dataset(
            "spike_count",
            data=residual_spike_count.astype(np.float32, copy=False),
            compression="gzip",
        )
        for key, value in src_attrs.items():
            out_hf.attrs[key] = value
        out_hf.attrs["n_bins"] = np.int64(residual_spike_count.shape[0])
        out_hf.attrs["n_cells"] = np.int64(residual_spike_count.shape[1])
        out_hf.attrs["rate_hz"] = float(FS_HZ)
        out_hf.attrs["content"] = "nonnegative_residual_spike_count"
        out_hf.attrs["model_vars"] = ",".join(MODEL_VARS)
        out_hf.attrs["residual_transform"] = "clip_min_0"
        out_hf.attrs["matched_indoor_outdoor_lengths"] = False
        out_hf.attrs["min_speed_cm_s"] = float(MIN_SPEED_CM_S)
        out_hf.attrs["fit_frames"] = int(meta["fit_frames"])
        out_hf.attrs["common_frames"] = int(meta["t_common"])
        out_hf.attrs["covariate_frames"] = int(meta["t_cov"])
        out_hf.attrs["original_spike_frames"] = int(meta["t_spk"])
        out_hf.attrs["nan_tail_frames"] = int(meta["nan_tail_frames"])
        out_hf.attrs["tail_fill_value"] = 0.0
        out_hf.attrs["source_session"] = session

    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return out_path


def process_one_session(session: str, output_root: Path, overwrite: bool = False) -> Tuple[bool, str]:
    out_path = output_root / Path(session_paths(session)["spike"]).name
    if out_path.exists() and not overwrite:
        return True, f"exists: {out_path.name}"

    data_common, y_common, y_spike, fit_mask, meta = prepare_session_position_speed(session)
    residual_spike_count = fit_position_speed_residuals(
        data_common=data_common,
        y_common=y_common,
        y_spike=y_spike,
        fit_mask=fit_mask,
    )
    save_residual_spike_count(session, residual_spike_count, output_root, meta)
    return True, "OK"


def _list_target_sessions() -> list[str]:
    set_spk = list_sessions_spike(SPIKE_INPUT_ROOT)
    set_dlc = list_sessions_dlc_final(DLC_ROOT)
    set_pos = list_sessions_position(POSITION_ROOT)
    sessions = sorted(set_spk & set_dlc & set_pos)
    if KEEP_DAYS:
        sessions = [s for s in sessions if any(day in s for day in KEEP_DAYS)]
    return sessions


def main(output_root: Path | None = None, overwrite: bool = False) -> None:
    if SPIKE_INPUT_MODE != "count":
        raise ValueError(
            "This script writes residual spike_count files and expects "
            f"SPIKE_INPUT_MODE='count', got {SPIKE_INPUT_MODE!r}."
        )

    if output_root is None:
        output_root = RESIDUAL_SPIKE_COUNT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    all_present = _list_target_sessions()
    if not all_present:
        print("[FATAL] No sessions found with spike + position + speed inputs present.")
        return

    _write_lines(output_root / "sessions_all_present.txt", all_present)
    todo = []
    already_done = []
    for session in all_present:
        out_path = output_root / Path(session_paths(session)["spike"]).name
        if out_path.exists() and not overwrite:
            already_done.append(session)
        else:
            todo.append(session)

    _write_lines(output_root / "sessions_already_done.txt", already_done)
    _write_lines(output_root / "sessions_todo.txt", todo)

    print(f"[INFO] Output root: {output_root}")
    print(f"[INFO] Found {len(all_present)} sessions with required inputs present.")
    print(f"[INFO] Already done: {len(already_done)}")
    print(f"[INFO] To compute:   {len(todo)}")

    if not todo:
        print("[INFO] No sessions left to compute. Exiting.")
        return

    processed = []
    skipped = []
    for session in todo:
        try:
            ok, msg = process_one_session(session, output_root=output_root, overwrite=overwrite)
        except Exception as exc:  # pragma: no cover - runtime logging
            ok, msg = False, str(exc)

        if ok:
            processed.append(session)
            print(f"[DONE] {session}: {msg}")
        else:
            skipped.append((session, msg))
            print(f"[SKIP] {session}: {msg}")

    _write_lines(output_root / "sessions_processed.txt", processed)
    with open(output_root / "sessions_skipped.txt", "w", encoding="utf-8") as f:
        for session, reason in skipped:
            f.write(f"{session}\t{reason}\n")

    print("\n=== Residual export complete ===")
    print(f"All-present list: {output_root / 'sessions_all_present.txt'}")
    print(f"Already done:     {output_root / 'sessions_already_done.txt'}")
    print(f"To compute:       {output_root / 'sessions_todo.txt'}")
    print(f"Processed:        {output_root / 'sessions_processed.txt'}")
    print(f"Skipped:          {output_root / 'sessions_skipped.txt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Fit Position+Speed Poisson GLMs for each session/neuron and export "
            "residual spike_count files without indoor/outdoor matched-length truncation."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RESIDUAL_SPIKE_COUNT_ROOT,
        help=f"Directory for residual spike_count files (default: {RESIDUAL_SPIKE_COUNT_ROOT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite residual files that already exist.",
    )
    args = parser.parse_args()
    main(output_root=args.output_root, overwrite=args.overwrite)
