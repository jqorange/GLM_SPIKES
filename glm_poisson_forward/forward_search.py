import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.model_selection import KFold
from tqdm import tqdm

from .config import (
    ALPHA,
    CV_FOLDS,
    ANGLE_N_BINS,
    MAX_MISMATCH_FRAMES_50HZ,
    MIN_SPEED_CM_S,
    N_JOBS,
    PLOT_END_SEC,
    PLOT_START_SEC,
    PLOT_SMOOTH_MS,
    PLOT_ZSCORE,
    VARS_ALL,
    WEIGHTS_BASE,
)
from .design_matrix import build_design_matrix, model_key_from_vars
from .io_utils import (
    apply_residual_angle,
    apply_residual_speed,
    filter_by_min_speed,
    load_spikes_50hz_counts,
    rebuild_inputs_50hz,
    session_paths,
)
from .metrics import (
    build_oof_constant_mu,
    compute_delta_ll_poisson_vs_baseline,
    wilcoxon_greater,
)
from .plotting_utils import load_oof_from_neuron_dir, plot_fitting_curve
from .training import (
    fit_predict_one_fold_poisson,
    save_full_fit_weights_for_all_neurons,
    save_neuron_artifacts_for_model,
)


def _build_design_cache(data_dict: Dict[str, np.ndarray]):
    cache: Dict[str, Tuple[sparse.csr_matrix, List[str]]] = {}

    def get_X_and_feats(model_vars: List[str]) -> Tuple[sparse.csr_matrix, List[str]]:
        mk = model_key_from_vars(model_vars)
        if mk in cache:
            return cache[mk]
        X, feats = build_design_matrix(model_vars, data_dict)
        cache[mk] = (X, feats)
        return X, feats

    return get_X_and_feats


@dataclass
class StepRecord:
    step: int
    model: List[str]
    mean_delta_ll: float
    fold_delta_ll: List[float]
    p_value_vs_prev: float = None
    stat_vs_prev: float = None
    n_pairs: int = None
    accepted: bool = True


def _delta_ll_cv_for_neuron(
    model_vars: List[str],
    neuron_idx: int,
    Y_all: np.ndarray,
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
    get_X_and_feats,
) -> Tuple[float, List[float]]:
    X_all_m, _feat = get_X_and_feats(model_vars)
    y = Y_all[:, neuron_idx].astype(np.float64)

    fold_delta_ll: List[float] = []
    mu_oof = np.full_like(y, np.nan, dtype=np.float32)

    for (tr, va) in folds_idx:
        mu_va, _llhi = fit_predict_one_fold_poisson(X_all_m, y, tr, va)
        mean_tr = float(np.mean(y[tr]))
        base_rate = max(mean_tr, 1e-12)
        mu_base = np.full_like(y[va], base_rate, dtype=np.float64)
        delta_ll = compute_delta_ll_poisson_vs_baseline(y[va], mu_va, mu_base)
        fold_delta_ll.append(float(delta_ll))
        mu_oof[va] = mu_va

    mu_base_oof = build_oof_constant_mu(y, folds_idx)
    delta_ll_oof = compute_delta_ll_poisson_vs_baseline(y, mu_oof, mu_base_oof)
    return float(delta_ll_oof), fold_delta_ll


def _save_accepted_step(
    neuron_idx: int,
    model_vars: List[str],
    OUT_ROOT: Path,
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
    Y_all: np.ndarray,
    get_X_and_feats,
):
    model_dir = OUT_ROOT / model_key_from_vars(model_vars)
    X_all_m, feat_names = get_X_and_feats(model_vars)
    y = Y_all[:, neuron_idx].astype(np.float64)
    neuron_dir = model_dir / f"neuron_{neuron_idx+1}"
    return save_neuron_artifacts_for_model(
        model_vars=model_vars,
        model_dir=model_dir,
        neuron_dir=neuron_dir,
        neuron_index=neuron_idx,
        folds=folds_idx,
        X_all=X_all_m,
        y_all=y,
        feature_names=feat_names,
    )


def _forward_select_one_neuron(
    neuron_idx: int,
    Y_all: np.ndarray,
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
    OUT_ROOT: Path,
    get_X_and_feats,
):
    path_records: List[StepRecord] = []
    remaining = VARS_ALL.copy()

    single_candidates = []
    for v in remaining:
        oof_delta_ll, fold_delta_ll = _delta_ll_cv_for_neuron(
            [v], neuron_idx, Y_all, folds_idx, get_X_and_feats
        )
        single_candidates.append((v, oof_delta_ll, fold_delta_ll))

    single_candidates.sort(key=lambda x: (x[1] if np.isfinite(x[1]) else -np.inf), reverse=True)
    best_v, best_oof_delta_ll, best_fold = single_candidates[0]

    stat, p, n = wilcoxon_greater(best_fold, b=None)
    accepted = (p < ALPHA)

    path_records.append(
        StepRecord(
            step=1,
            model=[best_v],
            mean_delta_ll=best_oof_delta_ll,
            fold_delta_ll=list(map(float, best_fold)),
            p_value_vs_prev=p,
            stat_vs_prev=stat,
            n_pairs=n,
            accepted=accepted,
        )
    )

    if not accepted:
        return {
            "neuron": f"neuron_{neuron_idx+1}",
            "final_model": [],
            "classified": False,
            "path": [vars(s) for s in path_records],
        }

    _save_accepted_step(neuron_idx, [best_v], OUT_ROOT, folds_idx, Y_all, get_X_and_feats)

    selected = [best_v]
    remaining.remove(best_v)
    fold_delta_ll_prev = list(best_fold)

    step = 2
    while remaining:
        cand_list = []
        for cand in remaining:
            trial_vars = selected + [cand]
            oof_delta_ll, fold_delta_ll = _delta_ll_cv_for_neuron(
                trial_vars, neuron_idx, Y_all, folds_idx, get_X_and_feats
            )
            cand_list.append((cand, trial_vars, oof_delta_ll, fold_delta_ll))

        cand_list.sort(key=lambda x: (x[2] if np.isfinite(x[2]) else -np.inf), reverse=True)
        best_cand, best_trial_vars, best_trial_oof_delta_ll, best_trial_fold = cand_list[0]

        stat, p, n = wilcoxon_greater(best_trial_fold, fold_delta_ll_prev)
        accepted = (p < ALPHA)

        path_records.append(
            StepRecord(
                step=step,
                model=best_trial_vars,
                mean_delta_ll=best_trial_oof_delta_ll,
                fold_delta_ll=list(map(float, best_trial_fold)),
                p_value_vs_prev=p,
                stat_vs_prev=stat,
                n_pairs=n,
                accepted=accepted,
            )
        )

        if not accepted:
            break

        _save_accepted_step(neuron_idx, best_trial_vars, OUT_ROOT, folds_idx, Y_all, get_X_and_feats)
        selected = best_trial_vars
        remaining.remove(best_cand)
        fold_delta_ll_prev = list(best_trial_fold)
        step += 1

        if len(selected) == len(VARS_ALL):
            break

    const_p = None
    const_stat = None
    const_n = None
    if selected:
        _delta_ll, fold_delta_ll = _delta_ll_cv_for_neuron(
            selected, neuron_idx, Y_all, folds_idx, get_X_and_feats
        )
        const_stat, const_p, const_n = wilcoxon_greater(fold_delta_ll, b=None)
        if const_p >= ALPHA:
            return {
                "neuron": f"neuron_{neuron_idx+1}",
                "final_model": None,
                "classified": False,
                "path": [vars(s) for s in path_records],
                "const_rate_p_value": const_p,
                "const_rate_stat": const_stat,
                "const_rate_n_pairs": const_n,
            }

    return {
        "neuron": f"neuron_{neuron_idx+1}",
        "final_model": selected,
        "classified": True,
        "path": [vars(s) for s in path_records],
        "const_rate_p_value": const_p,
        "const_rate_stat": const_stat,
        "const_rate_n_pairs": const_n,
    }


def _plot_selected_models(
    rows,
    OUT_ROOT: Path,
    session: str,
    Y_all: np.ndarray,
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
):
    fig_dir = OUT_ROOT / "figures"
    for rec in rows:
        neuron_name = rec["neuron"]
        model_key = rec["final_model"]
        model_dir = OUT_ROOT / model_key
        neuron_dir = model_dir / neuron_name

        try:
            neuron_idx = int(neuron_name.split("_")[-1]) - 1
            y_full = Y_all[:, neuron_idx].astype(np.float64)
            y_oof, mu_oof = load_oof_from_neuron_dir(neuron_dir)
            mu_base_oof = build_oof_constant_mu(y_full, folds_idx)
            delta_ll = compute_delta_ll_poisson_vs_baseline(y_oof, mu_oof, mu_base_oof)
            title = (
                f"{session} | {neuron_name} | PoissonGLM | vars={model_key.replace('_','+')} | "
                f"ΔLL={delta_ll:.4f}"
            )
            out_png = fig_dir / f"{neuron_name}__{model_key}.png"
            plot_fitting_curve(
                out_png,
                title,
                y_oof,
                mu_oof,
                smooth_ms=PLOT_SMOOTH_MS,
                start_sec=PLOT_START_SEC,
                end_sec=PLOT_END_SEC,
                do_zscore=PLOT_ZSCORE,
            )
        except Exception:
            continue


def run_one_session(
    session: str,
    use_residual_speed: bool = False,
    use_residual_roll: bool = False,
    use_residual_yaw: bool = False,
    use_residual_pitch: bool = False,
    weights_base: Path | None = None,
) -> Tuple[bool, str]:
    if weights_base is None:
        weights_base = WEIGHTS_BASE
    OUT_ROOT = weights_base / session
    (OUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)

    paths = session_paths(session)
    for k in ["imu", "spike", "dlc_final", "position"]:
        if not paths[k].exists():
            return False, f"Missing input {k}: {paths[k]}"

    data_dict = rebuild_inputs_50hz(session, paths)

    Y50 = load_spikes_50hz_counts(paths["spike"])  # (T50_spk, N)
    T_spk, N_NEURONS = Y50.shape

    T_cov = int(data_dict["T"])
    T = min(T_cov, T_spk)
    if abs(T_cov - T_spk) > MAX_MISMATCH_FRAMES_50HZ:
        return False, f"Length mismatch @50Hz (> {MAX_MISMATCH_FRAMES_50HZ}): cov={T_cov}, spk={T_spk}"

    for k in [
        "position",
        "head_v",
        "head_v_bin",
        "roll",
        "yaw",
        "pitch",
        "roll_bin",
        "yaw_bin",
        "pitch_bin",
    ]:
        data_dict[k] = data_dict[k][:T]
    Y_all = Y50[:T].astype(np.float64)
    data_dict, Y_all, speed_mask = filter_by_min_speed(data_dict, Y_all, MIN_SPEED_CM_S)
    if speed_mask is not None and not speed_mask.any():
        return False, f"No samples >= min speed {MIN_SPEED_CM_S:g} cm/s"

    rng = np.random.default_rng(0)
    data_dict, Y_all, subsample_msg = _subsample_outdoor_angles_to_indoor(session, data_dict, Y_all, rng)
    if subsample_msg:
        print(f"[{session}] {subsample_msg}")

    if use_residual_speed:
        data_dict = apply_residual_speed(data_dict)
    if use_residual_roll:
        data_dict = apply_residual_angle(data_dict, "roll")
    if use_residual_yaw:
        data_dict = apply_residual_angle(data_dict, "yaw")
    if use_residual_pitch:
        data_dict = apply_residual_angle(data_dict, "pitch")

    T = int(data_dict["T"])
    kf = KFold(n_splits=CV_FOLDS, shuffle=False)
    folds_idx = list(kf.split(np.arange(T)))

    get_X_and_feats = _build_design_cache(data_dict)

    X_full, feats_full = get_X_and_feats(VARS_ALL)
    save_full_fit_weights_for_all_neurons(
        out_root=OUT_ROOT,
        model_vars=VARS_ALL,
        X_all=X_full,
        feature_names=feats_full,
        Y_all=Y_all,
        folds_idx=folds_idx,
        n_jobs=N_JOBS,
    )

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_forward_select_one_neuron)(i, Y_all, folds_idx, OUT_ROOT, get_X_and_feats)
        for i in tqdm(range(N_NEURONS), desc=f"{session} | forward search (Poisson)")
    )

    logs_dir = OUT_ROOT / "logs"
    with open(logs_dir / "neuron_forward_paths.jsonl", "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rows, unclassified = [], []
    for rec in results:
        if rec["classified"]:
            rows.append({"neuron": rec["neuron"], "final_model": "_".join(rec["final_model"])})
        else:
            unclassified.append(rec["neuron"])

    pd.DataFrame(rows).to_csv(OUT_ROOT / "selected_models.csv", index=False)
    with open(OUT_ROOT / "unclassified_neurons.txt", "w", encoding="utf-8") as f:
        for n in unclassified:
            f.write(n + "\n")

    _plot_selected_models(rows, OUT_ROOT, session, Y_all, folds_idx)

    with open(OUT_ROOT / "_SUCCESS", "w", encoding="utf-8") as f:
        f.write(f"OK\t{datetime.now().isoformat(timespec='seconds')}\n")

    return True, f"OK (T50={T}, N={N_NEURONS})"


def _apply_index_subsample(
    data_dict: Dict[str, np.ndarray],
    Y_all: np.ndarray,
    keep_idx: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    base_len = int(data_dict.get("T", keep_idx.size))
    filtered = {}
    for k, v in data_dict.items():
        if isinstance(v, np.ndarray) and v.shape[0] == base_len:
            filtered[k] = v[keep_idx]
        else:
            filtered[k] = v
    filtered["T"] = int(keep_idx.size)
    return filtered, Y_all[keep_idx]


def _subsample_outdoor_angles_to_indoor(
    session: str,
    data_dict: Dict[str, np.ndarray],
    Y_all: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, str | None]:
    if not session.endswith("_outdoor"):
        return data_dict, Y_all, None

    indoor_session = f"{session[: -len('_outdoor')]}_indoor"
    indoor_paths = session_paths(indoor_session)
    if not all(indoor_paths[k].exists() for k in ["imu", "spike", "dlc_final", "position"]):
        return data_dict, Y_all, f"Skip subsample: missing indoor inputs for {indoor_session}"

    indoor_dict = rebuild_inputs_50hz(indoor_session, indoor_paths)
    Y50_indoor = load_spikes_50hz_counts(indoor_paths["spike"])
    T_spk, _ = Y50_indoor.shape
    T_cov = int(indoor_dict["T"])
    T_indoor = min(T_cov, T_spk)
    for k in [
        "position",
        "head_v",
        "head_v_bin",
        "roll",
        "yaw",
        "pitch",
        "roll_bin",
        "yaw_bin",
        "pitch_bin",
    ]:
        indoor_dict[k] = indoor_dict[k][:T_indoor]
    indoor_dict, _Y_indoor, _mask = filter_by_min_speed(indoor_dict, Y50_indoor[:T_indoor], MIN_SPEED_CM_S)

    indoor_bins = (
        indoor_dict["roll_bin"] * (ANGLE_N_BINS * ANGLE_N_BINS)
        + indoor_dict["yaw_bin"] * ANGLE_N_BINS
        + indoor_dict["pitch_bin"]
    )
    outdoor_bins = (
        data_dict["roll_bin"] * (ANGLE_N_BINS * ANGLE_N_BINS)
        + data_dict["yaw_bin"] * ANGLE_N_BINS
        + data_dict["pitch_bin"]
    )
    n_joint = ANGLE_N_BINS**3
    indoor_counts = np.bincount(indoor_bins, minlength=n_joint)
    outdoor_counts = np.bincount(outdoor_bins, minlength=n_joint)

    valid = (indoor_counts > 0) & (outdoor_counts > 0)
    ratios = outdoor_counts[valid] / indoor_counts[valid]
    if ratios.size == 0:
        return data_dict, Y_all, f"Skip subsample: no overlapping roll/yaw/pitch bins for {indoor_session}"

    alpha = min(1.0, float(ratios.min()))
    target_counts = np.floor(indoor_counts * alpha).astype(int)
    target_bins = np.nonzero(target_counts)[0]
    if target_bins.size == 0:
        return data_dict, Y_all, f"Skip subsample: target counts empty for {indoor_session}"

    keep_indices = []
    for b in target_bins:
        idx = np.flatnonzero(outdoor_bins == b)
        n_keep = target_counts[b]
        if n_keep <= 0:
            continue
        if idx.size <= n_keep:
            keep_indices.append(idx)
        else:
            keep_indices.append(rng.choice(idx, size=n_keep, replace=False))
    if not keep_indices:
        return data_dict, Y_all, f"Skip subsample: no indices selected for {indoor_session}"

    keep_idx = np.sort(np.concatenate(keep_indices))
    data_dict, Y_all = _apply_index_subsample(data_dict, Y_all, keep_idx)
    msg = f"Subsampled outdoor to match indoor roll/yaw/pitch distribution: kept {keep_idx.size}/{outdoor_bins.size}"
    return data_dict, Y_all, msg
