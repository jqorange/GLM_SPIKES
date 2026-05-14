import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from tqdm import tqdm

from .config import (
    ALPHA,
    CV_FOLDS,
    CV_SEGMENTS_PER_GROUP,
    CV_SPLIT_MODE,
    CV_TOTAL_SEGMENTS,
    CV_VAL_FOLDS,
    FORWARD_SEARCH_METRIC,
    INPUT_FILES,
    N_JOBS,
    PLOT_N_JOBS,
    PLOT_END_SEC,
    PLOT_START_SEC,
    PLOT_SMOOTH_MS,
    PLOT_ZSCORE,
    SKIP_SMALL_ARTIFACT_WRITES,
    VARS_ALL,
    WEIGHTS_BASE,
)
from .design_matrix import build_design_matrix, model_key_from_vars
from .io_utils import (
    prepare_session_for_modeling,
    session_paths,
)
from .metrics import (
    build_oof_constant_mu,
    compute_deviance_explained_poisson_vs_baseline,
    compute_llhi_bps_poisson_vs_baseline,
    wilcoxon_greater,
)
from .plotting_utils import load_oof_from_neuron_dir, plot_fitting_curve
from .training import (
    fit_predict_one_fold_poisson,
    save_full_fit_weights_for_all_neurons,
    save_neuron_artifacts_for_model,
)


def _build_design_cache(data_dict: Dict[str, np.ndarray]):
    cache: Dict[str, Tuple[sparse.csr_matrix, List[str], np.ndarray | None]] = {}

    def get_X_and_feats(model_vars: List[str]) -> Tuple[sparse.csr_matrix, List[str], np.ndarray | None]:
        mk = model_key_from_vars(model_vars)
        if mk in cache:
            return cache[mk]
        X, feats = build_design_matrix(model_vars, data_dict)
        pos_xy = data_dict.get("position_xy_by_idx") if "Position" in model_vars else None
        cache[mk] = (X, feats, pos_xy)
        return X, feats, pos_xy

    return get_X_and_feats


@dataclass
class StepRecord:
    step: int
    model: List[str]
    metric_name: str
    mean_metric: float
    fold_metric: List[float]
    p_value_vs_prev: float = None
    stat_vs_prev: float = None
    n_pairs: int = None
    accepted: bool = True


def _build_grouped_segment_folds(T: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if T < CV_TOTAL_SEGMENTS:
        raise ValueError(
            f"T={T} is smaller than CV_TOTAL_SEGMENTS={CV_TOTAL_SEGMENTS}; "
            "cannot build non-empty grouped segments."
        )

    idx = np.arange(T, dtype=np.int64)
    segments = [seg for seg in np.array_split(idx, CV_TOTAL_SEGMENTS) if seg.size > 0]
    if len(segments) != CV_TOTAL_SEGMENTS:
        raise ValueError(
            f"Expected {CV_TOTAL_SEGMENTS} non-empty segments, got {len(segments)}"
        )

    n_groups = CV_TOTAL_SEGMENTS // CV_SEGMENTS_PER_GROUP
    grouped_segments = [
        segments[g * CV_SEGMENTS_PER_GROUP:(g + 1) * CV_SEGMENTS_PER_GROUP]
        for g in range(n_groups)
    ]

    folds_idx_full: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(CV_SEGMENTS_PER_GROUP):
        va_idx = np.concatenate(
            [group[fold_idx] for group in grouped_segments],
            axis=0,
        )
        tr_idx = np.concatenate(
            [
                seg
                for group in grouped_segments
                for seg_idx, seg in enumerate(group)
                if seg_idx != fold_idx
            ],
            axis=0,
        )
        folds_idx_full.append((tr_idx, va_idx))
    return folds_idx_full


def build_cv_folds(T: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if CV_SPLIT_MODE == "grouped_segments":
        return _build_grouped_segment_folds(T)
    raise ValueError(f"Unsupported CV_SPLIT_MODE: {CV_SPLIT_MODE!r}")


def _compute_forward_search_metric(
    y_true: np.ndarray,
    mu_pred: np.ndarray,
    mu_base: np.ndarray,
) -> float:
    if FORWARD_SEARCH_METRIC == "deviance_explained":
        return compute_deviance_explained_poisson_vs_baseline(y_true, mu_pred, mu_base)
    if FORWARD_SEARCH_METRIC == "llhi":
        return compute_llhi_bps_poisson_vs_baseline(y_true, mu_pred, mu_base)
    raise ValueError(f"Unsupported FORWARD_SEARCH_METRIC: {FORWARD_SEARCH_METRIC!r}")


def _forward_search_metric_cv_for_neuron(
    model_vars: List[str],
    neuron_idx: int,
    Y_all: np.ndarray,
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
    get_X_and_feats,
) -> Tuple[float, List[float]]:
    X_all_m, feat, pos_xy = get_X_and_feats(model_vars)
    y = Y_all[:, neuron_idx].astype(np.float64)
    fold_metric: List[float] = []

    for (tr, va) in folds_idx:
        mu_val, _ = fit_predict_one_fold_poisson(
            X_all_m,
            y,
            tr,
            va,
            feat,
            position_xy_by_idx=pos_xy,
        )
        mu_base = np.full_like(y[va], fill_value=max(float(np.mean(y[tr])), 1e-12), dtype=np.float64)
        metric_val = _compute_forward_search_metric(y[va], mu_val, mu_base)
        fold_metric.append(float(metric_val))

    if not fold_metric:
        mean_metric = float("nan")
    else:
        finite_metric = np.asarray(fold_metric, dtype=np.float64)
        finite_metric = finite_metric[np.isfinite(finite_metric)]
        mean_metric = float(np.mean(finite_metric)) if finite_metric.size else float("nan")
    return mean_metric, fold_metric


def _save_accepted_step(
    neuron_idx: int,
    model_vars: List[str],
    OUT_ROOT: Path,
    folds_idx_full: List[Tuple[np.ndarray, np.ndarray]],
    Y_all: np.ndarray,
    get_X_and_feats,
):
    model_dir = OUT_ROOT / model_key_from_vars(model_vars)
    X_all_m, feat_names, pos_xy = get_X_and_feats(model_vars)
    y = Y_all[:, neuron_idx].astype(np.float64)
    neuron_dir = model_dir / f"neuron_{neuron_idx+1}"
    return save_neuron_artifacts_for_model(
        model_vars=model_vars,
        model_dir=model_dir,
        neuron_dir=neuron_dir,
        neuron_index=neuron_idx,
        folds=folds_idx_full,
        X_all=X_all_m,
        y_all=y,
        feature_names=feat_names,
        position_xy_by_idx=pos_xy,
    )


def _forward_select_one_neuron(
    neuron_idx: int,
    Y_all: np.ndarray,
    folds_idx_train: List[Tuple[np.ndarray, np.ndarray]],
    folds_idx_full: List[Tuple[np.ndarray, np.ndarray]],
    OUT_ROOT: Path,
    get_X_and_feats,
):
    path_records: List[StepRecord] = []
    remaining = VARS_ALL.copy()
    single_candidates = []
    for v in remaining:
        mean_metric, fold_metric = _forward_search_metric_cv_for_neuron(
            [v],
            neuron_idx,
            Y_all,
            folds_idx_train,
            get_X_and_feats,
        )
        single_candidates.append((v, mean_metric, fold_metric))

    single_candidates.sort(key=lambda x: (x[1] if np.isfinite(x[1]) else -np.inf), reverse=True)
    best_v, best_mean_metric, best_fold = single_candidates[0]

    stat, p, n = wilcoxon_greater(best_fold, b=None)
    accepted = (p < ALPHA)

    path_records.append(
        StepRecord(
            step=1,
            model=[best_v],
            metric_name=FORWARD_SEARCH_METRIC,
            mean_metric=best_mean_metric,
            fold_metric=list(map(float, best_fold)),
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

    _save_accepted_step(neuron_idx, [best_v], OUT_ROOT, folds_idx_full, Y_all, get_X_and_feats)

    selected = [best_v]
    remaining.remove(best_v)
    fold_metric_prev = list(best_fold)

    step = 2
    while remaining:
        cand_list = []
        for cand in remaining:
            trial_vars = selected + [cand]
            mean_metric, fold_metric = _forward_search_metric_cv_for_neuron(
                trial_vars,
                neuron_idx,
                Y_all,
                folds_idx_train,
                get_X_and_feats,
            )
            cand_list.append((cand, trial_vars, mean_metric, fold_metric))

        cand_list.sort(key=lambda x: (x[2] if np.isfinite(x[2]) else -np.inf), reverse=True)
        best_cand, best_trial_vars, best_trial_mean_metric, best_trial_fold = cand_list[0]

        stat, p, n = wilcoxon_greater(best_trial_fold, fold_metric_prev)
        accepted = (p < ALPHA)

        path_records.append(
            StepRecord(
                step=step,
                model=best_trial_vars,
                metric_name=FORWARD_SEARCH_METRIC,
                mean_metric=best_trial_mean_metric,
                fold_metric=list(map(float, best_trial_fold)),
                p_value_vs_prev=p,
                stat_vs_prev=stat,
                n_pairs=n,
                accepted=accepted,
            )
        )

        if not accepted:
            break

        _save_accepted_step(
            neuron_idx,
            best_trial_vars,
            OUT_ROOT,
            folds_idx_full,
            Y_all,
            get_X_and_feats,
        )
        selected = best_trial_vars
        remaining.remove(best_cand)
        fold_metric_prev = list(best_trial_fold)
        step += 1

        if len(selected) == len(VARS_ALL):
            break

    const_p = None
    const_stat = None
    const_n = None
    if selected:
        _metric, fold_metric = _forward_search_metric_cv_for_neuron(
            selected,
            neuron_idx,
            Y_all,
            folds_idx_train,
            get_X_and_feats,
        )
        const_stat, const_p, const_n = wilcoxon_greater(fold_metric, b=None)
        if const_p >= ALPHA:
            return {
                "neuron": f"neuron_{neuron_idx+1}",
                "final_model": None,
                "classified": False,
                "forward_search_metric": FORWARD_SEARCH_METRIC,
                "path": [vars(s) for s in path_records],
                "const_rate_p_value": const_p,
                "const_rate_stat": const_stat,
                "const_rate_n_pairs": const_n,
            }

    return {
        "neuron": f"neuron_{neuron_idx+1}",
        "final_model": selected,
        "classified": True,
        "forward_search_metric": FORWARD_SEARCH_METRIC,
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
    if SKIP_SMALL_ARTIFACT_WRITES:
        return

    fig_dir = OUT_ROOT / "figures"
    if not rows:
        return

    n_plot_jobs = max(1, min(int(PLOT_N_JOBS), len(rows)))

    def _plot_one(rec):
        neuron_name = rec["neuron"]
        model_key = rec["final_model"]
        model_dir = OUT_ROOT / model_key
        neuron_dir = model_dir / neuron_name

        try:
            neuron_idx = int(neuron_name.split("_")[-1]) - 1
            y_full = Y_all[:, neuron_idx].astype(np.float64)
            y_oof, mu_oof = load_oof_from_neuron_dir(neuron_dir)
            mu_base_oof = build_oof_constant_mu(y_full, folds_idx)
            dev_exp = compute_deviance_explained_poisson_vs_baseline(y_oof, mu_oof, mu_base_oof)
            title = f"{session} | {neuron_name} | PoissonGLM | vars={model_key.replace('_','+')} | DevExp={dev_exp:.4f}"
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
            return

    with ThreadPoolExecutor(max_workers=n_plot_jobs) as ex:
        list(ex.map(_plot_one, rows))


def run_one_session(
    session: str,
    weights_base: Path | None = None,
) -> Tuple[bool, str]:
    if weights_base is None:
        weights_base = WEIGHTS_BASE
    OUT_ROOT = weights_base / session
    (OUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)

    paths = session_paths(session)
    required_inputs = ["spike"] + list(INPUT_FILES.keys())
    for k in required_inputs:
        if not paths[k].exists():
            return False, f"Missing input {k}: {paths[k]}"
    try:
        data_dict, Y_all, prep_meta = prepare_session_for_modeling(session, paths)
    except Exception as exc:  # pylint: disable=broad-except
        return False, str(exc)

    matched_len = prep_meta.get("matched_len")
    if matched_len is not None:
        print(f"[pair_match] {session}: truncated to matched indoor/outdoor length {matched_len}")

    T = int(prep_meta["t_final"])
    N_NEURONS = int(prep_meta["n_neurons"])
    folds_idx_full = build_cv_folds(T)
    fold_val_indices = [va for _tr, va in folds_idx_full]
    if CV_VAL_FOLDS < 1 or CV_VAL_FOLDS >= CV_FOLDS:
        raise ValueError(f"CV_VAL_FOLDS must be in [1, {CV_FOLDS - 1}]")
    folds_idx_train = []
    for k in range(CV_FOLDS):
        val_folds = [(k + i) % CV_FOLDS for i in range(CV_VAL_FOLDS)]
        tr_parts, va_parts = [], []
        for fold_idx, fold_va in enumerate(fold_val_indices):
            if fold_idx in val_folds:
                va_parts.append(fold_va)
            else:
                tr_parts.append(fold_va)
        tr_idx = np.concatenate(tr_parts)
        va_idx = np.concatenate(va_parts)
        folds_idx_train.append((tr_idx, va_idx))

    get_X_and_feats = _build_design_cache(data_dict)

    # X_full, feats_full, pos_xy_full = get_X_and_feats(VARS_ALL)
    # save_full_fit_weights_for_all_neurons(
    #     out_root=OUT_ROOT,
    #     model_vars=VARS_ALL,
    #     X_all=X_full,
    #     feature_names=feats_full,
    #     position_xy_by_idx=pos_xy_full,
    #     Y_all=Y_all,
    #     folds_idx=folds_idx_full,
    #     n_jobs=N_JOBS,
    # )

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_forward_select_one_neuron)(
            i,
            Y_all,
            folds_idx_train,
            folds_idx_full,
            OUT_ROOT,
            get_X_and_feats,
        )
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

    _plot_selected_models(rows, OUT_ROOT, session, Y_all, folds_idx_full)

    with open(OUT_ROOT / "_SUCCESS", "w", encoding="utf-8") as f:
        f.write(f"OK\t{datetime.now().isoformat(timespec='seconds')}\n")

    return True, f"OK (T50={T}, N={N_NEURONS})"
