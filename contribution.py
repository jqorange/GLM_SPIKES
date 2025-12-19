#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute Poisson-GLM Deviance Explained (DevExpl) and drop-one (reduced-model) contributions
for PYRAMIDAL cells only, separately for indoor vs outdoor sessions.

What this script does (high-level):
1) For each session:
   - load 50 Hz covariates (Position/Speed/roll/yaw/pitch) and 50 Hz spike counts
   - identify pyramidal cells via cell_metrics.putativeCellType from a matching *cellinfo*.mat
   - for FULL model and each DROP-ONE model (remove one feature):
       * train Poisson GLM weights per fold (10-fold CV) for pyramidal neurons
         OR reuse saved weights if they already exist
       * compute out-of-fold predictions using saved weights
       * compute D_null, D_full, D_reduced and DevExpl per neuron
       * compute per-neuron "fraction of full-model deviance" contribution of each feature
2) Aggregate across sessions for indoor and outdoor separately:
   - hierarchical bootstrap (sessions -> neurons) to get mean and 90% CI (5th–95th)
3) Plot two summary figures (indoor & outdoor):
   - left panel: full-model DevExpl (bar + CI)
   - right panel: per-feature contribution fraction (bars + CI)

Directory layout (inside each session folder under WEIGHTS_BASE):
  <session>/
    DROPONE_FITS/
      FULL/                 # model_vars = Position_Speed_roll_yaw_pitch
        feature_names.json
        neuron_###/
          fold1/weights.csv
          ...
          fold10/weights.csv
          weights_mean.csv
      DROP_Speed/
      DROP_roll/
      DROP_yaw/
      DROP_pitch/
      DROP_Position/
    DROPONE_STATS/
      full_devexpl_pyr.csv
      dropone_contrib_pyr.csv

All paths and binning parameters are shared with ``glm_poisson_forward.config`` to stay
consistent with the forward-search pipeline.
"""

from __future__ import annotations
import faulthandler
faulthandler.enable()
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.io
from scipy import sparse
from scipy.special import gammaln

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold
from tqdm import tqdm

from glm_poisson_forward.config import (
    CV_FOLDS,
    DLC_ROOT,
    IMU_ROOT,
    MAX_ITER,
    MAX_MISMATCH_FRAMES_50HZ,
    N_JOBS,
    POISSON_ALPHA,
    POSITION_ROOT,
    SEED,
    SPIKE_ROOT,
    VARS_ALL,
    WEIGHTS_BASE,
)
from glm_poisson_forward.design_matrix import build_design_matrix
from glm_poisson_forward.io_utils import (
    list_sessions_dlc_final,
    list_sessions_imu,
    list_sessions_position,
    list_sessions_spike,
    load_spikes_50hz_counts,
    rebuild_inputs_50hz,
    session_paths,
)

# Where to search for cell_metrics/cellinfo mats (same logic as your pyramidal-only stats script)
DAY_SEARCH_DIRS = [
    r"W:\data\FieldRat\2024\F4\day1",
    r"W:\data\FieldRat\2024\F4\day4",
    r"W:\data\FieldRat\2024\F5\Merged\day2\121_day2",
    r"W:\data\FieldRat\2024\F5\Merged\day3\121_day3",
    r"W:\data\FieldRat\2024\F5\Merged\day5\121_day5",
    r"W:\data\FieldRat\2024\F5\Merged\day6\3E6_day6",
    r"W:\data\FieldRat\2024\F5\Merged\day10\121_day10",
    r"W:\data\FieldRat\2024\F6\Merged\day3\3E6_day3",
    r"W:\data\FieldRat\2024\F6\Merged\day5\3E6_day5",
    r"W:\data\FieldRat\2024\F6\Merged\day8\3E6_day8",
    r"W:\data\FieldRat\2024\F6\Merged\day9\3E6_day9",
    r"W:\data\FieldRat\2024\F6\Merged\day10\3E6_day10",
    r"W:\data\FieldRat\2024\F6\Merged\day2\3E6_day2",
    r"W:\data\FieldRat\2024\F6\Merged\day4\3E6_day4",
    r"W:\data\FieldRat\2024\F6\Merged\day6\121_day6",
]
DAY_SEARCH_DIRS = [Path(p) for p in DAY_SEARCH_DIRS]

# Output subfolders inside each session directory
DROPONE_FITS_DIRNAME  = "DROPONE_FITS"
DROPONE_STATS_DIRNAME = "DROPONE_STATS"

# Bootstrap settings
N_BOOT = 2000
CI_LO, CI_HI = 5, 95

# Numerical safety
MU_EPS = 1e-12


# ===============================
# Pyramidal filtering (cell_metrics)
# ===============================

def parse_day_id_from_session(session_name: str) -> Optional[str]:
    mF = re.search(r'F(\d+)', session_name, flags=re.IGNORECASE)
    mD = re.search(r'D(\d+)', session_name, flags=re.IGNORECASE)
    if mF and mD:
        return f"F{int(mF.group(1))}D{int(mD.group(1))}"
    return None

def parse_day_id_from_path(day_dir: Path) -> Optional[str]:
    s = str(day_dir)
    mF = re.search(r'F(\d+)', s, flags=re.IGNORECASE)
    mD = re.search(r'day\s*([0-9]+)', s, flags=re.IGNORECASE)
    if mF and mD:
        return f"F{int(mF.group(1))}D{int(mD.group(1))}"
    return None

def find_cellinfo_mat(day_dir: Path) -> Optional[Path]:
    for pat in ["*cellinfo*.mat", "*cell_metrics*.mat"]:
        hits = list(day_dir.glob(pat))
        if hits:
            return hits[0]
    return None

def build_dayid_to_cellinfo() -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for dd in DAY_SEARCH_DIRS:
        day_id = parse_day_id_from_path(dd)
        if not day_id:
            continue
        ci = find_cellinfo_mat(dd)
        if ci:
            mapping[day_id] = ci
    return mapping

def load_cell_types(cellinfo_mat: Path) -> List[str]:
    md = scipy.io.loadmat(str(cellinfo_mat), squeeze_me=True, struct_as_record=False)
    if "cell_metrics" not in md:
        raise KeyError(f"{cellinfo_mat} missing cell_metrics")
    cm = md["cell_metrics"]
    if hasattr(cm, "putativeCellType"):
        raw = cm.putativeCellType
    elif isinstance(cm, dict) and "putativeCellType" in cm:
        raw = cm["putativeCellType"]
    else:
        raise KeyError(f"{cellinfo_mat} missing putativeCellType")
    if isinstance(raw, np.ndarray):
        return [str(x).strip() for x in raw.tolist()]
    return [str(raw).strip()]

def pyramidal_indices_for_session(session: str, dayid2cellinfo: Dict[str, Path], n_neurons: int) -> Optional[np.ndarray]:
    day_id = parse_day_id_from_session(session)
    if not day_id or day_id not in dayid2cellinfo:
        return None
    try:
        types = load_cell_types(dayid2cellinfo[day_id])
    except Exception:
        return None
    mask = [(t.lower() == "pyramidal cell") for t in types]
    if len(mask) < n_neurons:
        # best effort: truncate n_neurons to available typing
        n_use = len(mask)
    else:
        n_use = n_neurons
    idx = np.array([i for i in range(n_use) if mask[i]], dtype=np.int32)
    return idx


def model_key_from_vars(model_vars: List[str]) -> str:
    if model_vars == VARS_ALL:
        return "FULL"
    # drop-one models
    missing = [v for v in VARS_ALL if v not in model_vars]
    if len(missing) == 1:
        return f"DROP_{missing[0]}"
    return "MODEL_" + "_".join(model_vars)

def ensure_feature_names_file(model_dir: Path, feature_names: List[str]):
    model_dir.mkdir(parents=True, exist_ok=True)
    p = model_dir / "feature_names.json"
    if not p.exists():
        with open(p, "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2)

def load_feature_names_file(model_dir: Path) -> Optional[List[str]]:
    p = model_dir / "feature_names.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ===============================
# Poisson deviance / DevExpl
# ===============================

def poisson_loglik(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Full Poisson log-likelihood including -log(y!), needed for deviance.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    mu = np.asarray(mu, dtype=np.float64).ravel()
    mu = np.clip(mu, MU_EPS, None)
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1.0)))

def poisson_loglik_saturated(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).ravel()
    # convention: 0*log(0)=0
    term = np.zeros_like(y, dtype=float)
    mask = y > 0
    term[mask] = y[mask] * np.log(y[mask])
    return float(np.sum(term - y - gammaln(y + 1.0)))

def deviance_from_ll(ll_sat: float, ll_model: float) -> float:
    return float(2.0 * (ll_sat - ll_model))

def devexpl_from_deviances(D_model: float, D_null: float) -> float:
    if not np.isfinite(D_model) or not np.isfinite(D_null) or D_null <= 0:
        return float("nan")
    return float(1.0 - (D_model / D_null))


# ===============================
# Training / caching weights
# ===============================

def fit_one_fold_weights(
    X_all: sparse.csr_matrix,
    y_all: np.ndarray,
    tr_idx: np.ndarray,
) -> np.ndarray:
    """
    Return w = [coef..., intercept] fitted on train indices only.
    """
    Xtr = X_all[tr_idx]
    ytr = y_all[tr_idx].astype(np.float64)

    mean_tr = float(np.mean(ytr))
    if mean_tr <= 0:
        w = np.zeros(Xtr.shape[1] + 1, dtype=np.float32)
        w[-1] = np.log(MU_EPS)
        return w

    mdl = PoissonRegressor(alpha=POISSON_ALPHA, max_iter=MAX_ITER, fit_intercept=True)
    mdl.fit(Xtr, ytr)
    w = np.concatenate([mdl.coef_.ravel().astype(np.float32), np.array([mdl.intercept_], dtype=np.float32)])
    return w

def weights_exist_for_neuron(model_dir: Path, neuron_idx1: int) -> bool:
    nd = model_dir / f"neuron_{neuron_idx1}"
    if not nd.exists():
        return False
    if not (nd / "weights_mean.csv").exists():
        return False
    for k in range(1, CV_FOLDS + 1):
        if not (nd / f"fold{k}" / "weights.csv").exists():
            return False
    return True

def save_weights_for_model(
    model_dir: Path,
    feature_names: List[str],
    X_all: sparse.csr_matrix,
    Y_all: np.ndarray,
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
    neuron_indices: np.ndarray,
    n_jobs: int,
):
    """
    Ensure weights exist for all neuron_indices for this model.
    Only trains missing neurons (caches per neuron).
    """
    ensure_feature_names_file(model_dir, feature_names)

    def _one_neuron(neuron_idx: int) -> Tuple[bool, str]:
        idx1 = neuron_idx + 1
        if weights_exist_for_neuron(model_dir, idx1):
            return True, "CACHED"

        try:
            y = Y_all[:, neuron_idx].astype(np.float64)
            nd = model_dir / f"neuron_{idx1}"
            nd.mkdir(parents=True, exist_ok=True)

            ws = []
            for k, (tr, _va) in enumerate(folds_idx, start=1):
                w = fit_one_fold_weights(X_all, y, tr)
                ws.append(w)

                fd = nd / f"fold{k}"
                fd.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    w.reshape(1, -1),
                    index=[f"neuron_{idx1}"],
                    columns=feature_names,
                ).to_csv(fd / "weights.csv")

            w_mean = np.mean(np.stack(ws, axis=0), axis=0).astype(np.float32)
            pd.DataFrame(
                w_mean.reshape(1, -1),
                index=[f"neuron_{idx1}"],
                columns=feature_names,
            ).to_csv(nd / "weights_mean.csv")

            return True, "OK"
        except Exception as e:
            return False, str(e)

    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_neuron)(i) for i in tqdm(neuron_indices.tolist(), desc=f"fit weights | {model_dir.parent.name}/{model_dir.name}")
    )

    bad = [(i, msg) for i, (ok, msg) in zip(neuron_indices.tolist(), res) if not ok]
    if bad:
        with open(model_dir / "failures.txt", "w", encoding="utf-8") as f:
            for i, msg in bad:
                f.write(f"neuron_{i+1}\t{msg}\n")


# ===============================
# Predict from saved weights (OOF)
# ===============================

def load_fold_weights(csv_path: Path, feature_names: Iterable[str]) -> np.ndarray:
    """
    Read weights saved by pandas.DataFrame(...).to_csv("weights.csv") (wide 1-row format),
    OR long key/value format. Return ordered weight vector aligned to feature_names.

    Expected feature_names includes intercept as the last name.
    """
    csv_path = Path(csv_path)
    if (not csv_path.exists()) or csv_path.stat().st_size < 8:
        raise ValueError(f"Missing or too-small weights file: {csv_path}")

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if len(r) > 0]

    if len(rows) < 2:
        raise ValueError(f"Corrupted weights CSV (need >=2 rows): {csv_path}")

    header = [c.strip() for c in rows[0]]
    first_data = [c.strip() for c in rows[1]]

    def to_float(x: str) -> float:
        if x is None or x == "":
            return 0.0
        return float(x)

    out: Dict[str, float] = {}

    # Case 1: long format: two columns (name/value)
    header_lc = [h.lower() for h in header]
    is_key_value = (
        len(header) == 2 and len(first_data) == 2 and
        (("name" in header_lc[0] and "value" in header_lc[1]) or
         ("feature" in header_lc[0] and "value" in header_lc[1]) or
         ("key" in header_lc[0] and "value" in header_lc[1]))
    )

    if is_key_value:
        for r in rows[1:]:
            if len(r) < 2:
                continue
            k = r[0].strip()
            v = r[1].strip()
            if k == "":
                continue
            out[k] = to_float(v)
    else:
        # Case 2: wide format: header are feature names, row are values
        # pandas often writes an index column first; commonly header[0] == "" or "Unnamed: 0"
        # We keep all columns but will only pick those in feature_names.
        if len(first_data) != len(header):
            if len(first_data) < len(header):
                first_data = first_data + [""] * (len(header) - len(first_data))
            else:
                first_data = first_data[:len(header)]

        for k, v in zip(header, first_data):
            if k == "":
                continue
            # ignore unnamed index columns
            if k.lower().startswith("unnamed"):
                continue
            out[k] = to_float(v)

    names: List[str] = list(feature_names)
    # fill missing with 0
    for n in names:
        if n not in out:
            out[n] = 0.0

    w_vec = np.asarray([out[n] for n in names], dtype=np.float64)
    return w_vec


# Backward-compatible wrapper if your code calls load_fold_weights(path, something)
def load_fold_weights_compat(csv_path: Path, *args: Any, **kwargs: Any):
    """
    Drop-in replacement signature: accepts extra positional args.
    If second positional arg exists, treat it as feature_names.
    """
    feature_names = args[0] if len(args) >= 1 else kwargs.pop("feature_names", None)
    debug = kwargs.pop("debug", False)
    return load_fold_weights(csv_path, feature_names=feature_names, debug=debug)

def predict_oof_from_saved_weights(
    model_dir: Path,
    X_all: sparse.csr_matrix,
    feature_names_now: List[str],
    folds_idx: List[Tuple[np.ndarray, np.ndarray]],
    neuron_idx: int,
) -> np.ndarray:
    idx1 = neuron_idx + 1
    neuron_dir = model_dir / f"neuron_{idx1}"

    saved_feats = load_feature_names_file(model_dir)
    if saved_feats is None:
        raise FileNotFoundError(f"Missing feature_names.json in {model_dir}")
    if saved_feats != feature_names_now:
        raise ValueError(
            f"Feature name mismatch for {model_dir}.\n"
            f"Saved: {len(saved_feats)} cols, Now: {len(feature_names_now)} cols."
        )

    T = X_all.shape[0]
    mu_oof = np.full(T, np.nan, dtype=np.float64)

    for k, (tr, va) in enumerate(folds_idx, start=1):
        csv_path = neuron_dir / f"fold{k}" / "weights.csv"
        w_vec = load_fold_weights(csv_path, feature_names=saved_feats)  # 1D vector

        coef = w_vec[:-1]
        intercept = w_vec[-1]

        Xva = X_all[va]
        eta = (Xva @ coef).astype(np.float64) + float(intercept)
        mu_va = np.exp(eta)
        mu_va = np.clip(mu_va, MU_EPS, None)
        mu_oof[va] = mu_va

    if np.any(~np.isfinite(mu_oof)):
        mu_oof = np.where(np.isfinite(mu_oof), mu_oof, MU_EPS)
    return mu_oof



# ===============================
# Per-session computation
# ===============================

@dataclass
class SessionResult:
    session: str
    group: str  # indoor/outdoor
    full_devexpl_by_neuron: Dict[int, float]           # neuron_idx -> DevExpl
    contrib_frac_by_feature_by_neuron: Dict[str, Dict[int, float]]  # feature -> neuron_idx -> frac


def compute_session_dropone(
    session: str,
    dayid2cellinfo: Dict[str, Path],
    *,
    n_jobs: int = N_JOBS,
) -> Optional[SessionResult]:

    sess_dir = WEIGHTS_BASE / session
    if not sess_dir.exists():
        # we still can run even if forward-selection wasn't run; create folder
        sess_dir.mkdir(parents=True, exist_ok=True)

    paths = session_paths(session)
    for k in ["imu", "spike", "dlc_final", "position"]:
        if not paths[k].exists():
            print(f"[SKIP] {session}: missing input {k}: {paths[k]}")
            return None

    # determine indoor/outdoor
    s_lower = session.lower()
    if "indoor" in s_lower:
        group = "indoor"
    elif "outdoor" in s_lower:
        group = "outdoor"
    else:
        print(f"[SKIP] {session}: cannot infer indoor/outdoor from name")
        return None

    # load data
    data_dict = rebuild_inputs_50hz(session, paths)
    Y50 = load_spikes_50hz_counts(paths["spike"])  # (T50_spk, N)
    T_spk, N_NEURONS = Y50.shape
    T_cov = int(data_dict["T"])
    if abs(T_cov - T_spk) > MAX_MISMATCH_FRAMES_50HZ:
        print(f"[SKIP] {session}: length mismatch @50Hz cov={T_cov} spk={T_spk}")
        return None

    T = min(T_cov, T_spk)
    for k in ["position", "head_v_bin", "roll_bin", "yaw_bin", "pitch_bin"]:
        data_dict[k] = data_dict[k][:T]
    Y_all = Y50[:T].astype(np.float64)

    # pyramidal filtering
    pyr_idx = pyramidal_indices_for_session(session, dayid2cellinfo, N_NEURONS)
    if pyr_idx is None or pyr_idx.size == 0:
        print(f"[SKIP] {session}: pyramidal cell info not found or empty")
        return None

    # CV folds (fixed)
    kf = KFold(n_splits=CV_FOLDS, shuffle=False)
    folds_idx = list(kf.split(np.arange(T)))

    # cache design matrices per model
    X_cache: Dict[str, Tuple[sparse.csr_matrix, List[str], List[str]]] = {}

    def get_X(model_vars: List[str]) -> Tuple[sparse.csr_matrix, List[str], str]:
        mk = model_key_from_vars(model_vars)
        if mk in X_cache:
            X, feats, _ = X_cache[mk]
            return X, feats, mk
        X, feats = build_design_matrix(model_vars, data_dict)
        X_cache[mk] = (X, feats, mk)
        return X, feats, mk

    # model set: full + drop-one
    model_vars_list: List[List[str]] = []
    model_vars_list.append(VARS_ALL)
    for v in VARS_ALL:
        model_vars_list.append([x for x in VARS_ALL if x != v])

    # fit/cache weights for each model
    fits_root = sess_dir / DROPONE_FITS_DIRNAME
    fits_root.mkdir(parents=True, exist_ok=True)

    for mv in model_vars_list:
        X, feats, mk = get_X(mv)
        model_dir = fits_root / mk
        save_weights_for_model(
            model_dir=model_dir,
            feature_names=feats,
            X_all=X,
            Y_all=Y_all,
            folds_idx=folds_idx,
            neuron_indices=pyr_idx,
            n_jobs=n_jobs,
        )

    # compute DevExpl / contributions (cacheable)
    stats_root = sess_dir / DROPONE_STATS_DIRNAME
    stats_root.mkdir(parents=True, exist_ok=True)

    full_csv = stats_root / "full_devexpl_pyr.csv"
    contrib_csv = stats_root / "dropone_contrib_pyr.csv"

    if full_csv.exists() and contrib_csv.exists():
        # load cached results
        df_full = pd.read_csv(full_csv)
        df_con = pd.read_csv(contrib_csv)
        full_map = {int(r["neuron_idx"]): float(r["devexpl_full"]) for _, r in df_full.iterrows()}
        contrib: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}
        for _, r in df_con.iterrows():
            contrib[str(r["feature"])][int(r["neuron_idx"])] = float(r["frac_full_dev"])
        return SessionResult(session=session, group=group, full_devexpl_by_neuron=full_map, contrib_frac_by_feature_by_neuron=contrib)

    # compute from weights
    # full model deviances
    X_full, feats_full, mk_full = get_X(VARS_ALL)
    model_dir_full = fits_root / mk_full

    full_devexpl_by_neuron: Dict[int, float] = {}
    D_full_by_neuron: Dict[int, float] = {}
    D_null_by_neuron: Dict[int, float] = {}

    for ni in pyr_idx.tolist():
        y = Y_all[:, ni].astype(np.float64)
        mu_oof = predict_oof_from_saved_weights(model_dir_full, X_full, feats_full, folds_idx, ni)

        ll_sat = poisson_loglik_saturated(y)
        ll_full = poisson_loglik(y, mu_oof)
        mu0 = np.full_like(y, fill_value=max(float(np.mean(y)), MU_EPS), dtype=np.float64)
        ll_null = poisson_loglik(y, mu0)

        D_full = deviance_from_ll(ll_sat, ll_full)
        D_null = deviance_from_ll(ll_sat, ll_null)

        D_full_by_neuron[ni] = float(D_full)
        D_null_by_neuron[ni] = float(D_null)
        full_devexpl_by_neuron[ni] = devexpl_from_deviances(D_full, D_null)

    # drop-one contributions: fraction of full-model deviance attributable to each feature
    contrib: Dict[str, Dict[int, float]] = {v: {} for v in VARS_ALL}

    for v in VARS_ALL:
        mv = [x for x in VARS_ALL if x != v]
        X_red, feats_red, mk_red = get_X(mv)
        model_dir_red = fits_root / mk_red

        for ni in pyr_idx.tolist():
            y = Y_all[:, ni].astype(np.float64)
            mu_oof_red = predict_oof_from_saved_weights(model_dir_red, X_red, feats_red, folds_idx, ni)

            ll_sat = poisson_loglik_saturated(y)
            ll_red = poisson_loglik(y, mu_oof_red)
            D_red = deviance_from_ll(ll_sat, ll_red)

            D_full = D_full_by_neuron[ni]
            D_null = D_null_by_neuron[ni]
            denom = (D_null - D_full)

            if not np.isfinite(D_red) or not np.isfinite(denom) or denom <= 0:
                frac = float("nan")
            else:
                frac = float((D_red - D_full) / denom)
            contrib[v][ni] = frac

    # save caches
    df_full = pd.DataFrame(
        [{"session": session, "group": group, "neuron_idx": ni, "devexpl_full": full_devexpl_by_neuron[ni]}
         for ni in sorted(full_devexpl_by_neuron.keys())]
    )
    df_full.to_csv(full_csv, index=False)

    rows = []
    for v in VARS_ALL:
        for ni, frac in contrib[v].items():
            rows.append({"session": session, "group": group, "feature": v, "neuron_idx": ni, "frac_full_dev": frac})
    pd.DataFrame(rows).to_csv(contrib_csv, index=False)

    return SessionResult(session=session, group=group, full_devexpl_by_neuron=full_devexpl_by_neuron, contrib_frac_by_feature_by_neuron=contrib)


# ===============================
# Hierarchical bootstrap + plotting
# ===============================

def hierarchical_bootstrap_mean(
    session_to_values: Dict[str, np.ndarray],
    n_boot: int = N_BOOT,
    ci_lo: float = CI_LO,
    ci_hi: float = CI_HI,
    seed: int = SEED,
) -> Tuple[float, float, float]:
    """
    session_to_values: session -> 1D array of neuron-level values
    Bootstrap:
      sample sessions with replacement
      within each sampled session sample neurons with replacement
      compute grand mean over pooled sampled neurons
    Returns: (mean, lo, hi)
    """
    rng = np.random.default_rng(seed)
    sessions = list(session_to_values.keys())
    if len(sessions) == 0:
        return float("nan"), float("nan"), float("nan")

    # point estimate: mean over all neurons pooled
    pooled = np.concatenate([session_to_values[s] for s in sessions], axis=0)
    pooled = pooled[np.isfinite(pooled)]
    point = float(np.mean(pooled)) if pooled.size else float("nan")

    boots = []
    for _ in range(int(n_boot)):
        ss = rng.choice(sessions, size=len(sessions), replace=True)
        vals = []
        for s in ss:
            arr = session_to_values[s]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            draw = rng.choice(arr, size=arr.size, replace=True)
            vals.append(draw)
        if not vals:
            boots.append(np.nan)
        else:
            vv = np.concatenate(vals, axis=0)
            boots.append(float(np.mean(vv)))
    boots = np.asarray(boots, dtype=np.float64)
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(boots, ci_lo))
    hi = float(np.percentile(boots, ci_hi))
    return point, lo, hi

def plot_summary_figure(
    out_png: Path,
    title: str,
    full_stat: Tuple[float, float, float],
    feature_stats: Dict[str, Tuple[float, float, float]],
):
    """
    Two-panel figure:
      left: full DevExpl
      right: per-feature contribution fraction
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    features = VARS_ALL[:]  # keep order
    means = [feature_stats[f][0] for f in features]
    los   = [feature_stats[f][1] for f in features]
    his   = [feature_stats[f][2] for f in features]

    # error bars (asymmetric)
    yerr_low = np.array(means) - np.array(los)
    yerr_high = np.array(his) - np.array(means)
    yerr = np.vstack([yerr_low, yerr_high])

    fig = plt.figure(figsize=(10, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.35)

    # left: full DevExpl
    ax0 = fig.add_subplot(gs[0, 0])
    m, lo, hi = full_stat
    ax0.bar([0], [m], width=0.6, edgecolor="black", linewidth=0.8)
    ax0.errorbar([0], [m], yerr=[[m-lo], [hi-m]], fmt="none", capsize=4, linewidth=1.2)
    ax0.set_xticks([0])
    ax0.set_xticklabels(["Full\nmodel"])
    ax0.set_ylabel("Deviance explained")
    ax0.set_title("Full model")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # right: contributions
    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(features))
    ax1.bar(x, means, width=0.65, edgecolor="black", linewidth=0.8)
    ax1.errorbar(x, means, yerr=yerr, fmt="none", capsize=4, linewidth=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=0)
    ax1.set_ylabel("Fraction of full-model dev.")
    ax1.set_title("Drop-one contribution (pyramidal only)")
    ax1.axhline(0.0, linewidth=0.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


# ===============================
# Main
# ===============================

def main():
    WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)

    # discover sessions from inputs (most robust)
    set_imu = list_sessions_imu(IMU_ROOT)
    set_spk = list_sessions_spike(SPIKE_ROOT)
    set_dlc = list_sessions_dlc_final(DLC_ROOT)
    set_pos = list_sessions_position(POSITION_ROOT)
    sessions = sorted(list(set_imu & set_spk & set_dlc & set_pos))

    if not sessions:
        print("[FATAL] No sessions found with all required inputs.")
        return

    dayid2cellinfo = build_dayid_to_cellinfo()

    results: List[SessionResult] = []
    for s in sessions:
        try:
            r = compute_session_dropone(s, dayid2cellinfo, n_jobs=N_JOBS)
        except Exception as e:
            print(f"[SKIP] {s}: exception {e}")
            r = None
        if r is not None:
            results.append(r)

    if not results:
        print("[FATAL] No sessions processed successfully.")
        return

    # group by indoor/outdoor
    for group in ["indoor", "outdoor"]:
        group_res = [r for r in results if r.group == group]
        if not group_res:
            print(f"[WARN] No sessions for group={group}")
            continue

        # full DevExpl per session
        sess_full: Dict[str, np.ndarray] = {}
        for r in group_res:
            arr = np.array(list(r.full_devexpl_by_neuron.values()), dtype=np.float64)
            sess_full[r.session] = arr

        full_stat = hierarchical_bootstrap_mean(sess_full, n_boot=N_BOOT, ci_lo=CI_LO, ci_hi=CI_HI, seed=SEED)

        # per feature contribution
        feature_stats: Dict[str, Tuple[float, float, float]] = {}
        for feat in VARS_ALL:
            sess_feat: Dict[str, np.ndarray] = {}
            for r in group_res:
                arr = np.array(list(r.contrib_frac_by_feature_by_neuron[feat].values()), dtype=np.float64)
                sess_feat[r.session] = arr
            feature_stats[feat] = hierarchical_bootstrap_mean(sess_feat, n_boot=N_BOOT, ci_lo=CI_LO, ci_hi=CI_HI, seed=SEED)

        out_png = WEIGHTS_BASE / f"DEVEXPL_DROPONE_PYR_{group}.png"
        plot_summary_figure(
            out_png=out_png,
            title=f"Poisson GLM | Pyramidal only | {group} | mean ± {CI_LO}-{CI_HI}th (hier bootstrap)",
            full_stat=full_stat,
            feature_stats=feature_stats,
        )
        print(f"[OK] Saved: {out_png}")

        # also save summary CSV
        rows = [{"group": group, "metric": "full_devexpl", "feature": "FULL",
                 "mean": full_stat[0], "ci_lo": full_stat[1], "ci_hi": full_stat[2]}]
        for feat in VARS_ALL:
            m, lo, hi = feature_stats[feat]
            rows.append({"group": group, "metric": "frac_full_dev", "feature": feat,
                         "mean": m, "ci_lo": lo, "ci_hi": hi})
        pd.DataFrame(rows).to_csv(WEIGHTS_BASE / f"DEVEXPL_DROPONE_PYR_{group}_summary.csv", index=False)

    print("[DONE]")

if __name__ == "__main__":
    main()
