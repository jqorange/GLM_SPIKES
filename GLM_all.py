import os
import shutil
import h5py
import numpy as np
import pandas as pd
from numpy import unwrap
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm

def create_raised_cosine_basis(n_basis=16, history_ms=160, dt=1):
    ttb = np.arange(dt, history_ms + dt, dt, dtype=np.float32)
    log_t = np.log(ttb)
    centers = np.linspace(log_t[0], log_t[-1], n_basis, dtype=np.float32)
    width = centers[1] - centers[0]
    basis = []
    for c in centers:
        arg = (log_t - c) * np.pi / (2 * width)
        basis.append(((np.cos(np.clip(arg, -np.pi, np.pi)) + 1) / 2).astype(np.float32))
    return np.stack(basis, axis=1)

def spike_history_design_fast(spikes, basis):
    T = spikes.shape[0]
    basis_rev = basis[::-1, :].astype(np.float32)
    Xh_cols = [
        np.convolve(spikes, basis_rev[:, k], mode="full")[:T].astype(np.float32)
        for k in range(basis_rev.shape[1])
    ]
    return np.stack(Xh_cols, axis=1)

# -----------------------------------------------------------------------------
# 1. Read and preprocess everything up to X_aligned, y_all
# -----------------------------------------------------------------------------
with h5py.File("spike_binary_1000Hz.h5", "r") as hf:
    spike_array = hf["spike_binary"][:].astype(np.float32)
neuron_names = [f"neuron_{i+1}" for i in range(spike_array.shape[1])]
spike_df = pd.DataFrame(spike_array, columns=neuron_names)

behavior_df = pd.read_csv("F5D10_outdoor_modified.csv").drop(columns=["Index"], errors="ignore")
position_df = pd.read_csv("positions_F5D10_outdoor.csv")[["head_x", "head_y"]]
dlc_df      = pd.read_csv("final_filtered_F5D10_outdoor_50hz.csv")[["head_v", "bodyCenter1_v"]]
imu_df      = pd.read_csv("F5D10_outdoor_IMU_features_basic.csv")[["roll","yaw","pitch","speed_z"]]

def bin_column(series, bins, min_val=None, max_val=None):
    if min_val is None: min_val = series.min()
    if max_val is None: max_val = series.max()
    edges = np.linspace(min_val, max_val, bins + 1, dtype=np.float32)
    binned = np.digitize(series, edges) - 1
    return np.clip(binned, 0, bins - 1).astype(np.int16)

# position binning
position_df["x_bin"] = (position_df["head_x"] // (4*4.611473)).astype(int)
position_df["y_bin"] = (position_df["head_y"] // (4*4.611473)).astype(int)
unique_pos = position_df[["x_bin","y_bin"]].drop_duplicates().reset_index(drop=True)
unique_pos["pos_idx"] = np.arange(len(unique_pos), dtype=np.int16)
position_df = position_df.merge(unique_pos, on=["x_bin","y_bin"], how="left")

# upsample / repeat to 1kHz
def repeat_df(df, factor=20):
    return df.loc[df.index.repeat(factor)].reset_index(drop=True)

def upsample_df(df, factor=20):
    old = np.arange(len(df))
    new = np.linspace(0, len(df)-1, len(df)*factor)
    return pd.DataFrame({c: np.interp(new, old, df[c].values).astype(np.float32)
                         for c in df.columns})

behavior_1k = repeat_df(behavior_df)
dlc_1k      = upsample_df(dlc_df)
imu_df["roll"]  = unwrap(imu_df["roll"])
imu_df["yaw"]   = unwrap(imu_df["yaw"])
imu_df["pitch"] = unwrap(imu_df["pitch"])
imu_1k      = upsample_df(imu_df)
pos_1k      = repeat_df(position_df[["pos_idx"]]).rename(columns={"pos_idx":"position"})

# bin DLC/IMU
dlc_binned = pd.DataFrame({
    c: bin_column(dlc_1k[c], 24, -180, 180) if "angle_" in c
       else bin_column(dlc_1k[c], 30)
    for c in dlc_1k.columns
})
imu_binned = pd.DataFrame({
    c: bin_column(imu_1k[c], 24, -np.pi, np.pi) if c in ["roll","yaw","pitch"]
       else bin_column(imu_1k[c], 30)
    for c in imu_1k.columns
})

# combine and one-hot encode
all_binned = pd.concat([behavior_1k, dlc_binned, imu_binned, pos_1k], axis=1).reset_index(drop=True)
encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore", drop='if_binary')
X_sparse = encoder.fit_transform(all_binned).astype(np.float32)

# build mappings
feat_names_out = encoder.get_feature_names_out(all_binned.columns.tolist())
history_basis = create_raised_cosine_basis(16, 160)  # float32
history_names = [f"history_{i+1}" for i in range(history_basis.shape[1])]
os.makedirs("cv_results", exist_ok=True)
with open("cv_results/feature_mapping.txt","w") as f1, \
     open("cv_results/coefficients_mapping.txt","w") as f2:
    for idx,name in enumerate(np.concatenate([feat_names_out, history_names])):
        f1.write(f"{idx}: {name}\n")
        f2.write(f"{idx}: {name}\n")
shutil.copy("cv_results/coefficients_mapping.txt","cv_results/pvalue_mapping.txt")

# align spike & X
start_idx = int(1.238591123957225e4 * 1000)
end_idx   = int(1.588404693733597e4 * 1000)
spike_aligned = spike_df.iloc[start_idx:end_idx].reset_index(drop=True)
X_aligned     = X_sparse[:end_idx-start_idx, :]
y_all         = spike_aligned.values.astype(np.float32)

# -----------------------------------------------------------------------------
# 2. Pre-split and save each fold to disk
# -----------------------------------------------------------------------------
kf = KFold(n_splits=5, shuffle=False)
splits = list(kf.split(X_aligned))
os.makedirs("cv_results/splits", exist_ok=True)

for fold, (tr_idx, va_idx) in enumerate(splits, 1):
    X_tr = X_aligned[tr_idx].toarray().astype(np.float32)
    X_va = X_aligned[va_idx].toarray().astype(np.float32)
    y_tr = y_all[tr_idx]
    y_va = y_all[va_idx]
    np.savez_compressed(f"cv_results/splits/fold{fold}.npz",
                        X_train=X_tr, y_train=y_tr,
                        X_val  =X_va, y_val  =y_va)
    del X_tr, X_va, y_tr, y_va

# free big arrays
del X_aligned, y_all, all_binned

# -----------------------------------------------------------------------------
# 3. Load each fold file and run GLM + history on-the-fly
# -----------------------------------------------------------------------------
P = None
K = history_basis.shape[1]
for fold in range(1, 6):
    print(f"\n=== Fold {fold} ===")
    data = np.load(f"cv_results/splits/fold{fold}.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    if P is None:
        P = X_train.shape[1]
        Xtr_base = np.empty((X_train.shape[0], P+K), dtype=np.float32)
        Xva_base = np.empty((X_val.shape[0],   P+K), dtype=np.float32)
    Xtr_base[:, :P] = X_train
    Xva_base[:, :P] = X_val

    train_betas, train_pvals = [], []
    train_r2s, val_r2s       = [], []
    pred_val_all             = np.zeros_like(y_val, dtype=np.float32)

    for i, neuron in enumerate(tqdm(neuron_names, desc="Neurons")):
        # on-the-fly history
        spikes = spike_aligned[neuron].values.astype(np.float32)
        hist_full = spike_history_design_fast(spikes, history_basis)
        tr_idx, va_idx = splits[fold-1]
        hist_tr = hist_full[tr_idx]
        hist_va = hist_full[va_idx]
        del hist_full

        Xtr_base[:, P:] = hist_tr
        Xva_base[:, P:] = hist_va

        y_tr = y_train[:, i]
        y_va = y_val[:, i]
        model = PoissonRegressor(alpha=0.0005, max_iter=10000)
        model.fit(Xtr_base, y_tr)

        mu_tr = model.predict(Xtr_base).astype(np.float32)
        mu_va = model.predict(Xva_base).astype(np.float32)
        train_betas.append(model.coef_.astype(np.float32))

        def _dev(y, yp):
            eps = 1e-8
            return 2 * np.sum(y*np.log((y+eps)/(yp+eps)) - (y-yp))
        train_r2s.append(1 - _dev(y_tr, mu_tr)/_dev(y_tr, y_tr.mean()))
        val_r2s  .append(1 - _dev(y_va, mu_va)/_dev(y_va, y_va.mean()))

        W = mu_tr
        try:
            Fisher = Xtr_base.T @ (W[:,None]*Xtr_base) + 1e-4*np.eye(P+K, dtype=np.float32)
            cov    = np.linalg.inv(Fisher)
            se     = np.sqrt(np.maximum(np.diag(cov),1e-12))
            z      = model.coef_.astype(np.float32)/(se+1e-8)
            train_pvals.append((2*(1-norm.cdf(np.abs(z)))).astype(np.float32))
        except np.linalg.LinAlgError:
            train_pvals.append(np.full((P+K,), np.nan, dtype=np.float32))

        pred_val_all[:, i] = mu_va

    # save results per fold
    pd.DataFrame(train_betas, index=neuron_names)\
      .to_csv(f"cv_results/fold{fold}_train_coefficients.csv")
    pd.DataFrame(train_pvals, index=neuron_names)\
      .to_csv(f"cv_results/fold{fold}_train_pvalues.csv")
    pd.DataFrame({
        "neuron": neuron_names,
        "pseudo_R2_train": train_r2s,
        "pseudo_R2_val":  val_r2s
    }).to_csv(f"cv_results/fold{fold}_r2.csv", index=False)

    with h5py.File(f"cv_results/fold{fold}_pred.h5","w") as hf:
        hf.create_dataset("pred", data=pred_val_all, compression="gzip")
        hf.create_dataset("true", data=y_val,           compression="gzip")

print("\n✅ All folds finished. Results in cv_results/.")
