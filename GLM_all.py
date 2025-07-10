import os
import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm
from numpy import unwrap
import tqdm
import shutil

# === 读取数据 ===
spike_df = pd.read_csv("spike_counts_5Hz.csv").drop(columns=["Index"], errors="ignore")
behavior_df = pd.read_csv("F5D10_outdoor_modified.csv").drop(columns=["Index"], errors="ignore")
position_df = pd.read_csv("positions_F5D10_outdoor.csv")[["head_x", "head_y"]]
dlc_df = pd.read_csv("final_filtered_F5D10_outdoor_50hz.csv")
imu_df = pd.read_csv("F5D10_outdoor_IMU_features_basic.csv")

# === 分 bin 函数 ===
def bin_column(series, bins, min_val=None, max_val=None):
    if min_val is None: min_val = series.min()
    if max_val is None: max_val = series.max()
    binned = np.digitize(series, np.linspace(min_val, max_val, bins+1)) - 1
    binned[binned >= bins] = bins - 1
    binned[binned < 0] = 0
    return binned

# === DLC 特征分 bin ===
angle_cols = [col for col in dlc_df.columns if "angle_" in col and "change" not in col]
angle_change_cols = [col for col in dlc_df.columns if "angle_change" in col]
linear_cols = list(set(dlc_df.columns) - set(angle_cols) - set(angle_change_cols))

dlc_binned = pd.DataFrame()
for col in angle_cols:
    dlc_binned[col] = bin_column(dlc_df[col], 24, -180, 180)
for col in angle_change_cols:
    dlc_binned[col] = bin_column(dlc_df[col], 30)
for col in linear_cols:
    dlc_binned[col] = bin_column(dlc_df[col], 30)

# === IMU 特征分 bin ===
imu_df["roll"] = unwrap(imu_df["roll"])
imu_df["yaw"] = unwrap(imu_df["yaw"])
imu_df["pitch"] = unwrap(imu_df["pitch"])

imu_binned = pd.DataFrame()
for col in imu_df.columns:
    if col in ["roll", "yaw", "pitch"]:
        imu_binned[col] = bin_column(imu_df[col], 24, -np.pi, np.pi)
    else:
        imu_binned[col] = bin_column(imu_df[col], 30)

# === 位置分 bin ===
position_df["x_bin"] = (position_df["head_x"] // 50).astype(int)
position_df["y_bin"] = (position_df["head_y"] // 50).astype(int)
unique_positions = position_df[["x_bin", "y_bin"]].drop_duplicates().reset_index(drop=True)
unique_positions["pos_idx"] = np.arange(len(unique_positions))
position_df = position_df.merge(unique_positions, on=["x_bin", "y_bin"], how="left")

# === 下采样到 5Hz ===
def downsample_df(df, factor=10):
    T = df.shape[0] // factor
    return df[:T*factor].values.reshape(T, factor, -1).mean(axis=1)

behavior_df_5hz = pd.DataFrame(
    downsample_df(behavior_df).round().astype(int),
    columns=behavior_df.columns
)
dlc_binned_5hz = pd.DataFrame(
    downsample_df(dlc_binned).astype(int),
    columns=dlc_binned.columns
)
imu_binned_5hz = pd.DataFrame(
    downsample_df(imu_binned).astype(int),
    columns=imu_binned.columns
)
position_idx_5hz = pd.DataFrame(
    downsample_df(position_df[["pos_idx"]]).astype(int),
    columns=["position"]
)

# === 编码非行为部分（强制固定 categories） ===
categories = []
for col in dlc_binned_5hz.columns:
    if "angle_" in col:
        categories.append(np.arange(24))
    elif "angle_change" in col:
        categories.append(np.arange(30))
    else:
        categories.append(np.arange(30))
for col in imu_binned_5hz.columns:
    if col in ["roll", "yaw", "pitch"]:
        categories.append(np.arange(24))
    else:
        categories.append(np.arange(30))
categories.append(np.arange(len(unique_positions)))  # position

encoder = OneHotEncoder(categories=categories, sparse_output=False, handle_unknown="ignore")
non_behavior_all = np.concatenate([
    dlc_binned_5hz,
    imu_binned_5hz,
    position_idx_5hz
], axis=1)
onehot_all = encoder.fit_transform(non_behavior_all)

# === 写 feature mapping ===
os.makedirs("cv_results", exist_ok=True)
with open("cv_results/feature_mapping.txt", "w") as f:
    dim = 0
    for col in behavior_df_5hz.columns:
        f.write(f"{dim}-{dim}: behavior.{col}\n")
        dim += 1
    feature_names = list(dlc_binned_5hz.columns) + list(imu_binned_5hz.columns) + ["position"]
    for col, cats in zip(feature_names, encoder.categories_):
        n_cats = len(cats)
        f.write(f"{dim}-{dim + n_cats - 1}: {col}\n")
        dim += n_cats

with open("cv_results/coefficients_mapping.txt", "w") as f:
    dim = 0
    for col in behavior_df_5hz.columns:
        f.write(f"{dim}-{dim}: behavior.{col}\n")
        dim += 1
    for col, cats in zip(feature_names, encoder.categories_):
        for i in range(len(cats)):
            f.write(f"{dim}: {col} (bin {cats[i]})\n")
            dim += 1
shutil.copy("cv_results/coefficients_mapping.txt", "cv_results/pvalue_mapping.txt")

# === 特征拼接并对齐 ===
all_features = np.concatenate([behavior_df_5hz, onehot_all], axis=1)
start_idx = int(1.238591123957225e4 * 5)
end_idx = int(1.588404693733597e4 * 5) - 1
spike_aligned = spike_df.iloc[start_idx:end_idx].reset_index(drop=True)
feature_aligned = pd.DataFrame(all_features[:spike_aligned.shape[0]])

X = feature_aligned.values
y_all = spike_aligned.values
neuron_names = spike_aligned.columns.tolist()

# === GLM + Cross Validation ===
kf = KFold(n_splits=5, shuffle=False)
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n=== Fold {fold} ===")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train_all, y_val_all = y_all[train_idx], y_all[val_idx]

    train_betas, train_pvals = [], []
    train_r2s, val_r2s = [], []

    for i, neuron in tqdm.tqdm(enumerate(neuron_names)):
        y_train = y_train_all[:, i]
        y_val = y_val_all[:, i]

        model = PoissonRegressor(alpha=0.0005, max_iter=10000)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        beta = model.coef_
        train_betas.append(beta)

        def deviance(y, y_pred):
            eps = 1e-8
            return 2 * np.sum(y * np.log((y + eps) / (y_pred + eps)) - (y - y_pred))

        r2_train = 1 - deviance(y_train, y_pred_train) / deviance(y_train, np.full_like(y_train, np.mean(y_train)))
        r2_val = 1 - deviance(y_val, y_pred_val) / deviance(y_val, np.full_like(y_val, np.mean(y_val)))
        train_r2s.append(r2_train)
        val_r2s.append(r2_val)

        mu = y_pred_train
        W_diag = mu
        try:
            reg_term = 1e-4 * np.eye(X_train.shape[1])
            Fisher = X_train.T @ (W_diag[:, None] * X_train) + reg_term
            cov = np.linalg.inv(Fisher)
            se = np.sqrt(np.maximum(np.diag(cov), 1e-12))
            z = beta / (se + 1e-8)
            p = 2 * (1 - norm.cdf(np.abs(z)))
        except np.linalg.LinAlgError:
            print(f"[Warning] Singular matrix for neuron {neuron}")
            p = np.full_like(beta, np.nan)
        train_pvals.append(p)

    pd.DataFrame(train_betas, index=neuron_names).to_csv(f"cv_results/fold{fold}_train_coefficients.csv")
    pd.DataFrame(train_pvals, index=neuron_names).to_csv(f"cv_results/fold{fold}_train_pvalues.csv")
    pd.DataFrame({"neuron": neuron_names, "pseudo_R2_train": train_r2s, "pseudo_R2_val": val_r2s}).to_csv(f"cv_results/fold{fold}_r2.csv", index=False)

print("\n✅ Finished all folds. Coefficients, p-values, and R² saved.")