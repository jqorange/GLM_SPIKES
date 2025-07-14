import os
import shutil
import h5py
import numpy as np
import pandas as pd
from numpy import unwrap
from scipy import sparse
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.linear_model import PoissonRegressor
import tqdm

# === 1. 读取 spike 二值数据 ===
with h5py.File("spike_binary_1000Hz.h5", "r") as hf:
    spike_array = hf["spike_binary"][:]  # shape: (T, N)
neuron_names = [f"neuron_{i+1}" for i in range(spike_array.shape[1])]
spike_df = pd.DataFrame(spike_array, columns=neuron_names)

# === 2. 读取其它原始数据 ===
behavior_df = pd.read_csv("F5D10_outdoor_modified.csv") \
                .drop(columns=["Index"], errors="ignore")
position_df = pd.read_csv("positions_F5D10_outdoor.csv")[["head_x", "head_y"]]
dlc_df      = pd.read_csv("final_filtered_F5D10_outdoor_50hz.csv")[["head_v", "bodyCenter1_v"]]
imu_df      = pd.read_csv("F5D10_outdoor_IMU_features_basic.csv")[["roll", "yaw", "pitch", "speed_z"]]

# === 3. 分 bin 工具函数 ===
def bin_column(series, bins, min_val=None, max_val=None):
    if min_val is None: min_val = series.min()
    if max_val is None: max_val = series.max()
    edges = np.linspace(min_val, max_val, bins + 1)
    binned = np.digitize(series, edges) - 1
    binned = np.clip(binned, 0, bins - 1)
    return binned

# === 4. 准备位置 bin ===
# 这里用 4*4.611473 单位分格大小
position_df["x_bin"] = (position_df["head_x"] // (4 * 4.611473)).astype(int)
position_df["y_bin"] = (position_df["head_y"] // (4 * 4.611473)).astype(int)
unique_pos = position_df[["x_bin","y_bin"]].drop_duplicates().reset_index(drop=True)
unique_pos["pos_idx"] = np.arange(len(unique_pos))
position_df = position_df.merge(unique_pos, on=["x_bin","y_bin"], how="left")

# === 5. 把所有 50Hz 数据 upsample/重复 到 1000Hz ===
def repeat_df(df, factor=20):
    return df.loc[df.index.repeat(factor)].reset_index(drop=True)

def upsample_df(df, factor=20):
    old_idx = np.arange(len(df))
    new_idx = np.linspace(0, len(df)-1, len(df)*factor)
    data = {col: np.interp(new_idx, old_idx, df[col].values) for col in df.columns}
    return pd.DataFrame(data)

behavior_1k = repeat_df(behavior_df)
dlc_1k      = upsample_df(dlc_df)
imu_df["roll"]  = unwrap(imu_df["roll"])
imu_df["yaw"]   = unwrap(imu_df["yaw"])
imu_df["pitch"] = unwrap(imu_df["pitch"])
imu_1k      = upsample_df(imu_df)
pos_1k      = repeat_df(position_df[["pos_idx"]]).rename(columns={"pos_idx":"position"})

# === 6. 对 DLC / IMU 做 binning ===
# 将角度与非角度分开 binned
angle_cols       = [c for c in dlc_1k.columns if "angle_" in c]
angle_change_cols= [c for c in dlc_1k.columns if "angle_change" in c]
linear_cols      = [c for c in dlc_1k.columns if c not in angle_cols + angle_change_cols]

dlc_binned = pd.DataFrame({
    col: bin_column(dlc_1k[col], 24, -180, 180) if col in angle_cols
         else bin_column(dlc_1k[col], 30)
    for col in dlc_1k.columns
})

imu_binned = pd.DataFrame({
    col: bin_column(imu_1k[col], 24, -np.pi, np.pi) if col in ["roll","yaw","pitch"]
         else bin_column(imu_1k[col], 30)
    for col in imu_1k.columns
})

# === 7. 合并所有离散化后的特征 DataFrame ===
all_binned = pd.concat([
    behavior_1k.reset_index(drop=True),
    dlc_binned.reset_index(drop=True),
    imu_binned.reset_index(drop=True),
    pos_1k.reset_index(drop=True)
], axis=1)

# === 8. 一次性 One-Hot 编码为稀疏矩阵 ===
encoder = OneHotEncoder(sparse_output=True, handle_unknown="ignore", drop='if_binary' )
X_sparse = encoder.fit_transform(all_binned)  # CSR matrix, shape=(T, total_bins)

# === 9. 生成 feature_mapping 和 coefficients_mapping ===
input_features = all_binned.columns.tolist()
feat_names_out = encoder.get_feature_names_out(input_features)

os.makedirs("cv_results", exist_ok=True)
with open("cv_results/feature_mapping.txt", "w") as f_feat, \
     open("cv_results/coefficients_mapping.txt", "w") as f_coef:
    for idx, fname in enumerate(feat_names_out):
        f_feat.write(f"{idx}: {fname}\n")
        f_coef.write(f"{idx}: {fname}\n")

# 复制一份给 pvalue_mapping
shutil.copy("cv_results/coefficients_mapping.txt", "cv_results/pvalue_mapping.txt")

# === 10. 按时间区间对齐 spike 与特征 ===
start_idx = int(1.238591123957225e4 * 1000)
end_idx   = int(1.588404693733597e4 * 1000)  # 不 -1，Python 切片自动开区间

spike_aligned = spike_df.iloc[start_idx:end_idx].reset_index(drop=True)
X_aligned     = X_sparse[start_idx:end_idx, :]

y_all = spike_aligned.values  # shape=(end_idx-start_idx, N)

# === 11. GLM + 5-fold CV ===
kf = KFold(n_splits=5, shuffle=False)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_aligned), 1):
    print(f"\n=== Fold {fold} ===")
    # 只在当前 fold toarray() 展开
    X_train = X_aligned[train_idx].toarray()
    X_val   = X_aligned[val_idx].toarray()
    y_train = y_all[train_idx]
    y_val   = y_all[val_idx]

    train_betas, train_pvals = [], []
    train_r2s, val_r2s = [], []

    for i, neuron in enumerate(neuron_names):
        y_tr = y_train[:, i]
        y_va = y_val[:, i]

        model = PoissonRegressor(alpha=0.0005, max_iter=10000)
        model.fit(X_train, y_tr)

        mu_tr = model.predict(X_train)
        mu_va = model.predict(X_val)
        beta = model.coef_
        train_betas.append(beta)

        # deviance-based pseudo-R²
        def dev(y, y_pred):
            eps = 1e-8
            return 2 * np.sum(y * np.log((y+eps)/(y_pred+eps)) - (y - y_pred))
        r2_tr = 1 - dev(y_tr, mu_tr) / dev(y_tr, np.full_like(y_tr, y_tr.mean()))
        r2_va = 1 - dev(y_va, mu_va) / dev(y_va, np.full_like(y_va, y_va.mean()))
        train_r2s.append(r2_tr)
        val_r2s.append(r2_va)

        # Fisher 信息矩阵计算标准误 & p-value
        W = mu_tr
        try:
            Fisher = X_train.T @ (W[:, None] * X_train) + 1e-4 * np.eye(X_train.shape[1])
            cov    = np.linalg.inv(Fisher)
            se     = np.sqrt(np.maximum(np.diag(cov), 1e-12))
            z      = beta / (se + 1e-8)
            pvals  = 2 * (1 - norm.cdf(np.abs(z)))
        except np.linalg.LinAlgError:
            print(f"[Warning] Fisher singular for {neuron}")
            pvals = np.full_like(beta, np.nan)
        train_pvals.append(pvals)

    # 保存本 fold 结果
    pd.DataFrame(train_betas, index=neuron_names) \
      .to_csv(f"cv_results/fold{fold}_train_coefficients.csv")
    pd.DataFrame(train_pvals, index=neuron_names) \
      .to_csv(f"cv_results/fold{fold}_train_pvalues.csv")
    pd.DataFrame({
        "neuron": neuron_names,
        "pseudo_R2_train": train_r2s,
        "pseudo_R2_val": val_r2s
    }).to_csv(f"cv_results/fold{fold}_r2.csv", index=False)

print("\n✅ All folds finished. Results in cv_results/.")
