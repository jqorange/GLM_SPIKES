import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 1. 加载系数矩阵与 feature_mapping.txt ===
coef_path = r"cv_results\fold1_train_coefficients.csv"
mapping_path = r"cv_results\feature_mapping.txt"
df_betas = pd.read_csv(coef_path, index_col=0)

# === 2. 提取 position one-hot 的维度范围 ===
pos_start, pos_end = None, None
with open(mapping_path, "r") as f:
    for line in f:
        if "position" in line.lower():
            parts = line.strip().split(":")[0].split("-")
            pos_start, pos_end = int(parts[0]), int(parts[1])
            break

if pos_start is None or pos_end is None:
    raise ValueError("⚠️ 未在 feature_mapping.txt 中找到 'position' 行。")

# === 3. 提取位置系数 ===
pos_cols = df_betas.columns[pos_start:pos_end + 1]
beta_pos = df_betas[pos_cols]

# === 4. 获取真实出现过的位置索引 ===
position_path = r"positions_F5D10_outdoor.csv"
position_df = pd.read_csv(position_path)[["head_x", "head_y"]]

# 分 bin
position_df["x_bin"] = (position_df["head_x"] // 50).astype(int)
position_df["y_bin"] = (position_df["head_y"] // 50).astype(int)
unique_positions = position_df[["x_bin", "y_bin"]].drop_duplicates().reset_index(drop=True)
unique_positions["pos_idx"] = np.arange(len(unique_positions))

# 映射每个位置为 pos_idx
position_df = position_df.merge(unique_positions, on=["x_bin", "y_bin"], how="left")

# 下采样 pos_idx
def downsample_df(df, factor=10):
    T = df.shape[0] // factor
    return df[:T*factor].values.reshape(T, factor, -1).mean(axis=1)

position_idx_5hz = downsample_df(position_df[["pos_idx"]]).astype(int).flatten()
used_pos_idx_set = sorted(np.unique(position_idx_5hz))

# 获取实际用到的 x_bin, y_bin
used_positions = unique_positions[unique_positions["pos_idx"].isin(used_pos_idx_set)].reset_index(drop=True)

# 检查维度一致性
if beta_pos.shape[1] != used_positions.shape[0]:
    raise ValueError(f"🚨 Beta 维度和 used_positions 不一致：{beta_pos.shape[1]} vs {used_positions.shape[0]}")

# === 5. 绘图 ===
os.makedirs("figure_effecients", exist_ok=True)

for neuron_name in beta_pos.index:
    beta = beta_pos.loc[neuron_name].values
    merged = used_positions.copy()
    merged["beta"] = beta

    x_max = merged["x_bin"].max() + 1
    y_max = merged["y_bin"].max() + 1
    heatmap = np.full((y_max, x_max), np.nan)

    for _, row in merged.iterrows():
        x, y = int(row["x_bin"]), int(row["y_bin"])
        heatmap[y, x] = row["beta"]

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap="coolwarm", origin="lower")
    plt.colorbar(label="GLM Coefficient")
    plt.title(f"Spatial Importance | {neuron_name}")
    plt.xlabel("x_bin")
    plt.ylabel("y_bin")
    plt.tight_layout()
    plt.savefig(f"figure_effecients/positions/{neuron_name}.png", dpi=300)
    plt.close()

print("✅ 所有 neuron 的空间热图已保存至 'figure_effecients/'。")
