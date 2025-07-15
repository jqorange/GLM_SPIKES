"""Plot coefficient profiles for all features produced by ``GLM_all.py``."""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 路径 ===
coef_path = r"cv_results/fold5_train_coefficients.csv"
mapping_path = r"cv_results/coefficients_mapping.txt"
output_dir = "figure_effecients"

df_betas = pd.read_csv(coef_path, index_col=0)

# === 分类函数 ===
def is_angle_feature(name: str) -> bool:
    """Return True if the feature represents an angle."""
    return ("angle_" in name.lower() or name.lower() in ["roll", "yaw", "pitch"])

# === 读取 mapping 文件 ("idx: name") ===
mapping = []
with open(mapping_path, "r") as f:
    for line in f:
        if not line.strip() or ":" not in line:
            continue
        idx_str, name = line.split(":", 1)
        idx = int(idx_str.strip())
        mapping.append((idx, name.strip()))

# 按索引排序并推断每个 feature 的维度范围
mapping.sort(key=lambda x: x[0])
feature_dims = {}
current = None
start_idx = None
for idx, name in mapping:
    base = re.sub(r"_\d+$", "", name)
    if base != current:
        if current is not None:
            feature_dims[current] = (start_idx, prev_idx)
        current = base
        start_idx = idx
    prev_idx = idx
if current is not None:
    feature_dims[current] = (start_idx, prev_idx)

# === 绘图 ===
for feature, (start, end) in feature_dims.items():
    if feature in ["head_v", "bodyCenter1_v", "distance_bodycenter1_boundary","roll","yaw","pitch","acc_x", "acc_y","acc_z"]:
        feature_folder = os.path.join(output_dir, feature.replace("/", "_"))
        os.makedirs(feature_folder, exist_ok=True)

        beta_block = df_betas.iloc[:, start:end+1]  # 所有神经元的该特征的系数矩阵
        num_bins = end - start + 1

        for neuron_name in beta_block.index:
            beta = beta_block.loc[neuron_name].values

            if is_angle_feature(feature):
                # === 角度型特征：极坐标扇形图 ===
                theta = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
                width = 2 * np.pi / num_bins
                cmap = plt.get_cmap("coolwarm")
                norm = plt.Normalize(vmin=beta.min(), vmax=beta.max())
                colors = cmap(norm(beta))

                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
                bars = ax.bar(theta, np.ones_like(beta), width=width, bottom=0.0, color=colors, edgecolor='black', linewidth=0.5)
                ax.set_yticklabels([])
                ax.set_xticks(np.linspace(0, 2*np.pi, 8, endpoint=False))
                ax.set_title(f"{feature} | {neuron_name}", fontsize=12)
                fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal", label="GLM Coefficient")
            else:
                # === 其他特征：折线图 + 曲线拟合 ===
                x = np.arange(num_bins)
                y = beta

                # 使用三次多项式拟合
                coeffs = np.polyfit(x, y, deg=3)
                y_fit = np.polyval(coeffs, x)

                plt.figure(figsize=(6, 4))
                plt.plot(x, y, marker='o', label="Original", linestyle='--', alpha=0.7)
                plt.plot(x, y_fit, color='red', label="Fitted Curve", linewidth=2)
                plt.title(f"{feature} | {neuron_name}")
                plt.xlabel("bin index")
                plt.ylabel("GLM Coefficient")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

            # === 保存图像 ===
            out_path = os.path.join(feature_folder, f"{neuron_name}.png")
            plt.savefig(out_path, dpi=300)
            plt.close()

print("✅ 所有特征图已保存至 'figure_effecients/'。")
