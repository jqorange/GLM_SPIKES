import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 配置路径 ===
coeff_csv = "cv_results/fold1_train_coefficients.csv"
mapping_txt = "cv_results/coefficients_mapping.txt"
pos_csv    = "positions_F5D10_outdoor.csv"
out_dir    = "figure_coefficients/positions"

# === 1. 加载系数矩阵和 mapping ===
df_coef = pd.read_csv(coeff_csv, index_col=0)
mapping = []
with open(mapping_txt, 'r') as f:
    for line in f:
        if ':' in line:
            _, name = line.split(':', 1)
            mapping.append(name.strip())

# === 2. 找到 position one-hot 列区间 ===
pos_indices = [i for i, nm in enumerate(mapping)
               if re.sub(r'_\d+$', '', nm).lower() == 'position']
if not pos_indices:
    raise ValueError("未在 mapping 中找到 'position' 特征。")
pos_start, pos_end = min(pos_indices), max(pos_indices)

# 提取 position 对应的列标签（按位置切片）
pos_cols = df_coef.columns[pos_start:pos_end+1]
beta_pos = df_coef[pos_cols]

# === 3. 读取位置数据并构建完整网格 ===
pos_df = pd.read_csv(pos_csv)[['head_x', 'head_y']]
bin_size = 4 * 4.611473
# 计算 bin
pos_df['x_bin'] = (pos_df['head_x'] // bin_size).astype(int)
pos_df['y_bin'] = (pos_df['head_y'] // bin_size).astype(int)
# 保持与训练时相同的 pos_idx 顺序: 首次出现顺序
uniq = pos_df[['x_bin', 'y_bin']].drop_duplicates().reset_index(drop=True)
uniq['pos_idx'] = np.arange(len(uniq))
# 检查一致性
n_bins = pos_end - pos_start + 1
if uniq.shape[0] != n_bins:
    print(f"警告: mapping 中 {n_bins} 个 position 特征, 但实际识别到 {uniq.shape[0]} 个格子。")

# 网格尺寸
x_min, x_max = uniq['x_bin'].min(), uniq['x_bin'].max()
y_min, y_max = uniq['y_bin'].min(), uniq['y_bin'].max()
grid_w = x_max - x_min + 1
grid_h = y_max - y_min + 1

# === 4. 绘图保存 ===
os.makedirs(out_dir, exist_ok=True)
for neuron in beta_pos.index:
    beta = beta_pos.loc[neuron].values
    # 初始化网格，默认 0
    heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
    # 填充已访问格子
    for _, row in uniq.iterrows():
        idx = int(row['pos_idx'])
        if idx < beta.size:
            xi = int(row['x_bin'] - x_min)
            yi = int(row['y_bin'] - y_min)
            heatmap[yi, xi] = beta[idx]
    # 绘制热图
    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        heatmap,
        origin='lower',
        cmap='coolwarm',
        vmin=np.nanmin(beta),
        vmax=np.nanmax(beta)
    )
    plt.colorbar(im, label='GLM Coefficient')
    plt.title(f"Spatial Importance | {neuron}")
    plt.xlabel('x_bin')
    plt.ylabel('y_bin')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{neuron}.png"), dpi=300)
    plt.close()

print(f"✅ 空间系数热图已保存至 {out_dir}")