import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# === 1. 加载数据 ===
coef_df = pd.read_csv(r"cv_results\fold1_train_coefficients.csv", index_col=0).iloc[:, :12]
pval_df = pd.read_csv(r"cv_results\fold1_train_pvalues.csv", index_col=0).iloc[:, :12]

# === 2. 仅保留同时在 x_bin 和 y_bin 上 p < 0.01 的 neuron
# significant_place_neurons = (pval_df["x_bin"] < 0.001) & (pval_df["y_bin"] < 0.001)
significant_place_neurons = (pval_df["1"] < 1)
selected_neurons = pval_df.index[significant_place_neurons]

# === 3. 提取这些 neuron 的所有行（包括 x_bin, y_bin 和所有行为标签）
coef_df = coef_df.loc[selected_neurons]
pval_df = pval_df.loc[selected_neurons]


# === 4. 去除 intercept 行 ===
coef_df = coef_df.drop(columns=["intercept"], errors="ignore")
pval_df = pval_df.drop(columns=["intercept"], errors="ignore")

# === 5. 转置：行为为行，神经元为列
coef_mat = coef_df.T
pval_mat = pval_df.T

# === 6. 设置 mask: p >= 0.01 为遮罩
mask = pval_mat >= 0.05

# === 5. 创建自定义颜色映射 - 更鲜艳的颜色
colors = [
    '#0d47a1',  # 深蓝
    '#1565c0',
    '#42a5f5',  # 浅蓝
    '#bbdefb',  # 更浅蓝
    '#eceff1',  # 浅灰 (原先是白色 '#ffffff')
    '#ffcccb',  # 肉粉色
    '#ef5350',  # 浅红
    '#d32f2f',  # 深红
    '#b71c1c'   # 更深红
]
n_bins = 256
cmap = LinearSegmentedColormap.from_list('vivid_coolwarm_no_white', colors, N=256)

# === 6. 绘图设置 ===
plt.style.use('default')  # 使用默认样式确保一致性
fig, ax = plt.subplots(figsize=(max(12, coef_mat.shape[1] * 0.3), max(8, coef_mat.shape[0] * 0.4)))

# 创建热力图
heatmap = sns.heatmap(
    coef_mat,
    mask=mask,
    cmap=cmap,
    center=0,
    linewidths=0.8,
    linecolor='white',
    cbar_kws={
        "label": "GLM Coefficient",
        "shrink": 0.8,
        "aspect": 30
    },
    square=False,
    ax=ax
)

# === 7. 为遮罩区域添加灰色背景和斜线纹理 ===
# 首先添加灰色背景
for i in range(coef_mat.shape[0]):
    for j in range(coef_mat.shape[1]):
        if mask.iloc[i, j]:
            # 添加灰色矩形背景
            rect = patches.Rectangle((j, i), 1, 1, linewidth=0,
                                     edgecolor='none', facecolor='#e0e0e0',
                                     alpha=0.7, zorder=1)
            ax.add_patch(rect)

            # 添加斜线纹理
            # 从左下到右上的斜线
            line1 = patches.Polygon([(j, i + 1), (j + 0.1, i + 1), (j + 1, i + 0.1), (j + 1, i)],
                                    closed=True, fill=True,
                                    facecolor='#bdbdbd', alpha=0.6, zorder=2)
            ax.add_patch(line1)

            # 从左上到右下的斜线（交叉效果）
            line2 = patches.Polygon([(j, i), (j + 0.1, i), (j + 1, i + 0.9), (j + 1, i + 1)],
                                    closed=True, fill=True,
                                    facecolor='#bdbdbd', alpha=0.4, zorder=2)
            ax.add_patch(line2)

# === 8. 美化设置 ===
ax.set_xlabel("Neuron", fontsize=14, fontweight='bold')
ax.set_ylabel("Behavior", fontsize=14, fontweight='bold')
ax.set_title("GLM Coefficients Heatmap\n(Significant coefficients p < 0.01)",
             fontsize=16, fontweight='bold', pad=20)

# 调整刻度标签
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# 调整颜色条
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=12)
cbar.set_label('GLM Coefficient', fontsize=12, fontweight='bold')

# 添加网格线增强对比度
ax.grid(True, which='major', color='white', linewidth=1.5, alpha=0.3)

# === 9. 添加图例说明 ===
# 创建自定义图例
legend_elements = [
    patches.Patch(facecolor=cmap(0.9), edgecolor='white', linewidth=1, label='Positive coefficient (p < 0.01)'),
    patches.Patch(facecolor=cmap(0.1), edgecolor='white', linewidth=1, label='Negative coefficient (p < 0.01)'),
    patches.Patch(facecolor='#e0e0e0', edgecolor='#bdbdbd', linewidth=1,
                  label='Non-significant (p ≥ 0.01)', hatch='///')
]

ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1),
          frameon=True, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig("glm_heatmap_enhanced.jpg", dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()

print(f"热力图尺寸: {coef_mat.shape}")
print(f"显著系数数量: {(~mask).sum().sum()}")
print(f"非显著系数数量: {mask.sum().sum()}")