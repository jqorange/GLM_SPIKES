import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm
import h5py

# === 读取mat文件 ===
mat = scipy.io.loadmat(
    r"121_day10.cell_metrics.cellinfo.mat",
    squeeze_me=True,
    struct_as_record=False
)

# === 提取spike时间 ===
spike_struct_array = mat['cell_metrics']
spike_cells = list(spike_struct_array.spikes.times)

# === 计算总时长 ===
max_time = max(np.max(cell) if len(cell) > 0 else 0 for cell in spike_cells)

# === 定义采样率 ===
sampling_rates = {
    "1000Hz": 1000  # 1ms bin
}

for name, rate in sampling_rates.items():
    bin_width = 1.0 / rate
    num_bins = int(np.ceil(max_time * rate))

    # 初始化结果矩阵为布尔值
    result = np.zeros((num_bins, len(spike_cells)), dtype=bool)

    # 填充数据（有 spike 就设为 True）
    for neuron_idx, spike_times in tqdm(enumerate(spike_cells), total=len(spike_cells)):
        bin_indices = (spike_times / bin_width).astype(int)
        bin_indices = bin_indices[bin_indices < num_bins]  # 边界检查
        result[bin_indices, neuron_idx] = True

    # 保存为 HDF5 文件
    h5_filename = f"spike_binary_{name}.h5"
    with h5py.File(h5_filename, "w") as hf:
        hf.create_dataset("spike_binary", data=result.astype(np.uint8), compression="gzip")

    print(f"Saved {h5_filename} successfully!")
