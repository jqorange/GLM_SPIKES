from pathlib import Path

# Input roots
IMU_ROOT = Path(r"/home/js3785/Dataset/GLM_Data/IMU_results")
SPIKE_ROOT = Path(r"/home/js3785/Dataset/GLM_Data/spike_binary")
SPIKE_COUNT_ROOT = Path(r"/home/js3785/Dataset/GLM_Data/spike_count")
DLC_ROOT = Path(r"/home/js3785/Dataset/GLM_Data/DLC_results")
POSITION_ROOT = Path(r"/home/js3785/Dataset/GLM_Data/position_50hz")

# Spike input mode:
# - "binary": read 1000 Hz spike_binary and aggregate to FS_HZ inside the GLM code
# - "count":  read precomputed spike_count already binned at FS_HZ
SPIKE_INPUT_MODE = "count"
if SPIKE_INPUT_MODE not in {"binary", "count"}:
    raise ValueError(f"SPIKE_INPUT_MODE must be 'binary' or 'count', got {SPIKE_INPUT_MODE!r}")
SPIKE_INPUT_ROOT = SPIKE_ROOT if SPIKE_INPUT_MODE == "binary" else SPIKE_COUNT_ROOT

# Output root
WEIGHTS_BASE = Path("weights_Poisson_forward")
WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)

# Directories
FULL_FIT_DIRNAME = "FULL_FIT"

# Parallel / CV
N_JOBS = 128
SEED = 0
CV_FOLDS = 10
CV_VAL_FOLDS = 2

# Bin definitions
BIN_MS = 20
FS_HZ = 50.0
BASE_FS = 1000.0
AGG_FACTOR = int(BASE_FS / FS_HZ)  # 1000 Hz -> 50 Hz means 20 bins per GLM frame
MAX_MISMATCH_FRAMES_50HZ = 50  # tolerance in 50 Hz frames

# Optional indoor/outdoor paired-length matching before model fitting.
# When enabled, each indoor/outdoor pair is truncated to the shorter session length
# after loading data at 50 Hz and before design-matrix construction.
# MATCHED_SESSION_ALIGN:
#   - "start": keep the first target_len frames
#   - "end":   keep the last target_len frames
MATCH_INDOOR_OUTDOOR_LENGTHS = True
MATCHED_SESSION_ALIGN = "end"
if MATCHED_SESSION_ALIGN not in {"start", "end"}:
    raise ValueError(
        f"MATCHED_SESSION_ALIGN must be 'start' or 'end', got {MATCHED_SESSION_ALIGN!r}"
    )

# Optional time covariate:
# Time is represented as a single continuous feature.
# It is generated with 1-second bins and linearly mapped from 0 to 1.
INCLUDE_TIME_VARIABLE = False
TIME_BIN_SEC = 60

# -----------------------------------------------------------------------------
# Input / variable mapping config
# -----------------------------------------------------------------------------
# 每个文件可在此处定义采样率(fs_hz)；每个变量可指定来自哪个文件、读取哪一列。
# design matrix 前的数据构建会读取这些配置并自动对齐到 FS_HZ。
INPUT_FILES = {
    "position": {
        "filename": "positions_{session}.csv",
        "root": POSITION_ROOT,
        "fs_hz": 50.0,
    },
    "dlc_final": {
        "filename": "final_filtered_{session}_50hz.csv",
        "root": DLC_ROOT,
        "parent_dir_is_session": True,
        "fs_hz": 50.0,
    },
    "imu": {
        "filename": "{session}_IMU_features.csv",
        "root": IMU_ROOT,
        "parent_dir_is_session": True,
        "fs_hz": 50.0,
    },
}

VARIABLE_SPECS = {
    "Position": {
        "kind": "position2d",
        "source": "position",
        "columns": {"head_x": "head_x", "head_y": "head_y"},
        "design_key": "position",
        "cell_cm": 8.0,
    },
    "Speed": {
        "kind": "continuous",
        "source": "dlc_final",
        "column": ["bodyCenter1_v"],
        "n_bins": 30,
        "bin_range": (0.0, 1.5),
        "design_key": "speed",
    },
    "roll": {
        "kind": "continuous",
        "source": "imu",
        "column": "roll",
        "n_bins": 5,
        "trim_percentiles": (1.0, 99.0),
    },
    "yaw": {
        "kind": "continuous",
        "source": "imu",
        "column": "yaw",
        "n_bins": 12,
    },

    "pitch": {
        "kind": "continuous",
        "source": "imu",
        "column": "pitch",
        "n_bins": 8,
        "trim_percentiles": (1.0, 99.0),
    },

}

if INCLUDE_TIME_VARIABLE:
    VARIABLE_SPECS["Time"] = {
        "kind": "time",
        "design_key": "time",
        "value_key": "time",
        "time_bin_sec": TIME_BIN_SEC,
    }

# 画图时的合成变量（key 为新变量名，value 为待合成的基础变量列表）
VARIABLE_COMPOSITES = {
    "H": ["roll", "yaw", "pitch"],
}

# Forward-selection test threshold
ALPHA = 0.05
FORWARD_SEARCH_METRIC = "llhi"  # options: "deviance_explained", "llhi"
FORWARD_SEARCH_METRIC_CHOICES = ("deviance_explained", "llhi")
if FORWARD_SEARCH_METRIC not in FORWARD_SEARCH_METRIC_CHOICES:
    raise ValueError(
        f"FORWARD_SEARCH_METRIC must be one of {FORWARD_SEARCH_METRIC_CHOICES}, "
        f"got {FORWARD_SEARCH_METRIC!r}"
    )

# PoissonRegressor params
MAX_ITER = 1000
POISSON_ALPHA = 1e-5  # IMPORTANT: small alpha to avoid over-shrinking for one-hot high-dim X

# Approximate L1 via warm-start proximal-gradient refinement on top of PoissonRegressor.
# If L1_PROX_STEPS <= 0 or L1_LAMBDA <= 0, this refinement is skipped.
L1_LAMBDA = 1e-2
L1_PROX_STEPS = 500
L1_PROX_LR = 0.05

# Smoothness regularization (pseudo-observation trick)
# Set per-variable lambdas (>0) to enable smoothing per feature group.
SMOOTH_LAMBDAS = {
    "Position": 300.0,
    "Time": 0,
    "Speed": 1.0,
    "roll": 1.0,
    "yaw": 1.0,
    "pitch": 1.0,
}
SMOOTH_VARS = list(SMOOTH_LAMBDAS.keys())

# Candidate variable set
VARS_ALL = list(VARIABLE_SPECS.keys())

MIN_SPEED_CM_S = 0.0  # minimum head speed (cm/s) to include in fitting; <=0 keeps all

# Fitting-curve plots
PLOT_N_JOBS = 64  # plotting threads
PLOT_SMOOTH_MS = 500
PLOT_START_SEC = 600.0
PLOT_END_SEC = 1200.0
PLOT_ZSCORE = False
