import math
from pathlib import Path

# Input roots
IMU_ROOT = Path(r"D:\Jiaqi\Projects\IMU_Preprocess\IMU_results")
SPIKE_ROOT = Path(r"D:\Jiaqi\Projects\GLM_File\spike_binary")
DLC_ROOT = Path(r"D:\Jiaqi\Projects\DLC_results_features")
POSITION_ROOT = Path(r"D:\Jiaqi\Projects\ACC_DATA\DLC_Process\position_50hz")

# Output root
WEIGHTS_BASE = Path("weights_Poisson_forward")
WEIGHTS_BASE.mkdir(parents=True, exist_ok=True)

# Directories
FULL_FIT_DIRNAME = "FULL_FIT"

# Parallel / CV
N_JOBS = 29
SEED = 0
CV_FOLDS = 10

# Bin definitions
BIN_MS = 20
FS_HZ = 50.0
BASE_FS = 200.0
AGG_FACTOR = int(BASE_FS / FS_HZ)  # 4
MAX_MISMATCH_FRAMES_50HZ = 5  # tolerance in 50 Hz frames

# Forward-selection test threshold
ALPHA = 0.05

# PoissonRegressor params
MAX_ITER = 500
POISSON_ALPHA = 1e-6  # IMPORTANT: small alpha to avoid over-shrinking for one-hot high-dim X

# Smoothness regularization (pseudo-observation trick)
# Set per-variable lambdas (>0) to enable smoothing per feature group.
SMOOTH_LAMBDAS = {
    "Position": 0.0,
    "Speed": 0.0,
    "roll": 0.0,
    "yaw": 0.0,
    "pitch": 0.0,
}
SMOOTH_VARS = list(SMOOTH_LAMBDAS.keys())

# Candidate variable set
VARS_ALL = ["Position", "Speed", "roll", "yaw", "pitch"]

# Discretization bins
POSITION_CELL_CM = 8.0
SPEED_N_BINS = 15
MIN_SPEED_CM_S = 0.0  # minimum head speed (cm/s) to include in fitting; <=0 keeps all

# Angle binning (ranges are in radians). For multi-interval ranges, order controls
# bin layout and adjacency (used for smoothing/plotting).
ANGLE_RANGES_BY_VAR = {
    "yaw": [(0.0, 2.0 * math.pi)],
    "roll": [(3.0 * math.pi / 2.0, 2.0 * math.pi), (0.0, math.pi / 2.0)],
    "pitch": [(3.0 * math.pi / 2.0, 2.0 * math.pi), (0.0, math.pi / 2.0)],
}
ANGLE_BINS_BY_VAR = {
    "yaw": 12,
    "roll": 6,
    "pitch": 6,
}

# Fitting-curve plots
PLOT_SMOOTH_MS = 1000
PLOT_START_SEC = 0.0
PLOT_END_SEC = 600.0
PLOT_ZSCORE = False
