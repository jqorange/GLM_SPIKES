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
BIN_MS = 5
FS_HZ = 200.0
BASE_FS = 200.0
COV_FS_HZ = 50.0
UPSAMPLE_FACTOR = int(FS_HZ / COV_FS_HZ)  # 4
MAX_MISMATCH_FRAMES_200HZ = 20  # tolerance in 200 Hz frames

# Forward-selection test threshold
ALPHA = 0.05

# Bernoulli (logistic) params
MAX_ITER = 500
LOGISTIC_C = 1e6  # inverse of regularization strength (large C -> weak regularization)

# Candidate variable set
VARS_ALL = ["Position", "Speed", "roll", "yaw", "pitch"]

# Discretization bins
POSITION_CELL_CM = 8.0
SPEED_N_BINS = 15
ANGLE_N_BINS = 36  # roll/yaw/pitch bins
MIN_SPEED_CM_S = 0.0  # minimum head speed (cm/s) to include in fitting; <=0 keeps all

# Fitting-curve plots
PLOT_SMOOTH_MS = 1000
PLOT_START_SEC = 0.0
PLOT_END_SEC = 600.0
PLOT_ZSCORE = False
