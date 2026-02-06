import os
from pathlib import Path

from glm_poisson_forward.config import BIN_MS

OUT_ROOT = Path("tuning_curves_traditional")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Movement speed mask (m/s) corresponding to 2-100 cm/s
SPEED_MIN_M_S = 0.02
SPEED_MAX_M_S = 1.5

# Shuffle settings
SHUFFLE_N = 200
SHUFFLE_MIN_SEC = 20.0

# Score thresholds (percentiles of shuffle distributions)
SCORE_PERCENTILES = {
    "hd_score": 99,
    "roll_score": 99,
    "pitch_score": 99,
    "speed_score": 99,
    "speed_stability": 99,
}

# Smoothing
# Unmasked rate smoothing (time series) for speed score
RATE_SMOOTH_SIGMA_BINS = 10.0
# Bin-wise smoothing for tuning curves (fixed for visualization)
BIN_SMOOTH_SIGMA_BINS = 2.0
MIN_BIN_OCCUPANCY_SEC = 5.0

# Angular k-fold vector length (1..K)
ANGULAR_K_MAX = 5

# Plotting
PLOT_MAX_NEURONS = 10
REBUILD_PAIRED_POLAR_PLOTS = True
EQUALIZE_POLAR_AREA = False
RESCORE_MODE = "scores"

# Angular binning (degrees per bin)
ROLL_BIN_DEG = 3.0
YAW_BIN_DEG = 3.0
PITCH_BIN_DEG = 3.0
ROLL_N_BINS = int(round(360.0 / ROLL_BIN_DEG))
YAW_N_BINS = int(round(360.0 / YAW_BIN_DEG))
PITCH_N_BINS = int(round(360.0 / PITCH_BIN_DEG))

# Parallelism
N_WORKERS = 28

# Derived
BIN_SEC = BIN_MS / 1000.0
