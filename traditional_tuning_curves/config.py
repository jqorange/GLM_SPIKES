from pathlib import Path

from glm_poisson_forward.config import BIN_MS

OUT_ROOT = Path("tuning_curves_traditional")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Movement speed mask (m/s) corresponding to 2-100 cm/s
SPEED_MIN_M_S = 0.02
SPEED_MAX_M_S = 1.0

# Shuffle settings
SHUFFLE_N = 200
SHUFFLE_MIN_SEC = 20.0
PERCENTILE = 99

# Smoothing / binning settings
HD_SMOOTH_DEG = 14.5
SPEED_SMOOTH_MS = 200.0
ADAPTIVE_SMOOTH_ALPHA = 200.0
SPATIAL_SMOOTH_SIGMA_BINS = 1.0
CURVE_SMOOTH_SIGMA_BINS = 1.0

# Plotting
PLOT_MAX_NEURONS = 10

# Derived
BIN_SEC = BIN_MS / 1000.0
