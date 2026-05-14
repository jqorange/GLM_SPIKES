from pathlib import Path

# Output subfolders inside each session directory
RLLR_FITS_DIRNAME = "RLLR_FITS"
RLLR_STATS_DIRNAME = "RLLR_STATS"

# Bootstrap settings
N_BOOT = 2000
CI_LO, CI_HI = 5, 95

# Numerical safety
MU_EPS = 1e-12

# Where to search for cell_metrics/cellinfo mats (shared with devexpl pipeline)
DAY_SEARCH_DIRS = [
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F4/day1"),
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F4/day4"),
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F5/Merged/day2/121_day2"),
    Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F5/Merged/day3/121_day3"),
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F5/Merged/day4/121_day4"),
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F5/Merged/day5/121_day5"),
    # # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F5/Merged/day6/3E6_day6"),
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F5/Merged/day7/121_day7"),
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F5/Merged/day10/121_day10"),
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day3/3E6_day3"),
    # # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day5/3E6_day5"),
    Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day7/3E6_day7"),
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day8/3E6_day8"),
    # # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day9/3E6_day9"),
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day10/3E6_day10"),
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day2/3E6_day2"),
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day4/3E6_day4"),
    # Path(r"/fs/ayadata1-afr77.nbb.cornell.edu/volume4/ayadata4/data/FieldRat/2024/F6/Merged/day6/121_day6"),
]
