"""
Helper utilities for the contribution workflow.
"""

from .cell_metrics import (
    build_dayid_to_cellinfo,
    find_cellinfo_mat,
    load_cell_types,
    parse_day_id_from_path,
    parse_day_id_from_session,
    pyramidal_indices_for_session,
)
from .constants import (
    CI_HI,
    CI_LO,
    DAY_SEARCH_DIRS,
    DROPONE_FITS_DIRNAME,
    DROPONE_STATS_DIRNAME,
    MU_EPS,
    N_BOOT,
)
from .plotting import (
    DroponePlotData,
    DroponeSessionStats,
    collect_dropone_plot_data,
    infer_group,
    load_dropone_session_stats,
    plot_dropone_suite,
    plot_summary_figure,
    suffix_for_threshold,
)
from .stats import (
    devexpl_from_deviances,
    deviance_from_ll,
    hierarchical_bootstrap_mean,
    poisson_loglik,
    poisson_loglik_saturated,
)
from .weights import (
    ensure_feature_names_file,
    load_feature_names_file,
    load_fold_weights,
    load_fold_weights_compat,
    model_key_from_vars,
    predict_oof_from_saved_weights,
    save_weights_for_model,
)

__all__ = [
    "build_dayid_to_cellinfo",
    "find_cellinfo_mat",
    "load_cell_types",
    "parse_day_id_from_path",
    "parse_day_id_from_session",
    "pyramidal_indices_for_session",
    "CI_HI",
    "CI_LO",
    "DAY_SEARCH_DIRS",
    "DROPONE_FITS_DIRNAME",
    "DROPONE_STATS_DIRNAME",
    "MU_EPS",
    "N_BOOT",
    "plot_summary_figure",
    "DroponePlotData",
    "DroponeSessionStats",
    "collect_dropone_plot_data",
    "infer_group",
    "load_dropone_session_stats",
    "plot_dropone_suite",
    "suffix_for_threshold",
    "devexpl_from_deviances",
    "deviance_from_ll",
    "hierarchical_bootstrap_mean",
    "poisson_loglik",
    "poisson_loglik_saturated",
    "ensure_feature_names_file",
    "load_feature_names_file",
    "load_fold_weights",
    "load_fold_weights_compat",
    "model_key_from_vars",
    "predict_oof_from_saved_weights",
    "save_weights_for_model",
]
