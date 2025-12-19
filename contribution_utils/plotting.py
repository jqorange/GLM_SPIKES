from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from glm_poisson_forward.config import VARS_ALL


def plot_summary_figure(
    out_png: Path,
    title: str,
    full_stat: Tuple[float, float, float],
    feature_stats: Dict[str, Tuple[float, float, float]],
):
    """
    Two-panel figure:
      left: full DevExpl
      right: per-feature contribution fraction
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)

    features = VARS_ALL[:]  # keep order
    means = [feature_stats[f][0] for f in features]
    los = [feature_stats[f][1] for f in features]
    his = [feature_stats[f][2] for f in features]

    yerr_low = np.array(means) - np.array(los)
    yerr_high = np.array(his) - np.array(means)
    yerr = np.vstack([yerr_low, yerr_high])

    fig = plt.figure(figsize=(10, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    m, lo, hi = full_stat
    ax0.bar([0], [m], width=0.6, edgecolor="black", linewidth=0.8)
    ax0.errorbar([0], [m], yerr=[[m - lo], [hi - m]], fmt="none", capsize=4, linewidth=1.2)
    ax0.set_xticks([0])
    ax0.set_xticklabels(["Full\nmodel"])
    ax0.set_ylabel("Deviance explained")
    ax0.set_title("Full model")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(features))
    ax1.bar(x, means, width=0.65, edgecolor="black", linewidth=0.8)
    ax1.errorbar(x, means, yerr=yerr, fmt="none", capsize=4, linewidth=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(features, rotation=0)
    ax1.set_ylabel("Fraction of full-model dev.")
    ax1.set_title("Drop-one contribution (pyramidal only)")
    ax1.axhline(0.0, linewidth=0.8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_png, dpi=250)
    plt.close(fig)
