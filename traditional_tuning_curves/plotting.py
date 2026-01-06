from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from glm_poisson_forward.config import ANGLE_N_BINS, POSITION_CELL_CM, SPEED_N_BINS


def _plot_rate_map(ax, rate_map: np.ndarray) -> None:
    im = ax.imshow(rate_map, origin="lower", cmap="viridis")
    ax.set_title("Spatial rate map (Hz)")
    ax.set_xlabel("X bin")
    ax.set_ylabel("Y bin")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_autocorr(ax, autocorr: np.ndarray) -> None:
    im = ax.imshow(autocorr, origin="lower", cmap="coolwarm")
    ax.set_title("Autocorrelation")
    ax.set_xlabel("X lag")
    ax.set_ylabel("Y lag")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_speed(ax, speed_curve: np.ndarray) -> None:
    edges = np.linspace(0.0, 1.5, SPEED_N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, speed_curve, color="tab:green")
    ax.set_title("Speed tuning")
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Rate (Hz)")


def _close_curve(theta_deg: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    theta_rad = np.deg2rad(theta_deg)
    return np.concatenate([theta_rad, theta_rad[:1]]), np.concatenate([r, r[:1]])


def plot_polar_curve(out_path: Path, theta_deg: np.ndarray, r: np.ndarray, title: str, color: str) -> None:
    theta_c, r_c = _close_curve(theta_deg, r)
    plt.figure(figsize=(6.0, 6.0))
    ax = plt.subplot(111, projection="polar")
    ax.plot(theta_c, r_c, linewidth=2, color=color)
    ax.fill(theta_c, r_c, alpha=0.22, color=color)
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_neuron_summary(
    out_path: Path,
    neuron_idx: int,
    score_dict: Dict[str, float],
    aux: Dict[str, np.ndarray],
) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax_rate = fig.add_subplot(2, 2, 1)
    ax_auto = fig.add_subplot(2, 2, 2)
    ax_hd = fig.add_subplot(2, 2, 3, projection="polar")
    ax_speed = fig.add_subplot(2, 2, 4)

    _plot_rate_map(ax_rate, aux["rate_map"])
    _plot_autocorr(ax_auto, aux["autocorr"])

    theta_deg = np.linspace(0.0, 360.0, ANGLE_N_BINS, endpoint=False)
    theta_rad, r = _close_curve(theta_deg, aux["hd_curve"])
    ax_hd.plot(theta_rad, r, linewidth=2, color="#1f77b4")
    ax_hd.fill(theta_rad, r, alpha=0.22, color="#1f77b4")
    ax_hd.set_title("Yaw tuning (polar)")

    _plot_speed(ax_speed, aux["speed_curve"])

    fig.suptitle(
        "Neuron {} | grid={:.3f}, border={:.3f}, hd={:.3f}, speed={:.3f}".format(
            neuron_idx,
            score_dict.get("grid_score", float("nan")),
            score_dict.get("border_score", float("nan")),
            score_dict.get("hd_score", float("nan")),
            score_dict.get("speed_score", float("nan")),
        )
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    plot_polar_curve(
        out_path.parent / f"neuron_{neuron_idx:03d}_yaw_polar.png",
        theta_deg,
        aux["hd_curve"],
        f"Neuron {neuron_idx} | yaw tuning",
        color="#1f77b4",
    )
    plot_polar_curve(
        out_path.parent / f"neuron_{neuron_idx:03d}_roll_polar.png",
        theta_deg,
        aux["roll_curve"],
        f"Neuron {neuron_idx} | roll tuning",
        color="#ff7f0e",
    )
    plot_polar_curve(
        out_path.parent / f"neuron_{neuron_idx:03d}_pitch_polar.png",
        theta_deg,
        aux["pitch_curve"],
        f"Neuron {neuron_idx} | pitch tuning",
        color="#2ca02c",
    )


def binning_note(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Traditional tuning curves use the same binning rules as the GLM:\n"
            f"- Position bins: {POSITION_CELL_CM:.1f} cm square\n"
            f"- Speed bins: {SPEED_N_BINS} bins on [0, 1.5] m/s\n"
            f"- Angle bins (roll/yaw/pitch): {ANGLE_N_BINS} bins on [0, 2π) rad\n"
        )
