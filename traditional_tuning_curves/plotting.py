from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from glm_poisson_forward.config import ANGLE_N_BINS, POSITION_CELL_CM, SPEED_N_BINS


def _plot_rate_map(ax, rate_map: np.ndarray) -> None:
    display_map = np.nan_to_num(rate_map, nan=0.0)
    im = ax.imshow(display_map, origin="lower", cmap="viridis", vmin=0.0)
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
    r = np.nan_to_num(r, nan=0.0)
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


def _plot_single_rate_map(out_path: Path, rate_map: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    _plot_rate_map(ax, rate_map)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_single_speed(out_path: Path, speed_curve: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    _plot_speed(ax, speed_curve)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_neuron_summary(
    out_dir: Path,
    neuron_idx: int,
    score_dict: Dict[str, float],
    aux: Dict[str, np.ndarray],
) -> None:
    theta_deg = np.linspace(0.0, 360.0, ANGLE_N_BINS, endpoint=False)
    title_bits = (
        f"Neuron {neuron_idx} | grid={score_dict.get('grid_score', float('nan')):.3f}, "
        f"border={score_dict.get('border_score', float('nan')):.3f}"
    )
    _plot_single_rate_map(
        out_dir / "position" / f"neuron_{neuron_idx:03d}.png",
        aux["rate_map"],
        title_bits,
    )
    _plot_single_speed(
        out_dir / "speed" / f"neuron_{neuron_idx:03d}.png",
        aux["speed_curve"],
        f"Neuron {neuron_idx} | speed tuning",
    )

    plot_polar_curve(
        out_dir / "yaw" / f"neuron_{neuron_idx:03d}.png",
        theta_deg,
        aux["hd_curve"],
        f"Neuron {neuron_idx} | yaw tuning",
        color="#1f77b4",
    )
    plot_polar_curve(
        out_dir / "roll" / f"neuron_{neuron_idx:03d}.png",
        theta_deg,
        aux["roll_curve"],
        f"Neuron {neuron_idx} | roll tuning",
        color="#ff7f0e",
    )
    plot_polar_curve(
        out_dir / "pitch" / f"neuron_{neuron_idx:03d}.png",
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
