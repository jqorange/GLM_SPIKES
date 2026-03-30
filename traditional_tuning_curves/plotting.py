from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from glm_poisson_forward.config import VARIABLE_SPECS

from .config import (
    PITCH_N_BINS,
    ROLL_N_BINS,
    ROLL_PITCH_TRIM_PERCENTILES,
    SPEED_MAX_M_S,
    SPEED_MIN_M_S,
    YAW_N_BINS,
)
from .tuning_scores import AngleBinningRanges

POSITION_CELL_CM = float(VARIABLE_SPECS.get("Position", {}).get("cell_cm", 8.0))
SPEED_N_BINS = int(VARIABLE_SPECS.get("Speed", {}).get("n_bins", 15))


def _plot_speed(ax, speed_curve: np.ndarray) -> None:
    edges = np.linspace(SPEED_MIN_M_S, SPEED_MAX_M_S, SPEED_N_BINS + 1)
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
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(theta_c, r_c, linewidth=2, color=color)
    ax.fill(theta_c, r_c, alpha=0.22, color=color)
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_single_speed(out_path: Path, speed_curve: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    _plot_speed(ax, speed_curve)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_paired_speed_curve(
    out_path: Path,
    indoor_curve: np.ndarray,
    outdoor_curve: np.ndarray,
    title: str,
    indoor_color: str = "#1f77b4",
    outdoor_color: str = "#d62728",
) -> None:
    edges = np.linspace(SPEED_MIN_M_S, SPEED_MAX_M_S, SPEED_N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indoor = np.nan_to_num(indoor_curve, nan=0.0)
    outdoor = np.nan_to_num(outdoor_curve, nan=0.0)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot(centers, indoor, linewidth=2, color=indoor_color, label="indoor")
    ax.plot(centers, outdoor, linewidth=2, color=outdoor_color, label="outdoor")
    ax.fill_between(centers, indoor, alpha=0.15, color=indoor_color)
    ax.fill_between(centers, outdoor, alpha=0.15, color=outdoor_color)
    ax.set_title(title)
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Rate (Hz)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_neuron_summary(
    out_dir: Path,
    neuron_idx: int,
    aux: Dict[str, np.ndarray],
) -> None:
    title_bits = f"Neuron {neuron_idx}"
    _plot_single_speed(
        out_dir / "speed.png",
        aux["speed_curve"],
        f"Neuron {neuron_idx} | speed tuning",
    )

    plot_polar_curve(
        out_dir / "yaw.png",
        np.linspace(0.0, 360.0, len(aux["hd_curve"]), endpoint=False),
        aux["hd_curve"],
        f"Neuron {neuron_idx} | yaw tuning",
        color="#1f77b4",
    )
    plot_polar_curve(
        out_dir / "roll.png",
        np.linspace(0.0, 360.0, len(aux["roll_curve"]), endpoint=False),
        aux["roll_curve"],
        f"Neuron {neuron_idx} | roll tuning",
        color="#ff7f0e",
    )
    plot_polar_curve(
        out_dir / "pitch.png",
        np.linspace(0.0, 360.0, len(aux["pitch_curve"]), endpoint=False),
        aux["pitch_curve"],
        f"Neuron {neuron_idx} | pitch tuning",
        color="#2ca02c",
    )


def _equalize_polar_area(
    theta_rad: np.ndarray, indoor: np.ndarray, outdoor: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    dtheta = float(np.mean(np.diff(np.concatenate([theta_rad, theta_rad[:1] + 2 * np.pi]))))
    indoor_area = 0.5 * float(np.sum(indoor**2) * dtheta)
    outdoor_area = 0.5 * float(np.sum(outdoor**2) * dtheta)
    target_area = 0.5 * (indoor_area + outdoor_area)
    indoor_scale = np.sqrt(target_area / indoor_area) if indoor_area > 0 else 1.0
    outdoor_scale = np.sqrt(target_area / outdoor_area) if outdoor_area > 0 else 1.0
    return indoor * indoor_scale, outdoor * outdoor_scale


def plot_paired_polar_curve(
    out_path: Path,
    theta_deg: np.ndarray,
    indoor_curve: np.ndarray,
    outdoor_curve: np.ndarray,
    title: str,
    indoor_color: str = "#1f77b4",
    outdoor_color: str = "#d62728",
    equalize_area: bool = True,
) -> None:
    theta_rad = np.deg2rad(theta_deg)
    indoor = np.nan_to_num(indoor_curve, nan=0.0)
    outdoor = np.nan_to_num(outdoor_curve, nan=0.0)
    if equalize_area:
        indoor, outdoor = _equalize_polar_area(theta_rad, indoor, outdoor)
    theta_c = np.concatenate([theta_rad, theta_rad[:1]])
    indoor_c = np.concatenate([indoor, indoor[:1]])
    outdoor_c = np.concatenate([outdoor, outdoor[:1]])

    plt.figure(figsize=(6.0, 6.0))
    ax = plt.subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(theta_c, indoor_c, linewidth=2, color=indoor_color, label="indoor")
    ax.plot(theta_c, outdoor_c, linewidth=2, color=outdoor_color, label="outdoor")
    ax.fill(theta_c, indoor_c, alpha=0.15, color=indoor_color)
    ax.fill(theta_c, outdoor_c, alpha=0.15, color=outdoor_color)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.15))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def binning_note(out_path: Path, *, angle_ranges: AngleBinningRanges | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lower_pct, upper_pct = ROLL_PITCH_TRIM_PERCENTILES
    roll_line = f"- Roll bins: {ROLL_N_BINS} bins"
    pitch_line = f"- Pitch bins: {PITCH_N_BINS} bins"
    if angle_ranges is not None:
        roll_min = angle_ranges.roll_start
        roll_max = angle_ranges.roll_start + angle_ranges.roll_width
        pitch_min = angle_ranges.pitch_start
        pitch_max = angle_ranges.pitch_start + angle_ranges.pitch_width
        roll_line += f" on [{roll_min:.3f}, {roll_max:.3f}) rad (trim {lower_pct:g}–{upper_pct:g} pct)"
        pitch_line += f" on [{pitch_min:.3f}, {pitch_max:.3f}) rad (trim {lower_pct:g}–{upper_pct:g} pct)"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            "Traditional tuning curves use these binning rules:\n"
            f"- Position bins: {POSITION_CELL_CM:.1f} cm square\n"
            f"- Speed bins: {SPEED_N_BINS} bins on [0, 1.5] m/s\n"
            f"- Yaw bins: {YAW_N_BINS} bins on [0, 2π) rad\n"
            f"{roll_line}\n"
            f"{pitch_line}\n"
        )
