"""Reusable visualization utilities (12 September 2026 task, Rajashree workstream).

ADDITIVE module. Centralizes all plotting so the end-to-end notebook calls
small functions instead of duplicating matplotlib logic. No data is modified
for visualization; legend entries only cover classes actually present unless
the caller explicitly passes the full project taxonomy.
"""
from __future__ import annotations

from typing import Dict, Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

PROJECT_CLASS_COLORS: Dict[str, str] = {
    "vehicle": "#d62728",
    "pedestrian_vru": "#ff7f0e",
    "static_manmade": "#7f7f7f",
    "vegetation": "#2ca02c",
    "road_driveable": "#1f77b4",
    "unknown": "#bcbd22",
}


def _resolve_ax(ax, figsize=(9, 8)):
    if ax is not None:
        return ax, False
    fig, ax = plt.subplots(figsize=figsize)
    return ax, True


def plot_lidar_xy(points, title="LiDAR XY View", ax=None, **kwargs):
    """Scatter X vs Y of N x 4 points (intensity-coloured when available)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 4:
        raise ValueError(f"points must be N x 4. Received shape {pts.shape!r}")
    ax, _ = _resolve_ax(ax)
    c = pts[:, 3] if np.all(np.isfinite(pts[:, 3])) else None
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=1, c=c, cmap="viridis", **kwargs)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    if c is not None:
        plt.colorbar(sc, ax=ax, label="Intensity")
    ax.grid(True, alpha=0.3)
    return ax


def plot_semantic_regions(region_df, title="Semantic Map", ax=None):
    """XY scatter of region centroids coloured by semantic class present."""
    ax, _ = _resolve_ax(ax)
    for label, group in region_df.groupby("semantic_class"):
        ax.scatter(
            group["x"], group["y"], s=20,
            color=PROJECT_CLASS_COLORS.get(str(label), "#9467bd"),
            label=f"{label} ({len(group)})", alpha=0.85,
        )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(markerscale=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    return ax


def plot_importance_map(region_df, title="Region Importance — Real nuScenes Frame", ax=None):
    """XY scatter of region centroids coloured by importance [0,1]."""
    ax, _ = _resolve_ax(ax)
    sc = ax.scatter(region_df["x"], region_df["y"], c=region_df["importance"], s=20,
                    vmin=0.0, vmax=1.0, cmap="plasma")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    plt.colorbar(sc, ax=ax, label="Importance")
    return ax


def plot_resolution_map(region_df, title="Adaptive 2.5D Resolution Map", ax=None):
    """XY scatter of region centroids coloured by resolution (m)."""
    ax, _ = _resolve_ax(ax)
    sc = ax.scatter(region_df["x"], region_df["y"], c=region_df["resolution"], s=20,
                    cmap="viridis_r")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    plt.colorbar(sc, ax=ax, label="Resolution (m)")
    return ax


def plot_elevation_map(region_df, title="Adaptive 2.5D Elevation Map", ax=None):
    """XY scatter of region centroids coloured by elevation (m).

    This view demonstrates elevation-aware mapping rather than a purely
    2D occupancy representation.
    """
    ax, _ = _resolve_ax(ax)
    sc = ax.scatter(region_df["x"], region_df["y"], c=region_df["elevation"], s=20,
                    cmap="terrain")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    plt.colorbar(sc, ax=ax, label="Elevation (m)")
    return ax


def plot_combined_demo(points, region_df, suptitle="End-to-End Adaptive 2.5D Prototype"):
    """Four-view demo: raw/processed LiDAR | semantic | importance | resolution."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    plot_lidar_xy(points, title="Real nuScenes LiDAR Input", ax=axes[0, 0])
    plot_semantic_regions(region_df, title="Semantic Map", ax=axes[0, 1])
    plot_importance_map(region_df, title="Region Importance", ax=axes[1, 0])
    plot_resolution_map(region_df, title="Adaptive 2.5D Resolution Map", ax=axes[1, 1])
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    return fig


def available_backend() -> str:
    """Return the active matplotlib backend (diagnostic helper)."""
    return matplotlib.get_backend()
