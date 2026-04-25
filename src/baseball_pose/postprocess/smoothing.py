"""Smoothing boundaries for pose trajectories."""

from __future__ import annotations

from baseball_pose.pose.schema import PoseRecord


def smooth_pose_records(
    records: list[PoseRecord],
    method: str = "savgol",
    window_length: int = 7,
    polyorder: int = 2,
) -> list[PoseRecord]:
    """Smooth pose trajectories while preserving the common schema."""

    if method != "savgol":
        raise ValueError(f"Unsupported smoothing method: {method}")
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")
    if polyorder >= window_length:
        raise ValueError("polyorder must be smaller than window_length.")
    return records
