"""Schema for relative 3D pose records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pose3DRecord:
    """One joint of one frame in a lifted 3D skeleton sequence."""

    clip_id: str
    condition_id: str
    frame_index: int
    timestamp_sec: float
    joint_name: str
    x_3d: float
    y_3d: float
    z_3d: float
    scale_mode: str
    lift_backend: str
    input_quality_score: float | None = None
