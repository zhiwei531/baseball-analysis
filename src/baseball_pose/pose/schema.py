"""Common pose output schema."""

from __future__ import annotations

from dataclasses import dataclass


CANONICAL_JOINTS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class PoseRecord:
    clip_id: str
    condition_id: str
    frame_index: int
    timestamp_sec: float
    joint_name: str
    x: float | None
    y: float | None
    visibility: float | None
    confidence: float | None
    backend: str
    inference_time_ms: float | None = None


def validate_joint_name(joint_name: str) -> None:
    if joint_name not in CANONICAL_JOINTS:
        raise ValueError(f"Unsupported joint name: {joint_name}")
