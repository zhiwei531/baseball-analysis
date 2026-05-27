"""Pose-quality helpers shared across filtering, smoothing, and reporting."""

from __future__ import annotations

from typing import Any


TORSO_JOINTS = {"nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip"}
DISTAL_JOINTS = {"left_wrist", "right_wrist", "left_ankle", "right_ankle"}
MID_LIMB_JOINTS = {"left_elbow", "right_elbow", "left_knee", "right_knee"}

LIMB_SEGMENTS = (
    ("left_shoulder", "left_elbow", "left_elbow"),
    ("right_shoulder", "right_elbow", "right_elbow"),
    ("left_elbow", "left_wrist", "left_wrist"),
    ("right_elbow", "right_wrist", "right_wrist"),
    ("left_hip", "left_knee", "left_knee"),
    ("right_hip", "right_knee", "right_knee"),
    ("left_knee", "left_ankle", "left_ankle"),
    ("right_knee", "right_ankle", "right_ankle"),
)


def joint_group(joint_name: str) -> str:
    if joint_name in TORSO_JOINTS:
        return "torso"
    if joint_name in DISTAL_JOINTS:
        return "distal"
    if joint_name in MID_LIMB_JOINTS:
        return "mid_limb"
    return "default"


def threshold_for_joint(
    joint_name: str,
    default_threshold: float,
    config: dict[str, Any] | None = None,
) -> float:
    if not config:
        return float(default_threshold)
    overrides = config.get("joint_overrides", {})
    if isinstance(overrides, dict) and joint_name in overrides:
        return float(overrides[joint_name])
    group = joint_group(joint_name)
    if group in config:
        return float(config[group])
    return float(config.get("default", default_threshold))


def gap_for_joint(
    joint_name: str,
    default_gap: int,
    config: dict[str, Any] | None = None,
) -> int:
    if not config:
        return int(default_gap)
    overrides = config.get("joint_overrides", {})
    if isinstance(overrides, dict) and joint_name in overrides:
        return int(overrides[joint_name])
    group = joint_group(joint_name)
    if group in config:
        return int(config[group])
    return int(config.get("default", default_gap))


def multiplier_for_joint(
    joint_name: str,
    default_multiplier: float,
    config: dict[str, Any] | None = None,
) -> float:
    if not config:
        return float(default_multiplier)
    overrides = config.get("joint_overrides", {})
    if isinstance(overrides, dict) and joint_name in overrides:
        return float(overrides[joint_name])
    group = joint_group(joint_name)
    if group in config:
        return float(config[group])
    return float(config.get("default", default_multiplier))
