"""Confidence filtering helpers."""

from __future__ import annotations

from baseball_pose.pose.quality import threshold_for_joint
from baseball_pose.pose.schema import PoseRecord, pose_score


def is_confident(
    record: PoseRecord,
    threshold: float,
    threshold_config: dict[str, object] | None = None,
) -> bool:
    score = pose_score(record)
    joint_threshold = threshold_for_joint(record.joint_name, threshold, threshold_config)
    return score is not None and score >= joint_threshold
