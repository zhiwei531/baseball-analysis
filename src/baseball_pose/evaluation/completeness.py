"""Keypoint completeness metrics."""

from __future__ import annotations

from baseball_pose.pose.schema import PoseRecord


def keypoint_completeness(records: list[PoseRecord], required_joints: set[str]) -> float:
    if not required_joints:
        raise ValueError("required_joints cannot be empty.")
    present = {
        record.joint_name
        for record in records
        if record.joint_name in required_joints and record.x is not None and record.y is not None
    }
    return len(present) / len(required_joints)
