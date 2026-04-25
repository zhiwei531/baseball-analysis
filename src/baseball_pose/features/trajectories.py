"""Trajectory feature boundaries."""

from __future__ import annotations

from baseball_pose.pose.schema import PoseRecord


def select_joint_trajectory(records: list[PoseRecord], joint_name: str) -> list[PoseRecord]:
    return [record for record in records if record.joint_name == joint_name]
