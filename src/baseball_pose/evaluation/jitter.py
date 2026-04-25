"""Temporal jitter metric boundaries."""

from __future__ import annotations

from baseball_pose.pose.schema import PoseRecord


def temporal_jitter(records: list[PoseRecord], joint_name: str) -> float | None:
    trajectory = [
        record for record in sorted(records, key=lambda item: item.frame_index)
        if record.joint_name == joint_name and record.x is not None and record.y is not None
    ]
    if len(trajectory) < 2:
        return None
    total = 0.0
    for previous, current in zip(trajectory, trajectory[1:]):
        dx = current.x - previous.x
        dy = current.y - previous.y
        total += (dx * dx + dy * dy) ** 0.5
    return total / (len(trajectory) - 1)
