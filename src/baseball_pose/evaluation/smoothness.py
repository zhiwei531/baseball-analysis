"""Trajectory smoothness metric boundaries."""

from __future__ import annotations

from baseball_pose.pose.schema import PoseRecord


def second_difference_smoothness(records: list[PoseRecord], joint_name: str) -> float | None:
    trajectory = [
        record for record in sorted(records, key=lambda item: item.frame_index)
        if record.joint_name == joint_name and record.x is not None and record.y is not None
    ]
    if len(trajectory) < 3:
        return None
    total = 0.0
    count = 0
    for first, second, third in zip(trajectory, trajectory[1:], trajectory[2:]):
        ddx = third.x - 2 * second.x + first.x
        ddy = third.y - 2 * second.y + first.y
        total += (ddx * ddx + ddy * ddy) ** 0.5
        count += 1
    return total / count
