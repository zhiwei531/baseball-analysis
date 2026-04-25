"""Runtime aggregation helpers."""

from __future__ import annotations

from baseball_pose.pose.schema import PoseRecord


def mean_inference_time_ms(records: list[PoseRecord]) -> float | None:
    values = [record.inference_time_ms for record in records if record.inference_time_ms is not None]
    if not values:
        return None
    return sum(values) / len(values)
