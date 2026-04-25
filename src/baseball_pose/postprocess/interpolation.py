"""Interpolation boundaries for short missing keypoint gaps."""

from __future__ import annotations

from baseball_pose.pose.schema import PoseRecord


def interpolate_short_gaps(
    records: list[PoseRecord],
    max_gap_frames: int,
) -> list[PoseRecord]:
    """Return records with short missing gaps interpolated.

    Detailed interpolation is planned for Phase 4.
    """

    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative.")
    return records
