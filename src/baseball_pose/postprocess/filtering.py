"""Confidence filtering helpers."""

from __future__ import annotations

from baseball_pose.pose.schema import PoseRecord


def is_confident(record: PoseRecord, threshold: float) -> bool:
    score = record.confidence if record.confidence is not None else record.visibility
    return score is not None and score >= threshold
