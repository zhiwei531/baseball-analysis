"""Pose overlay rendering boundary."""

from __future__ import annotations

from typing import Any

from baseball_pose.pose.schema import PoseRecord


def draw_pose_overlay(image: Any, records: list[PoseRecord]) -> Any:
    raise NotImplementedError("Overlay rendering will be implemented after pose inference.")
