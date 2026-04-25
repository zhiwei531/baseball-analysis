"""Pose estimator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from baseball_pose.io.video import FrameRecord
from baseball_pose.pose.schema import PoseRecord


class PoseEstimator(ABC):
    backend_name: str

    @abstractmethod
    def estimate_frame(self, image: Any, frame: FrameRecord, condition_id: str) -> list[PoseRecord]:
        """Estimate pose records for one frame."""
