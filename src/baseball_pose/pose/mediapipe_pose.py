"""MediaPipe pose backend placeholder."""

from __future__ import annotations

from typing import Any

from baseball_pose.io.video import FrameRecord
from baseball_pose.pose.base import PoseEstimator
from baseball_pose.pose.schema import PoseRecord


class MediaPipePoseEstimator(PoseEstimator):
    backend_name = "mediapipe"

    def estimate_frame(self, image: Any, frame: FrameRecord, condition_id: str) -> list[PoseRecord]:
        raise NotImplementedError("MediaPipe inference will be implemented in Phase 2.")
