"""Optional RTMPose backend boundary."""

from __future__ import annotations

from typing import Any

from baseball_pose.io.video import FrameRecord
from baseball_pose.pose.base import PoseEstimator
from baseball_pose.pose.schema import PoseRecord


class RTMPoseEstimator(PoseEstimator):
    backend_name = "rtmpose"

    def estimate_frame(self, image: Any, frame: FrameRecord, condition_id: str) -> list[PoseRecord]:
        raise NotImplementedError("RTMPose is a stretch backend and is not part of the MVP.")
