"""MediaPipe pose backend placeholder."""

from __future__ import annotations

import time
from typing import Any

from baseball_pose.io.video import FrameRecord
from baseball_pose.pose.base import PoseEstimator
from baseball_pose.pose.schema import PoseRecord


MEDIAPIPE_TO_CANONICAL = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


class MediaPipePoseEstimator(PoseEstimator):
    backend_name = "mediapipe"

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MediaPipe is required for pose estimation. Install project dependencies first."
            ) from exc

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate_frame(self, image: Any, frame: FrameRecord, condition_id: str) -> list[PoseRecord]:
        cv2 = _require_cv2()
        start = time.perf_counter()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        inference_time_ms = (time.perf_counter() - start) * 1000

        records: list[PoseRecord] = []
        landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
        for index, joint_name in MEDIAPIPE_TO_CANONICAL.items():
            landmark = landmarks[index] if landmarks else None
            records.append(
                PoseRecord(
                    clip_id=frame.clip_id,
                    condition_id=condition_id,
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    joint_name=joint_name,
                    x=landmark.x if landmark else None,
                    y=landmark.y if landmark else None,
                    visibility=landmark.visibility if landmark else None,
                    confidence=landmark.presence if landmark else None,
                    backend=self.backend_name,
                    inference_time_ms=inference_time_ms,
                )
            )
        return records

    def close(self) -> None:
        self._pose.close()


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for pose estimation. Install project dependencies first."
        ) from exc

    return cv2
