"""MediaPipe pose backend."""

from __future__ import annotations

import os
import time
from pathlib import Path
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
        model_asset_path: str | Path = "models/pose_landmarker_lite.task",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "MediaPipe is required for pose estimation. Install project dependencies first."
            ) from exc

        model_path = Path(model_asset_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe pose model not found: {model_path}. "
                "Download pose_landmarker_lite.task into models/ before running pose estimation."
            )

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._pose = vision.PoseLandmarker.create_from_options(options)

    def estimate_frame(self, image: Any, frame: FrameRecord, condition_id: str) -> list[PoseRecord]:
        cv2 = _require_cv2()
        start = time.perf_counter()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._pose.detect(mp_image)
        inference_time_ms = (time.perf_counter() - start) * 1000

        records: list[PoseRecord] = []
        landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
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
                    visibility=getattr(landmark, "visibility", None) if landmark else None,
                    confidence=getattr(landmark, "presence", None) if landmark else None,
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
