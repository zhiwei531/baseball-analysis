"""MediaPipe world-landmark backend for relative 3D skeleton output."""

from __future__ import annotations

import time
from typing import Any

from baseball_pose.io.video import FrameRecord
from baseball_pose.pose.mediapipe_pose import MEDIAPIPE_TO_CANONICAL, MediaPipePoseEstimator, _require_cv2
from baseball_pose.pose3d.schema import Pose3DRecord


class MediaPipeWorldPoseEstimator(MediaPipePoseEstimator):
    """Expose MediaPipe Pose world landmarks as relative 3D joint records."""

    backend_name = "mediapipe_world"

    def estimate_frame_3d(
        self,
        image: Any,
        frame: FrameRecord,
        condition_id: str,
    ) -> list[Pose3DRecord]:
        start = time.perf_counter()
        cv2 = _require_cv2()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._pose.detect(mp_image)
        _ = (time.perf_counter() - start) * 1000

        pose_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
        world_landmarks = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        records: list[Pose3DRecord] = []
        for index, joint_name in MEDIAPIPE_TO_CANONICAL.items():
            world = world_landmarks[index] if world_landmarks else None
            landmark = pose_landmarks[index] if pose_landmarks else None
            if world is None:
                continue
            visibility = getattr(landmark, "visibility", None) if landmark else None
            confidence = getattr(landmark, "presence", None) if landmark else None
            quality = None
            if visibility is not None and confidence is not None:
                quality = min(float(visibility), float(confidence))
            elif visibility is not None:
                quality = float(visibility)
            elif confidence is not None:
                quality = float(confidence)
            records.append(
                Pose3DRecord(
                    clip_id=frame.clip_id,
                    condition_id=condition_id,
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    joint_name=joint_name,
                    x_3d=float(world.x),
                    y_3d=float(world.y),
                    z_3d=float(world.z),
                    scale_mode="mediapipe_world",
                    lift_backend=self.backend_name,
                    input_quality_score=quality,
                )
            )
        return records
