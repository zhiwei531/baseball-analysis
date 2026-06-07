"""RTMPose backend using RTMLib ONNX inference."""

from __future__ import annotations

import time
from typing import Any

from baseball_pose.io.video import FrameRecord
from baseball_pose.pose.base import PoseEstimator
from baseball_pose.pose.schema import COCO17_JOINTS, HALPE26_JOINTS, PoseRecord


RTMPOSE_BODY_WITH_FEET_MODELS = {
    "performance": {
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip"
        ),
        "pose_input_size": (288, 384),
    },
    "balanced": {
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip"
        ),
        "pose_input_size": (192, 256),
    },
    "lightweight": {
        "pose": (
            "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
            "rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip"
        ),
        "pose_input_size": (192, 256),
    },
}


class RTMPoseEstimator(PoseEstimator):
    backend_name = "rtmpose"

    def __init__(
        self,
        mode: str = "performance",
        backend: str = "onnxruntime",
        device: str = "cpu",
        model_url: str | None = None,
        model_input_size: tuple[int, int] | None = None,
        keypoint_schema: str = "halpe26",
        min_keypoint_score: float = 0.0,
    ) -> None:
        try:
            from rtmlib import RTMPose
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "RTMPose backend requires rtmlib and onnxruntime. "
                "Install with `python -m pip install rtmlib onnxruntime`."
            ) from exc

        if mode not in RTMPOSE_BODY_WITH_FEET_MODELS:
            raise ValueError(f"Unsupported RTMPose mode: {mode}")
        model_config = RTMPOSE_BODY_WITH_FEET_MODELS[mode]
        pose_model = model_url or model_config["pose"]
        pose_input_size = model_input_size or model_config["pose_input_size"]
        self._pose = RTMPose(
            pose_model,
            model_input_size=pose_input_size,
            to_openpose=False,
            backend=backend,
            device=device,
        )
        if keypoint_schema == "halpe26":
            self._joint_names = HALPE26_JOINTS
        elif keypoint_schema == "coco17":
            self._joint_names = COCO17_JOINTS
        else:
            raise ValueError(f"Unsupported RTMPose keypoint schema: {keypoint_schema}")
        self._min_keypoint_score = float(min_keypoint_score)

    def estimate_frame(self, image: Any, frame: FrameRecord, condition_id: str) -> list[PoseRecord]:
        height, width = image.shape[:2]
        start = time.perf_counter()
        keypoints, scores = self._pose(image, bboxes=[[0, 0, width, height]])
        inference_time_ms = (time.perf_counter() - start) * 1000

        if len(keypoints) == 0:
            return [
                self._empty_record(frame, condition_id, joint_name, inference_time_ms)
                for joint_name in self._joint_names
            ]

        instance_index = _best_instance_index(scores)
        instance_keypoints = keypoints[instance_index]
        instance_scores = scores[instance_index]

        records: list[PoseRecord] = []
        for index, joint_name in enumerate(self._joint_names):
            x_px = float(instance_keypoints[index][0])
            y_px = float(instance_keypoints[index][1])
            score = float(instance_scores[index])
            has_coordinate = (
                score >= self._min_keypoint_score and 0 <= x_px < width and 0 <= y_px < height
            )
            records.append(
                PoseRecord(
                    clip_id=frame.clip_id,
                    condition_id=condition_id,
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    joint_name=joint_name,
                    x=x_px / width if has_coordinate else None,
                    y=y_px / height if has_coordinate else None,
                    visibility=score,
                    confidence=score,
                    backend=self.backend_name,
                    inference_time_ms=inference_time_ms,
                )
            )
        return records

    def close(self) -> None:
        return None

    def _empty_record(
        self,
        frame: FrameRecord,
        condition_id: str,
        joint_name: str,
        inference_time_ms: float,
    ) -> PoseRecord:
        return PoseRecord(
            clip_id=frame.clip_id,
            condition_id=condition_id,
            frame_index=frame.frame_index,
            timestamp_sec=frame.timestamp_sec,
            joint_name=joint_name,
            x=None,
            y=None,
            visibility=None,
            confidence=None,
            backend=self.backend_name,
            inference_time_ms=inference_time_ms,
        )


def _best_instance_index(scores: Any) -> int:
    if len(scores) == 1:
        return 0
    return max(range(len(scores)), key=lambda index: float(scores[index].mean()))
