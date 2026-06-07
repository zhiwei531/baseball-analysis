"""Pose backend factory."""

from __future__ import annotations

from typing import Any

from baseball_pose.pose.base import PoseEstimator
from baseball_pose.pose.mediapipe_pose import MediaPipePoseEstimator
from baseball_pose.pose.rtmpose_pose import RTMPoseEstimator


def create_pose_estimator(raw_config: dict[str, Any]) -> PoseEstimator:
    pose_config = raw_config.get("pose", {})
    backend = str(pose_config.get("backend", "mediapipe")).lower()
    if backend == "mediapipe":
        return MediaPipePoseEstimator(
            model_asset_path=pose_config.get(
                "model_asset_path",
                "models/pose_landmarker_heavy.task",
            ),
            min_detection_confidence=float(pose_config.get("min_detection_confidence", 0.5)),
            min_tracking_confidence=float(pose_config.get("min_tracking_confidence", 0.5)),
        )
    if backend == "rtmpose":
        model_input_size = pose_config.get("model_input_size")
        if isinstance(model_input_size, list):
            model_input_size = tuple(int(value) for value in model_input_size)
        body_model_input_size = pose_config.get("body_model_input_size")
        if isinstance(body_model_input_size, list):
            body_model_input_size = tuple(int(value) for value in body_model_input_size)
        return RTMPoseEstimator(
            mode=str(pose_config.get("mode", "performance")),
            backend=str(pose_config.get("runtime_backend", "onnxruntime")),
            device=str(pose_config.get("device", "cpu")),
            model_url=pose_config.get("model_url"),
            model_input_size=model_input_size,
            body_model_url=pose_config.get("body_model_url"),
            body_model_input_size=body_model_input_size,
            source_keypoint_schema=str(
                pose_config.get("source_keypoint_schema", pose_config.get("keypoint_schema", "halpe26"))
            ),
            keypoint_schema=str(pose_config.get("keypoint_schema", "halpe26")),
            min_keypoint_score=float(pose_config.get("min_keypoint_score", 0.0)),
        )
    raise ValueError(f"Unsupported pose backend: {backend}")


def pose_prefers_unmasked_input(raw_config: dict[str, Any]) -> bool:
    pose_config = raw_config.get("pose", {})
    if "use_masked_input" in pose_config:
        return not bool(pose_config["use_masked_input"])
    return str(pose_config.get("backend", "mediapipe")).lower() == "rtmpose"
