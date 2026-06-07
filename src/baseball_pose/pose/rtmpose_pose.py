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
        body_model_url: str | None = None,
        body_model_input_size: tuple[int, int] | None = None,
        source_keypoint_schema: str = "halpe26",
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
        self._body_pose = (
            RTMPose(
                body_model_url,
                model_input_size=body_model_input_size or (288, 384),
                to_openpose=False,
                backend=backend,
                device=device,
            )
            if body_model_url
            else None
        )
        self._source_keypoint_schema = source_keypoint_schema
        self._keypoint_schema = keypoint_schema
        if keypoint_schema == "halpe26":
            self._joint_names = HALPE26_JOINTS
        elif keypoint_schema == "coco17":
            self._joint_names = COCO17_JOINTS
        else:
            raise ValueError(f"Unsupported RTMPose keypoint schema: {keypoint_schema}")
        if source_keypoint_schema not in {"halpe26", "coco17", "coco_wholebody133"}:
            raise ValueError(f"Unsupported RTMPose source keypoint schema: {source_keypoint_schema}")
        if keypoint_schema != source_keypoint_schema and (
            source_keypoint_schema,
            keypoint_schema,
        ) != ("coco_wholebody133", "halpe26"):
            raise ValueError(
                "RTMPose keypoint schema conversion is only supported from "
                "coco_wholebody133 to halpe26."
            )
        self._min_keypoint_score = float(min_keypoint_score)

    def estimate_frame(self, image: Any, frame: FrameRecord, condition_id: str) -> list[PoseRecord]:
        height, width = image.shape[:2]
        start = time.perf_counter()
        keypoints, scores = self._pose(image, bboxes=[[0, 0, width, height]])
        body_keypoints = body_scores = None
        if self._body_pose is not None:
            body_keypoints, body_scores = self._body_pose(image, bboxes=[[0, 0, width, height]])
        inference_time_ms = (time.perf_counter() - start) * 1000

        if len(keypoints) == 0:
            return [
                self._empty_record(frame, condition_id, joint_name, inference_time_ms)
                for joint_name in self._joint_names
            ]

        instance_index = _best_instance_index(scores)
        instance_keypoints = keypoints[instance_index]
        instance_scores = scores[instance_index]
        if (self._source_keypoint_schema, self._keypoint_schema) == (
            "coco_wholebody133",
            "halpe26",
        ):
            if self._body_pose is not None and body_keypoints is not None and len(body_keypoints) > 0:
                body_instance_index = _best_instance_index(body_scores)
                instance_keypoints, instance_scores = _fuse_coco17_body_with_wholebody_feet(
                    body_keypoints[body_instance_index],
                    body_scores[body_instance_index],
                    instance_keypoints,
                    instance_scores,
                )
            else:
                instance_keypoints, instance_scores = _coco_wholebody133_to_halpe26(
                    instance_keypoints, instance_scores
                )

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


def _coco_wholebody133_to_halpe26(keypoints: Any, scores: Any) -> tuple[list[list[float]], list[float]]:
    """Project COCO-WholeBody output to the Halpe26 body-with-feet schema."""

    def point(index: int) -> list[float]:
        return [float(keypoints[index][0]), float(keypoints[index][1])]

    def score(index: int) -> float:
        return float(scores[index])

    def midpoint(left_index: int, right_index: int) -> list[float]:
        left = point(left_index)
        right = point(right_index)
        return [(left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0]

    def mean_score(left_index: int, right_index: int) -> float:
        return (score(left_index) + score(right_index)) / 2.0

    body = [point(index) for index in range(17)]
    body_scores = [score(index) for index in range(17)]

    head = point(0)
    head_score = score(0)
    neck = midpoint(5, 6)
    neck_score = mean_score(5, 6)
    hip = midpoint(11, 12)
    hip_score = mean_score(11, 12)

    # COCO-WholeBody foot order: left toe/small toe/heel, then right toe/small toe/heel.
    foot_order = [17, 20, 18, 21, 19, 22]
    feet = [point(index) for index in foot_order]
    feet_scores = [score(index) for index in foot_order]

    return (
        [*body, head, neck, hip, *feet],
        [*body_scores, head_score, neck_score, hip_score, *feet_scores],
    )


def _fuse_coco17_body_with_wholebody_feet(
    body_keypoints: Any,
    body_scores: Any,
    wholebody_keypoints: Any,
    wholebody_scores: Any,
) -> tuple[list[list[float]], list[float]]:
    """Use a stable COCO17 body model and only take feet from COCO-WholeBody."""

    def body_point(index: int) -> list[float]:
        return [float(body_keypoints[index][0]), float(body_keypoints[index][1])]

    def body_score(index: int) -> float:
        return float(body_scores[index])

    def wholebody_point(index: int) -> list[float]:
        return [float(wholebody_keypoints[index][0]), float(wholebody_keypoints[index][1])]

    def wholebody_score(index: int) -> float:
        return float(wholebody_scores[index])

    def body_midpoint(left_index: int, right_index: int) -> list[float]:
        left = body_point(left_index)
        right = body_point(right_index)
        return [(left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0]

    def body_mean_score(left_index: int, right_index: int) -> float:
        return (body_score(left_index) + body_score(right_index)) / 2.0

    body = [body_point(index) for index in range(17)]
    body_confidences = [body_score(index) for index in range(17)]
    head = body_point(0)
    head_score = body_score(0)
    neck = body_midpoint(5, 6)
    neck_score = body_mean_score(5, 6)
    hip = body_midpoint(11, 12)
    hip_score = body_mean_score(11, 12)
    left_feet, left_foot_scores = _stable_side_foot_points(
        ankle=body_point(15),
        knee=body_point(13),
        ankle_score=body_score(15),
        wholebody_points=[wholebody_point(index) for index in (17, 18, 19)],
        wholebody_scores=[wholebody_score(index) for index in (17, 18, 19)],
    )
    right_feet, right_foot_scores = _stable_side_foot_points(
        ankle=body_point(16),
        knee=body_point(14),
        ankle_score=body_score(16),
        wholebody_points=[wholebody_point(index) for index in (20, 21, 22)],
        wholebody_scores=[wholebody_score(index) for index in (20, 21, 22)],
    )
    feet = [
        left_feet[0],
        right_feet[0],
        left_feet[1],
        right_feet[1],
        left_feet[2],
        right_feet[2],
    ]
    feet_scores = [
        left_foot_scores[0],
        right_foot_scores[0],
        left_foot_scores[1],
        right_foot_scores[1],
        left_foot_scores[2],
        right_foot_scores[2],
    ]
    return (
        [*body, head, neck, hip, *feet],
        [*body_confidences, head_score, neck_score, hip_score, *feet_scores],
    )


def _stable_side_foot_points(
    ankle: list[float],
    knee: list[float],
    ankle_score: float,
    wholebody_points: list[list[float]],
    wholebody_scores: list[float],
) -> tuple[list[list[float]], list[float]]:
    lower_leg_length = max(float(((ankle[0] - knee[0]) ** 2 + (ankle[1] - knee[1]) ** 2) ** 0.5), 1.0)
    ankle_distances = [
        float(((point[0] - ankle[0]) ** 2 + (point[1] - ankle[1]) ** 2) ** 0.5)
        for point in wholebody_points
    ]
    foot_spread = max(
        float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
        for index, a in enumerate(wholebody_points)
        for b in wholebody_points[index + 1 :]
    )
    max_distance = max(ankle_distances)
    plausible = max_distance <= lower_leg_length * 0.75 and foot_spread <= lower_leg_length * 0.55
    if plausible:
        return wholebody_points, wholebody_scores

    anchored = [[ankle[0], ankle[1]], [ankle[0], ankle[1]], [ankle[0], ankle[1]]]
    anchored_score = min(float(ankle_score), 0.35)
    return anchored, [anchored_score, anchored_score, anchored_score]
