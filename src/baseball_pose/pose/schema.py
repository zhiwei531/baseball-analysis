"""Common pose output schema."""

from __future__ import annotations

from dataclasses import dataclass


CANONICAL_JOINTS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

MEDIAPIPE_JOINTS = (
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)

HALPE26_JOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
    "neck",
    "hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
)

COCO17_JOINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

SUPPORTED_JOINTS = tuple(dict.fromkeys((*MEDIAPIPE_JOINTS, *HALPE26_JOINTS, *COCO17_JOINTS)))

POSE_CONNECTIONS = (
    ("nose", "left_eye_inner"),
    ("left_eye_inner", "left_eye"),
    ("left_eye", "left_eye_outer"),
    ("left_eye_outer", "left_ear"),
    ("nose", "right_eye_inner"),
    ("right_eye_inner", "right_eye"),
    ("right_eye", "right_eye_outer"),
    ("right_eye_outer", "right_ear"),
    ("mouth_left", "mouth_right"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_pinky"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_thumb"),
    ("left_pinky", "left_index"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_pinky"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_thumb"),
    ("right_pinky", "right_index"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
    ("head", "neck"),
    ("neck", "hip"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_hip", "hip"),
    ("right_hip", "hip"),
    ("left_ankle", "left_big_toe"),
    ("left_ankle", "left_small_toe"),
    ("right_ankle", "right_big_toe"),
    ("right_ankle", "right_small_toe"),
)


@dataclass(frozen=True)
class PoseRecord:
    clip_id: str
    condition_id: str
    frame_index: int
    timestamp_sec: float
    joint_name: str
    x: float | None
    y: float | None
    visibility: float | None
    confidence: float | None
    backend: str
    inference_time_ms: float | None = None


def pose_score(record: PoseRecord) -> float | None:
    """Return a conservative trust score for one pose record.

    MediaPipe can emit high presence/confidence even when a joint is heavily
    occluded. For downstream filtering and rendering, visibility is therefore
    the safer signal. When both values are available, use the lower one.
    """

    if record.visibility is not None and record.confidence is not None:
        return min(record.visibility, record.confidence)
    if record.visibility is not None:
        return record.visibility
    return record.confidence


def validate_joint_name(joint_name: str) -> None:
    if joint_name not in SUPPORTED_JOINTS:
        raise ValueError(f"Unsupported joint name: {joint_name}")
