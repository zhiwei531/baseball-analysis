"""Frame-level motion feature extraction from pose landmarks."""

from __future__ import annotations

from dataclasses import dataclass
import math

from baseball_pose.features.angles import (
    angle_degrees,
    segment_angle_degrees,
    signed_angle_delta_degrees,
)
from baseball_pose.pose.schema import PoseRecord


ANGLE_DEFINITIONS = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder_angle": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder_angle": ("right_elbow", "right_shoulder", "right_hip"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
}
WRIST_JOINTS = ("left_wrist", "right_wrist")


@dataclass(frozen=True)
class MotionFeatureRow:
    clip_id: str
    condition_id: str
    frame_index: int
    timestamp_sec: float
    left_elbow_angle: float | None
    right_elbow_angle: float | None
    left_shoulder_angle: float | None
    right_shoulder_angle: float | None
    left_knee_angle: float | None
    right_knee_angle: float | None
    trunk_tilt_deg: float | None
    pelvis_rotation_deg: float | None
    shoulder_rotation_deg: float | None
    hip_shoulder_separation_deg: float | None
    pelvis_rotation_velocity_deg_s: float | None
    trunk_rotation_velocity_deg_s: float | None
    center_of_mass_x: float | None
    center_of_mass_y: float | None
    left_knee_extension_from_start_deg: float | None
    right_knee_extension_from_start_deg: float | None
    left_knee_angular_velocity_deg_s: float | None
    right_knee_angular_velocity_deg_s: float | None
    left_wrist_x: float | None
    left_wrist_y: float | None
    right_wrist_x: float | None
    right_wrist_y: float | None
    left_wrist_speed: float | None
    right_wrist_speed: float | None
    hand_speed_proxy: float | None


def extract_motion_features(
    records: list[PoseRecord],
    confidence_threshold: float = 0.5,
) -> list[MotionFeatureRow]:
    """Convert long-form pose landmarks into one feature row per frame."""

    if not records:
        return []

    by_frame = _records_by_frame(records)
    previous_wrist_points: dict[str, tuple[float, float, float]] = {}
    previous_rotation_angles: dict[str, tuple[float, float]] = {}
    previous_knee_angles: dict[str, tuple[float, float]] = {}
    starting_knee_angles: dict[str, float] = {}
    rows: list[MotionFeatureRow] = []

    for frame_index in sorted(by_frame):
        frame_records = by_frame[frame_index]
        clip_id = frame_records[0].clip_id
        condition_id = frame_records[0].condition_id
        timestamp_sec = frame_records[0].timestamp_sec
        points = _confident_points(frame_records, confidence_threshold)
        angles = {
            feature_name: _angle_for_points(points, joints)
            for feature_name, joints in ANGLE_DEFINITIONS.items()
        }
        wrist_speeds = {
            joint_name: _wrist_speed(
                joint_name,
                points.get(joint_name),
                timestamp_sec,
                previous_wrist_points,
            )
            for joint_name in WRIST_JOINTS
        }
        pelvis_rotation_deg = _segment_angle(points, "left_hip", "right_hip")
        shoulder_rotation_deg = _segment_angle(points, "left_shoulder", "right_shoulder")
        left_knee_angle = angles["left_knee_angle"]
        right_knee_angle = angles["right_knee_angle"]
        for joint_name, knee_angle in (
            ("left_knee", left_knee_angle),
            ("right_knee", right_knee_angle),
        ):
            if knee_angle is not None and joint_name not in starting_knee_angles:
                starting_knee_angles[joint_name] = knee_angle

        rows.append(
            MotionFeatureRow(
                clip_id=clip_id,
                condition_id=condition_id,
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                left_elbow_angle=angles["left_elbow_angle"],
                right_elbow_angle=angles["right_elbow_angle"],
                left_shoulder_angle=angles["left_shoulder_angle"],
                right_shoulder_angle=angles["right_shoulder_angle"],
                left_knee_angle=left_knee_angle,
                right_knee_angle=right_knee_angle,
                trunk_tilt_deg=_trunk_tilt(points),
                pelvis_rotation_deg=pelvis_rotation_deg,
                shoulder_rotation_deg=shoulder_rotation_deg,
                hip_shoulder_separation_deg=_hip_shoulder_separation(
                    pelvis_rotation_deg,
                    shoulder_rotation_deg,
                ),
                pelvis_rotation_velocity_deg_s=_angular_velocity(
                    "pelvis",
                    pelvis_rotation_deg,
                    timestamp_sec,
                    previous_rotation_angles,
                ),
                trunk_rotation_velocity_deg_s=_angular_velocity(
                    "trunk",
                    shoulder_rotation_deg,
                    timestamp_sec,
                    previous_rotation_angles,
                ),
                center_of_mass_x=_center_of_mass(points, axis=0),
                center_of_mass_y=_center_of_mass(points, axis=1),
                left_knee_extension_from_start_deg=_knee_extension_from_start(
                    "left_knee",
                    left_knee_angle,
                    starting_knee_angles,
                ),
                right_knee_extension_from_start_deg=_knee_extension_from_start(
                    "right_knee",
                    right_knee_angle,
                    starting_knee_angles,
                ),
                left_knee_angular_velocity_deg_s=_linear_angle_velocity(
                    "left_knee",
                    left_knee_angle,
                    timestamp_sec,
                    previous_knee_angles,
                ),
                right_knee_angular_velocity_deg_s=_linear_angle_velocity(
                    "right_knee",
                    right_knee_angle,
                    timestamp_sec,
                    previous_knee_angles,
                ),
                left_wrist_x=_coordinate(points, "left_wrist", 0),
                left_wrist_y=_coordinate(points, "left_wrist", 1),
                right_wrist_x=_coordinate(points, "right_wrist", 0),
                right_wrist_y=_coordinate(points, "right_wrist", 1),
                left_wrist_speed=wrist_speeds["left_wrist"],
                right_wrist_speed=wrist_speeds["right_wrist"],
                hand_speed_proxy=_max_optional(
                    wrist_speeds["left_wrist"],
                    wrist_speeds["right_wrist"],
                ),
            )
        )

    return rows


def _records_by_frame(records: list[PoseRecord]) -> dict[int, list[PoseRecord]]:
    by_frame: dict[int, list[PoseRecord]] = {}
    for record in records:
        by_frame.setdefault(record.frame_index, []).append(record)
    return by_frame


def _confident_points(
    records: list[PoseRecord],
    confidence_threshold: float,
) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    for record in records:
        if record.x is None or record.y is None:
            continue
        score = record.confidence if record.confidence is not None else record.visibility
        if score is not None and score < confidence_threshold:
            continue
        points[record.joint_name] = (record.x, record.y)
    return points


def _angle_for_points(
    points: dict[str, tuple[float, float]],
    joints: tuple[str, str, str],
) -> float | None:
    try:
        return angle_degrees(points[joints[0]], points[joints[1]], points[joints[2]])
    except (KeyError, ValueError):
        return None


def _trunk_tilt(points: dict[str, tuple[float, float]]) -> float | None:
    shoulder_mid = _midpoint(points, "left_shoulder", "right_shoulder")
    hip_mid = _midpoint(points, "left_hip", "right_hip")
    if shoulder_mid is None or hip_mid is None:
        return None
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(dx, -dy))


def _segment_angle(
    points: dict[str, tuple[float, float]],
    first: str,
    second: str,
) -> float | None:
    if first not in points or second not in points:
        return None
    try:
        return segment_angle_degrees(points[first], points[second])
    except ValueError:
        return None


def _hip_shoulder_separation(
    pelvis_rotation_deg: float | None,
    shoulder_rotation_deg: float | None,
) -> float | None:
    if pelvis_rotation_deg is None or shoulder_rotation_deg is None:
        return None
    return signed_angle_delta_degrees(pelvis_rotation_deg, shoulder_rotation_deg)


def _angular_velocity(
    key: str,
    angle_deg: float | None,
    timestamp_sec: float,
    previous_angles: dict[str, tuple[float, float]],
) -> float | None:
    if angle_deg is None:
        return None
    previous = previous_angles.get(key)
    previous_angles[key] = (angle_deg, timestamp_sec)
    if previous is None:
        return None
    dt = timestamp_sec - previous[1]
    if dt <= 0:
        return None
    return signed_angle_delta_degrees(previous[0], angle_deg) / dt


def _linear_angle_velocity(
    key: str,
    angle_deg: float | None,
    timestamp_sec: float,
    previous_angles: dict[str, tuple[float, float]],
) -> float | None:
    if angle_deg is None:
        return None
    previous = previous_angles.get(key)
    previous_angles[key] = (angle_deg, timestamp_sec)
    if previous is None:
        return None
    dt = timestamp_sec - previous[1]
    if dt <= 0:
        return None
    return (angle_deg - previous[0]) / dt


def _knee_extension_from_start(
    key: str,
    angle_deg: float | None,
    starting_knee_angles: dict[str, float],
) -> float | None:
    if angle_deg is None or key not in starting_knee_angles:
        return None
    return angle_deg - starting_knee_angles[key]


def _center_of_mass(points: dict[str, tuple[float, float]], axis: int) -> float | None:
    joints = (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    )
    values = [points[joint][axis] for joint in joints if joint in points]
    if not values:
        return None
    return sum(values) / len(values)


def _max_optional(*values: float | None) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return max(valid_values)


def _midpoint(
    points: dict[str, tuple[float, float]],
    first: str,
    second: str,
) -> tuple[float, float] | None:
    if first not in points or second not in points:
        return None
    return ((points[first][0] + points[second][0]) / 2, (points[first][1] + points[second][1]) / 2)


def _coordinate(
    points: dict[str, tuple[float, float]],
    joint_name: str,
    axis: int,
) -> float | None:
    point = points.get(joint_name)
    return None if point is None else point[axis]


def _wrist_speed(
    joint_name: str,
    point: tuple[float, float] | None,
    timestamp_sec: float,
    previous_wrist_points: dict[str, tuple[float, float, float]],
) -> float | None:
    if point is None:
        return None
    previous = previous_wrist_points.get(joint_name)
    previous_wrist_points[joint_name] = (point[0], point[1], timestamp_sec)
    if previous is None:
        return None
    dt = timestamp_sec - previous[2]
    if dt <= 0:
        return None
    return math.hypot(point[0] - previous[0], point[1] - previous[1]) / dt
