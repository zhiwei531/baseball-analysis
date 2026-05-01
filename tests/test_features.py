from baseball_pose.features.angles import (
    angle_degrees,
    segment_angle_degrees,
    signed_angle_delta_degrees,
)
from baseball_pose.features.extraction import extract_motion_features
from baseball_pose.pose.schema import PoseRecord


def test_angle_degrees_right_angle():
    assert angle_degrees((1, 0), (0, 0), (0, 1)) == 90


def test_segment_angle_and_signed_delta_use_image_coordinates():
    assert segment_angle_degrees((0, 0), (1, -1)) == 45
    assert signed_angle_delta_degrees(170, -170) == 20


def test_extract_motion_features_angles_and_wrist_speed():
    records = [
        _record(0, 0.0, "left_shoulder", 0, 1),
        _record(0, 0.0, "left_elbow", 0, 0),
        _record(0, 0.0, "left_wrist", 1, 0),
        _record(0, 0.0, "right_wrist", 3, 4),
        _record(1, 0.5, "left_shoulder", 0, 1),
        _record(1, 0.5, "left_elbow", 0, 0),
        _record(1, 0.5, "left_wrist", 4, 0),
        _record(1, 0.5, "right_wrist", 3, 8),
    ]

    rows = extract_motion_features(records, confidence_threshold=0.5)

    assert len(rows) == 2
    assert rows[0].left_elbow_angle == 90
    assert rows[0].left_wrist_speed is None
    assert rows[1].left_wrist_speed == 6
    assert rows[1].right_wrist_speed == 8


def test_extract_motion_features_posture_analysis_proxies():
    records = [
        _record(0, 0.0, "left_hip", 0, 0),
        _record(0, 0.0, "right_hip", 2, 0),
        _record(0, 0.0, "left_shoulder", 0, -2),
        _record(0, 0.0, "right_shoulder", 2, -2),
        _record(0, 0.0, "left_knee", 0, 2),
        _record(0, 0.0, "left_ankle", 2, 2),
        _record(1, 0.5, "left_hip", 0, 0),
        _record(1, 0.5, "right_hip", 2, 0),
        _record(1, 0.5, "left_shoulder", 0, -2),
        _record(1, 0.5, "right_shoulder", 1, -3),
        _record(1, 0.5, "left_knee", 0, 2),
        _record(1, 0.5, "left_ankle", 0, 4),
    ]

    rows = extract_motion_features(records, confidence_threshold=0.5)

    assert rows[0].pelvis_rotation_deg == 0
    assert rows[0].shoulder_rotation_deg == 0
    assert rows[0].hip_shoulder_separation_deg == 0
    assert rows[1].shoulder_rotation_deg == 45
    assert rows[1].hip_shoulder_separation_deg == 45
    assert rows[1].trunk_rotation_velocity_deg_s == 90
    assert rows[0].center_of_mass_x == 1
    assert rows[1].left_knee_extension_from_start_deg > 0


def _record(
    frame_index: int,
    timestamp_sec: float,
    joint_name: str,
    x: float,
    y: float,
) -> PoseRecord:
    return PoseRecord(
        clip_id="clip",
        condition_id="condition",
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        joint_name=joint_name,
        x=x,
        y=y,
        visibility=1.0,
        confidence=1.0,
        backend="test",
    )
