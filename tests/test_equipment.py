from baseball_pose.equipment.features import extract_object_motion_features
from baseball_pose.equipment.detection import (
    _BallCandidate,
    EquipmentTrackingConfig,
    _YoloDetection,
    _bat_swing_window_frames,
    _create_yolo_detector,
    _detect_ball_yolo,
    _detect_bat_yolo,
    _filter_short_object_tracks,
    _interpolate_object_records,
    _resolve_yolo_device,
    _smooth_bat_records,
)
from baseball_pose.equipment.schema import ObjectTrackRecord
from baseball_pose.io.object_csv import read_object_tracks, write_object_tracks
from baseball_pose.io.video import FrameRecord


def test_extract_object_motion_features_computes_bat_and_ball_speed():
    records = [
        _record(0, 0.0, "bat", 0.10, 0.20, 0.05, 0.25),
        _record(0, 0.0, "ball", 0.40, 0.40),
        _record(1, 0.5, "bat", 0.20, 0.20, 0.10, 0.25),
        _record(1, 0.5, "ball", 0.40, 0.50),
    ]

    rows = extract_object_motion_features(records)

    assert len(rows) == 2
    assert rows[0].bat_speed_norm_s is None
    assert round(rows[1].bat_speed_norm_s, 2) == 0.20
    assert round(rows[1].bat_speed_px_s, 2) == 128.00
    assert round(rows[1].ball_speed_norm_s, 2) == 0.20


def test_object_track_csv_round_trip(tmp_path):
    path = tmp_path / "objects.csv"
    records = [_record(0, 0.0, "bat", 0.1, 0.2, 0.0, 0.3)]

    write_object_tracks(path, records)
    loaded = read_object_tracks(path)

    assert loaded == records


def test_motion_backend_does_not_require_yolo():
    assert _create_yolo_detector(EquipmentTrackingConfig(detector_backend="motion")) is None


def test_yolo_auto_device_resolves_to_available_backend():
    assert _resolve_yolo_device("auto") in {"cuda", "mps", "cpu", None}


def test_yolo_bat_candidate_uses_baseball_bat_class_and_wrist_prior():
    config = EquipmentTrackingConfig(detector_backend="yolo")
    detections = [
        _YoloDetection(class_id=32, confidence=0.95, xyxy=(10.0, 10.0, 20.0, 20.0)),
        _YoloDetection(class_id=34, confidence=0.80, xyxy=(100.0, 80.0, 180.0, 96.0)),
    ]

    bat = _detect_bat_yolo(
        image=_bat_test_image(),
        detections=detections,
        wrist_points=[(96.0, 88.0)],
        previous=None,
        width=640,
        height=480,
        config=config,
    )

    assert bat is not None
    assert abs(bat.handle[0] - 100.0) < 5
    assert abs(bat.handle[1] - 88.0) < 5
    assert abs(bat.barrel[0] - 180.0) < 5
    assert abs(bat.barrel[1] - 88.0) < 5
    assert bat.confidence > 0.6


def test_yolo_bat_candidate_does_not_require_pose_prior():
    config = EquipmentTrackingConfig(detector_backend="yolo")
    detections = [_YoloDetection(class_id=34, confidence=0.80, xyxy=(100.0, 80.0, 180.0, 96.0))]

    bat = _detect_bat_yolo(
        image=_bat_test_image(),
        detections=detections,
        wrist_points=[],
        previous=None,
        width=640,
        height=480,
        config=config,
    )

    assert bat is not None
    assert abs(bat.handle[0] - 100.0) < 5
    assert abs(bat.barrel[0] - 180.0) < 5


def test_yolo_ball_candidate_uses_sports_ball_class_and_anchor():
    config = EquipmentTrackingConfig(detector_backend="yolo")
    detections = [
        _YoloDetection(class_id=34, confidence=0.95, xyxy=(100.0, 80.0, 180.0, 96.0)),
        _YoloDetection(class_id=32, confidence=0.70, xyxy=(300.0, 100.0, 310.0, 110.0)),
    ]

    ball = _detect_ball_yolo(
        detections=detections,
        previous=None,
        anchors=[(304.0, 106.0)],
        width=640,
        height=480,
        config=config,
    )

    assert ball is not None
    assert ball.center == (305.0, 105.0)
    assert ball.radius_px == 5.0
    assert ball.confidence > 0.6


def test_yolo_ball_candidate_does_not_require_pose_or_object_anchor():
    config = EquipmentTrackingConfig(detector_backend="yolo")
    detections = [_YoloDetection(class_id=32, confidence=0.70, xyxy=(300.0, 100.0, 310.0, 110.0))]

    ball = _detect_ball_yolo(
        detections=detections,
        previous=None,
        anchors=[],
        width=640,
        height=480,
        config=config,
    )

    assert ball is not None
    assert ball.center == (305.0, 105.0)


def test_yolo_ball_high_confidence_can_reacquire_after_bad_previous():
    config = EquipmentTrackingConfig(detector_backend="yolo", ball_max_y_ratio=1.0)
    previous = _BallCandidate(center=(50.0, 50.0), radius_px=5.0, confidence=0.40)
    detections = [_YoloDetection(class_id=32, confidence=0.75, xyxy=(620.0, 410.0, 650.0, 432.0))]

    ball = _detect_ball_yolo(
        detections=detections,
        previous=previous,
        anchors=[],
        width=1280,
        height=720,
        config=config,
    )

    assert ball is not None
    assert ball.center == (635.0, 421.0)


def test_interpolate_object_records_fills_short_gap():
    frames = [
        FrameRecord("clip", index, index / 30.0, frame_path="unused.png", condition_id="condition")
        for index in range(3)
    ]
    records = [
        _record(0, 0.0, "ball", 0.10, 0.20),
        _record(2, 2 / 30.0, "ball", 0.30, 0.40),
    ]

    interpolated = _interpolate_object_records(
        records,
        frames,
        EquipmentTrackingConfig(interpolate_max_gap_frames=1),
    )

    assert len(interpolated) == 3
    middle = interpolated[1]
    assert middle.frame_index == 1
    assert middle.object_name == "ball"
    assert round(middle.x, 2) == 0.20
    assert round(middle.y, 2) == 0.30
    assert middle.source == "temporal_interpolation"


def test_filter_short_ball_tracks_removes_isolated_false_positives():
    records = [
        _record(0, 0.0, "bat", 0.10, 0.20),
        _record(10, 10 / 30.0, "ball", 0.10, 0.20),
        _record(30, 30 / 30.0, "ball", 0.30, 0.40),
        _record(31, 31 / 30.0, "ball", 0.31, 0.41),
        _record(32, 32 / 30.0, "ball", 0.32, 0.42),
    ]

    filtered = _filter_short_object_tracks(
        records,
        object_name="ball",
        min_track_length=3,
        max_gap_frames=2,
    )

    assert [(record.object_name, record.frame_index) for record in filtered] == [
        ("bat", 0),
        ("ball", 30),
        ("ball", 31),
        ("ball", 32),
    ]


def test_smooth_bat_records_reduces_jitter_without_moving_ball():
    records = [
        _record(0, 0.0, "bat", 0.10, 0.20, 0.00, 0.25),
        _record(0, 0.0, "ball", 0.80, 0.10),
        _record(1, 1 / 30.0, "bat", 0.20, 0.10, 0.10, 0.15),
        _record(1, 1 / 30.0, "ball", 0.70, 0.20),
        _record(2, 2 / 30.0, "bat", 0.10, 0.20, 0.00, 0.25),
    ]

    smoothed = _smooth_bat_records(records, EquipmentTrackingConfig(bat_smoothing_window_frames=3))
    bat_frame_1 = next(record for record in smoothed if record.object_name == "bat" and record.frame_index == 1)
    ball_records = [record for record in smoothed if record.object_name == "ball"]

    assert bat_frame_1.x is not None
    assert bat_frame_1.x < 0.20
    assert bat_frame_1.y is not None
    assert bat_frame_1.y > 0.10
    assert [(record.x, record.y) for record in ball_records] == [(0.80, 0.10), (0.70, 0.20)]


def test_smooth_bat_records_preserves_line_geometry():
    records = [
        _record(0, 0.0, "bat", 0.20, 0.20, 0.00, 0.20),
        _record(1, 1 / 30.0, "bat", 0.24, 0.22, 0.04, 0.18),
        _record(2, 2 / 30.0, "bat", 0.20, 0.20, 0.00, 0.20),
    ]

    smoothed = _smooth_bat_records(records, EquipmentTrackingConfig(bat_smoothing_window_frames=3))
    middle = next(record for record in smoothed if record.frame_index == 1)

    assert middle.x is not None
    assert middle.x2 is not None
    assert round(middle.x - middle.x2, 2) == 0.20
    assert middle.y is not None
    assert middle.y2 is not None
    assert abs(middle.y - middle.y2) < 0.03


def test_smooth_bat_records_preserves_contact_motion_near_ball():
    records = [
        _record(0, 0.0, "bat", 0.10, 0.20, 0.00, 0.20),
        _record(1, 1 / 30.0, "bat", 0.20, 0.20, 0.10, 0.20),
        _record(1, 1 / 30.0, "ball", 0.60, 0.20),
        _record(2, 2 / 30.0, "bat", 0.30, 0.20, 0.20, 0.20),
    ]

    smoothed = _smooth_bat_records(
        records,
        EquipmentTrackingConfig(
            bat_smoothing_window_frames=3,
            bat_smoothing_passes=2,
            bat_smoothing_contact_margin_frames=0,
            bat_smoothing_contact_original_weight=0.35,
        ),
    )
    contact_bat = next(record for record in smoothed if record.object_name == "bat" and record.frame_index == 1)
    ball = next(record for record in smoothed if record.object_name == "ball")

    assert contact_bat.x is not None
    assert contact_bat.x > 0.15
    assert ball.x == 0.60
    assert ball.y == 0.20


def test_smooth_bat_records_preserves_fast_angular_motion_without_ball():
    records = [
        _record(0, 0.0, "bat", 0.30, 0.20, 0.10, 0.20),
        _record(1, 1 / 30.0, "bat", 0.20, 0.30, 0.20, 0.10),
        _record(2, 2 / 30.0, "bat", 0.10, 0.20, 0.30, 0.20),
    ]

    smoothed = _smooth_bat_records(
        records,
        EquipmentTrackingConfig(
            bat_smoothing_window_frames=3,
            bat_smoothing_passes=2,
            bat_smoothing_fast_angle_threshold_deg=20.0,
            bat_smoothing_fast_original_weight=0.85,
        ),
    )
    middle = next(record for record in smoothed if record.frame_index == 1)

    assert middle.y is not None
    assert middle.y2 is not None
    assert middle.y - middle.y2 > 0.12


def test_bat_swing_window_uses_ball_window_when_available():
    records = [
        _record(10, 10 / 30.0, "bat", 0.30, 0.20, 0.10, 0.20),
        _record(11, 11 / 30.0, "bat", 0.20, 0.30, 0.20, 0.10),
        _record(100, 100 / 30.0, "bat", 0.30, 0.20, 0.10, 0.20),
        _record(101, 101 / 30.0, "bat", 0.20, 0.30, 0.20, 0.10),
        _record(100, 100 / 30.0, "ball", 0.60, 0.20),
    ]
    bat_by_frame = {record.frame_index: record for record in records if record.object_name == "bat"}

    swing_frames = _bat_swing_window_frames(
        records,
        bat_by_frame,
        EquipmentTrackingConfig(
            bat_smoothing_fast_angle_threshold_deg=20.0,
            bat_swing_window_margin_frames=2,
        ),
    )

    assert 100 in swing_frames
    assert 101 in swing_frames
    assert 10 not in swing_frames
    assert 11 not in swing_frames


def _record(
    frame_index: int,
    timestamp_sec: float,
    object_name: str,
    x: float,
    y: float,
    x2: float | None = None,
    y2: float | None = None,
) -> ObjectTrackRecord:
    return ObjectTrackRecord(
        clip_id="clip",
        condition_id="condition",
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        object_name=object_name,
        x=x,
        y=y,
        x2=x2,
        y2=y2,
        confidence=0.8,
        width=640,
        height=480,
        source="test",
    )


def _bat_test_image():
    import cv2
    import numpy as np

    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(image, (100, 88), (180, 88), (255, 255, 255), 3, cv2.LINE_AA)
    return image
