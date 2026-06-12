from baseball_pose.equipment.features import extract_object_motion_features
from baseball_pose.equipment.detection import (
    EquipmentTrackingConfig,
    _YoloDetection,
    _create_yolo_detector,
    _detect_ball_yolo,
    _detect_bat_yolo,
    _interpolate_object_records,
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


def test_yolo_bat_candidate_uses_baseball_bat_class_and_wrist_prior():
    config = EquipmentTrackingConfig(detector_backend="yolo")
    detections = [
        _YoloDetection(class_id=32, confidence=0.95, xyxy=(10.0, 10.0, 20.0, 20.0)),
        _YoloDetection(class_id=34, confidence=0.80, xyxy=(100.0, 80.0, 180.0, 96.0)),
    ]

    bat = _detect_bat_yolo(
        detections=detections,
        wrist_points=[(96.0, 88.0)],
        previous=None,
        width=640,
        height=480,
        config=config,
    )

    assert bat is not None
    assert bat.handle == (100.0, 88.0)
    assert bat.barrel == (180.0, 88.0)
    assert bat.confidence > 0.6


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
