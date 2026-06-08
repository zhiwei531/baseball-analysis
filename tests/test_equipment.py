from baseball_pose.equipment.features import extract_object_motion_features
from baseball_pose.equipment.schema import ObjectTrackRecord
from baseball_pose.io.object_csv import read_object_tracks, write_object_tracks


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
