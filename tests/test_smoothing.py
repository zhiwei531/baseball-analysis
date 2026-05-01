from baseball_pose.pose.schema import PoseRecord
from baseball_pose.postprocess.smoothing import smooth_pose_records


def test_smooth_pose_records_rejects_isolated_jump():
    records = [
        _record(0, 0.00, 0.00),
        _record(1, 0.01, 0.01),
        _record(2, 0.50, 0.50),
        _record(3, 0.03, 0.03),
        _record(4, 0.04, 0.04),
        _record(5, 0.05, 0.05),
        _record(6, 0.06, 0.06),
    ]

    smoothed = smooth_pose_records(
        records,
        window_length=5,
        polyorder=2,
        confidence_threshold=0.5,
        max_gap_frames=1,
        jump_threshold_multiplier=3.0,
    )

    assert smoothed[2].x is not None
    assert smoothed[2].y is not None
    assert smoothed[2].x < 0.10
    assert smoothed[2].y < 0.10


def test_smooth_pose_records_hides_low_confidence_points():
    records = [
        _record(0, 0.0, 0.0),
        _record(1, 0.1, 0.1, confidence=0.1),
        _record(2, 0.2, 0.2),
        _record(3, 0.3, 0.3),
        _record(4, 0.4, 0.4),
    ]

    smoothed = smooth_pose_records(records, window_length=3, polyorder=1, confidence_threshold=0.5)

    assert smoothed[1].x is not None
    assert round(smoothed[1].x, 2) == 0.10


def test_smooth_pose_records_refine_window_reduces_wiggle():
    records = [
        _record(0, 0.00, 0.00),
        _record(1, 0.20, 0.20),
        _record(2, 0.05, 0.05),
        _record(3, 0.24, 0.24),
        _record(4, 0.10, 0.10),
        _record(5, 0.28, 0.28),
        _record(6, 0.15, 0.15),
        _record(7, 0.31, 0.31),
        _record(8, 0.20, 0.20),
    ]

    base = smooth_pose_records(records, window_length=5, polyorder=2, refine_window_length=1)
    refined = smooth_pose_records(records, window_length=5, polyorder=2, refine_window_length=5)

    base_total_variation = _total_variation(base)
    refined_total_variation = _total_variation(refined)

    assert refined_total_variation < base_total_variation


def _record(
    frame_index: int,
    x: float,
    y: float,
    confidence: float = 1.0,
) -> PoseRecord:
    return PoseRecord(
        clip_id="clip",
        condition_id="condition",
        frame_index=frame_index,
        timestamp_sec=frame_index / 30,
        joint_name="left_wrist",
        x=x,
        y=y,
        visibility=confidence,
        confidence=confidence,
        backend="test",
    )


def _total_variation(records: list[PoseRecord]) -> float:
    values = [record.x for record in records if record.x is not None]
    return sum(abs(curr - prev) for prev, curr in zip(values, values[1:]))
