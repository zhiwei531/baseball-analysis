from baseball_pose.pose.schema import PoseRecord
from baseball_pose.postprocess.completion import complete_pose_records


def test_complete_pose_records_fills_short_distal_gap_with_bone_length():
    records = []
    for frame in range(5):
        records.extend(
            [
                _record(frame, "right_elbow", 0.50 + frame * 0.01, 0.50),
                _record(
                    frame,
                    "right_wrist",
                    None if frame in {2, 3} else 0.60 + frame * 0.01,
                    None if frame in {2, 3} else 0.50,
                ),
            ]
        )

    completed = complete_pose_records(
        records,
        confidence_threshold=0.5,
        max_gap_frames=3,
        imputed_confidence=0.62,
    )
    wrists = {
        record.frame_index: record
        for record in completed
        if record.joint_name == "right_wrist"
    }

    assert wrists[2].x is not None
    assert wrists[3].x is not None
    assert round(wrists[2].x - 0.52, 2) == 0.10
    assert wrists[2].backend == "test+imputed"
    assert wrists[2].confidence == 0.62


def test_complete_pose_records_leaves_long_gap_missing():
    records = []
    for frame in range(7):
        records.extend(
            [
                _record(frame, "right_elbow", 0.50, 0.50),
                _record(
                    frame,
                    "right_wrist",
                    None if frame in {1, 2, 3, 4} else 0.60,
                    None if frame in {1, 2, 3, 4} else 0.50,
                ),
            ]
        )

    completed = complete_pose_records(records, confidence_threshold=0.5, max_gap_frames=3)
    wrists = [
        record
        for record in completed
        if record.joint_name == "right_wrist" and record.frame_index in {1, 2, 3, 4}
    ]

    assert all(record.x is None and record.y is None for record in wrists)


def test_complete_pose_records_rescues_low_confidence_biomechanical_candidate():
    records = []
    for frame in range(5):
        records.extend(
            [
                _record(frame, "right_elbow", 0.50 + frame * 0.01, 0.50),
                _record(
                    frame,
                    "right_wrist",
                    0.60 + frame * 0.01,
                    0.50,
                    confidence=0.08 if frame == 2 else 1.0,
                ),
            ]
        )

    completed = complete_pose_records(
        records,
        confidence_threshold=0.5,
        max_gap_frames=0,
        rescue_low_confidence=True,
        rescue_min_confidence=0.03,
        rescue_limb_tolerance_ratio=0.25,
        rescue_temporal_tolerance=0.03,
        imputed_confidence=0.62,
    )
    wrist = [
        record
        for record in completed
        if record.joint_name == "right_wrist" and record.frame_index == 2
    ][0]

    assert wrist.x == 0.62
    assert wrist.confidence == 0.62
    assert wrist.backend == "test+rescued"


def test_complete_pose_records_rejects_low_confidence_implausible_candidate():
    records = []
    for frame in range(5):
        records.extend(
            [
                _record(frame, "right_elbow", 0.50 + frame * 0.01, 0.50),
                _record(
                    frame,
                    "right_wrist",
                    0.90 if frame == 2 else 0.60 + frame * 0.01,
                    0.90 if frame == 2 else 0.50,
                    confidence=0.08 if frame == 2 else 1.0,
                ),
            ]
        )

    completed = complete_pose_records(
        records,
        confidence_threshold=0.5,
        max_gap_frames=0,
        rescue_low_confidence=True,
        rescue_min_confidence=0.03,
        rescue_limb_tolerance_ratio=0.25,
        rescue_temporal_tolerance=0.03,
        imputed_confidence=0.62,
    )
    wrist = [
        record
        for record in completed
        if record.joint_name == "right_wrist" and record.frame_index == 2
    ][0]

    assert wrist.confidence == 0.08
    assert wrist.backend == "test"


def _record(
    frame_index: int,
    joint_name: str,
    x: float | None,
    y: float | None,
    confidence: float = 1.0,
) -> PoseRecord:
    return PoseRecord(
        clip_id="clip",
        condition_id="cond",
        frame_index=frame_index,
        timestamp_sec=frame_index / 30,
        joint_name=joint_name,
        x=x,
        y=y,
        visibility=confidence,
        confidence=confidence,
        backend="test",
        inference_time_ms=1.0,
    )
