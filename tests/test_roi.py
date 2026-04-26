from baseball_pose.pose.schema import PoseRecord
from baseball_pose.preprocessing.roi import (
    RoiBox,
    estimate_pose_prior_roi,
    remap_pose_records_to_full_frame,
)


def test_roi_clamps_to_image_bounds():
    roi = RoiBox(-10, 5, 50, 100).clamped(80, 60)

    assert roi.x == 0
    assert roi.y == 5
    assert roi.width == 40
    assert roi.height == 55


def test_remap_pose_records_to_full_frame_normalized_coordinates():
    records = [
        PoseRecord("clip", "auto_roi_raw", 0, 0.0, "left_wrist", 0.5, 0.25, 1.0, 1.0, "test")
    ]
    remapped = remap_pose_records_to_full_frame(records, RoiBox(20, 10, 40, 80), 100, 200)

    assert remapped[0].x == 0.4
    assert remapped[0].y == 0.15


def test_pose_prior_roi_uses_confident_landmarks_and_padding():
    records = [
        PoseRecord("clip", "baseline_raw", 0, 0.0, "nose", 0.50, 0.20, 1.0, 1.0, "test"),
        PoseRecord("clip", "baseline_raw", 0, 0.0, "left_shoulder", 0.40, 0.40, 1.0, 1.0, "test"),
        PoseRecord("clip", "baseline_raw", 0, 0.0, "right_shoulder", 0.60, 0.40, 1.0, 1.0, "test"),
        PoseRecord("clip", "baseline_raw", 0, 0.0, "left_hip", 0.42, 0.70, 1.0, 1.0, "test"),
        PoseRecord("clip", "baseline_raw", 0, 0.0, "right_hip", 0.58, 0.70, 1.0, 1.0, "test"),
    ]

    result = estimate_pose_prior_roi(records, 1000, 1000)

    assert result.condition_id == "auto_roi_pose_prior"
    assert result.roi.x < 400
    assert result.roi.y < 200
    assert result.roi.width > 200
    assert result.roi.height > 500
