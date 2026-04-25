from baseball_pose.evaluation.completeness import keypoint_completeness
from baseball_pose.evaluation.jitter import temporal_jitter
from baseball_pose.pose.schema import PoseRecord


def test_keypoint_completeness_counts_present_required_joints():
    records = [
        PoseRecord("clip", "baseline", 0, 0.0, "left_wrist", 0.1, 0.2, 1.0, 1.0, "test"),
        PoseRecord("clip", "baseline", 0, 0.0, "right_wrist", None, None, 0.0, 0.0, "test"),
    ]

    assert keypoint_completeness(records, {"left_wrist", "right_wrist"}) == 0.5


def test_temporal_jitter_uses_mean_frame_to_frame_distance():
    records = [
        PoseRecord("clip", "baseline", 0, 0.0, "left_wrist", 0.0, 0.0, 1.0, 1.0, "test"),
        PoseRecord("clip", "baseline", 1, 0.1, "left_wrist", 3.0, 4.0, 1.0, 1.0, "test"),
        PoseRecord("clip", "baseline", 2, 0.2, "left_wrist", 6.0, 8.0, 1.0, 1.0, "test"),
    ]

    assert temporal_jitter(records, "left_wrist") == 5.0
