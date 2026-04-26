from baseball_pose.pose.schema import PoseRecord
from baseball_pose.preprocessing.roi import RoiBox, remap_pose_records_to_full_frame


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
