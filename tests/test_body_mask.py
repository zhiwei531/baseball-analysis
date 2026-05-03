import numpy as np

from baseball_pose.pose.schema import PoseRecord
from baseball_pose.preprocessing.body_mask import create_body_prior_masked_crop
from baseball_pose.preprocessing.roi import RoiBox


def test_body_prior_mask_blacks_out_pixels_outside_prior_body():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    records = [
        _record("left_shoulder", 0.45, 0.30),
        _record("right_shoulder", 0.55, 0.30),
        _record("left_hip", 0.45, 0.55),
        _record("right_hip", 0.55, 0.55),
        _record("left_knee", 0.45, 0.75),
        _record("right_knee", 0.55, 0.75),
    ]

    result = create_body_prior_masked_crop(
        image=image,
        prior_records=records,
        image_width=100,
        image_height=100,
        fallback_roi=RoiBox(0, 0, 100, 100),
        padding_ratio=0.20,
        min_width_ratio=0.30,
        min_height_ratio=0.50,
        limb_thickness_ratio=0.04,
        joint_radius_ratio=0.03,
    )

    assert result.prior_joint_count == 6
    assert result.image.sum() < image.sum()
    assert result.image.sum() > 0


def test_body_prior_mask_falls_back_without_enough_prior_joints():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    fallback = RoiBox(20, 10, 40, 50)

    result = create_body_prior_masked_crop(
        image=image,
        prior_records=[],
        image_width=100,
        image_height=100,
        fallback_roi=fallback,
    )

    assert result.roi == fallback
    assert result.image.shape[:2] == (50, 40)


def _record(joint_name: str, x: float, y: float) -> PoseRecord:
    return PoseRecord(
        clip_id="clip",
        condition_id="center_prior_roi_smooth",
        frame_index=0,
        timestamp_sec=0.0,
        joint_name=joint_name,
        x=x,
        y=y,
        visibility=1.0,
        confidence=1.0,
        backend="test",
    )
