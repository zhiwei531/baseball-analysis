from baseball_pose.pose3d.schema import Pose3DRecord
from baseball_pose.postprocess.smoothing3d import smooth_pose3d_records


def test_smpl24_smoothing_preserves_low_quality_coordinates():
    joint_names = [
        "hip",
        "spine1",
        "spine2",
        "spine3",
        "left_collar",
        "right_collar",
        "left_wrist",
        "right_wrist",
    ]
    records = []
    for frame_index in range(7):
        for joint_index, joint_name in enumerate(joint_names):
            records.append(
                Pose3DRecord(
                    clip_id="clip_a",
                    condition_id="pose_3d",
                    frame_index=frame_index,
                    timestamp_sec=frame_index / 30.0,
                    joint_name=joint_name,
                    x_3d=float(frame_index + joint_index),
                    y_3d=float(joint_index),
                    z_3d=0.0,
                    scale_mode="smpl_world",
                    lift_backend="gvhmr",
                    input_quality_score=0.1,
                )
            )

    smoothed = smooth_pose3d_records(
        records,
        window_length=3,
        polyorder=1,
        confidence_threshold=0.5,
        max_gap_frames=1,
    )

    assert len(smoothed) == len(records)
    assert all(record.x_3d == record.x_3d for record in smoothed)
