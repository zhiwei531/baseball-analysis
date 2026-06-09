from math import isnan
from pathlib import Path

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.frame_csv import write_frame_records
from baseball_pose.io.pose_csv import write_pose_records
from baseball_pose.io.pose3d_csv import read_pose3d_records
from baseball_pose.io.video import FrameRecord
from baseball_pose.pose.schema import PoseRecord
from baseball_pose.io.metadata import ClipMetadata
from baseball_pose.pipeline.pose3d import build_pose3d_plan, lift_pose_sequence


def test_external_video_hmr_csv_imports_and_gates_with_2d_prior(tmp_path):
    data_dir = tmp_path / "data"
    external_csv = data_dir / "external_pose3d" / "gvhmr" / "clip_a.csv"
    external_csv.parent.mkdir(parents=True)
    external_csv.write_text(
        "\n".join(
            [
                "frame_index,joint_name,x_3d,y_3d,z_3d,confidence",
                "0,left_wrist,1.0,2.0,3.0,0.9",
                "0,right_wrist,4.0,5.0,6.0,0.8",
            ]
        ),
        encoding="utf-8",
    )
    write_frame_records(
        data_dir / "interim" / "frames" / "clip_a" / "image_center_motion_grabcut_pose.csv",
        [
            FrameRecord(
                clip_id="clip_a",
                condition_id="image_center_motion_grabcut_pose",
                frame_index=0,
                timestamp_sec=1.25,
                frame_path=Path("frame_0000.png"),
                width=640,
                height=480,
            )
        ],
    )
    write_pose_records(
        data_dir / "processed" / "poses" / "clip_a" / "image_center_motion_grabcut_pose_smooth.csv",
        [
            PoseRecord(
                clip_id="clip_a",
                condition_id="image_center_motion_grabcut_pose_smooth",
                frame_index=0,
                timestamp_sec=1.25,
                joint_name="left_wrist",
                x=0.4,
                y=0.5,
                visibility=0.9,
                confidence=0.9,
                backend="rtmpose",
            ),
            PoseRecord(
                clip_id="clip_a",
                condition_id="image_center_motion_grabcut_pose_smooth",
                frame_index=0,
                timestamp_sec=1.25,
                joint_name="right_wrist",
                x=0.6,
                y=0.5,
                visibility=0.1,
                confidence=0.1,
                backend="rtmpose",
            ),
        ],
    )
    config = RuntimeConfig(
        path=tmp_path / "config.yaml",
        raw={
            "project": {"data_dir": str(data_dir), "output_dir": str(tmp_path / "outputs")},
            "pose3d": {
                "backend": "gvhmr",
                "gate_with_2d_prior": True,
                "confidence_threshold": 0.5,
                "external_result_path": str(external_csv),
            },
        },
    )
    clip = ClipMetadata(
        clip_id="clip_a",
        source_path=Path("raw/clip_a.mp4"),
        action_type="batting",
        fps_target=30.0,
        difficulty_tags=(),
    )

    plan = build_pose3d_plan(
        clip,
        config,
        input_condition_id="image_center_motion_grabcut_pose_smooth",
    )
    count = lift_pose_sequence(plan, clip, config)
    records = read_pose3d_records(plan.output_pose3d_path)
    by_joint = {record.joint_name: record for record in records}

    assert count == 2
    assert by_joint["left_wrist"].x_3d == 1.0
    assert by_joint["left_wrist"].timestamp_sec == 1.25
    assert by_joint["left_wrist"].lift_backend == "gvhmr"
    assert isnan(by_joint["right_wrist"].x_3d)


def test_external_video_hmr_preserves_coordinates_without_hard_2d_gate(tmp_path):
    data_dir = tmp_path / "data"
    external_csv = data_dir / "external_pose3d" / "gvhmr" / "clip_a.csv"
    external_csv.parent.mkdir(parents=True)
    external_csv.write_text(
        "\n".join(
            [
                "frame_index,joint_name,x_3d,y_3d,z_3d,confidence",
                "0,right_wrist,4.0,5.0,6.0,0.8",
            ]
        ),
        encoding="utf-8",
    )
    write_frame_records(
        data_dir / "interim" / "frames" / "clip_a" / "image_center_motion_grabcut_pose.csv",
        [
            FrameRecord(
                clip_id="clip_a",
                condition_id="image_center_motion_grabcut_pose",
                frame_index=0,
                timestamp_sec=1.25,
                frame_path=Path("frame_0000.png"),
                width=640,
                height=480,
            )
        ],
    )
    write_pose_records(
        data_dir / "processed" / "poses" / "clip_a" / "image_center_motion_grabcut_pose_smooth.csv",
        [
            PoseRecord(
                clip_id="clip_a",
                condition_id="image_center_motion_grabcut_pose_smooth",
                frame_index=0,
                timestamp_sec=1.25,
                joint_name="right_wrist",
                x=0.6,
                y=0.5,
                visibility=0.1,
                confidence=0.1,
                backend="rtmpose",
            ),
        ],
    )
    config = RuntimeConfig(
        path=tmp_path / "config.yaml",
        raw={
            "project": {"data_dir": str(data_dir), "output_dir": str(tmp_path / "outputs")},
            "pose3d": {
                "backend": "gvhmr",
                "confidence_threshold": 0.5,
                "external_result_path": str(external_csv),
            },
        },
    )
    clip = ClipMetadata(
        clip_id="clip_a",
        source_path=Path("raw/clip_a.mp4"),
        action_type="batting",
        fps_target=30.0,
        difficulty_tags=(),
    )

    plan = build_pose3d_plan(
        clip,
        config,
        input_condition_id="image_center_motion_grabcut_pose_smooth",
    )
    lift_pose_sequence(plan, clip, config)
    records = read_pose3d_records(plan.output_pose3d_path)

    assert records[0].joint_name == "right_wrist"
    assert records[0].x_3d == 4.0
    assert records[0].input_quality_score == 0.1
