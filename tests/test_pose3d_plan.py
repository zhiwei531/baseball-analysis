from pathlib import Path

from baseball_pose.config import load_config, resolve_pose3d_config
from baseball_pose.io.metadata import ClipMetadata
from baseball_pose.io.paths import feature3d_path, pose3d_path
from baseball_pose.pipeline.pose3d import build_pose3d_plan


def test_default_pose3d_config_is_optional():
    config = load_config("configs/default.yaml")

    assert config.pose3d_enabled is False
    assert config.pose3d_backend == "mediapipe_world"
    assert config.pose3d_condition_ids == []


def test_resolve_pose3d_config_uses_base_values():
    config = load_config("configs/default.yaml")

    resolved = resolve_pose3d_config(config.raw, "batting_1")

    assert resolved["backend"] == "mediapipe_world"
    assert resolved["root_joint"] == "pelvis_center"


def test_pose3d_paths_follow_processed_conventions():
    assert pose3d_path("data_full", "clip_a", "cond_3d") == Path(
        "data_full/processed/poses3d/clip_a/cond_3d.csv"
    )
    assert feature3d_path("data_full", "clip_a", "cond_3d") == Path(
        "data_full/processed/features3d/clip_a/cond_3d.csv"
    )


def test_build_pose3d_plan_targets_smoothed_2d_outputs():
    config = load_config("configs/default.yaml")
    clip = ClipMetadata(
        clip_id="clip_a",
        source_path=Path("raw/clip_a.mov"),
        action_type="batting",
        fps_target=30.0,
        difficulty_tags=(),
    )

    plan = build_pose3d_plan(
        clip,
        config,
        input_condition_id="image_center_motion_grabcut_pose_smooth",
    )

    assert plan.input_pose_path == Path(
        "data/processed/poses/clip_a/image_center_motion_grabcut_pose_smooth.csv"
    )
    assert plan.output_pose3d_path == Path(
        "data/processed/poses3d/clip_a/image_center_motion_grabcut_pose_smooth_3d.csv"
    )
    assert plan.output_feature3d_path == Path(
        "data/processed/features3d/clip_a/image_center_motion_grabcut_pose_smooth_3d.csv"
    )
    assert plan.source_condition_id == "image_center_motion_grabcut_pose"
    assert plan.input_frames_path == Path(
        "data/interim/frames/clip_a/image_center_motion_grabcut_pose.csv"
    )
