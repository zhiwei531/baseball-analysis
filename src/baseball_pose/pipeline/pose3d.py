"""Planning and placeholder execution for 3D lifting stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig, resolve_pose3d_config
from baseball_pose.io.metadata import ClipMetadata
from baseball_pose.io.paths import feature3d_path, pose3d_path, pose_path


@dataclass(frozen=True)
class Pose3DPlan:
    clip_id: str
    input_condition_id: str
    output_condition_id: str
    input_pose_path: Path
    output_pose3d_path: Path
    output_feature3d_path: Path
    backend: str
    root_joint: str


def build_pose3d_plan(
    clip: ClipMetadata,
    config: RuntimeConfig,
    *,
    input_condition_id: str,
    output_condition_id: str | None = None,
) -> Pose3DPlan:
    pose3d_config = resolve_pose3d_config(config.raw, clip.clip_id)
    derived_output_condition = output_condition_id or f"{input_condition_id}_3d"
    return Pose3DPlan(
        clip_id=clip.clip_id,
        input_condition_id=input_condition_id,
        output_condition_id=derived_output_condition,
        input_pose_path=pose_path(config.data_dir, clip.clip_id, input_condition_id),
        output_pose3d_path=pose3d_path(config.data_dir, clip.clip_id, derived_output_condition),
        output_feature3d_path=feature3d_path(config.data_dir, clip.clip_id, derived_output_condition),
        backend=str(pose3d_config.get("backend", "temporal_lifter_stub")),
        root_joint=str(pose3d_config.get("root_joint", "pelvis_center")),
    )


def lift_pose_sequence_placeholder(plan: Pose3DPlan) -> None:
    """Fail loudly until a real temporal 3D backend is integrated."""

    raise NotImplementedError(
        "3D lifting is not implemented yet. Planned input:"
        f" {plan.input_pose_path} -> {plan.output_pose3d_path}"
    )
