"""Post-processing orchestration for existing 3D pose CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig, resolve_pose3d_config
from baseball_pose.io.paths import pose3d_path
from baseball_pose.io.pose3d_csv import read_pose3d_records, write_pose3d_records
from baseball_pose.postprocess.smoothing3d import smooth_pose3d_records


@dataclass(frozen=True)
class SmoothPose3DResult:
    clip_id: str
    source_condition_id: str
    condition_id: str
    pose3d_csv: Path
    pose_record_count: int


def smooth_pose3d_files(
    clip_ids: list[str],
    config: RuntimeConfig,
    source_conditions: list[str],
    output_suffix: str = "_smooth",
) -> list[SmoothPose3DResult]:
    results: list[SmoothPose3DResult] = []
    for clip_id in clip_ids:
        pose3d_config = resolve_pose3d_config(config.raw, clip_id)
        threshold_config = pose3d_config.get("confidence_thresholds", {})
        max_gap_config = pose3d_config.get("interpolate_max_gap_by_group", {})
        smoothing_config = pose3d_config.get("smoothing", {})
        for source_condition_id in source_conditions:
            if source_condition_id.endswith(output_suffix):
                continue
            source_path = pose3d_path(config.data_dir, clip_id, source_condition_id)
            if not source_path.exists():
                continue
            condition_id = f"{source_condition_id}{output_suffix}"
            records = read_pose3d_records(source_path)
            smoothed_records = [
                record.__class__(
                    clip_id=record.clip_id,
                    condition_id=condition_id,
                    frame_index=record.frame_index,
                    timestamp_sec=record.timestamp_sec,
                    joint_name=record.joint_name,
                    x_3d=record.x_3d,
                    y_3d=record.y_3d,
                    z_3d=record.z_3d,
                    scale_mode=record.scale_mode,
                    lift_backend=record.lift_backend,
                    input_quality_score=record.input_quality_score,
                )
                for record in smooth_pose3d_records(
                    records,
                    window_length=int(smoothing_config.get("window_length", 15)),
                    polyorder=int(smoothing_config.get("polyorder", 2)),
                    median_window_length=int(smoothing_config.get("median_window_length", 5)),
                    refine_window_length=int(smoothing_config.get("refine_window_length", 9)),
                    confidence_threshold=float(pose3d_config.get("confidence_threshold", 0.5)),
                    threshold_config=threshold_config if isinstance(threshold_config, dict) else {},
                    max_gap_frames=int(pose3d_config.get("interpolate_max_gap_frames", 3)),
                    max_gap_config=max_gap_config if isinstance(max_gap_config, dict) else {},
                    jump_threshold_multiplier=float(smoothing_config.get("jump_threshold_multiplier", 3.0)),
                    joint_jump_config=smoothing_config.get("joint_jump_thresholds", {}),
                )
            ]
            output_path = pose3d_path(config.data_dir, clip_id, condition_id)
            write_pose3d_records(output_path, smoothed_records)
            results.append(
                SmoothPose3DResult(
                    clip_id=clip_id,
                    source_condition_id=source_condition_id,
                    condition_id=condition_id,
                    pose3d_csv=output_path,
                    pose_record_count=len(smoothed_records),
                )
            )
    return results
