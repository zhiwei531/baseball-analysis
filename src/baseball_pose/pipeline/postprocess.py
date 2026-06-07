"""Post-processing orchestration for existing pose CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig, resolve_postprocess_config
from baseball_pose.io.paths import pose_path
from baseball_pose.io.pose_csv import read_pose_records, write_pose_records
from baseball_pose.postprocess.smoothing import smooth_pose_records


@dataclass(frozen=True)
class SmoothPoseResult:
    clip_id: str
    source_condition_id: str
    condition_id: str
    pose_csv: Path
    pose_record_count: int


def smooth_pose_files(
    clip_ids: list[str],
    config: RuntimeConfig,
    source_conditions: list[str] | None = None,
    output_suffix: str = "_smooth",
) -> list[SmoothPoseResult]:
    condition_ids = source_conditions if source_conditions is not None else config.condition_ids
    condition_ids = [condition_id for condition_id in condition_ids if not condition_id.endswith(output_suffix)]
    results: list[SmoothPoseResult] = []

    for clip_id in clip_ids:
        postprocess_config = resolve_postprocess_config(config.raw, clip_id)
        smoothing_config = postprocess_config.get("smoothing", {})
        confidence_threshold = float(postprocess_config.get("confidence_threshold", 0.5))
        threshold_config = postprocess_config.get("confidence_thresholds", {})
        max_gap_frames = int(postprocess_config.get("interpolate_max_gap_frames", 3))
        max_gap_config = postprocess_config.get("interpolate_max_gap_by_group", {})
        for source_condition_id in condition_ids:
            source_path = pose_path(config.data_dir, clip_id, source_condition_id)
            if not source_path.exists():
                continue
            condition_id = f"{source_condition_id}{output_suffix}"
            records = read_pose_records(source_path)
            smoothed_records = [
                record.__class__(
                    clip_id=record.clip_id,
                    condition_id=condition_id,
                    frame_index=record.frame_index,
                    timestamp_sec=record.timestamp_sec,
                    joint_name=record.joint_name,
                    x=record.x,
                    y=record.y,
                    visibility=record.visibility,
                    confidence=record.confidence,
                    backend=record.backend,
                    inference_time_ms=record.inference_time_ms,
                )
                for record in smooth_pose_records(
                    records,
                    method=str(smoothing_config.get("method", "savgol")),
                    window_length=int(smoothing_config.get("window_length", 7)),
                    polyorder=int(smoothing_config.get("polyorder", 2)),
                    median_window_length=int(smoothing_config.get("median_window_length", 1)),
                    refine_window_length=int(smoothing_config.get("refine_window_length", 1)),
                    confidence_threshold=confidence_threshold,
                    threshold_config=threshold_config if isinstance(threshold_config, dict) else {},
                    max_gap_frames=max_gap_frames,
                    max_gap_config=max_gap_config if isinstance(max_gap_config, dict) else {},
                    jump_threshold_multiplier=float(
                        smoothing_config.get("jump_threshold_multiplier", 6.0)
                    ),
                    joint_jump_config=smoothing_config.get("joint_jump_thresholds", {}),
                    torso_gate_enabled=bool(smoothing_config.get("torso_gate_enabled", True)),
                    torso_jump_threshold_multiplier=float(
                        smoothing_config.get("torso_jump_threshold_multiplier", 8.0)
                    ),
                    min_torso_jump_distance=float(
                        smoothing_config.get("min_torso_jump_distance", 0.08)
                    ),
                    limb_length_tolerance_ratio=float(
                        smoothing_config.get("limb_length_tolerance_ratio", 0.28)
                    ),
                    foot_length_tolerance_ratio=float(
                        smoothing_config.get("foot_length_tolerance_ratio", 1.20)
                    ),
                )
            ]
            output_path = pose_path(config.data_dir, clip_id, condition_id)
            write_pose_records(output_path, smoothed_records)
            results.append(
                SmoothPoseResult(
                    clip_id=clip_id,
                    source_condition_id=source_condition_id,
                    condition_id=condition_id,
                    pose_csv=output_path,
                    pose_record_count=len(smoothed_records),
                )
            )

    return results
