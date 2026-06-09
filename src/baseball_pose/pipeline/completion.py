"""Pose completion orchestration for existing pose CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig, resolve_postprocess_config
from baseball_pose.io.paths import pose_path
from baseball_pose.io.pose_csv import read_pose_records, write_pose_records
from baseball_pose.postprocess.completion import complete_pose_records


@dataclass(frozen=True)
class CompletePoseResult:
    clip_id: str
    source_condition_id: str
    condition_id: str
    pose_csv: Path
    pose_record_count: int


def complete_pose_files(
    clip_ids: list[str],
    config: RuntimeConfig,
    source_conditions: list[str] | None = None,
    output_suffix: str = "_complete",
) -> list[CompletePoseResult]:
    completion_config = config.raw.get("completion", {})
    condition_ids = source_conditions if source_conditions is not None else config.condition_ids
    condition_ids = [
        condition_id
        for condition_id in condition_ids
        if not condition_id.endswith(output_suffix) and not condition_id.endswith("_smooth")
    ]
    results: list[CompletePoseResult] = []

    for clip_id in clip_ids:
        postprocess_config = resolve_postprocess_config(config.raw, clip_id)
        confidence_threshold = float(
            completion_config.get("confidence_threshold", postprocess_config.get("confidence_threshold", 0.5))
        )
        threshold_config = completion_config.get(
            "confidence_thresholds",
            postprocess_config.get("confidence_thresholds", {}),
        )
        max_gap_frames = int(completion_config.get("max_gap_frames", 5))
        max_gap_config = completion_config.get("max_gap_by_group", {})
        imputed_confidence = float(completion_config.get("imputed_confidence", 0.62))
        rescue_low_confidence = bool(completion_config.get("rescue_low_confidence", True))
        rescue_min_confidence = float(completion_config.get("rescue_min_confidence", 0.03))
        rescue_temporal_only_min_confidence = float(
            completion_config.get("rescue_temporal_only_min_confidence", 0.30)
        )
        rescue_limb_tolerance_ratio = float(completion_config.get("rescue_limb_tolerance_ratio", 0.75))
        rescue_temporal_tolerance = float(completion_config.get("rescue_temporal_tolerance", 0.12))
        for source_condition_id in condition_ids:
            source_path = pose_path(config.data_dir, clip_id, source_condition_id)
            if not source_path.exists():
                continue
            condition_id = f"{source_condition_id}{output_suffix}"
            records = read_pose_records(source_path)
            completed_records = [
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
                for record in complete_pose_records(
                    records,
                    confidence_threshold=confidence_threshold,
                    threshold_config=threshold_config if isinstance(threshold_config, dict) else {},
                    max_gap_frames=max_gap_frames,
                    max_gap_config=max_gap_config if isinstance(max_gap_config, dict) else {},
                    imputed_confidence=imputed_confidence,
                    rescue_low_confidence=rescue_low_confidence,
                    rescue_min_confidence=rescue_min_confidence,
                    rescue_temporal_only_min_confidence=rescue_temporal_only_min_confidence,
                    rescue_limb_tolerance_ratio=rescue_limb_tolerance_ratio,
                    rescue_temporal_tolerance=rescue_temporal_tolerance,
                )
            ]
            output_path = pose_path(config.data_dir, clip_id, condition_id)
            write_pose_records(output_path, completed_records)
            results.append(
                CompletePoseResult(
                    clip_id=clip_id,
                    source_condition_id=source_condition_id,
                    condition_id=condition_id,
                    pose_csv=output_path,
                    pose_record_count=len(completed_records),
                )
            )

    return results
