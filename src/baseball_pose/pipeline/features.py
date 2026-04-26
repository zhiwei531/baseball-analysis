"""Feature extraction orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig
from baseball_pose.features.extraction import extract_motion_features
from baseball_pose.io.feature_csv import write_feature_rows
from baseball_pose.io.paths import feature_path, pose_path
from baseball_pose.io.pose_csv import read_pose_records


@dataclass(frozen=True)
class FeatureExtractionResult:
    clip_id: str
    condition_id: str
    feature_csv: Path
    frame_count: int


def extract_feature_files(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str] | None = None,
) -> list[FeatureExtractionResult]:
    condition_ids = conditions if conditions is not None else config.condition_ids
    confidence_threshold = float(config.raw["postprocess"].get("confidence_threshold", 0.5))
    results: list[FeatureExtractionResult] = []

    for clip_id in clip_ids:
        for condition_id in condition_ids:
            source_path = pose_path(config.data_dir, clip_id, condition_id)
            if not source_path.exists():
                continue
            records = read_pose_records(source_path)
            rows = extract_motion_features(records, confidence_threshold=confidence_threshold)
            output_path = feature_path(config.data_dir, clip_id, condition_id)
            write_feature_rows(output_path, rows)
            results.append(
                FeatureExtractionResult(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    feature_csv=output_path,
                    frame_count=len(rows),
                )
            )

    return results
