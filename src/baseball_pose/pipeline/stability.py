"""Pose-stability summary tables for comparing conditions and clips."""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from baseball_pose.config import RuntimeConfig, resolve_postprocess_config
from baseball_pose.io.paths import metric_path, pose_path
from baseball_pose.io.pose_csv import read_pose_records
from baseball_pose.pose.quality import LIMB_SEGMENTS, threshold_for_joint
from baseball_pose.pose.schema import PoseRecord, pose_score


@dataclass(frozen=True)
class StabilitySummaryResult:
    clip_id: str
    joint_csv: Path
    limb_csv: Path
    row_count: int


def summarize_pose_stability(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str] | None = None,
) -> list[StabilitySummaryResult]:
    condition_ids = conditions if conditions is not None else config.condition_ids
    joint_rows: list[dict[str, object]] = []
    limb_rows: list[dict[str, object]] = []
    results: list[StabilitySummaryResult] = []

    for clip_id in clip_ids:
        postprocess_config = resolve_postprocess_config(config.raw, clip_id)
        confidence_threshold = float(postprocess_config.get("confidence_threshold", 0.5))
        threshold_config = postprocess_config.get("confidence_thresholds", {})
        smoothing_config = postprocess_config.get("smoothing", {})
        limb_tolerance = float(smoothing_config.get("limb_length_tolerance_ratio", 0.28))
        clip_joint_count = 0
        for condition_id in condition_ids:
            source_path = pose_path(config.data_dir, clip_id, condition_id)
            if not source_path.exists():
                continue
            records = read_pose_records(source_path)
            joint_rows.extend(
                _joint_rows(
                    records,
                    confidence_threshold=confidence_threshold,
                    threshold_config=threshold_config if isinstance(threshold_config, dict) else {},
                )
            )
            limb_rows.extend(
                _limb_rows(
                    records,
                    confidence_threshold=confidence_threshold,
                    threshold_config=threshold_config if isinstance(threshold_config, dict) else {},
                    tolerance_ratio=limb_tolerance,
                )
            )
            clip_joint_count += 1
        if clip_joint_count:
            joint_csv = metric_path(config.data_dir, "pose_stability")
            limb_csv = metric_path(config.data_dir, "pose_stability_limbs")
            results.append(
                StabilitySummaryResult(
                    clip_id=clip_id,
                    joint_csv=joint_csv,
                    limb_csv=limb_csv,
                    row_count=clip_joint_count,
                )
            )

    joint_csv = metric_path(config.data_dir, "pose_stability")
    limb_csv = metric_path(config.data_dir, "pose_stability_limbs")
    _write_csv(joint_csv, joint_rows)
    _write_csv(limb_csv, limb_rows)
    return results


def _joint_rows(
    records: list[PoseRecord],
    confidence_threshold: float,
    threshold_config: dict[str, object],
) -> list[dict[str, object]]:
    if not records:
        return []
    clip_id = records[0].clip_id
    condition_id = records[0].condition_id
    total_frames = len({record.frame_index for record in records})
    by_joint: dict[str, list[PoseRecord]] = {}
    for record in records:
        by_joint.setdefault(record.joint_name, []).append(record)

    rows: list[dict[str, object]] = []
    for joint_name, joint_records in sorted(by_joint.items()):
        threshold = threshold_for_joint(joint_name, confidence_threshold, threshold_config)
        valid_points = [
            record
            for record in joint_records
            if record.x is not None and record.y is not None and (pose_score(record) or 0.0) >= threshold
        ]
        low_visibility_frames = sum(1 for record in joint_records if (record.visibility or 0.0) < threshold)
        steps = _steps_for_records(valid_points)
        median_score = statistics.median(
            [score for score in (pose_score(record) for record in joint_records) if score is not None]
        ) if any(pose_score(record) is not None for record in joint_records) else None
        jump_threshold = _jump_threshold(steps)
        jump_outliers = sum(1 for step in steps if jump_threshold is not None and step > jump_threshold)
        rows.append(
            {
                "clip_id": clip_id,
                "condition_id": condition_id,
                "joint_name": joint_name,
                "frames_total": total_frames,
                "valid_frames": len(valid_points),
                "valid_ratio": round(len(valid_points) / total_frames, 4) if total_frames else 0.0,
                "low_visibility_frames": low_visibility_frames,
                "median_score": round(median_score, 4) if median_score is not None else "",
                "mean_step": round(float(np.mean(steps)), 6) if steps else "",
                "p95_step": round(float(np.percentile(steps, 95)), 6) if steps else "",
                "jump_outlier_frames": jump_outliers,
            }
        )
    return rows


def _limb_rows(
    records: list[PoseRecord],
    confidence_threshold: float,
    threshold_config: dict[str, object],
    tolerance_ratio: float,
) -> list[dict[str, object]]:
    if not records:
        return []
    clip_id = records[0].clip_id
    condition_id = records[0].condition_id
    by_frame: dict[int, dict[str, PoseRecord]] = {}
    for record in records:
        by_frame.setdefault(record.frame_index, {})[record.joint_name] = record

    rows: list[dict[str, object]] = []
    for proximal, distal, reject_joint in LIMB_SEGMENTS:
        lengths: list[float] = []
        for frame_records in by_frame.values():
            proximal_record = frame_records.get(proximal)
            distal_record = frame_records.get(distal)
            if not _record_is_confident(proximal_record, confidence_threshold, threshold_config):
                continue
            if not _record_is_confident(distal_record, confidence_threshold, threshold_config):
                continue
            assert proximal_record is not None and distal_record is not None
            lengths.append(
                float(
                    np.hypot(
                        distal_record.x - proximal_record.x,
                        distal_record.y - proximal_record.y,
                    )
                )
            )
        if not lengths:
            continue
        median_length = statistics.median(lengths)
        stdev = statistics.pstdev(lengths) if len(lengths) > 1 else 0.0
        outliers = sum(1 for length in lengths if abs(length - median_length) / max(median_length, 1e-6) > tolerance_ratio)
        rows.append(
            {
                "clip_id": clip_id,
                "condition_id": condition_id,
                "segment": f"{proximal}->{distal}",
                "reject_joint": reject_joint,
                "samples": len(lengths),
                "median_length": round(median_length, 6),
                "stdev_length": round(stdev, 6),
                "cv_length": round(stdev / median_length, 6) if median_length else "",
                "outlier_samples": outliers,
            }
        )
    return rows


def _record_is_confident(
    record: PoseRecord | None,
    confidence_threshold: float,
    threshold_config: dict[str, object],
) -> bool:
    if record is None or record.x is None or record.y is None:
        return False
    score = pose_score(record)
    threshold = threshold_for_joint(record.joint_name, confidence_threshold, threshold_config)
    return score is not None and score >= threshold


def _steps_for_records(records: list[PoseRecord]) -> list[float]:
    steps: list[float] = []
    previous: PoseRecord | None = None
    for record in records:
        if previous is not None:
            steps.append(float(np.hypot(record.x - previous.x, record.y - previous.y)))
        previous = record
    return steps


def _jump_threshold(steps: list[float]) -> float | None:
    nonzero_steps = [step for step in steps if step > 0]
    if len(nonzero_steps) < 3:
        return None
    return max(statistics.median(nonzero_steps) * 3.5, 0.03)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
