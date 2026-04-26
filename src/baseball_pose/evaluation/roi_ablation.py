"""ROI ablation metric aggregation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from baseball_pose.evaluation.jitter import temporal_jitter
from baseball_pose.evaluation.runtime import mean_inference_time_ms
from baseball_pose.evaluation.smoothness import second_difference_smoothness
from baseball_pose.io.paths import metric_path, pose_path
from baseball_pose.io.pose_csv import read_pose_records
from baseball_pose.pose.schema import CANONICAL_JOINTS, PoseRecord


ROI_ABLATION_CONDITIONS = ("baseline_raw", "auto_roi_raw", "auto_roi_pose_prior")
UPPER_BODY_JOINTS = {
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
}
LOWER_BODY_JOINTS = {
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}
WRIST_JOINTS = ("left_wrist", "right_wrist")


@dataclass(frozen=True)
class RoiMetricRow:
    clip_id: str
    condition_id: str
    metric_name: str
    joint_group: str
    value: float | None
    aggregation: str
    notes: str = ""


def summarize_roi_ablation(
    clip_ids: list[str],
    data_dir: str | Path,
    output_path: str | Path | None = None,
    conditions: tuple[str, ...] = ROI_ABLATION_CONDITIONS,
    confidence_threshold: float = 0.5,
) -> list[RoiMetricRow]:
    rows: list[RoiMetricRow] = []
    for clip_id in clip_ids:
        for condition_id in conditions:
            source_path = pose_path(data_dir, clip_id, condition_id)
            if not source_path.exists():
                rows.append(
                    RoiMetricRow(
                        clip_id,
                        condition_id,
                        "missing_pose_file",
                        "all",
                        None,
                        "file",
                        str(source_path),
                    )
                )
                continue
            records = read_pose_records(source_path)
            rows.extend(_summarize_records(records, confidence_threshold))

    if output_path is None:
        output_path = metric_path(data_dir, "roi_ablation")
    write_roi_metric_rows(output_path, rows)
    return rows


def write_roi_metric_rows(path: str | Path, rows: list[RoiMetricRow]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "clip_id",
                "condition_id",
                "metric_name",
                "joint_group",
                "value",
                "aggregation",
                "notes",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "clip_id": row.clip_id,
                    "condition_id": row.condition_id,
                    "metric_name": row.metric_name,
                    "joint_group": row.joint_group,
                    "value": "" if row.value is None else row.value,
                    "aggregation": row.aggregation,
                    "notes": row.notes,
                }
            )


def _summarize_records(
    records: list[PoseRecord],
    confidence_threshold: float,
) -> list[RoiMetricRow]:
    if not records:
        return []
    clip_id = records[0].clip_id
    condition_id = records[0].condition_id
    rows = [
        RoiMetricRow(
            clip_id,
            condition_id,
            "keypoint_completeness",
            "all",
            _mean_frame_completeness(records, set(CANONICAL_JOINTS), confidence_threshold),
            "mean_per_frame",
        ),
        RoiMetricRow(
            clip_id,
            condition_id,
            "keypoint_completeness",
            "upper_body",
            _mean_frame_completeness(records, UPPER_BODY_JOINTS, confidence_threshold),
            "mean_per_frame",
        ),
        RoiMetricRow(
            clip_id,
            condition_id,
            "keypoint_completeness",
            "lower_body",
            _mean_frame_completeness(records, LOWER_BODY_JOINTS, confidence_threshold),
            "mean_per_frame",
        ),
        RoiMetricRow(
            clip_id,
            condition_id,
            "runtime_ms_per_frame",
            "all",
            _mean_runtime_per_frame(records),
            "mean_unique_frame",
        ),
    ]

    for joint_name in WRIST_JOINTS:
        rows.append(
            RoiMetricRow(
                clip_id,
                condition_id,
                "missing_rate",
                joint_name,
                _missing_rate(records, joint_name, confidence_threshold),
                "frames",
            )
        )
        rows.append(
            RoiMetricRow(
                clip_id,
                condition_id,
                "temporal_jitter",
                joint_name,
                temporal_jitter(_confident_joint_records(records, joint_name, confidence_threshold), joint_name),
                "mean_frame_to_frame_distance",
            )
        )
        rows.append(
            RoiMetricRow(
                clip_id,
                condition_id,
                "trajectory_smoothness",
                joint_name,
                second_difference_smoothness(
                    _confident_joint_records(records, joint_name, confidence_threshold),
                    joint_name,
                ),
                "mean_second_difference",
            )
        )

    return rows


def _mean_frame_completeness(
    records: list[PoseRecord],
    required_joints: set[str],
    confidence_threshold: float,
) -> float:
    by_frame = _records_by_frame(records)
    if not by_frame:
        return 0.0
    total = 0.0
    for frame_records in by_frame.values():
        present = {
            record.joint_name
            for record in frame_records
            if record.joint_name in required_joints
            and record.x is not None
            and record.y is not None
            and _is_confident(record, confidence_threshold)
        }
        total += len(present) / len(required_joints)
    return total / len(by_frame)


def _missing_rate(records: list[PoseRecord], joint_name: str, confidence_threshold: float) -> float:
    by_frame = _records_by_frame(records)
    if not by_frame:
        return 1.0
    missing = 0
    for frame_records in by_frame.values():
        joint_records = [record for record in frame_records if record.joint_name == joint_name]
        if not joint_records or not _has_confident_coordinate(joint_records[0], confidence_threshold):
            missing += 1
    return missing / len(by_frame)


def _mean_runtime_per_frame(records: list[PoseRecord]) -> float | None:
    by_frame = _records_by_frame(records)
    frame_records = [frame_items[0] for frame_items in by_frame.values() if frame_items]
    return mean_inference_time_ms(frame_records)


def _confident_joint_records(
    records: list[PoseRecord],
    joint_name: str,
    confidence_threshold: float,
) -> list[PoseRecord]:
    return [
        record
        for record in records
        if record.joint_name == joint_name and _has_confident_coordinate(record, confidence_threshold)
    ]


def _records_by_frame(records: list[PoseRecord]) -> dict[int, list[PoseRecord]]:
    by_frame: dict[int, list[PoseRecord]] = {}
    for record in records:
        by_frame.setdefault(record.frame_index, []).append(record)
    return by_frame


def _has_confident_coordinate(record: PoseRecord, confidence_threshold: float) -> bool:
    return record.x is not None and record.y is not None and _is_confident(record, confidence_threshold)


def _is_confident(record: PoseRecord, confidence_threshold: float) -> bool:
    score = record.confidence if record.confidence is not None else record.visibility
    return score is None or score >= confidence_threshold
