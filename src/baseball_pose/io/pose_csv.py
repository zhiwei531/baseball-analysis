"""CSV serialization for pose records."""

from __future__ import annotations

import csv
from pathlib import Path

from baseball_pose.pose.schema import PoseRecord


POSE_FIELDNAMES = (
    "clip_id",
    "condition_id",
    "frame_index",
    "timestamp_sec",
    "joint_name",
    "x",
    "y",
    "visibility",
    "confidence",
    "backend",
    "inference_time_ms",
)


def write_pose_records(path: str | Path, records: list[PoseRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POSE_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "clip_id": record.clip_id,
                    "condition_id": record.condition_id,
                    "frame_index": record.frame_index,
                    "timestamp_sec": record.timestamp_sec,
                    "joint_name": record.joint_name,
                    "x": record.x,
                    "y": record.y,
                    "visibility": record.visibility,
                    "confidence": record.confidence,
                    "backend": record.backend,
                    "inference_time_ms": record.inference_time_ms,
                }
            )


def read_pose_records(path: str | Path) -> list[PoseRecord]:
    records: list[PoseRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                PoseRecord(
                    clip_id=row["clip_id"],
                    condition_id=row["condition_id"],
                    frame_index=int(row["frame_index"]),
                    timestamp_sec=float(row["timestamp_sec"]),
                    joint_name=row["joint_name"],
                    x=_optional_float(row["x"]),
                    y=_optional_float(row["y"]),
                    visibility=_optional_float(row["visibility"]),
                    confidence=_optional_float(row["confidence"]),
                    backend=row["backend"],
                    inference_time_ms=_optional_float(row["inference_time_ms"]),
                )
            )
    return records


def _optional_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)
