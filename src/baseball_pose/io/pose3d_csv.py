"""CSV serialization for 3D pose records."""

from __future__ import annotations

import csv
from pathlib import Path

from baseball_pose.pose3d.schema import Pose3DRecord


POSE3D_FIELDNAMES = (
    "clip_id",
    "condition_id",
    "frame_index",
    "timestamp_sec",
    "joint_name",
    "x_3d",
    "y_3d",
    "z_3d",
    "scale_mode",
    "lift_backend",
    "input_quality_score",
)


def write_pose3d_records(path: str | Path, records: list[Pose3DRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POSE3D_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "clip_id": record.clip_id,
                    "condition_id": record.condition_id,
                    "frame_index": record.frame_index,
                    "timestamp_sec": record.timestamp_sec,
                    "joint_name": record.joint_name,
                    "x_3d": record.x_3d,
                    "y_3d": record.y_3d,
                    "z_3d": record.z_3d,
                    "scale_mode": record.scale_mode,
                    "lift_backend": record.lift_backend,
                    "input_quality_score": record.input_quality_score,
                }
            )


def read_pose3d_records(path: str | Path) -> list[Pose3DRecord]:
    records: list[Pose3DRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                Pose3DRecord(
                    clip_id=row["clip_id"],
                    condition_id=row["condition_id"],
                    frame_index=int(row["frame_index"]),
                    timestamp_sec=float(row["timestamp_sec"]),
                    joint_name=row["joint_name"],
                    x_3d=float(row["x_3d"]),
                    y_3d=float(row["y_3d"]),
                    z_3d=float(row["z_3d"]),
                    scale_mode=row["scale_mode"],
                    lift_backend=row["lift_backend"],
                    input_quality_score=_optional_float(row["input_quality_score"]),
                )
            )
    return records


def _optional_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)
