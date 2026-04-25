"""CSV serialization for sampled frame records."""

from __future__ import annotations

import csv
from pathlib import Path

from baseball_pose.io.video import FrameRecord


FRAME_FIELDNAMES = (
    "clip_id",
    "condition_id",
    "frame_index",
    "timestamp_sec",
    "frame_path",
    "width",
    "height",
)


def write_frame_records(path: str | Path, records: list[FrameRecord]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_FIELDNAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "clip_id": record.clip_id,
                    "condition_id": record.condition_id,
                    "frame_index": record.frame_index,
                    "timestamp_sec": record.timestamp_sec,
                    "frame_path": record.frame_path,
                    "width": record.width,
                    "height": record.height,
                }
            )


def read_frame_records(path: str | Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                FrameRecord(
                    clip_id=row["clip_id"],
                    frame_index=int(row["frame_index"]),
                    timestamp_sec=float(row["timestamp_sec"]),
                    frame_path=Path(row["frame_path"]),
                    condition_id=row["condition_id"],
                    width=_optional_int(row["width"]),
                    height=_optional_int(row["height"]),
                )
            )
    return records


def _optional_int(value: str) -> int | None:
    if value == "":
        return None
    return int(value)
