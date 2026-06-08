"""CSV serialization for tracked equipment objects."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from baseball_pose.equipment.schema import ObjectFeatureRow, ObjectTrackRecord


OBJECT_TRACK_FIELDNAMES = (
    "clip_id",
    "condition_id",
    "frame_index",
    "timestamp_sec",
    "object_name",
    "x",
    "y",
    "x2",
    "y2",
    "confidence",
    "width",
    "height",
    "source",
)

OBJECT_FEATURE_FIELDNAMES = (
    "clip_id",
    "condition_id",
    "frame_index",
    "timestamp_sec",
    "bat_barrel_x",
    "bat_barrel_y",
    "bat_handle_x",
    "bat_handle_y",
    "bat_speed_px_s",
    "bat_speed_norm_s",
    "bat_angle_deg",
    "ball_x",
    "ball_y",
    "ball_speed_px_s",
    "ball_speed_norm_s",
)


def write_object_tracks(path: str | Path, records: list[ObjectTrackRecord]) -> None:
    _write_rows(path, OBJECT_TRACK_FIELDNAMES, records)


def read_object_tracks(path: str | Path) -> list[ObjectTrackRecord]:
    records: list[ObjectTrackRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                ObjectTrackRecord(
                    clip_id=row["clip_id"],
                    condition_id=row["condition_id"],
                    frame_index=int(row["frame_index"]),
                    timestamp_sec=float(row["timestamp_sec"]),
                    object_name=row["object_name"],
                    x=_optional_float(row["x"]),
                    y=_optional_float(row["y"]),
                    x2=_optional_float(row["x2"]),
                    y2=_optional_float(row["y2"]),
                    confidence=_optional_float(row["confidence"]),
                    width=_optional_int(row["width"]),
                    height=_optional_int(row["height"]),
                    source=row["source"],
                )
            )
    return records


def write_object_features(path: str | Path, rows: list[ObjectFeatureRow]) -> None:
    _write_rows(path, OBJECT_FEATURE_FIELDNAMES, rows)


def read_object_features(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: str | Path, fieldnames: tuple[str, ...], rows: list[object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            writer.writerow({key: "" if values[key] is None else values[key] for key in fieldnames})


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _optional_int(value: str) -> int | None:
    return None if value == "" else int(value)
