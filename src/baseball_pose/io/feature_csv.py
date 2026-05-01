"""CSV serialization for frame-level motion features."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from baseball_pose.features.extraction import MotionFeatureRow


FEATURE_FIELDNAMES = (
    "clip_id",
    "condition_id",
    "frame_index",
    "timestamp_sec",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_knee_angle",
    "right_knee_angle",
    "trunk_tilt_deg",
    "pelvis_rotation_deg",
    "shoulder_rotation_deg",
    "hip_shoulder_separation_deg",
    "pelvis_rotation_velocity_deg_s",
    "trunk_rotation_velocity_deg_s",
    "center_of_mass_x",
    "center_of_mass_y",
    "left_knee_extension_from_start_deg",
    "right_knee_extension_from_start_deg",
    "left_knee_angular_velocity_deg_s",
    "right_knee_angular_velocity_deg_s",
    "left_wrist_x",
    "left_wrist_y",
    "right_wrist_x",
    "right_wrist_y",
    "left_wrist_speed",
    "right_wrist_speed",
    "hand_speed_proxy",
)


def write_feature_rows(path: str | Path, rows: list[MotionFeatureRow]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            values = asdict(row)
            writer.writerow({key: "" if values[key] is None else values[key] for key in FEATURE_FIELDNAMES})


def read_feature_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
