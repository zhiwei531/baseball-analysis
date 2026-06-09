"""Import 3D joints produced by external video-HMR backends.

This adapter keeps research-model execution outside the core pipeline. Tools
such as GVHMR or WHAM can run in their own environment and export per-frame
3D joints here as CSV or NPZ.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from baseball_pose.io.video import FrameRecord
from baseball_pose.pose3d.schema import Pose3DRecord


CSV_REQUIRED_FIELDS = {"frame_index", "joint_name", "x_3d", "y_3d", "z_3d"}
NPZ_JOINT_KEYS = ("joints_3d", "pred_joints_3d", "world_joints", "joints")
NPZ_SCORE_KEYS = ("joint_scores", "scores", "confidence", "confidences")


def read_external_video_hmr_records(
    path: str | Path,
    *,
    clip_id: str,
    condition_id: str,
    frames: list[FrameRecord],
    backend_name: str,
    scale_mode: str = "external_video_hmr",
    joint_names: list[str] | None = None,
    joint_name_map: dict[str, str] | None = None,
) -> list[Pose3DRecord]:
    """Read external 3D joints and normalize them to the project CSV schema."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"External 3D result does not exist: {source}")
    frame_timestamps = {frame.frame_index: frame.timestamp_sec for frame in frames}
    if source.suffix.lower() == ".csv":
        return _read_external_csv(
            source,
            clip_id=clip_id,
            condition_id=condition_id,
            frame_timestamps=frame_timestamps,
            backend_name=backend_name,
            scale_mode=scale_mode,
            joint_name_map=joint_name_map or {},
        )
    if source.suffix.lower() == ".npz":
        return _read_external_npz(
            source,
            clip_id=clip_id,
            condition_id=condition_id,
            frame_timestamps=frame_timestamps,
            backend_name=backend_name,
            scale_mode=scale_mode,
            joint_names=joint_names,
            joint_name_map=joint_name_map or {},
        )
    raise ValueError(f"Unsupported external 3D result format: {source.suffix}")


def _read_external_csv(
    path: Path,
    *,
    clip_id: str,
    condition_id: str,
    frame_timestamps: dict[int, float],
    backend_name: str,
    scale_mode: str,
    joint_name_map: dict[str, str],
) -> list[Pose3DRecord]:
    records: list[Pose3DRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = CSV_REQUIRED_FIELDS - fields
        if missing:
            raise ValueError(f"External 3D CSV missing fields {sorted(missing)}: {path}")
        for row in reader:
            frame_index = int(row["frame_index"])
            joint_name = _map_joint_name(row["joint_name"], joint_name_map)
            records.append(
                Pose3DRecord(
                    clip_id=row.get("clip_id") or clip_id,
                    condition_id=condition_id,
                    frame_index=frame_index,
                    timestamp_sec=_optional_float(row.get("timestamp_sec", ""))
                    if row.get("timestamp_sec", "") != ""
                    else frame_timestamps.get(frame_index, float(frame_index)),
                    joint_name=joint_name,
                    x_3d=float(row["x_3d"]),
                    y_3d=float(row["y_3d"]),
                    z_3d=float(row["z_3d"]),
                    scale_mode=row.get("scale_mode") or scale_mode,
                    lift_backend=row.get("lift_backend") or backend_name,
                    input_quality_score=_optional_float(
                        row.get("input_quality_score") or row.get("confidence") or row.get("score") or ""
                    ),
                )
            )
    return records


def _read_external_npz(
    path: Path,
    *,
    clip_id: str,
    condition_id: str,
    frame_timestamps: dict[int, float],
    backend_name: str,
    scale_mode: str,
    joint_names: list[str] | None,
    joint_name_map: dict[str, str],
) -> list[Pose3DRecord]:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("NPZ external 3D import requires numpy.") from exc

    data = np.load(path, allow_pickle=True)
    joints = _first_present(data, NPZ_JOINT_KEYS)
    if joints is None:
        raise ValueError(f"External 3D NPZ missing one of {NPZ_JOINT_KEYS}: {path}")
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError(f"External 3D NPZ joints must have shape [frames, joints, 3]: {path}")

    names = joint_names or _npz_joint_names(data)
    if not names:
        raise ValueError("External 3D NPZ requires pose3d.external_joint_names or a joint_names array.")
    if len(names) != joints.shape[1]:
        raise ValueError(
            f"External 3D NPZ joint name count {len(names)} does not match joints axis {joints.shape[1]}."
        )

    scores = _first_present(data, NPZ_SCORE_KEYS)
    records: list[Pose3DRecord] = []
    for frame_offset in range(joints.shape[0]):
        frame_index = frame_offset
        timestamp_sec = frame_timestamps.get(frame_index, float(frame_index))
        for joint_index, source_joint_name in enumerate(names):
            joint_name = _map_joint_name(str(source_joint_name), joint_name_map)
            score = None
            if scores is not None:
                if scores.ndim == 2:
                    score = float(scores[frame_offset, joint_index])
                elif scores.ndim == 1:
                    score = float(scores[frame_offset])
            records.append(
                Pose3DRecord(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    joint_name=joint_name,
                    x_3d=float(joints[frame_offset, joint_index, 0]),
                    y_3d=float(joints[frame_offset, joint_index, 1]),
                    z_3d=float(joints[frame_offset, joint_index, 2]),
                    scale_mode=scale_mode,
                    lift_backend=backend_name,
                    input_quality_score=score,
                )
            )
    return records


def _first_present(data: Any, keys: tuple[str, ...]):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _npz_joint_names(data: Any) -> list[str]:
    if "joint_names" not in data:
        return []
    return [str(value) for value in data["joint_names"].tolist()]


def _map_joint_name(joint_name: str, joint_name_map: dict[str, str]) -> str:
    return joint_name_map.get(joint_name, joint_name)


def _optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
