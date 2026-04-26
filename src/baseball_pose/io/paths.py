"""Path conventions for generated artifacts."""

from __future__ import annotations

from pathlib import Path


def frame_dir(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "interim" / "frames" / clip_id / condition_id


def frame_manifest_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "interim" / "frames" / clip_id / f"{condition_id}.csv"


def pose_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "poses" / clip_id / f"{condition_id}.csv"


def feature_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "features" / clip_id / f"{condition_id}.csv"


def metric_path(data_dir: str | Path, experiment_id: str) -> Path:
    return Path(data_dir) / "processed" / "metrics" / f"{experiment_id}.csv"


def overlay_frame_dir(output_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(output_dir) / "overlays" / "frames" / clip_id / condition_id


def overlay_video_path(output_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(output_dir) / "overlays" / f"{clip_id}__{condition_id}.mp4"


def motion_preview_frame_dir(output_dir: str | Path, clip_id: str) -> Path:
    return Path(output_dir) / "motion_preview" / "frames" / clip_id


def motion_preview_video_path(output_dir: str | Path, clip_id: str) -> Path:
    return Path(output_dir) / "motion_preview" / f"{clip_id}__motion_preview.mp4"
