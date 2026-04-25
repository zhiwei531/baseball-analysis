"""Path conventions for generated artifacts."""

from __future__ import annotations

from pathlib import Path


def frame_dir(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "interim" / "frames" / clip_id / condition_id


def pose_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "poses" / clip_id / f"{condition_id}.csv"


def feature_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "features" / clip_id / f"{condition_id}.csv"


def metric_path(data_dir: str | Path, experiment_id: str) -> Path:
    return Path(data_dir) / "processed" / "metrics" / f"{experiment_id}.csv"
