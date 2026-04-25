"""Clip metadata loading and validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


VALID_ACTION_TYPES = {"pitching", "batting"}


@dataclass(frozen=True)
class ClipMetadata:
    clip_id: str
    source_path: Path
    action_type: str
    fps_target: float
    difficulty_tags: tuple[str, ...]
    notes: str = ""


def load_clips(path: str | Path, project_root: str | Path = ".") -> list[ClipMetadata]:
    """Load clip metadata from CSV."""

    csv_path = Path(path)
    root = Path(project_root)
    clips: list[ClipMetadata] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tags = tuple(tag for tag in row.get("difficulty_tags", "").split(";") if tag)
            clips.append(
                ClipMetadata(
                    clip_id=row["clip_id"],
                    source_path=root / row["source_path"],
                    action_type=row["action_type"],
                    fps_target=float(row["fps_target"]),
                    difficulty_tags=tags,
                    notes=row.get("notes", ""),
                )
            )
    validate_clips(clips)
    return clips


def validate_clips(clips: list[ClipMetadata]) -> None:
    if not clips:
        raise ValueError("No clips found in metadata.")

    seen: set[str] = set()
    for clip in clips:
        if clip.clip_id in seen:
            raise ValueError(f"Duplicate clip_id: {clip.clip_id}")
        seen.add(clip.clip_id)

        if clip.action_type not in VALID_ACTION_TYPES:
            raise ValueError(f"Invalid action_type for {clip.clip_id}: {clip.action_type}")

        if clip.fps_target <= 0:
            raise ValueError(f"fps_target must be positive for {clip.clip_id}")
