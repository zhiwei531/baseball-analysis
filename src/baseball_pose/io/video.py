"""Video ingestion interfaces.

Implementation will use OpenCV once dependencies are installed. This module
currently defines the data records and function boundaries used by the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameRecord:
    clip_id: str
    frame_index: int
    timestamp_sec: float
    frame_path: Path
    condition_id: str
    width: int | None = None
    height: int | None = None


def sample_video_frames(
    video_path: str | Path,
    clip_id: str,
    output_dir: str | Path,
    target_fps: float,
    resize_longest_side: int | None = None,
) -> list[FrameRecord]:
    """Sample frames from one video.

    This is a planned implementation boundary for Phase 1.
    """

    raise NotImplementedError("Frame sampling will be implemented in Phase 1.")
