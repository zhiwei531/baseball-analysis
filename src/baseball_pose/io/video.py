"""Video ingestion helpers."""

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
    condition_id: str = "baseline_raw",
    max_frames: int | None = None,
) -> list[FrameRecord]:
    """Sample frames from one video and write them as PNG files."""

    cv2 = _require_cv2()

    source = Path(video_path)
    if not source.exists():
        raise FileNotFoundError(source)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or target_fps
    if source_fps <= 0:
        source_fps = target_fps
    if target_fps <= 0:
        raise ValueError("target_fps must be positive.")

    frame_interval = max(1, round(source_fps / target_fps))
    records: list[FrameRecord] = []
    source_index = 0
    sampled_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if source_index % frame_interval != 0:
                source_index += 1
                continue

            frame = resize_frame(frame, resize_longest_side)
            height, width = frame.shape[:2]
            frame_path = output_path / f"{clip_id}__{condition_id}__frame_{sampled_index:06d}.png"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"Could not write frame: {frame_path}")

            records.append(
                FrameRecord(
                    clip_id=clip_id,
                    frame_index=sampled_index,
                    timestamp_sec=source_index / source_fps,
                    frame_path=frame_path,
                    condition_id=condition_id,
                    width=width,
                    height=height,
                )
            )
            sampled_index += 1
            source_index += 1

            if max_frames is not None and sampled_index >= max_frames:
                break
    finally:
        capture.release()

    return records


def read_frame(path: str | Path):
    cv2 = _require_cv2()
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Could not read frame: {path}")
    return image


def resize_frame(frame, longest_side: int | None):
    if not longest_side:
        return frame
    height, width = frame.shape[:2]
    current_longest = max(width, height)
    if current_longest <= longest_side:
        return frame
    scale = longest_side / current_longest
    new_size = (round(width * scale), round(height * scale))
    cv2 = _require_cv2()
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def write_video_from_frames(
    frame_paths: list[Path],
    output_path: str | Path,
    fps: float,
) -> None:
    """Write a video from image paths."""

    cv2 = _require_cv2()
    if not frame_paths:
        raise ValueError("frame_paths cannot be empty.")

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(target), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {target}")

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read frame: {frame_path}")
            writer.write(frame)
    finally:
        writer.release()


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for video processing. Install project dependencies first."
        ) from exc

    return cv2
