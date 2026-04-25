"""Run one clip through one condition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.frame_csv import write_frame_records
from baseball_pose.io.metadata import ClipMetadata
from baseball_pose.io.paths import (
    frame_dir,
    frame_manifest_path,
    overlay_frame_dir,
    overlay_video_path,
    pose_path,
)
from baseball_pose.io.pose_csv import write_pose_records
from baseball_pose.io.video import read_frame, sample_video_frames, write_video_from_frames
from baseball_pose.pose.mediapipe_pose import MediaPipePoseEstimator
from baseball_pose.pose.schema import PoseRecord
from baseball_pose.visualization.overlays import draw_pose_overlay

@dataclass(frozen=True)
class ClipRunRequest:
    clip_id: str
    condition_id: str
    config_path: str


def run_clip(request: ClipRunRequest) -> None:
    raise NotImplementedError("Clip execution will be implemented after stage modules are ready.")


@dataclass(frozen=True)
class ClipRunResult:
    clip_id: str
    condition_id: str
    frames_csv: Path
    poses_csv: Path
    overlay_video: Path
    frame_count: int
    pose_record_count: int


def run_baseline_clip(
    clip: ClipMetadata,
    config: RuntimeConfig,
    condition_id: str = "baseline_raw",
    max_frames: int | None = None,
) -> ClipRunResult:
    """Run raw-frame MediaPipe baseline for one clip."""

    if condition_id != "baseline_raw":
        raise ValueError("run_baseline_clip currently supports only baseline_raw.")

    frame_output_dir = frame_dir(config.data_dir, clip.clip_id, condition_id)
    frames = sample_video_frames(
        video_path=clip.source_path,
        clip_id=clip.clip_id,
        output_dir=frame_output_dir,
        target_fps=config.target_fps,
        resize_longest_side=config.resize_longest_side,
        condition_id=condition_id,
        max_frames=max_frames if max_frames is not None else config.max_frames_per_clip,
    )
    frames_csv = frame_manifest_path(config.data_dir, clip.clip_id, condition_id)
    write_frame_records(frames_csv, frames)

    estimator = MediaPipePoseEstimator(
        model_asset_path=config.raw["pose"].get("model_asset_path", "models/pose_landmarker_lite.task"),
        min_detection_confidence=float(config.raw["pose"].get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(config.raw["pose"].get("min_tracking_confidence", 0.5)),
    )
    all_pose_records: list[PoseRecord] = []
    overlay_paths: list[Path] = []
    tracks: dict[str, list[tuple[int, int]]] = {"left_wrist": [], "right_wrist": []}
    overlay_dir = overlay_frame_dir(config.output_dir, clip.clip_id, condition_id)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    try:
        for frame in frames:
            image = read_frame(frame.frame_path)
            records = estimator.estimate_frame(image, frame, condition_id)
            all_pose_records.extend(records)
            _update_tracks(tracks, records, frame.width, frame.height)
            overlay = draw_pose_overlay(
                image,
                records,
                confidence_threshold=float(config.raw["postprocess"].get("confidence_threshold", 0.5)),
                tracks=tracks,
            )
            overlay_path = overlay_dir / frame.frame_path.name
            _write_image(overlay_path, overlay)
            overlay_paths.append(overlay_path)
    finally:
        estimator.close()

    poses_csv = pose_path(config.data_dir, clip.clip_id, condition_id)
    write_pose_records(poses_csv, all_pose_records)

    video_path = overlay_video_path(config.output_dir, clip.clip_id, condition_id)
    write_video_from_frames(overlay_paths, video_path, fps=config.target_fps)

    return ClipRunResult(
        clip_id=clip.clip_id,
        condition_id=condition_id,
        frames_csv=frames_csv,
        poses_csv=poses_csv,
        overlay_video=video_path,
        frame_count=len(frames),
        pose_record_count=len(all_pose_records),
    )


def _update_tracks(
    tracks: dict[str, list[tuple[int, int]]],
    records: list[PoseRecord],
    width: int | None,
    height: int | None,
) -> None:
    if width is None or height is None:
        return
    for record in records:
        if record.joint_name not in tracks or record.x is None or record.y is None:
            continue
        x = int(record.x * width)
        y = int(record.y * height)
        if 0 <= x < width and 0 <= y < height:
            tracks[record.joint_name].append((x, y))


def _write_image(path: Path, image) -> None:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for visualization. Install project dependencies first."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")
