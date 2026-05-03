"""Render overlay videos from existing pose CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.frame_csv import read_frame_records
from baseball_pose.io.paths import frame_manifest_path, overlay_frame_dir, overlay_video_path, pose_path
from baseball_pose.io.pose_csv import read_pose_records
from baseball_pose.io.video import read_frame, write_video_from_frames
from baseball_pose.pose.schema import PoseRecord
from baseball_pose.visualization.overlays import draw_pose_overlay


@dataclass(frozen=True)
class OverlayRenderResult:
    clip_id: str
    condition_id: str
    overlay_video: Path
    frame_count: int


def render_pose_overlays(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str] | None = None,
    draw_tracks: bool = False,
) -> list[OverlayRenderResult]:
    condition_ids = conditions if conditions is not None else _preferred_overlay_conditions(config.condition_ids)
    confidence_threshold = float(config.raw["postprocess"].get("confidence_threshold", 0.5))
    results: list[OverlayRenderResult] = []

    for clip_id in clip_ids:
        for condition_id in condition_ids:
            source_condition_id = _frame_source_condition(condition_id)
            frames_csv = frame_manifest_path(config.data_dir, clip_id, source_condition_id)
            poses_csv = pose_path(config.data_dir, clip_id, condition_id)
            if not frames_csv.exists() or not poses_csv.exists():
                continue

            frames = read_frame_records(frames_csv)
            records_by_frame = _records_by_frame(read_pose_records(poses_csv))
            target_frame_dir = overlay_frame_dir(config.output_dir, clip_id, condition_id)
            target_frame_dir.mkdir(parents=True, exist_ok=True)
            overlay_paths: list[Path] = []
            tracks: dict[str, list[tuple[int, int]]] | None = (
                {"left_wrist": [], "right_wrist": []} if draw_tracks else None
            )

            for frame in frames:
                image = read_frame(frame.frame_path)
                frame_records = records_by_frame.get(frame.frame_index, [])
                if tracks is not None:
                    _update_tracks(tracks, frame_records, frame.width, frame.height)
                overlay = draw_pose_overlay(
                    image,
                    frame_records,
                    confidence_threshold=confidence_threshold,
                    tracks=tracks,
                )
                overlay_path = target_frame_dir / frame.frame_path.name.replace(
                    source_condition_id,
                    condition_id,
                )
                _write_image(overlay_path, overlay)
                overlay_paths.append(overlay_path)

            video_path = overlay_video_path(config.output_dir, clip_id, condition_id)
            write_video_from_frames(overlay_paths, video_path, fps=config.target_fps)
            results.append(
                OverlayRenderResult(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    overlay_video=video_path,
                    frame_count=len(overlay_paths),
                )
            )

    return results


def _preferred_overlay_conditions(condition_ids: list[str]) -> list[str]:
    body_mask_smoothed = [
        condition_id
        for condition_id in condition_ids
        if condition_id.startswith("body_prior_mask_roi") and condition_id.endswith("_smooth")
    ]
    if body_mask_smoothed:
        return body_mask_smoothed
    center_smoothed = [
        condition_id
        for condition_id in condition_ids
        if condition_id.startswith("center_prior_roi") and condition_id.endswith("_smooth")
    ]
    if center_smoothed:
        return center_smoothed
    smoothed_roi = [
        condition_id
        for condition_id in condition_ids
        if condition_id.endswith("_smooth") and "roi" in condition_id
    ]
    if smoothed_roi:
        return smoothed_roi
    smoothed = [condition_id for condition_id in condition_ids if condition_id.endswith("_smooth")]
    return smoothed if smoothed else condition_ids


def _frame_source_condition(condition_id: str) -> str:
    return condition_id.removesuffix("_smooth")


def _records_by_frame(records: list[PoseRecord]) -> dict[int, list[PoseRecord]]:
    by_frame: dict[int, list[PoseRecord]] = {}
    for record in records:
        by_frame.setdefault(record.frame_index, []).append(record)
    return by_frame


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


def _write_image(path: str | Path, image) -> None:
    cv2 = _require_cv2()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise RuntimeError(f"Could not write overlay frame: {target}")


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for visualization. Install project dependencies first."
        ) from exc

    return cv2
