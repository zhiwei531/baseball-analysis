"""Render preview videos from existing 3D pose CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.frame_csv import read_frame_records
from baseball_pose.io.paths import frame_manifest_path, overlay3d_frame_dir, overlay3d_video_path
from baseball_pose.io.pose3d_csv import read_pose3d_records
from baseball_pose.io.video import read_frame, write_video_from_frames
from baseball_pose.pipeline.pose3d import _source_condition_for_3d
from baseball_pose.visualization3d.overlays import build_projection_context, draw_pose3d_preview


@dataclass(frozen=True)
class Overlay3DRenderResult:
    clip_id: str
    condition_id: str
    overlay_video: Path
    frame_count: int


def render_pose3d_overlays(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str],
) -> list[Overlay3DRenderResult]:
    results: list[Overlay3DRenderResult] = []

    for clip_id in clip_ids:
        for condition_id in conditions:
            source_condition_id = _frame_source_condition_for_3d(condition_id)
            frames_csv = frame_manifest_path(config.data_dir, clip_id, source_condition_id)
            poses3d_csv = Path(config.data_dir) / "processed" / "poses3d" / clip_id / f"{condition_id}.csv"
            if not frames_csv.exists() or not poses3d_csv.exists():
                continue

            frames = read_frame_records(frames_csv)
            records_by_frame = _records_by_frame(read_pose3d_records(poses3d_csv))
            projection_context = build_projection_context(records_by_frame)
            target_frame_dir = overlay3d_frame_dir(config.output_dir, clip_id, condition_id)
            target_frame_dir.mkdir(parents=True, exist_ok=True)
            overlay_paths: list[Path] = []

            for frame in frames:
                frame_records = records_by_frame.get(frame.frame_index, [])
                if not frame_records:
                    continue
                image = read_frame(frame.frame_path)
                preview = draw_pose3d_preview(image, frame_records, context=projection_context)
                overlay_path = target_frame_dir / frame.frame_path.name.replace(source_condition_id, condition_id)
                _write_image(overlay_path, preview)
                overlay_paths.append(overlay_path)

            if not overlay_paths:
                continue
            video_path = overlay3d_video_path(config.output_dir, clip_id, condition_id)
            write_video_from_frames(overlay_paths, video_path, fps=config.target_fps)
            results.append(
                Overlay3DRenderResult(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    overlay_video=video_path,
                    frame_count=len(overlay_paths),
                )
            )

    return results


def _records_by_frame(records):
    by_frame = {}
    for record in records:
        by_frame.setdefault(record.frame_index, []).append(record)
    return by_frame


def _frame_source_condition_for_3d(condition_id: str) -> str:
    source = condition_id
    if source.endswith("_smooth"):
        source = source[: -len("_smooth")]
    if source.endswith("_3d"):
        source = source[: -len("_3d")]
    return _source_condition_for_3d(source)


def _write_image(path: str | Path, image) -> None:
    cv2 = _require_cv2()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise RuntimeError(f"Could not write 3D overlay frame: {target}")


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for 3D visualization. Install project dependencies first."
        ) from exc

    return cv2
