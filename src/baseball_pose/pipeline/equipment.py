"""Pipeline orchestration for bat and ball object tracking."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

from baseball_pose.config import RuntimeConfig, resolve_postprocess_config
from baseball_pose.equipment.detection import EquipmentTrackingConfig, detect_equipment_tracks
from baseball_pose.equipment.features import extract_object_motion_features
from baseball_pose.io.frame_csv import read_frame_records
from baseball_pose.io.object_csv import (
    read_object_tracks,
    write_object_features,
    write_object_tracks,
)
from baseball_pose.io.paths import (
    frame_manifest_path,
    object_feature_path,
    object_overlay_frame_dir,
    object_overlay_video_path,
    object_track_path,
    pose_path,
)
from baseball_pose.io.video import read_frame, write_video_from_frames
from baseball_pose.visualization.equipment import draw_equipment_overlay


@dataclass(frozen=True)
class EquipmentTrackingResult:
    clip_id: str
    condition_id: str
    object_csv: Path
    record_count: int


@dataclass(frozen=True)
class ObjectFeatureResult:
    clip_id: str
    condition_id: str
    feature_csv: Path
    frame_count: int


@dataclass(frozen=True)
class ObjectOverlayResult:
    clip_id: str
    condition_id: str
    overlay_video: Path
    frame_count: int


def track_equipment_files(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str] | None = None,
) -> list[EquipmentTrackingResult]:
    condition_ids = conditions if conditions is not None else _equipment_conditions(config)
    raw_tracking_config = config.raw.get("equipment_tracking", {})
    tracking_config = _tracking_config(raw_tracking_config)
    disable_bat_clip_ids = set(raw_tracking_config.get("disable_bat_clip_ids", []))
    action_types = _action_types_by_clip(config)
    results: list[EquipmentTrackingResult] = []
    for clip_id in clip_ids:
        postprocess_config = resolve_postprocess_config(config.raw, clip_id)
        pose_thresholds = postprocess_config.get("confidence_thresholds", {})
        clip_tracking_config = EquipmentTrackingConfig(
            **{
                **tracking_config.__dict__,
                "pose_thresholds": pose_thresholds if isinstance(pose_thresholds, dict) else {},
                "detect_bat": (
                    tracking_config.detect_bat
                    and action_types.get(clip_id) != "pitching"
                    and clip_id not in disable_bat_clip_ids
                ),
            }
        )
        for condition_id in condition_ids:
            frame_condition_id = _frame_source_condition(condition_id)
            frames_csv = frame_manifest_path(config.data_dir, clip_id, frame_condition_id)
            poses_csv = pose_path(config.data_dir, clip_id, condition_id)
            if not frames_csv.exists():
                continue
            records = detect_equipment_tracks(
                frames_csv=frames_csv,
                clip_id=clip_id,
                condition_id=condition_id,
                pose_csv=poses_csv if poses_csv.exists() else None,
                config=clip_tracking_config,
            )
            output_path = object_track_path(config.data_dir, clip_id, condition_id)
            write_object_tracks(output_path, records)
            results.append(
                EquipmentTrackingResult(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    object_csv=output_path,
                    record_count=len(records),
                )
            )
    return results


def extract_object_feature_files(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str] | None = None,
) -> list[ObjectFeatureResult]:
    condition_ids = conditions if conditions is not None else _equipment_conditions(config)
    raw_tracking_config = config.raw.get("equipment_tracking", {})
    bat_confidence_threshold = float(raw_tracking_config.get("feature_bat_confidence_threshold", 0.55))
    ball_confidence_threshold = float(raw_tracking_config.get("feature_ball_confidence_threshold", 0.45))
    max_bat_speed_px_s = _optional_float(raw_tracking_config.get("feature_max_bat_speed_px_s", 8000.0))
    max_ball_speed_px_s = _optional_float(raw_tracking_config.get("feature_max_ball_speed_px_s", 12000.0))
    results: list[ObjectFeatureResult] = []
    for clip_id in clip_ids:
        for condition_id in condition_ids:
            source_path = object_track_path(config.data_dir, clip_id, condition_id)
            if not source_path.exists():
                continue
            rows = extract_object_motion_features(
                read_object_tracks(source_path),
                bat_confidence_threshold=bat_confidence_threshold,
                ball_confidence_threshold=ball_confidence_threshold,
                max_bat_speed_px_s=max_bat_speed_px_s,
                max_ball_speed_px_s=max_ball_speed_px_s,
            )
            output_path = object_feature_path(config.data_dir, clip_id, condition_id)
            write_object_features(output_path, rows)
            results.append(
                ObjectFeatureResult(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    feature_csv=output_path,
                    frame_count=len(rows),
                )
            )
    return results


def render_object_overlays(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str] | None = None,
    draw_tracks: bool = False,
) -> list[ObjectOverlayResult]:
    condition_ids = conditions if conditions is not None else _equipment_conditions(config)
    results: list[ObjectOverlayResult] = []
    for clip_id in clip_ids:
        for condition_id in condition_ids:
            frame_condition_id = _frame_source_condition(condition_id)
            frames_csv = frame_manifest_path(config.data_dir, clip_id, frame_condition_id)
            objects_csv = object_track_path(config.data_dir, clip_id, condition_id)
            if not frames_csv.exists() or not objects_csv.exists():
                continue
            frames = read_frame_records(frames_csv)
            records_by_frame = _records_by_frame(read_object_tracks(objects_csv))
            target_frame_dir = object_overlay_frame_dir(config.output_dir, clip_id, condition_id)
            target_frame_dir.mkdir(parents=True, exist_ok=True)
            overlay_paths: list[Path] = []
            tracks: dict[str, list[tuple[int, int]]] | None = {"bat": [], "ball": []} if draw_tracks else None
            for frame in frames:
                image = read_frame(frame.frame_path)
                frame_records = records_by_frame.get(frame.frame_index, [])
                if tracks is not None:
                    _update_tracks(tracks, frame_records, frame.width, frame.height)
                overlay = draw_equipment_overlay(image, frame_records, tracks=tracks)
                overlay_path = target_frame_dir / frame.frame_path.name.replace(
                    frame_condition_id,
                    condition_id,
                )
                _write_image(overlay_path, overlay)
                overlay_paths.append(overlay_path)
            video_path = object_overlay_video_path(config.output_dir, clip_id, condition_id)
            write_video_from_frames(overlay_paths, video_path, fps=config.target_fps)
            results.append(
                ObjectOverlayResult(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    overlay_video=video_path,
                    frame_count=len(overlay_paths),
                )
            )
    return results


def _equipment_conditions(config: RuntimeConfig) -> list[str]:
    configured = config.raw.get("equipment_tracking", {}).get("conditions")
    if isinstance(configured, list) and configured:
        return [str(item) for item in configured]
    return [condition_id for condition_id in config.condition_ids if condition_id.endswith("_smooth")]


def _tracking_config(raw: dict[str, object]) -> EquipmentTrackingConfig:
    allowed = set(EquipmentTrackingConfig.__dataclass_fields__)
    values = {key: value for key, value in raw.items() if key in allowed}
    return EquipmentTrackingConfig(**values)


def _action_types_by_clip(config: RuntimeConfig) -> dict[str, str]:
    try:
        with config.clips_file.open("r", encoding="utf-8", newline="") as handle:
            return {
                row["clip_id"]: row.get("action_type", "")
                for row in csv.DictReader(handle)
                if row.get("clip_id")
            }
    except FileNotFoundError:
        return {}


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _frame_source_condition(condition_id: str) -> str:
    source_condition_id = condition_id.removesuffix("_smooth")
    return source_condition_id.removesuffix("_complete")


def _records_by_frame(records) -> dict[int, list]:
    by_frame: dict[int, list] = {}
    for record in records:
        by_frame.setdefault(record.frame_index, []).append(record)
    return by_frame


def _update_tracks(
    tracks: dict[str, list[tuple[int, int]]],
    records,
    width: int | None,
    height: int | None,
) -> None:
    if width is None or height is None:
        return
    for record in records:
        if record.object_name not in tracks or record.x is None or record.y is None:
            continue
        tracks[record.object_name].append((int(record.x * width), int(record.y * height)))


def _write_image(path: str | Path, image) -> None:
    cv2 = _require_cv2()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image):
        raise RuntimeError(f"Could not write object overlay frame: {target}")


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for object overlay rendering. Install project dependencies first."
        ) from exc

    return cv2
