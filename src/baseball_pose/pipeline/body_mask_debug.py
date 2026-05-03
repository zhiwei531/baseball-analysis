"""Render intermediate body-prior mask debug videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.frame_csv import read_frame_records
from baseball_pose.io.paths import (
    body_mask_debug_frame_dir,
    body_mask_debug_video_path,
    frame_manifest_path,
    pose_path,
)
from baseball_pose.io.pose_csv import read_pose_records
from baseball_pose.io.video import read_frame, write_video_from_frames
from baseball_pose.pose.schema import PoseRecord
from baseball_pose.preprocessing.body_mask import (
    create_body_prior_masked_crop,
    draw_body_prior_debug_overlay,
    paste_masked_crop_on_full_frame,
)
from baseball_pose.preprocessing.roi import estimate_center_prior_roi


@dataclass(frozen=True)
class BodyMaskDebugResult:
    clip_id: str
    condition_id: str
    proposal_video: Path
    masked_video: Path
    frame_count: int


def render_body_mask_debug_videos(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str] | None = None,
) -> list[BodyMaskDebugResult]:
    condition_ids = conditions if conditions is not None else ["body_prior_mask_roi"]
    results: list[BodyMaskDebugResult] = []

    for clip_id in clip_ids:
        for condition_id in condition_ids:
            condition_config = config.raw["conditions"][condition_id]
            roi_config = condition_config.get("roi", {})
            source_condition = roi_config.get("source_condition", "center_prior_roi_smooth")
            frames_csv = frame_manifest_path(config.data_dir, clip_id, condition_id)
            source_pose_csv = pose_path(config.data_dir, clip_id, source_condition)
            if not frames_csv.exists() or not source_pose_csv.exists():
                continue

            frames = read_frame_records(frames_csv)
            prior_records_by_frame = _records_by_frame(read_pose_records(source_pose_csv))
            if not frames:
                continue

            first_image = read_frame(frames[0].frame_path)
            image_height, image_width = first_image.shape[:2]
            fallback_roi = estimate_center_prior_roi(
                clip_id=clip_id,
                image_width=image_width,
                image_height=image_height,
                condition_id=condition_id,
                center_x=float(roi_config.get("fallback_center_x", 0.5)),
                center_y=float(roi_config.get("fallback_center_y", 0.5)),
                width_ratio=float(roi_config.get("fallback_width_ratio", 0.62)),
                height_ratio=float(roi_config.get("fallback_height_ratio", 1.0)),
            ).roi

            proposal_paths: list[Path] = []
            masked_paths: list[Path] = []
            proposal_dir = body_mask_debug_frame_dir(
                config.output_dir,
                clip_id,
                condition_id,
                "proposal_overlay",
            )
            masked_dir = body_mask_debug_frame_dir(
                config.output_dir,
                clip_id,
                condition_id,
                "masked_frame",
            )
            proposal_dir.mkdir(parents=True, exist_ok=True)
            masked_dir.mkdir(parents=True, exist_ok=True)

            for frame in frames:
                image = read_frame(frame.frame_path)
                masked_crop = create_body_prior_masked_crop(
                    image=image,
                    prior_records=prior_records_by_frame.get(frame.frame_index, []),
                    image_width=frame.width or image.shape[1],
                    image_height=frame.height or image.shape[0],
                    fallback_roi=fallback_roi,
                    confidence_threshold=float(roi_config.get("confidence_threshold", 0.5)),
                    padding_ratio=float(roi_config.get("padding_ratio", 0.55)),
                    min_width_ratio=float(roi_config.get("min_width_ratio", 0.28)),
                    min_height_ratio=float(roi_config.get("min_height_ratio", 0.50)),
                    limb_thickness_ratio=float(roi_config.get("limb_thickness_ratio", 0.045)),
                    joint_radius_ratio=float(roi_config.get("joint_radius_ratio", 0.035)),
                )
                proposal_path = proposal_dir / frame.frame_path.name
                masked_path = masked_dir / frame.frame_path.name
                _write_image(proposal_path, draw_body_prior_debug_overlay(image, masked_crop))
                _write_image(masked_path, paste_masked_crop_on_full_frame(image, masked_crop))
                proposal_paths.append(proposal_path)
                masked_paths.append(masked_path)

            proposal_video = body_mask_debug_video_path(
                config.output_dir,
                clip_id,
                condition_id,
                "proposal_overlay",
            )
            masked_video = body_mask_debug_video_path(
                config.output_dir,
                clip_id,
                condition_id,
                "masked_frame",
            )
            write_video_from_frames(proposal_paths, proposal_video, fps=config.target_fps)
            write_video_from_frames(masked_paths, masked_video, fps=config.target_fps)
            results.append(
                BodyMaskDebugResult(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    proposal_video=proposal_video,
                    masked_video=masked_video,
                    frame_count=len(proposal_paths),
                )
            )

    return results


def _records_by_frame(records: list[PoseRecord]) -> dict[int, list[PoseRecord]]:
    by_frame: dict[int, list[PoseRecord]] = {}
    for record in records:
        by_frame.setdefault(record.frame_index, []).append(record)
    return by_frame


def _write_image(path: Path, image) -> None:
    cv2 = _require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write body mask debug frame: {path}")


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for body-mask debug rendering. Install project dependencies first."
        ) from exc

    return cv2
