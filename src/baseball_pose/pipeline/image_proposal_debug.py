"""Render image-processing proposal debug videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.frame_csv import read_frame_records
from baseball_pose.io.paths import (
    frame_manifest_path,
    image_proposal_debug_frame_dir,
    image_proposal_debug_video_path,
)
from baseball_pose.io.video import read_frame, write_video_from_frames
from baseball_pose.preprocessing.image_proposal import (
    ImageProposalTracker,
    apply_image_proposal_mask,
    create_center_motion_grabcut_proposal,
    draw_image_proposal_overlay,
)
from baseball_pose.preprocessing.image_proposal_config import image_proposal_roi_config


@dataclass(frozen=True)
class ImageProposalDebugResult:
    clip_id: str
    condition_id: str
    proposal_video: Path
    masked_video: Path
    frame_count: int


def render_image_proposal_debug_videos(
    clip_ids: list[str],
    config: RuntimeConfig,
    source_condition: str = "center_prior_roi",
    condition_id: str = "image_center_motion_grabcut",
    center_x: float = 0.5,
    center_width_ratio: float = 0.54,
    min_area_ratio: float = 0.006,
    grabcut_iterations: int = 1,
    processing_scale: float = 0.45,
    vertical_body_width_ratio: float = 0.22,
    max_frames: int | None = None,
) -> list[ImageProposalDebugResult]:
    cv2 = _require_cv2()
    results: list[ImageProposalDebugResult] = []
    for clip_id in clip_ids:
        proposal_config = image_proposal_roi_config(
            config.raw,
            clip_id,
            "image_center_motion_grabcut_pose",
        )
        clip_center_x = float(proposal_config.get("center_x", center_x))
        clip_center_width_ratio = float(
            proposal_config.get("center_width_ratio", center_width_ratio)
        )
        clip_min_area_ratio = float(proposal_config.get("min_area_ratio", min_area_ratio))
        clip_grabcut_iterations = int(
            proposal_config.get("grabcut_iterations", grabcut_iterations)
        )
        clip_processing_scale = float(
            proposal_config.get("processing_scale", processing_scale)
        )
        clip_vertical_body_width_ratio = float(
            proposal_config.get("vertical_body_width_ratio", vertical_body_width_ratio)
        )
        clip_lower_body_width_ratio = (
            None
            if proposal_config.get("lower_body_width_ratio") is None
            else float(proposal_config["lower_body_width_ratio"])
        )
        clip_lower_body_left_width_ratio = (
            None
            if proposal_config.get("lower_body_left_width_ratio") is None
            else float(proposal_config["lower_body_left_width_ratio"])
        )
        clip_lower_body_right_width_ratio = (
            None
            if proposal_config.get("lower_body_right_width_ratio") is None
            else float(proposal_config["lower_body_right_width_ratio"])
        )
        frames_csv = frame_manifest_path(config.data_dir, clip_id, source_condition)
        if not frames_csv.exists():
            continue
        frames = read_frame_records(frames_csv)
        if max_frames is not None:
            frames = frames[:max_frames]
        if not frames:
            continue

        background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=80,
            varThreshold=24,
            detectShadows=False,
        )
        proposal_paths: list[Path] = []
        masked_paths: list[Path] = []
        proposal_dir = image_proposal_debug_frame_dir(
            config.output_dir,
            clip_id,
            condition_id,
            "proposal_overlay",
        )
        masked_dir = image_proposal_debug_frame_dir(
            config.output_dir,
            clip_id,
            condition_id,
            "masked_frame",
        )
        proposal_dir.mkdir(parents=True, exist_ok=True)
        masked_dir.mkdir(parents=True, exist_ok=True)

        previous_image = None
        previous_mask = None
        tracker = ImageProposalTracker(
            initial_center_x=clip_center_x,
            initial_width_ratio=clip_center_width_ratio,
            center_x=clip_center_x,
            center_width_ratio=clip_center_width_ratio,
            max_offset=float(proposal_config.get("tracker_max_offset", 0.12)),
            max_center_step=float(proposal_config.get("tracker_max_center_step", 0.015)),
            max_width_step=float(proposal_config.get("tracker_max_width_step", 0.025)),
            center_smoothing=float(proposal_config.get("tracker_center_smoothing", 0.55)),
            width_smoothing=float(proposal_config.get("tracker_width_smoothing", 0.45)),
            min_width_ratio=float(proposal_config.get("tracker_min_width_ratio", 0.56)),
            max_width_ratio=float(proposal_config.get("tracker_max_width_ratio", 0.72)),
            warmup_frames=int(proposal_config.get("tracker_warmup_frames", 90)),
        )
        for frame in frames:
            image = read_frame(frame.frame_path)
            proposal = create_center_motion_grabcut_proposal(
                image=image,
                previous_image=previous_image,
                previous_mask=previous_mask,
                background_subtractor=background_subtractor,
                center_x=tracker.center_x,
                center_width_ratio=tracker.center_width_ratio,
                min_area_ratio=clip_min_area_ratio,
                grabcut_iterations=clip_grabcut_iterations,
                processing_scale=clip_processing_scale,
                vertical_body_width_ratio=clip_vertical_body_width_ratio,
                lower_body_width_ratio=clip_lower_body_width_ratio,
                lower_body_left_width_ratio=clip_lower_body_left_width_ratio,
                lower_body_right_width_ratio=clip_lower_body_right_width_ratio,
            )
            proposal_path = proposal_dir / frame.frame_path.name.replace(
                source_condition,
                condition_id,
            )
            masked_path = masked_dir / frame.frame_path.name.replace(
                source_condition,
                condition_id,
            )
            _write_image(proposal_path, draw_image_proposal_overlay(image, proposal))
            _write_image(masked_path, apply_image_proposal_mask(image, proposal))
            proposal_paths.append(proposal_path)
            masked_paths.append(masked_path)
            tracker.update(proposal, image=image)
            previous_image = image
            previous_mask = proposal.mask

        proposal_video = image_proposal_debug_video_path(
            config.output_dir,
            clip_id,
            condition_id,
            "proposal_overlay",
        )
        masked_video = image_proposal_debug_video_path(
            config.output_dir,
            clip_id,
            condition_id,
            "masked_frame",
        )
        write_video_from_frames(proposal_paths, proposal_video, fps=config.target_fps)
        write_video_from_frames(masked_paths, masked_video, fps=config.target_fps)
        results.append(
            ImageProposalDebugResult(
                clip_id=clip_id,
                condition_id=condition_id,
                proposal_video=proposal_video,
                masked_video=masked_video,
                frame_count=len(proposal_paths),
            )
        )

    return results


def _write_image(path: Path, image) -> None:
    cv2 = _require_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image proposal debug frame: {path}")


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for image proposal debug rendering. Install dependencies first."
        ) from exc

    return cv2
