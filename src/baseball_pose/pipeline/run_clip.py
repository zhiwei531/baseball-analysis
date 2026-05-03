"""Run one clip through one condition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.frame_csv import write_frame_records
from baseball_pose.io.metadata import ClipMetadata
from baseball_pose.io.paths import (
    auto_roi_path,
    frame_dir,
    frame_manifest_path,
    motion_preview_frame_dir,
    motion_preview_video_path,
    overlay_frame_dir,
    overlay_video_path,
    pose_path,
    roi_debug_video_path,
)
from baseball_pose.io.pose_csv import read_pose_records, write_pose_records
from baseball_pose.io.video import read_frame, sample_video_frames, write_video_from_frames
from baseball_pose.pose.mediapipe_pose import MediaPipePoseEstimator
from baseball_pose.pose.schema import PoseRecord
from baseball_pose.preprocessing.body_mask import create_body_prior_masked_crop
from baseball_pose.preprocessing.image_proposal import (
    apply_image_proposal_mask,
    create_center_motion_grabcut_proposal,
)
from baseball_pose.preprocessing.roi import (
    crop_to_roi,
    estimate_clip_auto_roi,
    estimate_center_prior_roi,
    estimate_pose_prior_roi,
    remap_pose_records_to_full_frame,
    write_auto_roi_csv,
    write_roi_debug_video,
)
from baseball_pose.visualization.motion_preview import create_motion_preview
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


@dataclass(frozen=True)
class MotionPreviewResult:
    clip_id: str
    condition_id: str
    frames_csv: Path
    preview_video: Path
    frame_count: int


@dataclass(frozen=True)
class AutoRoiRunResult:
    clip_id: str
    condition_id: str
    frames_csv: Path
    roi_csv: Path
    roi_debug_video: Path
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
        model_asset_path=config.raw["pose"].get(
            "model_asset_path",
            "models/pose_landmarker_lite.task",
        ),
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


def run_motion_preview_clip(
    clip: ClipMetadata,
    config: RuntimeConfig,
    max_frames: int | None = None,
) -> MotionPreviewResult:
    """Sample frames and create an OpenCV point-track preview video."""

    condition_id = "motion_preview"
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

    preview_video = motion_preview_video_path(config.output_dir, clip.clip_id)
    create_motion_preview(
        frames=frames,
        output_dir=motion_preview_frame_dir(config.output_dir, clip.clip_id),
        output_video_path=preview_video,
        fps=config.target_fps,
    )

    return MotionPreviewResult(
        clip_id=clip.clip_id,
        condition_id=condition_id,
        frames_csv=frames_csv,
        preview_video=preview_video,
        frame_count=len(frames),
    )


def run_auto_roi_clip(
    clip: ClipMetadata,
    config: RuntimeConfig,
    condition_id: str = "auto_roi_raw",
    max_frames: int | None = None,
) -> AutoRoiRunResult:
    """Run MediaPipe on an automatically cropped fixed ROI and remap to full frame."""

    if condition_id != "auto_roi_raw":
        raise ValueError("run_auto_roi_clip currently supports only auto_roi_raw.")

    condition_config = config.raw["conditions"][condition_id]
    roi_config = condition_config.get("roi", {})
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

    roi_result = estimate_clip_auto_roi(
        frames,
        expansion=float(roi_config.get("expansion", 0.35)),
        max_frames=int(roi_config.get("proposal_frames", 60)),
        min_area_ratio=float(roi_config.get("min_area_ratio", 0.002)),
    )
    roi_csv = auto_roi_path(config.data_dir, clip.clip_id, condition_id)
    write_auto_roi_csv(roi_csv, roi_result)

    debug_video = roi_debug_video_path(config.output_dir, clip.clip_id, condition_id)
    write_roi_debug_video(frames, roi_result.roi, debug_video, fps=config.target_fps)

    estimator = MediaPipePoseEstimator(
        model_asset_path=config.raw["pose"].get(
            "model_asset_path",
            "models/pose_landmarker_lite.task",
        ),
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
            crop = crop_to_roi(image, roi_result.roi)
            crop_records = estimator.estimate_frame(crop, frame, condition_id)
            records = remap_pose_records_to_full_frame(
                crop_records,
                roi=roi_result.roi,
                image_width=frame.width or image.shape[1],
                image_height=frame.height or image.shape[0],
            )
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

    return AutoRoiRunResult(
        clip_id=clip.clip_id,
        condition_id=condition_id,
        frames_csv=frames_csv,
        roi_csv=roi_csv,
        roi_debug_video=debug_video,
        poses_csv=poses_csv,
        overlay_video=video_path,
        frame_count=len(frames),
        pose_record_count=len(all_pose_records),
    )


def run_pose_prior_roi_clip(
    clip: ClipMetadata,
    config: RuntimeConfig,
    condition_id: str = "auto_roi_pose_prior",
    max_frames: int | None = None,
) -> AutoRoiRunResult:
    """Run MediaPipe on an ROI estimated from baseline pose landmarks."""

    if condition_id != "auto_roi_pose_prior":
        raise ValueError("run_pose_prior_roi_clip currently supports only auto_roi_pose_prior.")

    condition_config = config.raw["conditions"][condition_id]
    roi_config = condition_config.get("roi", {})
    source_condition = roi_config.get("source_condition", "baseline_raw")
    source_pose_path = pose_path(config.data_dir, clip.clip_id, source_condition)
    if not source_pose_path.exists():
        raise FileNotFoundError(
            f"Pose-prior ROI requires existing baseline pose CSV: {source_pose_path}"
        )

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

    first_frame = read_frame(frames[0].frame_path)
    image_height, image_width = first_frame.shape[:2]
    baseline_records = read_pose_records(source_pose_path)
    roi_result = estimate_pose_prior_roi(
        baseline_records,
        image_width=image_width,
        image_height=image_height,
        condition_id=condition_id,
        confidence_threshold=float(roi_config.get("confidence_threshold", 0.5)),
        min_joints_per_frame=int(roi_config.get("min_joints_per_frame", 4)),
        horizontal_padding=float(roi_config.get("horizontal_padding", 0.35)),
        top_padding=float(roi_config.get("top_padding", 0.45)),
        bottom_padding=float(roi_config.get("bottom_padding", 0.25)),
    )
    roi_csv = auto_roi_path(config.data_dir, clip.clip_id, condition_id)
    write_auto_roi_csv(roi_csv, roi_result)

    debug_video = roi_debug_video_path(config.output_dir, clip.clip_id, condition_id)
    write_roi_debug_video(frames, roi_result.roi, debug_video, fps=config.target_fps)

    estimator = MediaPipePoseEstimator(
        model_asset_path=config.raw["pose"].get(
            "model_asset_path",
            "models/pose_landmarker_lite.task",
        ),
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
            crop = crop_to_roi(image, roi_result.roi)
            crop_records = estimator.estimate_frame(crop, frame, condition_id)
            records = remap_pose_records_to_full_frame(
                crop_records,
                roi=roi_result.roi,
                image_width=frame.width or image.shape[1],
                image_height=frame.height or image.shape[0],
            )
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

    return AutoRoiRunResult(
        clip_id=clip.clip_id,
        condition_id=condition_id,
        frames_csv=frames_csv,
        roi_csv=roi_csv,
        roi_debug_video=debug_video,
        poses_csv=poses_csv,
        overlay_video=video_path,
        frame_count=len(frames),
        pose_record_count=len(all_pose_records),
    )


def run_center_prior_roi_clip(
    clip: ClipMetadata,
    config: RuntimeConfig,
    condition_id: str = "center_prior_roi",
    max_frames: int | None = None,
) -> AutoRoiRunResult:
    """Run MediaPipe inside a hard-coded center ROI for centered-subject videos."""

    if condition_id != "center_prior_roi":
        raise ValueError("run_center_prior_roi_clip currently supports only center_prior_roi.")

    condition_config = config.raw["conditions"][condition_id]
    roi_config = condition_config.get("roi", {})
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

    first_frame = read_frame(frames[0].frame_path)
    image_height, image_width = first_frame.shape[:2]
    roi_result = estimate_center_prior_roi(
        clip_id=clip.clip_id,
        image_width=image_width,
        image_height=image_height,
        condition_id=condition_id,
        center_x=float(roi_config.get("center_x", 0.5)),
        center_y=float(roi_config.get("center_y", 0.5)),
        width_ratio=float(roi_config.get("width_ratio", 0.62)),
        height_ratio=float(roi_config.get("height_ratio", 1.0)),
    )
    roi_csv = auto_roi_path(config.data_dir, clip.clip_id, condition_id)
    write_auto_roi_csv(roi_csv, roi_result)

    debug_video = roi_debug_video_path(config.output_dir, clip.clip_id, condition_id)
    write_roi_debug_video(frames, roi_result.roi, debug_video, fps=config.target_fps)

    estimator = MediaPipePoseEstimator(
        model_asset_path=config.raw["pose"].get(
            "model_asset_path",
            "models/pose_landmarker_lite.task",
        ),
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
            crop = crop_to_roi(image, roi_result.roi)
            crop_records = estimator.estimate_frame(crop, frame, condition_id)
            records = remap_pose_records_to_full_frame(
                crop_records,
                roi=roi_result.roi,
                image_width=frame.width or image.shape[1],
                image_height=frame.height or image.shape[0],
            )
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

    return AutoRoiRunResult(
        clip_id=clip.clip_id,
        condition_id=condition_id,
        frames_csv=frames_csv,
        roi_csv=roi_csv,
        roi_debug_video=debug_video,
        poses_csv=poses_csv,
        overlay_video=video_path,
        frame_count=len(frames),
        pose_record_count=len(all_pose_records),
    )


def run_body_prior_mask_roi_clip(
    clip: ClipMetadata,
    config: RuntimeConfig,
    condition_id: str = "body_prior_mask_roi",
    max_frames: int | None = None,
) -> AutoRoiRunResult:
    """Run MediaPipe on an irregular body-prior masked ROI."""

    if condition_id != "body_prior_mask_roi":
        raise ValueError("run_body_prior_mask_roi_clip currently supports only body_prior_mask_roi.")

    condition_config = config.raw["conditions"][condition_id]
    roi_config = condition_config.get("roi", {})
    source_condition = roi_config.get("source_condition", "center_prior_roi_smooth")
    source_pose_path = pose_path(config.data_dir, clip.clip_id, source_condition)
    if not source_pose_path.exists():
        raise FileNotFoundError(
            f"Body-prior mask ROI requires existing source pose CSV: {source_pose_path}"
        )

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

    first_frame = read_frame(frames[0].frame_path)
    image_height, image_width = first_frame.shape[:2]
    fallback_roi_result = estimate_center_prior_roi(
        clip_id=clip.clip_id,
        image_width=image_width,
        image_height=image_height,
        condition_id=condition_id,
        center_x=float(roi_config.get("fallback_center_x", 0.5)),
        center_y=float(roi_config.get("fallback_center_y", 0.5)),
        width_ratio=float(roi_config.get("fallback_width_ratio", 0.62)),
        height_ratio=float(roi_config.get("fallback_height_ratio", 1.0)),
    )
    roi_csv = auto_roi_path(config.data_dir, clip.clip_id, condition_id)
    write_auto_roi_csv(roi_csv, fallback_roi_result)

    prior_records_by_frame = _records_by_frame(read_pose_records(source_pose_path))
    debug_video = roi_debug_video_path(config.output_dir, clip.clip_id, condition_id)
    write_roi_debug_video(frames, fallback_roi_result.roi, debug_video, fps=config.target_fps)

    estimator = MediaPipePoseEstimator(
        model_asset_path=config.raw["pose"].get(
            "model_asset_path",
            "models/pose_landmarker_lite.task",
        ),
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
            masked_crop = create_body_prior_masked_crop(
                image=image,
                prior_records=prior_records_by_frame.get(frame.frame_index, []),
                image_width=frame.width or image.shape[1],
                image_height=frame.height or image.shape[0],
                fallback_roi=fallback_roi_result.roi,
                confidence_threshold=float(roi_config.get("confidence_threshold", 0.5)),
                padding_ratio=float(roi_config.get("padding_ratio", 0.55)),
                min_width_ratio=float(roi_config.get("min_width_ratio", 0.28)),
                min_height_ratio=float(roi_config.get("min_height_ratio", 0.50)),
                limb_thickness_ratio=float(roi_config.get("limb_thickness_ratio", 0.045)),
                joint_radius_ratio=float(roi_config.get("joint_radius_ratio", 0.035)),
            )
            crop_records = estimator.estimate_frame(masked_crop.image, frame, condition_id)
            records = remap_pose_records_to_full_frame(
                crop_records,
                roi=masked_crop.roi,
                image_width=frame.width or image.shape[1],
                image_height=frame.height or image.shape[0],
            )
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

    return AutoRoiRunResult(
        clip_id=clip.clip_id,
        condition_id=condition_id,
        frames_csv=frames_csv,
        roi_csv=roi_csv,
        roi_debug_video=debug_video,
        poses_csv=poses_csv,
        overlay_video=video_path,
        frame_count=len(frames),
        pose_record_count=len(all_pose_records),
    )


def run_image_proposal_roi_clip(
    clip: ClipMetadata,
    config: RuntimeConfig,
    condition_id: str = "image_center_motion_grabcut_pose",
    max_frames: int | None = None,
) -> ClipRunResult:
    """Run MediaPipe on dynamic crops from the skeleton-free image proposal."""

    condition_config = config.raw["conditions"].get(condition_id, {})
    roi_config = condition_config.get("roi", {})
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
        model_asset_path=config.raw["pose"].get(
            "model_asset_path",
            "models/pose_landmarker_lite.task",
        ),
        min_detection_confidence=float(config.raw["pose"].get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(config.raw["pose"].get("min_tracking_confidence", 0.5)),
    )
    cv2 = _require_cv2()
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=80,
        varThreshold=24,
        detectShadows=False,
    )
    all_pose_records: list[PoseRecord] = []
    overlay_paths: list[Path] = []
    tracks: dict[str, list[tuple[int, int]]] = {"left_wrist": [], "right_wrist": []}
    overlay_dir = overlay_frame_dir(config.output_dir, clip.clip_id, condition_id)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    previous_image = None
    previous_mask = None

    try:
        for frame in frames:
            image = read_frame(frame.frame_path)
            proposal = create_center_motion_grabcut_proposal(
                image=image,
                previous_image=previous_image,
                previous_mask=previous_mask,
                background_subtractor=background_subtractor,
                center_x=float(roi_config.get("center_x", 0.5)),
                center_width_ratio=float(roi_config.get("center_width_ratio", 0.54)),
                min_area_ratio=float(roi_config.get("min_area_ratio", 0.006)),
                grabcut_iterations=int(roi_config.get("grabcut_iterations", 1)),
                processing_scale=float(roi_config.get("processing_scale", 0.45)),
                vertical_body_width_ratio=float(roi_config.get("vertical_body_width_ratio", 0.22)),
            )
            masked_image = apply_image_proposal_mask(image, proposal)
            crop = crop_to_roi(masked_image, proposal.roi)
            crop_records = estimator.estimate_frame(crop, frame, condition_id)
            records = remap_pose_records_to_full_frame(
                crop_records,
                roi=proposal.roi,
                image_width=frame.width or image.shape[1],
                image_height=frame.height or image.shape[0],
            )
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
            previous_image = image
            previous_mask = proposal.mask
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


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for image proposal ROI inference. Install dependencies first."
        ) from exc

    return cv2
