"""Run an experiment matrix."""

from __future__ import annotations

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.metadata import ClipMetadata
from baseball_pose.pipeline.run_clip import (
    AutoRoiRunResult,
    ClipRunResult,
    MotionPreviewResult,
    run_auto_roi_clip,
    run_baseline_clip,
    run_body_prior_mask_roi_clip,
    run_center_prior_roi_clip,
    run_image_proposal_roi_clip,
    run_motion_preview_clip,
    run_pose_prior_roi_clip,
)


def planned_runs(config: RuntimeConfig) -> list[tuple[str, str]]:
    return [(clip_id, condition_id) for clip_id in config.clip_ids for condition_id in config.condition_ids]


def run_baseline_experiment(
    clips: list[ClipMetadata],
    config: RuntimeConfig,
    clip_ids: set[str] | None = None,
    max_frames: int | None = None,
) -> list[ClipRunResult]:
    selected = [clip for clip in clips if clip_ids is None or clip.clip_id in clip_ids]
    results: list[ClipRunResult] = []
    for clip in selected:
        results.append(run_baseline_clip(clip, config, max_frames=max_frames))
    return results


def run_motion_preview_experiment(
    clips: list[ClipMetadata],
    config: RuntimeConfig,
    clip_ids: set[str] | None = None,
    max_frames: int | None = None,
) -> list[MotionPreviewResult]:
    selected = [clip for clip in clips if clip_ids is None or clip.clip_id in clip_ids]
    results: list[MotionPreviewResult] = []
    for clip in selected:
        results.append(run_motion_preview_clip(clip, config, max_frames=max_frames))
    return results


def run_auto_roi_experiment(
    clips: list[ClipMetadata],
    config: RuntimeConfig,
    clip_ids: set[str] | None = None,
    max_frames: int | None = None,
) -> list[AutoRoiRunResult]:
    selected = [clip for clip in clips if clip_ids is None or clip.clip_id in clip_ids]
    results: list[AutoRoiRunResult] = []
    for clip in selected:
        results.append(run_auto_roi_clip(clip, config, max_frames=max_frames))
    return results


def run_pose_prior_roi_experiment(
    clips: list[ClipMetadata],
    config: RuntimeConfig,
    clip_ids: set[str] | None = None,
    max_frames: int | None = None,
) -> list[AutoRoiRunResult]:
    selected = [clip for clip in clips if clip_ids is None or clip.clip_id in clip_ids]
    results: list[AutoRoiRunResult] = []
    for clip in selected:
        results.append(run_pose_prior_roi_clip(clip, config, max_frames=max_frames))
    return results


def run_center_prior_roi_experiment(
    clips: list[ClipMetadata],
    config: RuntimeConfig,
    clip_ids: set[str] | None = None,
    max_frames: int | None = None,
) -> list[AutoRoiRunResult]:
    selected = [clip for clip in clips if clip_ids is None or clip.clip_id in clip_ids]
    results: list[AutoRoiRunResult] = []
    for clip in selected:
        results.append(run_center_prior_roi_clip(clip, config, max_frames=max_frames))
    return results


def run_body_prior_mask_roi_experiment(
    clips: list[ClipMetadata],
    config: RuntimeConfig,
    clip_ids: set[str] | None = None,
    max_frames: int | None = None,
) -> list[AutoRoiRunResult]:
    selected = [clip for clip in clips if clip_ids is None or clip.clip_id in clip_ids]
    results: list[AutoRoiRunResult] = []
    for clip in selected:
        results.append(run_body_prior_mask_roi_clip(clip, config, max_frames=max_frames))
    return results


def run_image_proposal_roi_experiment(
    clips: list[ClipMetadata],
    config: RuntimeConfig,
    clip_ids: set[str] | None = None,
    max_frames: int | None = None,
) -> list[ClipRunResult]:
    selected = [clip for clip in clips if clip_ids is None or clip.clip_id in clip_ids]
    results: list[ClipRunResult] = []
    for clip in selected:
        results.append(run_image_proposal_roi_clip(clip, config, max_frames=max_frames))
    return results
