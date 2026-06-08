"""Path conventions for generated artifacts."""

from __future__ import annotations

from pathlib import Path


def frame_dir(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "interim" / "frames" / clip_id / condition_id


def frame_manifest_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "interim" / "frames" / clip_id / f"{condition_id}.csv"


def pose_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "poses" / clip_id / f"{condition_id}.csv"


def feature_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "features" / clip_id / f"{condition_id}.csv"


def object_track_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "objects" / clip_id / f"{condition_id}.csv"


def object_feature_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "object_features" / clip_id / f"{condition_id}.csv"


def pose3d_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "poses3d" / clip_id / f"{condition_id}.csv"


def feature3d_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "features3d" / clip_id / f"{condition_id}.csv"


def metric_path(data_dir: str | Path, experiment_id: str) -> Path:
    return Path(data_dir) / "processed" / "metrics" / f"{experiment_id}.csv"


def report_summary_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "metrics" / "report_summaries" / clip_id / f"{condition_id}.json"


def report_prompt_dir(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "metrics" / "report_prompts" / clip_id / condition_id


def report_llm_dir(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "processed" / "metrics" / "report_llm" / clip_id / condition_id


def overlay_frame_dir(output_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(output_dir) / "overlays" / "frames" / clip_id / condition_id


def overlay_video_path(output_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(output_dir) / "overlays" / f"{clip_id}__{condition_id}.mp4"


def object_overlay_frame_dir(output_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(output_dir) / "object_overlays" / "frames" / clip_id / condition_id


def object_overlay_video_path(output_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(output_dir) / "object_overlays" / f"{clip_id}__{condition_id}.mp4"


def action_window_video_path(output_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(output_dir) / "action_windows" / f"{clip_id}__{condition_id}__action_window.mp4"


def overlay3d_frame_dir(output_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(output_dir) / "overlays3d" / "frames" / clip_id / condition_id


def overlay3d_video_path(output_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(output_dir) / "overlays3d" / f"{clip_id}__{condition_id}.mp4"


def motion_preview_frame_dir(output_dir: str | Path, clip_id: str) -> Path:
    return Path(output_dir) / "motion_preview" / "frames" / clip_id


def motion_preview_video_path(output_dir: str | Path, clip_id: str) -> Path:
    return Path(output_dir) / "motion_preview" / f"{clip_id}__motion_preview.mp4"


def auto_roi_path(data_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(data_dir) / "interim" / "rois" / clip_id / f"{condition_id}.csv"


def roi_debug_video_path(output_dir: str | Path, clip_id: str, condition_id: str) -> Path:
    return Path(output_dir) / "roi_debug" / f"{clip_id}__{condition_id}.mp4"


def body_mask_debug_frame_dir(
    output_dir: str | Path,
    clip_id: str,
    condition_id: str,
    debug_type: str,
) -> Path:
    return Path(output_dir) / "body_mask_debug" / "frames" / clip_id / condition_id / debug_type


def body_mask_debug_video_path(
    output_dir: str | Path,
    clip_id: str,
    condition_id: str,
    debug_type: str,
) -> Path:
    return Path(output_dir) / "body_mask_debug" / f"{clip_id}__{condition_id}__{debug_type}.mp4"


def image_proposal_debug_frame_dir(
    output_dir: str | Path,
    clip_id: str,
    condition_id: str,
    debug_type: str,
) -> Path:
    return Path(output_dir) / "image_proposal_debug" / "frames" / clip_id / condition_id / debug_type


def image_proposal_debug_video_path(
    output_dir: str | Path,
    clip_id: str,
    condition_id: str,
    debug_type: str,
) -> Path:
    return Path(output_dir) / "image_proposal_debug" / f"{clip_id}__{condition_id}__{debug_type}.mp4"
