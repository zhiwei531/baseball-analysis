"""Export compact action-window videos from sampled frame records."""

from __future__ import annotations

from dataclasses import dataclass

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.feature_csv import read_feature_rows
from baseball_pose.io.frame_csv import read_frame_records
from baseball_pose.io.metadata import load_clips
from baseball_pose.io.paths import action_window_video_path, feature_path, frame_manifest_path
from baseball_pose.io.video import write_video_from_frames
from baseball_pose.pipeline.report_window import detect_action_video_window


@dataclass(frozen=True)
class ActionWindowVideoResult:
    clip_id: str
    condition_id: str
    source_condition_id: str
    video_path: str
    frame_count: int
    start_frame: int
    end_frame: int


def export_action_window_videos(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str] | None = None,
) -> list[ActionWindowVideoResult]:
    condition_ids = conditions if conditions is not None else config.condition_ids
    clip_lookup = {clip.clip_id: clip for clip in load_clips(config.clips_file)}
    results: list[ActionWindowVideoResult] = []

    for clip_id in clip_ids:
        clip = clip_lookup.get(clip_id)
        if clip is None:
            continue
        for condition_id in condition_ids:
            features_csv = feature_path(config.data_dir, clip_id, condition_id)
            source_condition_id = _frame_source_condition(condition_id)
            frames_csv = frame_manifest_path(config.data_dir, clip_id, source_condition_id)
            if not features_csv.exists() or not frames_csv.exists():
                continue

            rows = read_feature_rows(features_csv)
            window = detect_action_video_window(rows, action_type=clip.action_type)
            if window is None:
                continue

            frames = read_frame_records(frames_csv)
            selected_paths = [
                frame.frame_path
                for frame in frames
                if window.start_frame <= frame.frame_index <= window.end_frame
            ]
            if not selected_paths:
                continue

            output_path = action_window_video_path(config.output_dir, clip_id, condition_id)
            fps = clip.fps_target if clip.fps_target > 0 else config.target_fps
            write_video_from_frames(selected_paths, output_path, fps=fps)
            results.append(
                ActionWindowVideoResult(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    source_condition_id=source_condition_id,
                    video_path=str(output_path),
                    frame_count=len(selected_paths),
                    start_frame=window.start_frame,
                    end_frame=window.end_frame,
                )
            )

    return results


def _frame_source_condition(condition_id: str) -> str:
    return condition_id.removesuffix("_smooth")
