"""Report figure orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.paths import feature_path
from baseball_pose.visualization.plots import plot_posture_analysis, plot_wrist_trajectories


@dataclass(frozen=True)
class FigureResult:
    clip_id: str
    figure_path: Path
    condition_count: int


def make_report_figures(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str] | None = None,
) -> list[FigureResult]:
    condition_ids = conditions if conditions is not None else config.condition_ids
    prefer_smoothed = conditions is None
    output_dir = config.output_dir / "figures"
    results: list[FigureResult] = []

    for clip_id in clip_ids:
        available_feature_paths = {
            condition_id: feature_path(config.data_dir, clip_id, condition_id)
            for condition_id in condition_ids
            if feature_path(config.data_dir, clip_id, condition_id).exists()
        }
        feature_paths = _select_figure_feature_paths(available_feature_paths, prefer_smoothed)
        if not feature_paths:
            continue
        wrist_output_path = output_dir / f"{clip_id}__wrist_trajectories.png"
        plot_wrist_trajectories(
            feature_paths,
            wrist_output_path,
            title=f"{clip_id} wrist trajectories",
        )
        results.append(
            FigureResult(
                clip_id=clip_id,
                figure_path=wrist_output_path,
                condition_count=len(feature_paths),
            )
        )
        posture_output_path = output_dir / f"{clip_id}__posture_analysis.png"
        plot_posture_analysis(
            feature_paths,
            posture_output_path,
            title=f"{clip_id} posture analysis",
        )
        results.append(
            FigureResult(
                clip_id=clip_id,
                figure_path=posture_output_path,
                condition_count=len(feature_paths),
            )
        )

    return results


def _select_figure_feature_paths(
    feature_paths: dict[str, Path],
    prefer_smoothed: bool,
) -> dict[str, Path]:
    if not prefer_smoothed:
        return feature_paths
    smoothed = {
        condition_id: path
        for condition_id, path in feature_paths.items()
        if condition_id.endswith("_smooth")
    }
    if not smoothed:
        return feature_paths
    body_mask_smoothed = {
        condition_id: path
        for condition_id, path in smoothed.items()
        if condition_id.startswith("body_prior_mask_roi")
    }
    if body_mask_smoothed:
        return body_mask_smoothed
    center_smoothed = {
        condition_id: path
        for condition_id, path in smoothed.items()
        if condition_id.startswith("center_prior_roi")
    }
    if center_smoothed:
        return center_smoothed
    roi_smoothed = {
        condition_id: path
        for condition_id, path in smoothed.items()
        if "roi" in condition_id
    }
    return roi_smoothed if roi_smoothed else smoothed
