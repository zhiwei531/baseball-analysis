"""Report figure orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.feature_csv import read_feature_rows
from baseball_pose.io.metadata import load_clips
from baseball_pose.io.paths import feature_path
from baseball_pose.pipeline.report_window import filter_rows_to_action_window
from baseball_pose.visualization.plots import (
    plot_knee_balance_summary,
    plot_public_metric_dashboard,
    plot_side_to_side_summary,
)


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
    clip_lookup = {clip.clip_id: clip for clip in load_clips(config.clips_file)}

    for clip_id in clip_ids:
        clip = clip_lookup.get(clip_id)
        action_type = clip.action_type if clip is not None else "batting"
        available_feature_paths = {
            condition_id: feature_path(config.data_dir, clip_id, condition_id)
            for condition_id in condition_ids
            if feature_path(config.data_dir, clip_id, condition_id).exists()
        }
        feature_paths = _select_figure_feature_paths(available_feature_paths, prefer_smoothed)
        if not feature_paths:
            continue
        _write_action_window_feature_csvs(feature_paths, action_type)
        dashboard_output_path = output_dir / f"{clip_id}__movement_quality_dashboard.png"
        plot_public_metric_dashboard(
            feature_paths,
            dashboard_output_path,
            title=f"{clip_id} movement quality dashboard",
            action_type=action_type,
        )
        results.append(
            FigureResult(
                clip_id=clip_id,
                figure_path=dashboard_output_path,
                condition_count=len(feature_paths),
            )
        )

        knee_output_path = output_dir / f"{clip_id}__knee_balance_summary.png"
        plot_knee_balance_summary(
            feature_paths,
            knee_output_path,
            title=f"{clip_id} knee bend summary",
            action_type=action_type,
        )
        results.append(
            FigureResult(
                clip_id=clip_id,
                figure_path=knee_output_path,
                condition_count=len(feature_paths),
            )
        )

        side_balance_output_path = output_dir / f"{clip_id}__side_to_side_summary.png"
        plot_side_to_side_summary(
            feature_paths,
            side_balance_output_path,
            title=f"{clip_id} side-to-side comparison",
            action_type=action_type,
        )
        results.append(
            FigureResult(
                clip_id=clip_id,
                figure_path=side_balance_output_path,
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


def _write_action_window_feature_csvs(feature_paths: dict[str, Path], action_type: str) -> None:
    for condition_id, source_path in feature_paths.items():
        rows = read_feature_rows(source_path)
        window_rows, _window = filter_rows_to_action_window(
            rows,
            action_type=action_type,
            expanded=(action_type == "batting"),
        )
        if not window_rows:
            continue
        target_path = source_path.parent / f"{condition_id}__action_window.csv"
        header = list(window_rows[0].keys())
        import csv
        with target_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(window_rows)
