"""Plotting helpers for report figures."""

from __future__ import annotations

import csv
from pathlib import Path


def figure_name(clip_id: str, condition_id: str, figure_type: str) -> str:
    return f"{clip_id}__{condition_id}__{figure_type}.png"


def plot_wrist_trajectories(
    feature_paths: dict[str, str | Path],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot wrist coordinates over time for one clip across conditions."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    color_cycle = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    line_styles = ["-", "--", "-.", ":"]
    legend_handles = []
    legend_labels = []

    for condition_id, source_path in sorted(feature_paths.items()):
        rows = _read_rows(source_path)
        times = [_optional_float(row["timestamp_sec"]) for row in rows]
        if not times:
            continue
        color = color_cycle[len(legend_labels) % len(color_cycle)]
        line_style = line_styles[(len(legend_labels) // len(color_cycle)) % len(line_styles)]
        series = (
            (axes[0][0], "left_wrist_x", "Left wrist X", "x position", False),
            (axes[0][1], "left_wrist_y", "Left wrist Y", "y position", True),
            (axes[1][0], "right_wrist_x", "Right wrist X", "x position", False),
            (axes[1][1], "right_wrist_y", "Right wrist Y", "y position", True),
        )
        added_label = False
        for axis, field, axis_title, ylabel, invert_y in series:
            values = [_optional_float(row[field]) for row in rows]
            paired = [(time, value) for time, value in zip(times, values) if time is not None and value is not None]
            if not paired:
                continue
            (line,) = axis.plot(
                [time for time, _ in paired],
                [value for _, value in paired],
                linewidth=1.8,
                alpha=0.95,
                color=color,
                linestyle=line_style,
                label=condition_id,
            )
            if not added_label:
                legend_handles.append(line)
                legend_labels.append(condition_id)
                added_label = True
            axis.set_title(axis_title)
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
            if invert_y:
                axis.invert_yaxis()

    for axis in axes[1]:
        axis.set_xlabel("time (s)")

    _place_figure_header(fig, title, legend_handles, legend_labels)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def plot_posture_analysis(
    feature_paths: dict[str, str | Path],
    output_path: str | Path,
    title: str,
) -> None:
    """Plot rotation-chain and lower-body posture proxies for one clip."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    color_cycle = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    line_styles = ["-", "--", "-.", ":"]
    legend_handles = []
    legend_labels = []

    for condition_index, (condition_id, source_path) in enumerate(sorted(feature_paths.items())):
        rows = _read_rows(source_path)
        color = color_cycle[condition_index % len(color_cycle)]
        line_style = line_styles[(condition_index // len(color_cycle)) % len(line_styles)]
        label_added = False

        label_added = _plot_time_series(
            axes[0][0],
            rows,
            (
                ("pelvis_rotation_deg", "pelvis", color, line_style),
                ("shoulder_rotation_deg", "shoulder", color, ":"),
                ("hip_shoulder_separation_deg", "separation", color, "--"),
            ),
            condition_id,
            label_added,
            legend_handles,
            legend_labels,
        )
        label_added = _plot_time_series(
            axes[0][1],
            rows,
            (
                ("pelvis_rotation_velocity_deg_s", "pelvis velocity", color, line_style),
                ("trunk_rotation_velocity_deg_s", "trunk velocity", color, "--"),
            ),
            condition_id,
            label_added,
            legend_handles,
            legend_labels,
        )
        label_added = _plot_time_series(
            axes[1][1],
            rows,
            (
                ("left_knee_extension_from_start_deg", "left knee extension", color, line_style),
                ("right_knee_extension_from_start_deg", "right knee extension", color, "--"),
                ("hand_speed_proxy", "hand speed proxy", color, ":"),
            ),
            condition_id,
            label_added,
            legend_handles,
            legend_labels,
        )

        com_x = [_optional_float(row.get("center_of_mass_x", "")) for row in rows]
        com_y = [_optional_float(row.get("center_of_mass_y", "")) for row in rows]
        points = [(x, y) for x, y in zip(com_x, com_y) if x is not None and y is not None]
        if points:
            (line,) = axes[1][0].plot(
                [point[0] for point in points],
                [point[1] for point in points],
                linewidth=1.8,
                alpha=0.95,
                color=color,
                linestyle=line_style,
                label=condition_id,
            )
            if not label_added:
                legend_handles.append(line)
                legend_labels.append(condition_id)
                label_added = True
            axes[1][0].scatter(points[0][0], points[0][1], s=18, color=color)
            axes[1][0].scatter(points[-1][0], points[-1][1], s=22, color=color, marker="x")

    axes[0][0].set_title("Pelvis / shoulder rotation")
    axes[0][0].set_ylabel("degrees")
    axes[0][1].set_title("Rotation velocity")
    axes[0][1].set_ylabel("degrees / second")
    axes[1][0].set_title("Approximate COM path")
    axes[1][0].set_xlabel("x position")
    axes[1][0].set_ylabel("y position")
    axes[1][0].invert_yaxis()
    axes[1][1].set_title("Knee extension and hand speed")
    axes[1][1].set_xlabel("time (s)")
    axes[1][1].set_ylabel("degrees or pixels / second")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    for axis in axes[0]:
        axis.set_xlabel("time (s)")

    _place_figure_header(fig, title, legend_handles, legend_labels)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def _place_figure_header(
    fig,
    title: str,
    legend_handles: list[object],
    legend_labels: list[str],
) -> None:
    """Reserve separate vertical space for the legend, title, and subplots."""

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=min(4, len(legend_labels)),
            frameon=False,
        )

    fig.suptitle(title, y=0.94)
    fig.tight_layout(rect=(0, 0, 1, 0.88))


def _plot_time_series(
    axis,
    rows: list[dict[str, str]],
    series: tuple[tuple[str, str, str, str], ...],
    condition_id: str,
    label_added: bool,
    legend_handles: list[object],
    legend_labels: list[str],
) -> bool:
    times = [_optional_float(row.get("timestamp_sec", "")) for row in rows]
    for field, label, color, line_style in series:
        values = [_optional_float(row.get(field, "")) for row in rows]
        paired = [(time, value) for time, value in zip(times, values) if time is not None and value is not None]
        if not paired:
            continue
        (line,) = axis.plot(
            [time for time, _ in paired],
            [value for _, value in paired],
            linewidth=1.5,
            alpha=0.9,
            color=color,
            linestyle=line_style,
            label=f"{condition_id} {label}",
        )
        if not label_added:
            legend_handles.append(line)
            legend_labels.append(condition_id)
            label_added = True
    return label_added


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _optional_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)
