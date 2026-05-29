"""Plotting helpers for report figures."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from baseball_pose.pipeline.report_window import filter_rows_to_action_window


def figure_name(clip_id: str, condition_id: str, figure_type: str) -> str:
    return f"{clip_id}__{condition_id}__{figure_type}.png"


PREFERRED_FONT_FAMILIES = [
    "Arial Unicode MS",
    "PingFang SC",
    "Heiti SC",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]


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


def plot_public_metric_dashboard(
    feature_paths: dict[str, str | Path],
    output_path: str | Path,
    title: str,
    action_type: str = "batting",
) -> None:
    """Render coach-facing metric cards instead of raw trajectories."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = PREFERRED_FONT_FAMILIES

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    condition_id, rows = _primary_condition_rows(feature_paths, action_type=action_type)
    metrics = _public_metric_values(rows)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.patch.set_facecolor("white")
    card_specs = [
        (
            "躯干旋转速度",
            metrics["trunk_rotation_velocity"],
            "deg/s",
            (0.0, 1200.0),
            "上半身转动越快，通常越容易把力量传到动作末端。",
        ),
        (
            "骨盆旋转速度",
            metrics["pelvis_rotation_velocity"],
            "deg/s",
            (0.0, 1200.0),
            "下半身转动越积极，越能帮助形成更完整的动力链。",
        ),
        (
            "髋肩分离代理值",
            metrics["hip_shoulder_separation"],
            "deg",
            (0.0, 80.0),
            "数值越大，通常代表髋部和肩部错开的空间越明显。",
        ),
        (
            "双手速度代理值",
            metrics["hand_speed_proxy"],
            "px/s",
            (0.0, 1.5),
            "反映挥动末端的快慢，用于同一位运动员的重复对比。",
        ),
        (
            "左膝弯曲深度",
            metrics["left_knee_flexion"],
            "deg",
            (0.0, 90.0),
            "膝盖弯曲更明显时，通常说明下肢参与更充分。",
        ),
        (
            "右膝弯曲深度",
            metrics["right_knee_flexion"],
            "deg",
            (0.0, 90.0),
            "可与左侧比较，判断动作中左右下肢是否接近。",
        ),
    ]

    for axis, spec in zip(axes.flat, card_specs):
        title_text, value, unit, value_range, description = spec
        _draw_metric_card(axis, title_text, value, unit, value_range, description, patches)

    figure_title = title
    if condition_id:
        figure_title = f"{title} ({condition_id})"
    fig.suptitle(figure_title, y=0.98, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)


def plot_knee_balance_summary(
    feature_paths: dict[str, str | Path],
    output_path: str | Path,
    title: str,
    action_type: str = "batting",
) -> None:
    """Render left/right knee bend bars with easy-to-read zones."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = PREFERRED_FONT_FAMILIES

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    condition_id, rows = _primary_condition_rows(feature_paths, action_type=action_type)
    metrics = _public_metric_values(rows)
    left_value = metrics["left_knee_flexion"]
    right_value = metrics["right_knee_flexion"]
    balance_score = _symmetry_score(left_value, right_value)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.6), gridspec_kw={"height_ratios": [1, 1, 0.65]})
    fig.patch.set_facecolor("white")

    _draw_knee_gauge(
        axes[0],
        "左膝弯曲深度",
        left_value,
        marker_color="#1D4ED8",
    )
    _draw_knee_gauge(
        axes[1],
        "右膝弯曲深度",
        right_value,
        marker_color="#DC2626",
    )

    axes[2].axis("off")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].text(0.02, 0.75, "左右侧弯曲差值", fontsize=14, fontweight="bold")
    diff_text = _difference_text(left_value, right_value, "deg")
    axes[2].text(0.02, 0.40, diff_text, fontsize=22, fontweight="bold", color="#111827")
    axes[2].text(
        0.42,
        0.40,
        f"平衡分数: {balance_score:.0f}/100",
        fontsize=18,
        fontweight="bold",
        color=_score_color(balance_score),
    )
    axes[2].text(
        0.02,
        0.10,
        "解读方式：这张图不是医学诊断，而是帮助家长和教练快速判断左右膝在该动作里是否弯得接近。",
        fontsize=11,
        color="#4B5563",
    )

    figure_title = title
    if condition_id:
        figure_title = f"{title} ({condition_id})"
    fig.suptitle(figure_title, y=0.98, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(target, dpi=180, facecolor="white")
    plt.close(fig)


def plot_side_to_side_summary(
    feature_paths: dict[str, str | Path],
    output_path: str | Path,
    title: str,
    action_type: str = "batting",
) -> None:
    """Render side-to-side comparison bars for knees and wrists."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = PREFERRED_FONT_FAMILIES

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    condition_id, rows = _primary_condition_rows(feature_paths, action_type=action_type)
    metrics = _public_metric_values(rows)
    left_knee = metrics["left_knee_flexion"]
    right_knee = metrics["right_knee_flexion"]
    left_wrist = metrics["left_wrist_speed"]
    right_wrist = metrics["right_wrist_speed"]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8))
    fig.patch.set_facecolor("white")

    _draw_comparison_panel(
        axes[0],
        "下肢弯曲对比",
        ("左膝", left_knee, "#2563EB"),
        ("右膝", right_knee, "#DC2626"),
        max_value=90.0,
        unit="deg",
        footnote="数值越大，表示视频里观察到的膝盖弯曲越明显。",
    )
    _draw_comparison_panel(
        axes[1],
        "手部速度对比",
        ("左手", left_wrist, "#2563EB"),
        ("右手", right_wrist, "#DC2626"),
        max_value=1.5,
        unit="px/s",
        footnote="用于同一位运动员前后对比，不建议跨设备直接比较绝对值。",
    )

    figure_title = title
    if condition_id:
        figure_title = f"{title} ({condition_id})"
    fig.suptitle(figure_title, y=0.98, fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(target, dpi=180, facecolor="white")
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


def _primary_condition_rows(
    feature_paths: dict[str, str | Path],
    action_type: str,
) -> tuple[str, list[dict[str, str]]]:
    if not feature_paths:
        return "", []
    condition_id = sorted(feature_paths)[0]
    rows = _read_rows(feature_paths[condition_id])
    filtered_rows, _window = filter_rows_to_action_window(
        rows,
        action_type=action_type,
        expanded=(action_type == "batting"),
    )
    return condition_id, filtered_rows if filtered_rows else rows


def _public_metric_values(rows: list[dict[str, str]]) -> dict[str, float | None]:
    left_knee_flexion = [180.0 - value for value in _series(rows, "left_knee_angle")]
    right_knee_flexion = [180.0 - value for value in _series(rows, "right_knee_angle")]
    return {
        "trunk_rotation_velocity": _p95_abs(rows, "trunk_rotation_velocity_deg_s"),
        "pelvis_rotation_velocity": _p95_abs(rows, "pelvis_rotation_velocity_deg_s"),
        "hip_shoulder_separation": _p95_abs(rows, "hip_shoulder_separation_deg"),
        "hand_speed_proxy": _p95_abs(rows, "hand_speed_proxy"),
        "left_knee_flexion": _p95_value(left_knee_flexion),
        "right_knee_flexion": _p95_value(right_knee_flexion),
        "left_wrist_speed": _p95_value(_series(rows, "left_wrist_speed")),
        "right_wrist_speed": _p95_value(_series(rows, "right_wrist_speed")),
    }


def _series(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(field, "")
        if value in {"", None}:
            continue
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values


def _p95_abs(rows: list[dict[str, str]], field: str) -> float | None:
    values = [abs(value) for value in _series(rows, field)]
    return _p95_value(values)


def _p95_value(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * 0.95
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _draw_metric_card(axis, title_text: str, value: float | None, unit: str, value_range: tuple[float, float], description: str, patches) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    card = patches.FancyBboxPatch(
        (0.02, 0.04),
        0.96,
        0.92,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        linewidth=1.0,
        edgecolor="#D1D5DB",
        facecolor="#F9FAFB",
    )
    axis.add_patch(card)
    axis.text(0.07, 0.82, title_text, fontsize=14, fontweight="bold", color="#111827")
    if value is None:
        axis.text(0.07, 0.53, "N/A", fontsize=26, fontweight="bold", color="#9CA3AF")
        axis.text(0.07, 0.19, "当前帧段无法稳定提取该指标。", fontsize=10.5, color="#6B7280", wrap=True)
        return

    axis.text(0.07, 0.53, f"{value:.1f}", fontsize=28, fontweight="bold", color="#111827")
    axis.text(0.46, 0.54, unit, fontsize=13, color="#4B5563")
    _draw_progress_bar(axis, value, value_range, y=0.34, height=0.08)
    axis.text(0.07, 0.19, description, fontsize=10.3, color="#4B5563", wrap=True)


def _draw_progress_bar(axis, value: float, value_range: tuple[float, float], y: float, height: float) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    minimum, maximum = value_range
    span = maximum - minimum if maximum > minimum else 1.0
    normalized = max(0.0, min(1.0, (value - minimum) / span))
    axis.add_patch(
        Rectangle((0.07, y), 0.80, height, facecolor="#E5E7EB", edgecolor="none")
    )
    fill_color = _ratio_color(normalized)
    axis.add_patch(
        Rectangle((0.07, y), 0.80 * normalized, height, facecolor=fill_color, edgecolor="none")
    )
    axis.add_line(Line2D([0.07 + 0.80 * normalized, 0.07 + 0.80 * normalized], [y - 0.02, y + height + 0.02], color="#111827", linewidth=2))
    axis.text(0.07, y - 0.08, f"{minimum:.0f}", fontsize=9, color="#6B7280")
    axis.text(0.83, y - 0.08, f"{maximum:.0f}", fontsize=9, color="#6B7280", ha="right")


def _draw_knee_gauge(axis, label: str, value: float | None, marker_color: str) -> None:
    axis.set_xlim(0, 90)
    axis.set_ylim(0, 1)
    axis.spines[["top", "left", "right"]].set_visible(False)
    axis.spines["bottom"].set_color("#D1D5DB")
    axis.set_yticks([])
    axis.set_xticks([0, 20, 45, 70, 90])
    axis.tick_params(axis="x", labelsize=10, colors="#4B5563")
    axis.set_title(label, loc="left", fontsize=14, fontweight="bold")

    zones = [
        (0, 20, "#FEE2E2", "弯曲较少"),
        (20, 45, "#FEF3C7", "中等弯曲"),
        (45, 70, "#DCFCE7", "较深弯曲"),
        (70, 90, "#DBEAFE", "很深的弯曲"),
    ]
    for start, end, color, zone_label in zones:
        axis.axvspan(start, end, ymin=0.18, ymax=0.62, color=color, ec="white")
        axis.text((start + end) / 2, 0.80, zone_label, ha="center", va="center", fontsize=9.5, color="#374151")

    if value is not None:
        axis.plot([value, value], [0.12, 0.68], color=marker_color, linewidth=4)
        axis.scatter([value], [0.40], s=120, color=marker_color, zorder=3)
        axis.text(min(value + 2, 83), 0.28, f"{value:.1f} deg", fontsize=12, fontweight="bold", color=marker_color)
    else:
        axis.text(4, 0.35, "N/A", fontsize=18, fontweight="bold", color="#9CA3AF")


def _draw_comparison_panel(
    axis,
    title: str,
    first: tuple[str, float | None, str],
    second: tuple[str, float | None, str],
    max_value: float,
    unit: str,
    footnote: str,
) -> None:
    labels = [first[0], second[0]]
    values = [0.0 if first[1] is None else first[1], 0.0 if second[1] is None else second[1]]
    colors = [first[2], second[2]]

    axis.barh([1, 0], values, color=colors, height=0.45)
    axis.set_xlim(0, max_value)
    axis.set_yticks([1, 0], labels)
    axis.set_title(title, loc="left", fontsize=14, fontweight="bold")
    axis.grid(axis="x", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#D1D5DB")

    for y, value, color in zip([1, 0], values, colors):
        axis.text(min(value + max_value * 0.02, max_value * 0.96), y, f"{value:.1f} {unit}", va="center", fontsize=11, color=color, fontweight="bold")

    balance_score = _symmetry_score(first[1], second[1])
    axis.text(
        0.02 * max_value,
        -0.50,
        f"平衡分数: {balance_score:.0f}/100",
        fontsize=12,
        fontweight="bold",
        color=_score_color(balance_score),
    )
    axis.text(0.02 * max_value, -0.78, footnote, fontsize=9.8, color="#4B5563")


def _difference_text(first: float | None, second: float | None, unit: str) -> str:
    if first is None or second is None:
        return "无法稳定计算差值"
    return f"{abs(first - second):.1f} {unit}"


def _symmetry_score(first: float | None, second: float | None) -> float:
    if first is None or second is None:
        return 0.0
    scale = max(abs(first), abs(second), 1e-6)
    ratio = abs(first - second) / scale
    return max(0.0, 100.0 - ratio * 100.0)


def _ratio_color(ratio: float) -> str:
    if ratio < 0.33:
        return "#F59E0B"
    if ratio < 0.66:
        return "#10B981"
    return "#2563EB"


def _score_color(score: float) -> str:
    if score >= 85:
        return "#15803D"
    if score >= 65:
        return "#B45309"
    return "#B91C1C"
