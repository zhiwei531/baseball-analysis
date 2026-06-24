from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image

from build_vicon_2026_metrics import C3DTrial, clean_label, infer_action, is_reconstruction_point, read_c3d, trial_id


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTS = ROOT / "reports" / "vicon_2026_point_summary.csv"
DEFAULT_OUT_DIR = ROOT / "reports" / "assets" / "vicon_reconstruction"
DEFAULT_C3D_DIR = ROOT.parent / "vicon_2026"

PART_COLORS = {
    "头颈": "#7c3aed",
    "躯干": "#2563eb",
    "左臂": "#16a34a",
    "右臂": "#dc2626",
    "骨盆": "#0891b2",
    "左腿": "#f59e0b",
    "右腿": "#0ea5e9",
    "质心点": "#22c55e",
    "球棒": "#f97316",
}
BODY_SEGMENTS = [
    ("C7", "T10", "躯干"),
    ("LSHO", "RSHO", "躯干"),
    ("LASI", "RASI", "骨盆"),
    ("LSHO", "LELB", "左臂"),
    ("LELB", "LWRA", "左臂"),
    ("LELB", "LWRB", "左臂"),
    ("RSHO", "RELB", "右臂"),
    ("RELB", "RWRA", "右臂"),
    ("RELB", "RWRB", "右臂"),
    ("LASI", "LKNE", "左腿"),
    ("LKNE", "LANK", "左腿"),
    ("RASI", "RKNE", "右腿"),
    ("RKNE", "RANK", "右腿"),
]
BAT_SEGMENTS = [("Bat1", "Bat2"), ("Bat2", "Bat3"), ("Bat3", "Bat4"), ("Bat4", "Bat5")]
LABEL_POINTS = ["LFHD", "RFHD", "C7", "T10", "LSHO", "RSHO", "LASI", "RASI", "CentreOfMass", "Bat1", "Bat5"]
RAW_MARKERS = {
    "LFHD", "RFHD", "LBHD", "RBHD", "C7", "T10", "CLAV", "STRN", "RBAK",
    "LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN",
    "RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN",
    "LASI", "RASI", "LPSI", "RPSI", "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
    "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE",
}
RAW_MARKER_PARTS = {
    "头颈": {"LFHD", "RFHD", "LBHD", "RBHD"},
    "躯干": {"C7", "T10", "CLAV", "STRN", "RBAK"},
    "左臂": {"LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN"},
    "右臂": {"RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN"},
    "骨盆": {"LASI", "RASI", "LPSI", "RPSI"},
    "左腿": {"LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE"},
    "右腿": {"RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE"},
}
ZH_FONT_PATHS = [
    Path("/System/Library/Fonts/PingFang.ttc"),
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
]


def zh_font() -> FontProperties | None:
    for path in ZH_FONT_PATHS:
        if path.exists():
            return FontProperties(fname=str(path))
    return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def points_for_trial(rows: list[dict[str, str]]) -> dict[str, tuple[float, float, float]]:
    points = {}
    for row in rows:
        x = num(row.get("key_x_mm"))
        y = num(row.get("key_y_mm"))
        z = num(row.get("key_z_mm"))
        if x is None or y is None or z is None:
            continue
        points[row["point"]] = (x, y, z)
    return points


AxisLimits = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def axis_limits_from_coords(coords: np.ndarray, margin: float = 1.55) -> AxisLimits:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(float((maxs - mins).max()) / 2, 250.0) * margin
    return (
        (float(center[0] - radius), float(center[0] + radius)),
        (float(center[1] - radius), float(center[1] + radius)),
        (float(center[2] - radius), float(center[2] + radius)),
    )


def set_equal_axes(ax, points: dict[str, tuple[float, float, float]], limits: AxisLimits | None = None) -> None:
    if limits is None:
        coords = np.array(list(points.values()), dtype=float)
        limits = axis_limits_from_coords(coords)
    (xlim, ylim, zlim) = limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)


def is_render_point(name: str) -> bool:
    return name in RAW_MARKERS or name == "CentreOfMass" or name.startswith("Bat")


def trial_axis_limits(trial: C3DTrial, frame_indices: np.ndarray | None = None) -> AxisLimits:
    clean = [clean_label(label) for label in trial.labels]
    keep = [idx for idx, name in enumerate(clean) if is_reconstruction_point(name) and is_render_point(name)]
    if frame_indices is None:
        points = trial.points[:, keep, :3]
    else:
        points = trial.points[frame_indices, :, :][:, keep, :3]
    coords = points.reshape(-1, 3)
    coords = coords[np.isfinite(coords).all(axis=1)]
    if coords.size == 0:
        return ((-500.0, 500.0), (-500.0, 500.0), (-100.0, 1200.0))
    # Percentile bounds avoid one bad marker forcing the athlete to become tiny.
    lo = np.nanpercentile(coords, 1, axis=0)
    hi = np.nanpercentile(coords, 99, axis=0)
    return axis_limits_from_coords(np.vstack([lo, hi]), margin=1.35)


def draw_segment(ax, points: dict[str, tuple[float, float, float]], a: str, b: str, color: str, width: float) -> None:
    if a not in points or b not in points:
        return
    pa = points[a]
    pb = points[b]
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color=color, linewidth=width, solid_capstyle="round")


def draw_closed_shape(
    ax,
    points: dict[str, tuple[float, float, float]],
    faces: list[list[str]],
    edges: list[tuple[str, str]],
    color: str,
    width: float,
    alpha: float,
) -> None:
    polygons = []
    for face in faces:
        if all(name in points for name in face):
            polygons.append([points[name] for name in face])
    if polygons:
        collection = Poly3DCollection(
            polygons,
            facecolors=color,
            edgecolors=color,
            linewidths=width * 0.75,
            alpha=alpha,
        )
        ax.add_collection3d(collection)
    for a, b in edges:
        draw_segment(ax, points, a, b, color, width)


def draw_head_shape(ax, points: dict[str, tuple[float, float, float]]) -> None:
    color = PART_COLORS["头颈"]
    draw_closed_shape(
        ax,
        points,
        faces=[["LFHD", "RFHD", "RBHD"], ["LFHD", "RBHD", "LBHD"]],
        edges=[
            ("LFHD", "RFHD"),
            ("RFHD", "RBHD"),
            ("RBHD", "LBHD"),
            ("LBHD", "LFHD"),
            ("LFHD", "RBHD"),
            ("RFHD", "LBHD"),
            ("LFHD", "C7"),
            ("RFHD", "C7"),
            ("LBHD", "C7"),
            ("RBHD", "C7"),
        ],
        color=color,
        width=2.0,
        alpha=0.18,
    )


def draw_foot_shapes(ax, points: dict[str, tuple[float, float, float]]) -> None:
    draw_closed_shape(
        ax,
        points,
        faces=[["LANK", "LHEE", "LTOE"]],
        edges=[("LANK", "LHEE"), ("LHEE", "LTOE"), ("LTOE", "LANK")],
        color=PART_COLORS["左腿"],
        width=2.2,
        alpha=0.22,
    )
    draw_closed_shape(
        ax,
        points,
        faces=[["RANK", "RHEE", "RTOE"]],
        edges=[("RANK", "RHEE"), ("RHEE", "RTOE"), ("RTOE", "RANK")],
        color=PART_COLORS["右腿"],
        width=2.2,
        alpha=0.22,
    )


def marker_part(name: str) -> str:
    for part, labels in RAW_MARKER_PARTS.items():
        if name in labels:
            return part
    return "躯干"


def scatter_points(ax, points: dict[str, tuple[float, float, float]], label: str, color: str, size: float, alpha: float = 1.0) -> None:
    if not points:
        return
    coords = np.array(list(points.values()), dtype=float)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=size, c=color, alpha=alpha, depthshade=False, label=label)


def fixed_legend(ax, font: FontProperties | None, include_bat: bool) -> None:
    labels = ["头颈", "躯干", "骨盆", "左臂", "右臂", "左腿", "右腿", "质心点", "球棒点"]
    if not include_bat:
        labels = [label for label in labels if label != "球棒点"]
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PART_COLORS.get(label, PART_COLORS["球棒"]), markersize=6, label=label)
        for label in labels
    ]
    legend = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=8)
    if font is not None:
        for text in legend.get_texts():
            text.set_fontproperties(font)


def split_points(points: dict[str, tuple[float, float, float]]) -> tuple[
    dict[str, tuple[float, float, float]],
    dict[str, tuple[float, float, float]],
    dict[str, tuple[float, float, float]],
    dict[str, tuple[float, float, float]],
]:
    body_points = {k: v for k, v in points.items() if not k.startswith("Bat")}
    raw_points = {k: v for k, v in body_points.items() if k in RAW_MARKERS}
    model_points: dict[str, tuple[float, float, float]] = {}
    com_points = {k: v for k, v in body_points.items() if k == "CentreOfMass"}
    bat_points = {k: v for k, v in points.items() if k.startswith("Bat")}
    return raw_points, model_points, com_points, bat_points


def draw_reconstruction(
    ax,
    points: dict[str, tuple[float, float, float]],
    font: FontProperties | None,
    title: str,
    frame_label: str | None = None,
    show_labels: bool = True,
    axis_limits: AxisLimits | None = None,
    fixed_layout_legend: bool = False,
) -> None:
    raw_points, model_points, com_points, bat_points = split_points(points)
    visible_points = {**raw_points, **com_points, **bat_points}

    ax.set_facecolor("#ffffff")
    ax.view_init(elev=17, azim=-66)
    for a, b, part in BODY_SEGMENTS:
        draw_segment(ax, points, a, b, PART_COLORS[part], 2.4)
    draw_head_shape(ax, points)
    draw_foot_shapes(ax, points)
    for a, b in BAT_SEGMENTS:
        draw_segment(ax, points, a, b, PART_COLORS["球棒"], 3.4)

    for part in ("头颈", "躯干", "骨盆", "左臂", "右臂", "左腿", "右腿"):
        part_points = {name: value for name, value in raw_points.items() if marker_part(name) == part}
        scatter_points(ax, part_points, part, PART_COLORS[part], 18)
    scatter_points(ax, com_points, "质心点", PART_COLORS["质心点"], 34)
    scatter_points(ax, bat_points, "球棒点", PART_COLORS["球棒"], 34)

    if show_labels:
        for name in LABEL_POINTS:
            if name not in visible_points:
                continue
            x, y, z = visible_points[name]
            ax.text(x, y, z, name, fontsize=5.5, color="#475467")

    set_equal_axes(ax, visible_points, limits=axis_limits)
    ax.set_xlabel("X（毫米）", labelpad=8, fontsize=9, fontproperties=font)
    ax.set_ylabel("Y（毫米）", labelpad=8, fontsize=9, fontproperties=font)
    ax.set_zlabel("Z（毫米）", labelpad=8, fontsize=9, fontproperties=font)
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.grid(True, color="#e4e7ec", linewidth=0.7)
    if frame_label:
        ax.text2D(0.67, 0.92, frame_label, transform=ax.transAxes, fontsize=9, color="#344054", fontproperties=font)
    if fixed_layout_legend:
        fixed_legend(ax, font, include_bat=bool(bat_points))
    else:
        legend = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=8)
        if font is not None:
            for text in legend.get_texts():
                text.set_fontproperties(font)
    ax.set_title(title, fontsize=10, color="#101828", fontproperties=font)
    ax.set_box_aspect((1, 1, 1))


def render_trial(rows: list[dict[str, str]], out_dir: Path) -> Path | None:
    if not rows:
        return None
    first = rows[0]
    trial_id = first["trial_id"]
    sample = first.get("athlete", "")
    action = first.get("action_type", "")
    event = first.get("key_event", "关键动作")
    frame = first.get("key_frame_index", "")
    time_sec = num(first.get("key_time_sec"))
    points = points_for_trial(rows)
    if not points:
        return None

    fig = plt.figure(figsize=(8.0, 5.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    font = zh_font()
    fig.patch.set_facecolor("#ffffff")

    action_text = "投球" if action == "pitching" else "打击"
    time_text = f"{time_sec:.2f}秒" if time_sec is not None else "暂无时间"
    draw_reconstruction(
        ax,
        points,
        font,
        f"{sample} / {action_text} / {event} / 第{frame}帧 / {time_text}",
        show_labels=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trial_id}.png"
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def trial_frame_points(trial: C3DTrial, frame_idx: int, smooth_radius: int = 0) -> dict[str, tuple[float, float, float]]:
    clean = [clean_label(label) for label in trial.labels]
    points = {}
    start = max(0, frame_idx - smooth_radius)
    end = min(trial.points.shape[0], frame_idx + smooth_radius + 1)
    for idx, name in enumerate(clean):
        if not is_reconstruction_point(name):
            continue
        if not is_render_point(name):
            continue
        if smooth_radius:
            window = trial.points[start:end, idx, :3]
            valid = np.isfinite(window).all(axis=1)
            if not valid.any():
                continue
            xyz = np.nanmedian(window[valid], axis=0)
        else:
            xyz = trial.points[frame_idx, idx, :3]
        if np.isfinite(xyz).all():
            points[name] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return points


def key_frame_from_rows(rows: list[dict[str, str]], frame_count: int) -> tuple[int, str]:
    if not rows:
        return frame_count // 2, "关键动作"
    first = rows[0]
    key_frame = num(first.get("key_frame_index"))
    if key_frame is None:
        return frame_count // 2, first.get("key_event", "关键动作")
    idx = min(max(int(round(key_frame)), 0), frame_count - 1)
    return idx, first.get("key_event", "关键动作")


def key_action_frame_indices(
    trial: C3DTrial,
    rows: list[dict[str, str]],
    before_sec: float,
    after_sec: float,
    max_frames: int,
) -> tuple[np.ndarray, str]:
    frame_count = trial.points.shape[0]
    key_idx, event = key_frame_from_rows(rows, frame_count)
    before = max(1, int(round(before_sec * trial.rate_hz)))
    after = max(1, int(round(after_sec * trial.rate_hz)))
    start = max(0, key_idx - before)
    end = min(frame_count - 1, key_idx + after)
    if end <= start:
        start = max(0, key_idx - 1)
        end = min(frame_count - 1, key_idx + 1)
    count = min(max_frames, end - start + 1)
    return np.linspace(start, end, count, dtype=int), event


def render_trial_gif(
    trial: C3DTrial,
    rows: list[dict[str, str]],
    out_dir: Path,
    max_frames: int = 72,
    frame_duration_ms: int = 85,
    smooth_radius: int = 2,
    before_sec: float = 0.6,
    after_sec: float = 0.4,
) -> Path | None:
    frame_count = trial.points.shape[0]
    if frame_count == 0:
        return None
    frame_indices, event = key_action_frame_indices(trial, rows, before_sec, after_sec, max_frames)
    frames: list[Image.Image] = []
    font = zh_font()
    sample = trial.path.parent.name
    action = infer_action(trial.path)
    action_text = "投球" if action == "pitching" else "打击"
    title = f"{sample} / {action_text} / C3D骨架动图"
    limits = trial_axis_limits(trial, frame_indices=frame_indices)

    fig = plt.figure(figsize=(6.8, 4.6), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.94)
    for frame_idx in frame_indices:
        ax.clear()
        points = trial_frame_points(trial, int(frame_idx), smooth_radius=smooth_radius)
        if not points:
            continue
        draw_reconstruction(
            ax,
            points,
            font,
            title,
            frame_label=f"{event}窗口 / 第{int(frame_idx)}帧 / {frame_idx / trial.rate_hz:.2f}秒",
            show_labels=False,
            axis_limits=limits,
            fixed_layout_legend=True,
        )
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = Image.frombytes("RGBA", (width, height), fig.canvas.buffer_rgba())
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE))
    plt.close(fig)

    if not frames:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trial_id(trial.path)}.gif"
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Vicon C3D key-action reconstruction PNGs.")
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--c3d-dir", type=Path, default=DEFAULT_C3D_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-gif-frames", type=int, default=72)
    parser.add_argument("--gif-before-sec", type=float, default=0.6)
    parser.add_argument("--gif-after-sec", type=float, default=0.4)
    args = parser.parse_args()

    by_trial: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.points):
        by_trial[row["trial_id"]].append(row)

    outputs = []
    for trial_key in sorted(by_trial):
        out = render_trial(by_trial[trial_key], args.out_dir)
        if out is not None:
            outputs.append(out)

    for path in sorted(args.c3d_dir.glob("*/*.c3d")):
        if path.name.startswith("._"):
            continue
        trial = read_c3d(path)
        out = render_trial_gif(
            trial,
            by_trial.get(trial_id(path), []),
            args.out_dir,
            max_frames=args.max_gif_frames,
            before_sec=args.gif_before_sec,
            after_sec=args.gif_after_sec,
        )
        if out is not None:
            outputs.append(out)

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
