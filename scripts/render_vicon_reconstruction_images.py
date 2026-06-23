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
    "模型点": "#94a3b8",
    "质心点": "#22c55e",
    "球棒": "#f97316",
}
BODY_SEGMENTS = [
    ("LFHD", "RFHD", "头颈"),
    ("LFHD", "C7", "头颈"),
    ("RFHD", "C7", "头颈"),
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
MODEL_SEGMENTS = [
    ("PELO", "PELA", "骨盆"),
    ("PELO", "PELL", "骨盆"),
    ("PELO", "PELP", "骨盆"),
    ("TRXO", "TRXA", "躯干"),
    ("TRXO", "TRXL", "躯干"),
    ("TRXO", "TRXP", "躯干"),
    ("HEDO", "HEDA", "头颈"),
    ("HEDO", "HEDL", "头颈"),
    ("HEDO", "HEDP", "头颈"),
    ("LCLO", "LCLA", "左臂"),
    ("LCLO", "LCLL", "左臂"),
    ("RCLO", "RCLA", "右臂"),
    ("RCLO", "RCLL", "右臂"),
    ("LHUO", "LHUA", "左臂"),
    ("LHUO", "LHUL", "左臂"),
    ("LRAO", "LRAA", "左臂"),
    ("LRAO", "LRAL", "左臂"),
    ("LHNO", "LHNA", "左臂"),
    ("RHUO", "RHUA", "右臂"),
    ("RHUO", "RHUL", "右臂"),
    ("RRAO", "RRAA", "右臂"),
    ("RRAO", "RRAL", "右臂"),
    ("RHNO", "RHNA", "右臂"),
    ("LFEO", "LFEA", "左腿"),
    ("LFEO", "LFEL", "左腿"),
    ("LTIO", "LTIA", "左腿"),
    ("LTIO", "LTIL", "左腿"),
    ("LFOO", "LFOA", "左腿"),
    ("LTOO", "LTOA", "左腿"),
    ("RFEO", "RFEA", "右腿"),
    ("RFEO", "RFEL", "右腿"),
    ("RTIO", "RTIA", "右腿"),
    ("RTIO", "RTIL", "右腿"),
    ("RFOO", "RFOA", "右腿"),
    ("RTOO", "RTOA", "右腿"),
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


def set_equal_axes(ax, points: dict[str, tuple[float, float, float]]) -> None:
    coords = np.array(list(points.values()), dtype=float)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(float((maxs - mins).max()) / 2, 250.0) * 1.55
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_segment(ax, points: dict[str, tuple[float, float, float]], a: str, b: str, color: str, width: float) -> None:
    if a not in points or b not in points:
        return
    pa = points[a]
    pb = points[b]
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color=color, linewidth=width, solid_capstyle="round")


def marker_part(name: str) -> str:
    for part, labels in RAW_MARKER_PARTS.items():
        if name in labels:
            return part
    return "模型点"


def scatter_points(ax, points: dict[str, tuple[float, float, float]], label: str, color: str, size: float, alpha: float = 1.0) -> None:
    if not points:
        return
    coords = np.array(list(points.values()), dtype=float)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=size, c=color, alpha=alpha, depthshade=False, label=label)


def split_points(points: dict[str, tuple[float, float, float]]) -> tuple[
    dict[str, tuple[float, float, float]],
    dict[str, tuple[float, float, float]],
    dict[str, tuple[float, float, float]],
    dict[str, tuple[float, float, float]],
]:
    body_points = {k: v for k, v in points.items() if not k.startswith("Bat")}
    raw_points = {k: v for k, v in body_points.items() if k in RAW_MARKERS}
    model_points = {k: v for k, v in body_points.items() if k not in RAW_MARKERS and not k.startswith("CentreOfMass")}
    com_points = {k: v for k, v in body_points.items() if k.startswith("CentreOfMass")}
    bat_points = {k: v for k, v in points.items() if k.startswith("Bat")}
    return raw_points, model_points, com_points, bat_points


def draw_reconstruction(
    ax,
    points: dict[str, tuple[float, float, float]],
    font: FontProperties | None,
    title: str,
    frame_label: str | None = None,
    show_labels: bool = True,
) -> None:
    raw_points, model_points, com_points, bat_points = split_points(points)

    ax.set_facecolor("#ffffff")
    ax.view_init(elev=17, azim=-66)
    for a, b, part in MODEL_SEGMENTS:
        draw_segment(ax, points, a, b, PART_COLORS[part], 1.2)
    for a, b, part in BODY_SEGMENTS:
        draw_segment(ax, points, a, b, PART_COLORS[part], 2.4)
    for a, b in BAT_SEGMENTS:
        draw_segment(ax, points, a, b, PART_COLORS["球棒"], 3.4)

    if model_points:
        scatter_points(ax, model_points, "模型点", PART_COLORS["模型点"], 9, alpha=0.45)
    for part in ("头颈", "躯干", "骨盆", "左臂", "右臂", "左腿", "右腿"):
        part_points = {name: value for name, value in raw_points.items() if marker_part(name) == part}
        scatter_points(ax, part_points, part, PART_COLORS[part], 18)
    scatter_points(ax, com_points, "质心点", PART_COLORS["质心点"], 34)
    scatter_points(ax, bat_points, "球棒点", PART_COLORS["球棒"], 34)

    if show_labels:
        for name in LABEL_POINTS:
            if name not in points:
                continue
            x, y, z = points[name]
            ax.text(x, y, z, name, fontsize=5.5, color="#475467")

    set_equal_axes(ax, points)
    ax.set_xlabel("X（毫米）", labelpad=8, fontsize=9, fontproperties=font)
    ax.set_ylabel("Y（毫米）", labelpad=8, fontsize=9, fontproperties=font)
    ax.set_zlabel("Z（毫米）", labelpad=8, fontsize=9, fontproperties=font)
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.grid(True, color="#e4e7ec", linewidth=0.7)
    if frame_label:
        ax.text2D(0.67, 0.92, frame_label, transform=ax.transAxes, fontsize=9, color="#344054", fontproperties=font)
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


def trial_frame_points(trial: C3DTrial, frame_idx: int) -> dict[str, tuple[float, float, float]]:
    clean = [clean_label(label) for label in trial.labels]
    points = {}
    for idx, name in enumerate(clean):
        if not is_reconstruction_point(name):
            continue
        xyz = trial.points[frame_idx, idx, :3]
        if np.isfinite(xyz).all():
            points[name] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return points


def render_trial_gif(trial: C3DTrial, out_dir: Path, max_frames: int = 72, frame_duration_ms: int = 85) -> Path | None:
    frame_count = trial.points.shape[0]
    if frame_count == 0:
        return None
    frame_indices = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
    frames: list[Image.Image] = []
    font = zh_font()
    sample = trial.path.parent.name
    action = infer_action(trial.path)
    action_text = "投球" if action == "pitching" else "打击"
    title = f"{sample} / {action_text} / C3D完整模型动图"

    fig = plt.figure(figsize=(6.8, 4.6), dpi=110)
    fig.patch.set_facecolor("#ffffff")
    ax = fig.add_subplot(111, projection="3d")
    for frame_idx in frame_indices:
        ax.clear()
        points = trial_frame_points(trial, int(frame_idx))
        if not points:
            continue
        draw_reconstruction(
            ax,
            points,
            font,
            title,
            frame_label=f"第{int(frame_idx)}帧 / {frame_idx / trial.rate_hz:.2f}秒",
            show_labels=False,
        )
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
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
    args = parser.parse_args()

    by_trial: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.points):
        by_trial[row["trial_id"]].append(row)

    outputs = []
    for trial_id in sorted(by_trial):
        out = render_trial(by_trial[trial_id], args.out_dir)
        if out is not None:
            outputs.append(out)

    for path in sorted(args.c3d_dir.glob("*/*.c3d")):
        if path.name.startswith("._"):
            continue
        trial = read_c3d(path)
        out = render_trial_gif(trial, args.out_dir, max_frames=args.max_gif_frames)
        if out is not None:
            outputs.append(out)

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
