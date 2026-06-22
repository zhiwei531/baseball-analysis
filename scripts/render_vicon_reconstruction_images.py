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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POINTS = ROOT / "reports" / "vicon_2026_point_summary.csv"
DEFAULT_OUT_DIR = ROOT / "reports" / "assets" / "vicon_reconstruction"

BODY_SEGMENTS = [
    ("LFHD", "RFHD"),
    ("LFHD", "C7"),
    ("RFHD", "C7"),
    ("C7", "T10"),
    ("LSHO", "RSHO"),
    ("LSHO", "LELB"),
    ("LELB", "LWRA"),
    ("LELB", "LWRB"),
    ("RSHO", "RELB"),
    ("RELB", "RWRA"),
    ("RELB", "RWRB"),
    ("LASI", "RASI"),
    ("LASI", "LKNE"),
    ("LKNE", "LANK"),
    ("RASI", "RKNE"),
    ("RKNE", "RANK"),
]
BAT_SEGMENTS = [("Bat1", "Bat5")]
LABEL_POINTS = ["LFHD", "RFHD", "C7", "T10", "LSHO", "RSHO", "LASI", "RASI", "Bat1", "Bat5"]
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
    radius = max(float((maxs - mins).max()) / 2, 250.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_segment(ax, points: dict[str, tuple[float, float, float]], a: str, b: str, color: str, width: float) -> None:
    if a not in points or b not in points:
        return
    pa = points[a]
    pb = points[b]
    ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color=color, linewidth=width, solid_capstyle="round")


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

    body_points = {k: v for k, v in points.items() if not k.startswith("Bat")}
    bat_points = {k: v for k, v in points.items() if k.startswith("Bat")}

    fig = plt.figure(figsize=(8.0, 5.2), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    font = zh_font()
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    ax.view_init(elev=17, azim=-66)

    for a, b in BODY_SEGMENTS:
        draw_segment(ax, points, a, b, "#2563eb", 3.2)
    for a, b in BAT_SEGMENTS:
        draw_segment(ax, points, a, b, "#f97316", 5.2)

    if body_points:
        body = np.array(list(body_points.values()), dtype=float)
        ax.scatter(body[:, 0], body[:, 1], body[:, 2], s=34, c="#101828", depthshade=False, label="身体点")
    if bat_points:
        bat = np.array(list(bat_points.values()), dtype=float)
        ax.scatter(bat[:, 0], bat[:, 1], bat[:, 2], s=58, c="#f97316", depthshade=False, label="球棒点")

    for name in LABEL_POINTS:
        if name not in points:
            continue
        x, y, z = points[name]
        ax.text(x, y, z, name, fontsize=6.5, color="#475467")

    set_equal_axes(ax, points)
    ax.set_xlabel("X（毫米）", labelpad=8, fontsize=9, fontproperties=font)
    ax.set_ylabel("Y（毫米）", labelpad=8, fontsize=9, fontproperties=font)
    ax.set_zlabel("Z（毫米）", labelpad=8, fontsize=9, fontproperties=font)
    ax.tick_params(axis="both", labelsize=7, pad=1)
    ax.grid(True, color="#e4e7ec", linewidth=0.7)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=8)
    if font is not None:
        for text in legend.get_texts():
            text.set_fontproperties(font)

    action_text = "投球" if action == "pitching" else "打击"
    time_text = f"{time_sec:.2f}秒" if time_sec is not None else "暂无时间"
    fig.suptitle(f"{sample} / {action_text} / {event} / 第{frame}帧 / {time_text}", fontsize=10, color="#101828", fontproperties=font)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{trial_id}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Vicon C3D key-action reconstruction PNGs.")
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    by_trial: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.points):
        by_trial[row["trial_id"]].append(row)

    outputs = []
    for trial_id in sorted(by_trial):
        out = render_trial(by_trial[trial_id], args.out_dir)
        if out is not None:
            outputs.append(out)

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
