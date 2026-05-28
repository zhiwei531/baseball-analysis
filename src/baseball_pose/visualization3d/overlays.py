"""Render readable 3D skeleton preview panels."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from baseball_pose.pose.schema import POSE_CONNECTIONS
from baseball_pose.pose3d.schema import Pose3DRecord


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    axes: tuple[str, str]


PROJECTIONS = (
    ProjectionSpec(name="Front", axes=("x_3d", "y_3d")),
    ProjectionSpec(name="Side", axes=("z_3d", "y_3d")),
    ProjectionSpec(name="Top", axes=("x_3d", "z_3d")),
)


def draw_pose3d_preview(source_image: Any, records: list[Pose3DRecord]) -> Any:
    """Draw original frame plus three orthographic 3D skeleton projections."""

    cv2 = _require_cv2()
    import numpy as np

    canvas = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _draw_source_frame(canvas, source_image)
    _draw_projection_header(canvas)

    points = {record.joint_name: record for record in records}
    if not points:
        return canvas

    pelvis_center = _pelvis_center(points)
    normalized = _normalize_points(points, pelvis_center)
    for idx, spec in enumerate(PROJECTIONS):
        panel = _panel_rect(idx)
        _draw_projection_panel(canvas, panel, spec, normalized)

    return canvas


def _draw_source_frame(canvas, source_image) -> None:
    cv2 = _require_cv2()
    src_h, src_w = source_image.shape[:2]
    target_x, target_y, target_w, target_h = 40, 60, 560, 620
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = cv2.resize(source_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = target_x + (target_w - new_w) // 2
    y = target_y + (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    cv2.rectangle(canvas, (target_x, target_y), (target_x + target_w, target_y + target_h), (210, 210, 210), 2)
    cv2.putText(canvas, "Original Frame", (target_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)


def _draw_projection_header(canvas) -> None:
    cv2 = _require_cv2()
    cv2.putText(canvas, "3D Skeleton Views", (690, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)


def _panel_rect(index: int) -> tuple[int, int, int, int]:
    x = 660
    y = 60 + index * 210
    return (x, y, 560, 180)


def _draw_projection_panel(canvas, panel, spec: ProjectionSpec, points: dict[str, tuple[float, float, float]]) -> None:
    cv2 = _require_cv2()
    x0, y0, w, h = panel
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (210, 210, 210), 2)
    cv2.putText(canvas, spec.name, (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)

    projected = _project_points(points, spec.axes, x0 + 26, y0 + 36, w - 52, h - 52)
    for start, end in POSE_CONNECTIONS:
        if start in projected and end in projected:
            cv2.line(canvas, projected[start], projected[end], (36, 180, 255), 2, cv2.LINE_AA)
    for joint_name, pt in projected.items():
        color = (80, 220, 120) if "wrist" not in joint_name else (80, 120, 255)
        cv2.circle(canvas, pt, 4, color, -1, cv2.LINE_AA)


def _pelvis_center(points: dict[str, Pose3DRecord]) -> tuple[float, float, float]:
    left = points.get("left_hip")
    right = points.get("right_hip")
    if left and right:
        return ((left.x_3d + right.x_3d) / 2, (left.y_3d + right.y_3d) / 2, (left.z_3d + right.z_3d) / 2)
    if left:
        return (left.x_3d, left.y_3d, left.z_3d)
    if right:
        return (right.x_3d, right.y_3d, right.z_3d)
    nose = points["nose"]
    return (nose.x_3d, nose.y_3d, nose.z_3d)


def _normalize_points(
    points: dict[str, Pose3DRecord],
    pelvis_center: tuple[float, float, float],
) -> dict[str, tuple[float, float, float]]:
    px, py, pz = pelvis_center
    normalized: dict[str, tuple[float, float, float]] = {}
    for joint_name, record in points.items():
        normalized[joint_name] = (record.x_3d - px, -(record.y_3d - py), record.z_3d - pz)
    return normalized


def _project_points(
    points: dict[str, tuple[float, float, float]],
    axes: tuple[str, str],
    x0: int,
    y0: int,
    w: int,
    h: int,
) -> dict[str, tuple[int, int]]:
    axis_index = {"x_3d": 0, "y_3d": 1, "z_3d": 2}
    ax0 = axis_index[axes[0]]
    ax1 = axis_index[axes[1]]
    coords = [(values[ax0], values[ax1]) for values in points.values()]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max(max_x - min_x, 1e-5)
    range_y = max(max_y - min_y, 1e-5)
    scale = min(w / range_x, h / range_y) * 0.82
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    mapped: dict[str, tuple[int, int]] = {}
    for joint_name, values in points.items():
        px = int(x0 + w / 2 + (values[ax0] - center_x) * scale)
        py = int(y0 + h / 2 - (values[ax1] - center_y) * scale)
        mapped[joint_name] = (px, py)
    return mapped


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for 3D visualization. Install project dependencies first."
        ) from exc

    return cv2
