"""Render readable 3D skeleton preview panels."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any

from baseball_pose.pose.schema import POSE_CONNECTIONS
from baseball_pose.pose3d.schema import Pose3DRecord


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    axes: tuple[str, str]


@dataclass(frozen=True)
class ProjectionContext:
    body_scale: float
    bounds_by_projection: dict[str, tuple[float, float, float, float]]


PROJECTIONS = (
    ProjectionSpec(name="Front", axes=("x_3d", "y_3d")),
    ProjectionSpec(name="Side", axes=("z_3d", "y_3d")),
    ProjectionSpec(name="Top", axes=("x_3d", "z_3d")),
)

SMPL24_CONNECTIONS = (
    ("hip", "spine1"),
    ("spine1", "spine2"),
    ("spine2", "spine3"),
    ("spine3", "neck"),
    ("neck", "head"),
    ("hip", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("hip", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
    ("spine3", "left_collar"),
    ("left_collar", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand"),
    ("spine3", "right_collar"),
    ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand"),
)


def draw_pose3d_preview(
    source_image: Any,
    records: list[Pose3DRecord],
    *,
    context: ProjectionContext | None = None,
) -> Any:
    """Draw original frame plus three orthographic 3D skeleton projections."""

    cv2 = _require_cv2()
    import numpy as np

    canvas = np.full((720, 1280, 3), 255, dtype=np.uint8)
    _draw_source_frame(canvas, source_image)
    _draw_projection_header(canvas)

    points = {
        record.joint_name: record
        for record in records
        if _is_finite_record(record)
    }
    if not points:
        return canvas

    pelvis_center = _pelvis_center(points)
    body_scale = _body_scale(points)
    y_axis_sign = _vertical_axis_sign(points)
    normalized = _normalize_points(
        points,
        pelvis_center,
        body_scale if context is None else context.body_scale,
        y_axis_sign=y_axis_sign,
    )
    for idx, spec in enumerate(PROJECTIONS):
        panel = _panel_rect(idx)
        _draw_projection_panel(canvas, panel, spec, normalized, context)

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


def _draw_projection_panel(
    canvas,
    panel,
    spec: ProjectionSpec,
    points: dict[str, tuple[float, float, float]],
    context: ProjectionContext | None,
) -> None:
    cv2 = _require_cv2()
    x0, y0, w, h = panel
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (210, 210, 210), 2)
    cv2.putText(canvas, spec.name, (x0 + 12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"SMPL24 valid {len(points)}/24",
        (x0 + w - 190, y0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (90, 90, 90),
        1,
        cv2.LINE_AA,
    )

    bounds = None if context is None else context.bounds_by_projection.get(spec.name)
    projected = _project_points(points, spec.axes, x0 + 26, y0 + 36, w - 52, h - 52, bounds=bounds)
    for start, end in _connections_for_points(projected):
        if start in projected and end in projected:
            cv2.line(canvas, projected[start], projected[end], (36, 180, 255), 2, cv2.LINE_AA)
    for joint_name, pt in projected.items():
        color = (80, 220, 120) if "wrist" not in joint_name else (80, 120, 255)
        cv2.circle(canvas, pt, 4, color, -1, cv2.LINE_AA)


def _pelvis_center(points: dict[str, Pose3DRecord]) -> tuple[float, float, float]:
    root = points.get("hip")
    if root:
        return (root.x_3d, root.y_3d, root.z_3d)
    left = points.get("left_hip")
    right = points.get("right_hip")
    if left and right:
        return ((left.x_3d + right.x_3d) / 2, (left.y_3d + right.y_3d) / 2, (left.z_3d + right.z_3d) / 2)
    if left:
        return (left.x_3d, left.y_3d, left.z_3d)
    if right:
        return (right.x_3d, right.y_3d, right.z_3d)
    nose = points.get("nose")
    if nose:
        return (nose.x_3d, nose.y_3d, nose.z_3d)
    first = next(iter(points.values()))
    return (first.x_3d, first.y_3d, first.z_3d)


def _normalize_points(
    points: dict[str, Pose3DRecord],
    pelvis_center: tuple[float, float, float],
    body_scale: float,
    *,
    y_axis_sign: float,
) -> dict[str, tuple[float, float, float]]:
    px, py, pz = pelvis_center
    scale = max(body_scale, 1e-6)
    normalized: dict[str, tuple[float, float, float]] = {}
    for joint_name, record in points.items():
        normalized[joint_name] = (
            (record.x_3d - px) / scale,
            y_axis_sign * (record.y_3d - py) / scale,
            (record.z_3d - pz) / scale,
        )
    return normalized


def _project_points(
    points: dict[str, tuple[float, float, float]],
    axes: tuple[str, str],
    x0: int,
    y0: int,
    w: int,
    h: int,
    *,
    bounds: tuple[float, float, float, float] | None = None,
) -> dict[str, tuple[int, int]]:
    axis_index = {"x_3d": 0, "y_3d": 1, "z_3d": 2}
    ax0 = axis_index[axes[0]]
    ax1 = axis_index[axes[1]]
    coords = [(values[ax0], values[ax1]) for values in points.values()]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    if bounds is None:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x, max_x, min_y, max_y = bounds
    range_x = max(max_x - min_x, 1e-5)
    range_y = max(max_y - min_y, 1e-5)
    scale = min(w / range_x, h / range_y) * 1.05
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    mapped: dict[str, tuple[int, int]] = {}
    for joint_name, values in points.items():
        px = int(x0 + w / 2 + (values[ax0] - center_x) * scale)
        py = int(y0 + h / 2 - (values[ax1] - center_y) * scale)
        mapped[joint_name] = (px, py)
    return mapped


def _is_finite_record(record: Pose3DRecord) -> bool:
    return (
        record.x_3d is not None
        and record.y_3d is not None
        and record.z_3d is not None
        and math.isfinite(record.x_3d)
        and math.isfinite(record.y_3d)
        and math.isfinite(record.z_3d)
    )


def build_projection_context(records_by_frame: dict[int, list[Pose3DRecord]]) -> ProjectionContext | None:
    centered_frames: list[dict[str, tuple[float, float, float]]] = []
    body_scales: list[float] = []
    for frame_records in records_by_frame.values():
        points = {
            record.joint_name: record
            for record in frame_records
            if _is_finite_record(record)
        }
        if not points:
            continue
        pelvis_center = _pelvis_center(points)
        body_scale = _body_scale(points)
        y_axis_sign = _vertical_axis_sign(points)
        body_scales.append(body_scale)
        centered_frames.append(_normalize_points(points, pelvis_center, 1.0, y_axis_sign=y_axis_sign))
    if not centered_frames:
        return None

    stable_scale = _median(body_scales, fallback=1.0)
    bounds_by_projection: dict[str, tuple[float, float, float, float]] = {}
    for spec in PROJECTIONS:
        xs: list[float] = []
        ys: list[float] = []
        ax0 = {"x_3d": 0, "y_3d": 1, "z_3d": 2}[spec.axes[0]]
        ax1 = {"x_3d": 0, "y_3d": 1, "z_3d": 2}[spec.axes[1]]
        for points in centered_frames:
            for values in points.values():
                xs.append(values[ax0] / stable_scale)
                ys.append(values[ax1] / stable_scale)
        if not xs or not ys:
            continue
        bounds_by_projection[spec.name] = _robust_bounds(xs, ys)
    return ProjectionContext(body_scale=stable_scale, bounds_by_projection=bounds_by_projection)


def _body_scale(points: dict[str, Pose3DRecord]) -> float:
    candidates: list[float] = []
    for left_name, right_name in (
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
    ):
        left = points.get(left_name)
        right = points.get(right_name)
        if left and right:
            candidates.append(
                math.dist(
                    (left.x_3d, left.y_3d, left.z_3d),
                    (right.x_3d, right.y_3d, right.z_3d),
                )
            )
    left_shoulder = points.get("left_shoulder")
    right_shoulder = points.get("right_shoulder")
    left_hip = points.get("left_hip")
    right_hip = points.get("right_hip")
    if left_shoulder and left_hip:
        candidates.append(
            math.dist(
                (left_shoulder.x_3d, left_shoulder.y_3d, left_shoulder.z_3d),
                (left_hip.x_3d, left_hip.y_3d, left_hip.z_3d),
            )
        )
    if right_shoulder and right_hip:
        candidates.append(
            math.dist(
                (right_shoulder.x_3d, right_shoulder.y_3d, right_shoulder.z_3d),
                (right_hip.x_3d, right_hip.y_3d, right_hip.z_3d),
            )
        )
    return _median(candidates, fallback=1.0)


def _vertical_axis_sign(points: dict[str, Pose3DRecord]) -> float:
    upper_values: list[float] = []
    lower_values: list[float] = []
    for joint_name in ("head", "nose", "neck", "left_shoulder", "right_shoulder"):
        record = points.get(joint_name)
        if record:
            upper_values.append(record.y_3d)
    for joint_name in ("hip", "left_hip", "right_hip", "left_ankle", "right_ankle", "left_foot", "right_foot"):
        record = points.get(joint_name)
        if record:
            lower_values.append(record.y_3d)
    if not upper_values or not lower_values:
        return 1.0
    return 1.0 if _median(upper_values, fallback=0.0) >= _median(lower_values, fallback=0.0) else -1.0


def _connections_for_points(points: dict[str, tuple[int, int]]):
    joint_names = set(points)
    smpl_hits = sum(1 for joint in ("spine1", "spine2", "spine3", "left_collar", "right_collar") if joint in joint_names)
    if smpl_hits >= 2:
        return SMPL24_CONNECTIONS
    return POSE_CONNECTIONS


def _median(values: list[float], *, fallback: float) -> float:
    cleaned = [value for value in values if math.isfinite(value) and value > 0]
    if not cleaned:
        return fallback
    cleaned.sort()
    mid = len(cleaned) // 2
    if len(cleaned) % 2 == 1:
        return cleaned[mid]
    return (cleaned[mid - 1] + cleaned[mid]) / 2


def _robust_bounds(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    xs_sorted = sorted(xs)
    ys_sorted = sorted(ys)
    min_x = _percentile(xs_sorted, 0.05)
    max_x = _percentile(xs_sorted, 0.95)
    min_y = _percentile(ys_sorted, 0.05)
    max_y = _percentile(ys_sorted, 0.95)
    pad_x = max((max_x - min_x) * 0.15, 1e-4)
    pad_y = max((max_y - min_y) * 0.15, 1e-4)
    return (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for 3D visualization. Install project dependencies first."
        ) from exc

    return cv2
