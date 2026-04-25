"""Pose overlay rendering boundary."""

from __future__ import annotations

from typing import Any

from baseball_pose.pose.schema import POSE_CONNECTIONS, PoseRecord


def draw_pose_overlay(
    image: Any,
    records: list[PoseRecord],
    confidence_threshold: float = 0.5,
    tracks: dict[str, list[tuple[int, int]]] | None = None,
) -> Any:
    """Draw skeleton and optional joint tracks on one BGR image."""

    cv2 = _require_cv2()
    output = image.copy()
    height, width = output.shape[:2]
    points: dict[str, tuple[int, int]] = {}

    for record in records:
        if record.x is None or record.y is None:
            continue
        score = record.confidence if record.confidence is not None else record.visibility
        if score is not None and score < confidence_threshold:
            continue
        x = int(record.x * width)
        y = int(record.y * height)
        if 0 <= x < width and 0 <= y < height:
            points[record.joint_name] = (x, y)

    for start, end in POSE_CONNECTIONS:
        if start in points and end in points:
            cv2.line(output, points[start], points[end], (36, 180, 255), 2, cv2.LINE_AA)

    for joint_name, point in points.items():
        color = (80, 220, 120) if "wrist" not in joint_name else (80, 120, 255)
        cv2.circle(output, point, 4, color, -1, cv2.LINE_AA)

    if tracks:
        for joint_name, path in tracks.items():
            if len(path) < 2:
                continue
            color = (255, 80, 80) if "left" in joint_name else (180, 80, 255)
            for previous, current in zip(path, path[1:]):
                cv2.line(output, previous, current, color, 2, cv2.LINE_AA)

    return output


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for visualization. Install project dependencies first."
        ) from exc

    return cv2
