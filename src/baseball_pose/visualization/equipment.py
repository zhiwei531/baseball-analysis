"""Overlay rendering for tracked bat and ball objects."""

from __future__ import annotations

from typing import Any

from baseball_pose.equipment.schema import ObjectTrackRecord


def draw_equipment_overlay(
    image: Any,
    records: list[ObjectTrackRecord],
    tracks: dict[str, list[tuple[int, int]]] | None = None,
) -> Any:
    cv2 = _require_cv2()
    output = image.copy()
    height, width = output.shape[:2]

    for record in records:
        if record.x is None or record.y is None:
            continue
        point = (int(record.x * width), int(record.y * height))
        if record.object_name == "bat" and record.x2 is not None and record.y2 is not None:
            handle = (int(record.x2 * width), int(record.y2 * height))
            cv2.line(output, handle, point, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(output, point, 6, (0, 128, 255), -1, cv2.LINE_AA)
            cv2.circle(output, handle, 5, (255, 255, 0), -1, cv2.LINE_AA)
            _label(output, "bat", point)
        elif record.object_name == "ball":
            cv2.circle(output, point, 7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(output, point, 3, (0, 0, 255), -1, cv2.LINE_AA)
            _label(output, "ball", point)

    if tracks:
        colors = {"bat": (0, 128, 255), "ball": (255, 255, 255)}
        for object_name, path in tracks.items():
            if len(path) < 2:
                continue
            color = colors.get(object_name, (200, 200, 200))
            for previous, current in zip(path, path[1:]):
                cv2.line(output, previous, current, color, 2, cv2.LINE_AA)
    return output


def _label(image: Any, text: str, point: tuple[int, int]) -> None:
    cv2 = _require_cv2()
    x, y = point
    cv2.putText(image, text, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for equipment visualization. Install project dependencies first."
        ) from exc

    return cv2
