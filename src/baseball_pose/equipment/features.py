"""Feature extraction for tracked bat and ball objects."""

from __future__ import annotations

import math

from baseball_pose.equipment.schema import ObjectFeatureRow, ObjectTrackRecord


def extract_object_motion_features(
    records: list[ObjectTrackRecord],
    bat_confidence_threshold: float = 0.55,
    ball_confidence_threshold: float = 0.45,
    max_bat_speed_px_s: float | None = 8000.0,
    max_ball_speed_px_s: float | None = 12000.0,
) -> list[ObjectFeatureRow]:
    if not records:
        return []

    by_frame: dict[int, dict[str, ObjectTrackRecord]] = {}
    for record in records:
        by_frame.setdefault(record.frame_index, {})[record.object_name] = record

    previous_points: dict[str, tuple[float, float, float, int | None, int | None]] = {}
    rows: list[ObjectFeatureRow] = []
    for frame_index in sorted(by_frame):
        frame_records = by_frame[frame_index]
        sample = next(iter(frame_records.values()))
        bat = _trusted(frame_records.get("bat"), bat_confidence_threshold)
        ball = _trusted(frame_records.get("ball"), ball_confidence_threshold)
        bat_speed_px_s, bat_speed_norm_s = _speed("bat", bat, previous_points, max_bat_speed_px_s)
        ball_speed_px_s, ball_speed_norm_s = _speed("ball", ball, previous_points, max_ball_speed_px_s)
        rows.append(
            ObjectFeatureRow(
                clip_id=sample.clip_id,
                condition_id=sample.condition_id,
                frame_index=frame_index,
                timestamp_sec=sample.timestamp_sec,
                bat_barrel_x=bat.x if bat else None,
                bat_barrel_y=bat.y if bat else None,
                bat_handle_x=bat.x2 if bat else None,
                bat_handle_y=bat.y2 if bat else None,
                bat_speed_px_s=bat_speed_px_s,
                bat_speed_norm_s=bat_speed_norm_s,
                bat_angle_deg=_bat_angle(bat),
                ball_x=ball.x if ball else None,
                ball_y=ball.y if ball else None,
                ball_speed_px_s=ball_speed_px_s,
                ball_speed_norm_s=ball_speed_norm_s,
            )
        )
    return rows


def _trusted(record: ObjectTrackRecord | None, confidence_threshold: float) -> ObjectTrackRecord | None:
    if record is None:
        return None
    if record.confidence is not None and record.confidence < confidence_threshold:
        return None
    return record


def _speed(
    key: str,
    record: ObjectTrackRecord | None,
    previous_points: dict[str, tuple[float, float, float, int | None, int | None]],
    max_speed_px_s: float | None,
) -> tuple[float | None, float | None]:
    if record is None or record.x is None or record.y is None:
        return None, None
    previous = previous_points.get(key)
    if previous is None:
        previous_points[key] = (record.x, record.y, record.timestamp_sec, record.width, record.height)
        return None, None
    dt = record.timestamp_sec - previous[2]
    if dt <= 0:
        return None, None
    norm_speed = math.hypot(record.x - previous[0], record.y - previous[1]) / dt
    if record.width is None or record.height is None or previous[3] is None or previous[4] is None:
        return None, norm_speed
    x_px = record.x * record.width
    y_px = record.y * record.height
    previous_x_px = previous[0] * previous[3]
    previous_y_px = previous[1] * previous[4]
    speed_px_s = math.hypot(x_px - previous_x_px, y_px - previous_y_px) / dt
    if max_speed_px_s is not None and speed_px_s > max_speed_px_s:
        return None, None
    previous_points[key] = (record.x, record.y, record.timestamp_sec, record.width, record.height)
    return speed_px_s, norm_speed


def _bat_angle(record: ObjectTrackRecord | None) -> float | None:
    if record is None or None in {record.x, record.y, record.x2, record.y2}:
        return None
    return math.degrees(math.atan2(record.y - record.y2, record.x - record.x2))
