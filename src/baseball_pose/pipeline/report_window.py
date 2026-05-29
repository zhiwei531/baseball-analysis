"""Detect a compact action window for report-facing summaries and visuals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ActionWindow:
    start_index: int
    end_index: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    peak_index: int
    peak_frame: int
    peak_score: float
    frame_count: int


def detect_action_window(
    rows: list[dict[str, Any]],
    action_type: str = "batting",
) -> ActionWindow | None:
    """Return a compact motion window centered on the strongest action burst."""

    if not rows:
        return None

    parsed = [_coerce_row(row) for row in rows]
    scores = _smoothed_motion_scores(parsed, action_type)
    if not scores or max(scores) <= 0:
        return _full_window(parsed)

    peak_index, peak_score, start_index, end_index = _core_motion_window(scores)

    pad_before, pad_after, min_frames, max_frames = _window_parameters(action_type)
    start_index, end_index = _apply_window_constraints(
        start_index=start_index,
        end_index=end_index,
        peak_index=peak_index,
        total_count=len(parsed),
        pad_before=pad_before,
        pad_after=pad_after,
        min_frames=min_frames,
        max_frames=max_frames,
    )

    return _build_window(parsed, start_index, end_index, peak_index, peak_score)


def detect_action_video_window(
    rows: list[dict[str, Any]],
    action_type: str = "batting",
) -> ActionWindow | None:
    """Return a wider action window for exporting a fuller motion clip."""

    if not rows:
        return None

    parsed = [_coerce_row(row) for row in rows]
    scores = _smoothed_motion_scores(parsed, action_type)
    if not scores or max(scores) <= 0:
        return _full_window(parsed)

    peak_index, peak_score, core_start, core_end = _core_motion_window(scores)
    search_before, search_after, pad_before, pad_after, min_frames, max_frames = _video_window_parameters(action_type)
    start_index = _valley_index(scores, max(0, core_start - search_before), core_start)
    end_index = _valley_index(scores, core_end, min(len(scores) - 1, core_end + search_after))
    start_index, end_index = _apply_window_constraints(
        start_index=start_index,
        end_index=end_index,
        peak_index=peak_index,
        total_count=len(parsed),
        pad_before=pad_before,
        pad_after=pad_after,
        min_frames=min_frames,
        max_frames=max_frames,
    )

    return _build_window(parsed, start_index, end_index, peak_index, peak_score)


def _core_motion_window(scores: list[float]) -> tuple[int, float, int, int]:
    peak_index = max(range(len(scores)), key=lambda index: scores[index])
    peak_score = scores[peak_index]
    active_threshold = max(peak_score * 0.42, 0.28)

    start_index = peak_index
    while start_index > 0 and scores[start_index - 1] >= active_threshold:
        start_index -= 1

    end_index = peak_index
    while end_index < len(scores) - 1 and scores[end_index + 1] >= active_threshold:
        end_index += 1
    return peak_index, peak_score, start_index, end_index


def _apply_window_constraints(
    *,
    start_index: int,
    end_index: int,
    peak_index: int,
    total_count: int,
    pad_before: int,
    pad_after: int,
    min_frames: int,
    max_frames: int,
) -> tuple[int, int]:
    start_index = max(0, start_index - pad_before)
    end_index = min(total_count - 1, end_index + pad_after)

    current_count = end_index - start_index + 1
    if current_count < min_frames:
        deficit = min_frames - current_count
        extra_before = deficit // 2
        extra_after = deficit - extra_before
        start_index = max(0, start_index - extra_before)
        end_index = min(len(parsed) - 1, end_index + extra_after)

    current_count = end_index - start_index + 1
    if current_count > max_frames:
        half = max_frames // 2
        start_index = max(0, peak_index - half)
        end_index = min(total_count - 1, start_index + max_frames - 1)
        if end_index - start_index + 1 < max_frames:
            start_index = max(0, end_index - max_frames + 1)
    return start_index, end_index


def _build_window(
    parsed: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    peak_index: int,
    peak_score: float,
) -> ActionWindow:
    start_row = parsed[start_index]
    end_row = parsed[end_index]
    peak_row = parsed[peak_index]
    return ActionWindow(
        start_index=start_index,
        end_index=end_index,
        start_frame=int(start_row.get("frame_index", start_index)),
        end_frame=int(end_row.get("frame_index", end_index)),
        start_time_sec=float(start_row.get("timestamp_sec", 0.0) or 0.0),
        end_time_sec=float(end_row.get("timestamp_sec", 0.0) or 0.0),
        peak_index=peak_index,
        peak_frame=int(peak_row.get("frame_index", peak_index)),
        peak_score=float(peak_score),
        frame_count=end_index - start_index + 1,
    )


def filter_rows_to_action_window(
    rows: list[dict[str, Any]],
    action_type: str = "batting",
    expanded: bool = False,
) -> tuple[list[dict[str, Any]], ActionWindow | None]:
    detector = detect_action_video_window if expanded else detect_action_window
    window = detector(rows, action_type=action_type)
    if window is None:
        return rows, None
    return rows[window.start_index : window.end_index + 1], window


def frame_indices_in_action_window(
    rows: list[dict[str, Any]],
    action_type: str = "batting",
    expanded: bool = False,
) -> set[int]:
    detector = detect_action_video_window if expanded else detect_action_window
    window = detector(rows, action_type=action_type)
    if window is None:
        return set()
    parsed = [_coerce_row(row) for row in rows]
    return {
        int(parsed[index].get("frame_index", index))
        for index in range(window.start_index, window.end_index + 1)
    }


def _window_parameters(action_type: str) -> tuple[int, int, int, int]:
    if action_type == "pitching":
        return (10, 10, 30, 72)
    return (8, 8, 24, 54)


def _video_window_parameters(action_type: str) -> tuple[int, int, int, int, int, int]:
    if action_type == "pitching":
        return (55, 45, 14, 14, 48, 108)
    return (45, 35, 18, 4, 48, 90)


def _valley_index(scores: list[float], left: int, right: int) -> int:
    if left >= right:
        return left
    valley_score = min(scores[left : right + 1])
    tolerance = max(0.02, valley_score * 0.1)
    candidates = [
        index
        for index in range(left, right + 1)
        if scores[index] <= valley_score + tolerance
    ]
    return candidates[-1] if candidates else left


def _full_window(rows: list[dict[str, Any]]) -> ActionWindow:
    start_row = rows[0]
    end_row = rows[-1]
    return ActionWindow(
        start_index=0,
        end_index=len(rows) - 1,
        start_frame=int(start_row.get("frame_index", 0)),
        end_frame=int(end_row.get("frame_index", len(rows) - 1)),
        start_time_sec=float(start_row.get("timestamp_sec", 0.0) or 0.0),
        end_time_sec=float(end_row.get("timestamp_sec", 0.0) or 0.0),
        peak_index=0,
        peak_frame=int(start_row.get("frame_index", 0)),
        peak_score=0.0,
        frame_count=len(rows),
    )


def _smoothed_motion_scores(rows: list[dict[str, Any]], action_type: str) -> list[float]:
    hand_values = [abs(_optional_float(row.get("hand_speed_proxy"))) for row in rows]
    trunk_values = [abs(_optional_float(row.get("trunk_rotation_velocity_deg_s"))) for row in rows]
    pelvis_values = [abs(_optional_float(row.get("pelvis_rotation_velocity_deg_s"))) for row in rows]

    hand_scale = _robust_scale(hand_values)
    trunk_scale = _robust_scale(trunk_values)
    pelvis_scale = _robust_scale(pelvis_values)

    if action_type == "pitching":
        hand_weight, trunk_weight, pelvis_weight = (0.30, 0.38, 0.32)
    else:
        hand_weight, trunk_weight, pelvis_weight = (0.55, 0.25, 0.20)

    raw_scores = []
    for hand, trunk, pelvis in zip(hand_values, trunk_values, pelvis_values):
        raw_scores.append(
            hand_weight * min(hand / hand_scale, 2.0)
            + trunk_weight * min(trunk / trunk_scale, 2.0)
            + pelvis_weight * min(pelvis / pelvis_scale, 2.0)
        )
    return _moving_average(raw_scores, radius=3)


def _robust_scale(values: list[float]) -> float:
    positives = sorted(value for value in values if value > 0 and not math.isnan(value))
    if not positives:
        return 1.0
    if len(positives) == 1:
        return max(positives[0], 1.0)
    position = (len(positives) - 1) * 0.95
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return max(positives[lower], 1.0)
    lower_value = positives[lower]
    upper_value = positives[upper]
    weight = position - lower
    return max(lower_value + (upper_value - lower_value) * weight, 1.0)


def _moving_average(values: list[float], radius: int) -> list[float]:
    if not values:
        return []
    averaged: list[float] = []
    for index in range(len(values)):
        left = max(0, index - radius)
        right = min(len(values), index + radius + 1)
        window = values[left:right]
        averaged.append(sum(window) / len(window))
    return averaged


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    coerced: dict[str, Any] = {}
    for key, value in row.items():
        if value in {"", None}:
            coerced[key] = None
            continue
        if isinstance(value, (int, float)):
            coerced[key] = value
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            coerced[key] = value
            continue
        coerced[key] = int(number) if key == "frame_index" else number
    return coerced


def _optional_float(value: Any) -> float:
    if value in {"", None}:
        return 0.0
    return float(value)
