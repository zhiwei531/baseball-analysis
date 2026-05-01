"""Temporal smoothing for pose trajectories."""

from __future__ import annotations

from dataclasses import replace
import statistics

import numpy as np
from scipy.signal import savgol_filter

from baseball_pose.pose.schema import CANONICAL_JOINTS
from baseball_pose.pose.schema import PoseRecord


def smooth_pose_records(
    records: list[PoseRecord],
    method: str = "savgol",
    window_length: int = 7,
    polyorder: int = 2,
    refine_window_length: int = 1,
    confidence_threshold: float = 0.5,
    max_gap_frames: int = 3,
    jump_threshold_multiplier: float = 6.0,
) -> list[PoseRecord]:
    """Smooth each joint trajectory while preserving the common pose schema.

    Low-confidence coordinates are treated as missing values. Short missing
    gaps are linearly interpolated before applying a Savitzky-Golay filter.
    Isolated large jumps are rejected before interpolation because they usually
    indicate a temporary wrong-person lock or detector swap.
    """

    if method != "savgol":
        raise ValueError(f"Unsupported smoothing method: {method}")
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")
    if polyorder >= window_length:
        raise ValueError("polyorder must be smaller than window_length.")
    if refine_window_length < 1:
        raise ValueError("refine_window_length must be at least 1.")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative.")
    if jump_threshold_multiplier <= 0:
        raise ValueError("jump_threshold_multiplier must be positive.")

    by_key = _records_by_key(records)
    smoothed_by_identity: dict[tuple[int, str], PoseRecord] = {}
    for key_records in by_key.values():
        smoothed = _smooth_joint_records(
            key_records,
            window_length=window_length,
            polyorder=polyorder,
            refine_window_length=refine_window_length,
            confidence_threshold=confidence_threshold,
            max_gap_frames=max_gap_frames,
            jump_threshold_multiplier=jump_threshold_multiplier,
        )
        for record in smoothed:
            smoothed_by_identity[(record.frame_index, record.joint_name)] = record

    return [
        smoothed_by_identity.get((record.frame_index, record.joint_name), record)
        for record in records
    ]


def _records_by_key(records: list[PoseRecord]) -> dict[tuple[str, str, str], list[PoseRecord]]:
    grouped: dict[tuple[str, str, str], list[PoseRecord]] = {}
    for record in records:
        if record.joint_name not in CANONICAL_JOINTS:
            continue
        grouped.setdefault((record.clip_id, record.condition_id, record.joint_name), []).append(record)
    for key_records in grouped.values():
        key_records.sort(key=lambda item: item.frame_index)
    return grouped


def _smooth_joint_records(
    records: list[PoseRecord],
    window_length: int,
    polyorder: int,
    refine_window_length: int,
    confidence_threshold: float,
    max_gap_frames: int,
    jump_threshold_multiplier: float,
) -> list[PoseRecord]:
    if not records:
        return []

    x_values = np.array([_confident_value(record, "x", confidence_threshold) for record in records], dtype=float)
    y_values = np.array([_confident_value(record, "y", confidence_threshold) for record in records], dtype=float)
    x_values, y_values = _remove_jump_outliers(x_values, y_values, jump_threshold_multiplier)
    x_values = _interpolate_short_gaps(x_values, max_gap_frames)
    y_values = _interpolate_short_gaps(y_values, max_gap_frames)
    x_values = _savgol_valid_segments(x_values, window_length, polyorder)
    y_values = _savgol_valid_segments(y_values, window_length, polyorder)
    x_values = _moving_average_valid_segments(x_values, refine_window_length)
    y_values = _moving_average_valid_segments(y_values, refine_window_length)

    output: list[PoseRecord] = []
    for record, x_value, y_value in zip(records, x_values, y_values):
        output.append(
            replace(
                record,
                x=None if np.isnan(x_value) else float(x_value),
                y=None if np.isnan(y_value) else float(y_value),
            )
        )
    return output


def _confident_value(record: PoseRecord, axis: str, confidence_threshold: float) -> float:
    score = record.confidence if record.confidence is not None else record.visibility
    value = getattr(record, axis)
    if value is None or (score is not None and score < confidence_threshold):
        return np.nan
    return float(value)


def _remove_jump_outliers(
    x_values: np.ndarray,
    y_values: np.ndarray,
    threshold_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = [
        (float(x), float(y))
        for x, y in zip(x_values, y_values)
        if not np.isnan(x) and not np.isnan(y)
    ]
    if len(points) < 5:
        return x_values, y_values

    steps = [
        float(np.hypot(curr[0] - prev[0], curr[1] - prev[1]))
        for prev, curr in zip(points, points[1:])
    ]
    nonzero_steps = [step for step in steps if step > 0]
    if not nonzero_steps:
        return x_values, y_values

    median_step = statistics.median(nonzero_steps)
    threshold = max(median_step * threshold_multiplier, 0.03)
    cleaned_x = x_values.copy()
    cleaned_y = y_values.copy()
    for index in range(1, len(x_values) - 1):
        if any(np.isnan(value) for value in (x_values[index], y_values[index])):
            continue
        previous = _nearest_valid_point(x_values, y_values, index, -1)
        next_point = _nearest_valid_point(x_values, y_values, index, 1)
        if previous is None or next_point is None:
            continue
        current = (x_values[index], y_values[index])
        previous_distance = float(np.hypot(current[0] - previous[0], current[1] - previous[1]))
        next_distance = float(np.hypot(current[0] - next_point[0], current[1] - next_point[1]))
        neighbor_distance = float(np.hypot(next_point[0] - previous[0], next_point[1] - previous[1]))
        if previous_distance > threshold and next_distance > threshold and neighbor_distance < threshold:
            cleaned_x[index] = np.nan
            cleaned_y[index] = np.nan
    return cleaned_x, cleaned_y


def _nearest_valid_point(
    x_values: np.ndarray,
    y_values: np.ndarray,
    start_index: int,
    direction: int,
) -> tuple[float, float] | None:
    index = start_index + direction
    while 0 <= index < len(x_values):
        if not np.isnan(x_values[index]) and not np.isnan(y_values[index]):
            return float(x_values[index]), float(y_values[index])
        index += direction
    return None


def _interpolate_short_gaps(values: np.ndarray, max_gap_frames: int) -> np.ndarray:
    output = values.copy()
    index = 0
    while index < len(output):
        if not np.isnan(output[index]):
            index += 1
            continue
        start = index
        while index < len(output) and np.isnan(output[index]):
            index += 1
        end = index
        gap_length = end - start
        if (
            gap_length <= max_gap_frames
            and start > 0
            and end < len(output)
            and not np.isnan(output[start - 1])
            and not np.isnan(output[end])
        ):
            output[start:end] = np.linspace(output[start - 1], output[end], gap_length + 2)[1:-1]
    return output


def _savgol_valid_segments(values: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    output = values.copy()
    index = 0
    while index < len(output):
        if np.isnan(output[index]):
            index += 1
            continue
        start = index
        while index < len(output) and not np.isnan(output[index]):
            index += 1
        end = index
        segment_length = end - start
        segment_window = min(window_length, segment_length if segment_length % 2 == 1 else segment_length - 1)
        if segment_window <= polyorder or segment_window < 3:
            continue
        output[start:end] = savgol_filter(output[start:end], segment_window, polyorder)
    return output


def _moving_average_valid_segments(values: np.ndarray, window_length: int) -> np.ndarray:
    if window_length <= 1:
        return values

    output = values.copy()
    index = 0
    while index < len(output):
        if np.isnan(output[index]):
            index += 1
            continue
        start = index
        while index < len(output) and not np.isnan(output[index]):
            index += 1
        end = index
        segment_length = end - start
        if segment_length < 2:
            continue
        segment_window = min(window_length, segment_length)
        pad_left = segment_window // 2
        pad_right = segment_window - 1 - pad_left
        padded = np.pad(output[start:end], (pad_left, pad_right), mode="edge")
        kernel = np.ones(segment_window, dtype=float) / segment_window
        output[start:end] = np.convolve(padded, kernel, mode="valid")
    return output
