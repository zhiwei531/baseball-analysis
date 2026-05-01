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
    median_window_length: int = 1,
    refine_window_length: int = 1,
    confidence_threshold: float = 0.5,
    max_gap_frames: int = 3,
    jump_threshold_multiplier: float = 6.0,
    torso_gate_enabled: bool = True,
    torso_jump_threshold_multiplier: float = 8.0,
    min_torso_jump_distance: float = 0.08,
) -> list[PoseRecord]:
    """Smooth each joint trajectory while preserving the common pose schema.

    Low-confidence coordinates are treated as missing values. A torso continuity
    gate can mask whole frames that jump too far from the previous accepted body
    center, which helps when the detector briefly locks onto the wrong person.
    Short missing gaps are interpolated before median, Savitzky-Golay, and
    moving-average smoothing.
    """

    if method != "savgol":
        raise ValueError(f"Unsupported smoothing method: {method}")
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")
    if polyorder >= window_length:
        raise ValueError("polyorder must be smaller than window_length.")
    if median_window_length < 1:
        raise ValueError("median_window_length must be at least 1.")
    if refine_window_length < 1:
        raise ValueError("refine_window_length must be at least 1.")
    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative.")
    if jump_threshold_multiplier <= 0:
        raise ValueError("jump_threshold_multiplier must be positive.")
    if torso_jump_threshold_multiplier <= 0:
        raise ValueError("torso_jump_threshold_multiplier must be positive.")
    if min_torso_jump_distance <= 0:
        raise ValueError("min_torso_jump_distance must be positive.")

    gated_records = (
        _apply_torso_continuity_gate(
            records,
            confidence_threshold=confidence_threshold,
            jump_threshold_multiplier=torso_jump_threshold_multiplier,
            min_jump_distance=min_torso_jump_distance,
        )
        if torso_gate_enabled
        else records
    )
    by_key = _records_by_key(gated_records)
    smoothed_by_identity: dict[tuple[int, str], PoseRecord] = {}
    for key_records in by_key.values():
        smoothed = _smooth_joint_records(
            key_records,
            window_length=window_length,
            polyorder=polyorder,
            median_window_length=median_window_length,
            refine_window_length=refine_window_length,
            confidence_threshold=confidence_threshold,
            max_gap_frames=max_gap_frames,
            jump_threshold_multiplier=jump_threshold_multiplier,
        )
        for record in smoothed:
            smoothed_by_identity[(record.frame_index, record.joint_name)] = record

    return [
        smoothed_by_identity.get((record.frame_index, record.joint_name), record)
        for record in gated_records
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
    median_window_length: int,
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
    x_values = _median_valid_segments(x_values, median_window_length)
    y_values = _median_valid_segments(y_values, median_window_length)
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


def _apply_torso_continuity_gate(
    records: list[PoseRecord],
    confidence_threshold: float,
    jump_threshold_multiplier: float,
    min_jump_distance: float,
) -> list[PoseRecord]:
    by_frame: dict[int, list[PoseRecord]] = {}
    for record in records:
        by_frame.setdefault(record.frame_index, []).append(record)

    centers = {
        frame_index: _torso_center(frame_records, confidence_threshold)
        for frame_index, frame_records in by_frame.items()
    }
    valid_centers = [center for _, center in sorted(centers.items()) if center is not None]
    if len(valid_centers) < 5:
        return records

    steps = [
        float(np.hypot(curr[0] - prev[0], curr[1] - prev[1]))
        for prev, curr in zip(valid_centers, valid_centers[1:])
    ]
    nonzero_steps = [step for step in steps if step > 0]
    if not nonzero_steps:
        return records

    threshold = max(statistics.median(nonzero_steps) * jump_threshold_multiplier, min_jump_distance)
    rejected_frames: set[int] = set()
    last_accepted_center: tuple[float, float] | None = None
    last_accepted_frame: int | None = None

    for frame_index in sorted(by_frame):
        center = centers[frame_index]
        if center is None:
            continue
        if last_accepted_center is None or last_accepted_frame is None:
            last_accepted_center = center
            last_accepted_frame = frame_index
            continue
        frame_gap = max(1, frame_index - last_accepted_frame)
        allowed_distance = threshold * (frame_gap**0.5)
        distance = float(
            np.hypot(center[0] - last_accepted_center[0], center[1] - last_accepted_center[1])
        )
        if distance > allowed_distance:
            rejected_frames.add(frame_index)
            continue
        last_accepted_center = center
        last_accepted_frame = frame_index

    if not rejected_frames:
        return records

    return [
        replace(record, x=None, y=None) if record.frame_index in rejected_frames else record
        for record in records
    ]


def _torso_center(
    records: list[PoseRecord],
    confidence_threshold: float,
) -> tuple[float, float] | None:
    points: dict[str, tuple[float, float]] = {}
    torso_joints = {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}
    for record in records:
        if record.joint_name not in torso_joints:
            continue
        score = record.confidence if record.confidence is not None else record.visibility
        if record.x is None or record.y is None or (score is not None and score < confidence_threshold):
            continue
        points[record.joint_name] = (record.x, record.y)
    if "left_hip" in points and "right_hip" in points:
        return (
            (points["left_hip"][0] + points["right_hip"][0]) / 2,
            (points["left_hip"][1] + points["right_hip"][1]) / 2,
        )
    if "left_shoulder" in points and "right_shoulder" in points:
        return (
            (points["left_shoulder"][0] + points["right_shoulder"][0]) / 2,
            (points["left_shoulder"][1] + points["right_shoulder"][1]) / 2,
        )
    if points:
        return (
            sum(point[0] for point in points.values()) / len(points),
            sum(point[1] for point in points.values()) / len(points),
        )
    return None


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


def _median_valid_segments(values: np.ndarray, window_length: int) -> np.ndarray:
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
        if segment_window % 2 == 0 and segment_window > 1:
            segment_window -= 1
        if segment_window < 3:
            continue
        pad = segment_window // 2
        padded = np.pad(output[start:end], (pad, pad), mode="edge")
        output[start:end] = np.array(
            [
                statistics.median(padded[offset : offset + segment_window])
                for offset in range(segment_length)
            ],
            dtype=float,
        )
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
