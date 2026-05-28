"""Temporal smoothing for relative 3D pose trajectories."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import statistics
from math import nan

import numpy as np
from scipy.signal import savgol_filter

from baseball_pose.pose.quality import LIMB_SEGMENTS
from baseball_pose.pose.quality import gap_for_joint, multiplier_for_joint, threshold_for_joint
from baseball_pose.pose.schema import CANONICAL_JOINTS
from baseball_pose.pose3d.schema import Pose3DRecord


def smooth_pose3d_records(
    records: list[Pose3DRecord],
    *,
    window_length: int = 15,
    polyorder: int = 2,
    median_window_length: int = 5,
    refine_window_length: int = 9,
    confidence_threshold: float = 0.5,
    threshold_config: dict[str, object] | None = None,
    max_gap_frames: int = 3,
    max_gap_config: dict[str, object] | None = None,
    jump_threshold_multiplier: float = 3.0,
    joint_jump_config: dict[str, object] | None = None,
    limb_length_tolerance_ratio: float = 0.28,
) -> list[Pose3DRecord]:
    """Smooth each joint trajectory in 3D while preserving the flat record schema."""

    if window_length % 2 == 0:
        raise ValueError("window_length must be odd.")
    if polyorder >= window_length:
        raise ValueError("polyorder must be smaller than window_length.")
    if limb_length_tolerance_ratio <= 0:
        raise ValueError("limb_length_tolerance_ratio must be positive.")

    gated_records = _apply_limb_length_consistency_gate(
        records,
        confidence_threshold=confidence_threshold,
        threshold_config=threshold_config,
        tolerance_ratio=limb_length_tolerance_ratio,
    )
    by_key = _records_by_key(gated_records)
    smoothed_by_identity: dict[tuple[int, str], Pose3DRecord] = {}
    for key_records in by_key.values():
        smoothed = _smooth_joint_records(
            key_records,
            window_length=window_length,
            polyorder=polyorder,
            median_window_length=median_window_length,
            refine_window_length=refine_window_length,
            confidence_threshold=confidence_threshold,
            threshold_config=threshold_config,
            max_gap_frames=max_gap_frames,
            max_gap_config=max_gap_config,
            jump_threshold_multiplier=jump_threshold_multiplier,
            joint_jump_config=joint_jump_config,
        )
        for record in smoothed:
            smoothed_by_identity[(record.frame_index, record.joint_name)] = record

    return [
        smoothed_by_identity.get((record.frame_index, record.joint_name), record)
        for record in gated_records
    ]


def _apply_limb_length_consistency_gate(
    records: list[Pose3DRecord],
    *,
    confidence_threshold: float,
    threshold_config: dict[str, object] | None,
    tolerance_ratio: float,
) -> list[Pose3DRecord]:
    by_frame: dict[int, list[Pose3DRecord]] = defaultdict(list)
    for record in records:
        by_frame[record.frame_index].append(record)

    baseline_lengths: dict[tuple[str, str], float] = {}
    for proximal, distal, _ in LIMB_SEGMENTS:
        lengths: list[float] = []
        for frame_records in by_frame.values():
            points = _confident_points(
                frame_records,
                confidence_threshold=confidence_threshold,
                threshold_config=threshold_config,
            )
            if proximal not in points or distal not in points:
                continue
            lengths.append(float(np.linalg.norm(points[distal] - points[proximal])))
        if lengths:
            baseline_lengths[(proximal, distal)] = statistics.median(lengths)

    if not baseline_lengths:
        return records

    rejected: set[tuple[int, str]] = set()
    for frame_index, frame_records in by_frame.items():
        points = _confident_points(
            frame_records,
            confidence_threshold=confidence_threshold,
            threshold_config=threshold_config,
        )
        for proximal, distal, reject_joint in LIMB_SEGMENTS:
            baseline = baseline_lengths.get((proximal, distal))
            if baseline is None or baseline <= 0:
                continue
            if proximal not in points or distal not in points:
                continue
            length = float(np.linalg.norm(points[distal] - points[proximal]))
            if abs(length - baseline) / baseline > tolerance_ratio:
                rejected.add((frame_index, reject_joint))

    if not rejected:
        return records

    return [
        replace(
            record,
            x_3d=nan,
            y_3d=nan,
            z_3d=nan,
            input_quality_score=None,
        )
        if (record.frame_index, record.joint_name) in rejected
        else record
        for record in records
    ]


def _records_by_key(records: list[Pose3DRecord]) -> dict[tuple[str, str, str], list[Pose3DRecord]]:
    grouped: dict[tuple[str, str, str], list[Pose3DRecord]] = defaultdict(list)
    for record in records:
        if record.joint_name not in CANONICAL_JOINTS:
            continue
        grouped[(record.clip_id, record.condition_id, record.joint_name)].append(record)
    for key_records in grouped.values():
        key_records.sort(key=lambda item: item.frame_index)
    return grouped


def _smooth_joint_records(
    records: list[Pose3DRecord],
    *,
    window_length: int,
    polyorder: int,
    median_window_length: int,
    refine_window_length: int,
    confidence_threshold: float,
    threshold_config: dict[str, object] | None,
    max_gap_frames: int,
    max_gap_config: dict[str, object] | None,
    jump_threshold_multiplier: float,
    joint_jump_config: dict[str, object] | None,
) -> list[Pose3DRecord]:
    if not records:
        return []

    joint_name = records[0].joint_name
    x_values = np.array([_confident_value(record, "x_3d", confidence_threshold, threshold_config) for record in records], dtype=float)
    y_values = np.array([_confident_value(record, "y_3d", confidence_threshold, threshold_config) for record in records], dtype=float)
    z_values = np.array([_confident_value(record, "z_3d", confidence_threshold, threshold_config) for record in records], dtype=float)

    joint_jump_threshold = multiplier_for_joint(
        joint_name,
        jump_threshold_multiplier,
        joint_jump_config if isinstance(joint_jump_config, dict) else None,
    )
    x_values, y_values, z_values = _remove_jump_outliers(x_values, y_values, z_values, joint_jump_threshold)
    joint_gap = gap_for_joint(joint_name, max_gap_frames, max_gap_config if isinstance(max_gap_config, dict) else None)
    x_values = _interpolate_short_gaps(x_values, joint_gap)
    y_values = _interpolate_short_gaps(y_values, joint_gap)
    z_values = _interpolate_short_gaps(z_values, joint_gap)
    x_values = _median_valid_segments(x_values, median_window_length)
    y_values = _median_valid_segments(y_values, median_window_length)
    z_values = _median_valid_segments(z_values, median_window_length)
    x_values = _savgol_valid_segments(x_values, window_length, polyorder)
    y_values = _savgol_valid_segments(y_values, window_length, polyorder)
    z_values = _savgol_valid_segments(z_values, window_length, polyorder)
    x_values = _moving_average_valid_segments(x_values, refine_window_length)
    y_values = _moving_average_valid_segments(y_values, refine_window_length)
    z_values = _moving_average_valid_segments(z_values, refine_window_length)

    output: list[Pose3DRecord] = []
    for record, x_value, y_value, z_value in zip(records, x_values, y_values, z_values):
        if np.isnan(x_value) or np.isnan(y_value) or np.isnan(z_value):
            output.append(
                replace(
                    record,
                    x_3d=nan,
                    y_3d=nan,
                    z_3d=nan,
                    input_quality_score=None,
                )
            )
            continue
        output.append(
            replace(
                record,
                x_3d=float(x_value),
                y_3d=float(y_value),
                z_3d=float(z_value),
            )
        )
    return output


def _confident_points(
    records: list[Pose3DRecord],
    *,
    confidence_threshold: float,
    threshold_config: dict[str, object] | None,
) -> dict[str, np.ndarray]:
    points: dict[str, np.ndarray] = {}
    for record in records:
        score = record.input_quality_score
        joint_threshold = threshold_for_joint(record.joint_name, confidence_threshold, threshold_config)
        if (
            record.x_3d is None
            or record.y_3d is None
            or record.z_3d is None
            or any(np.isnan(value) for value in (record.x_3d, record.y_3d, record.z_3d))
            or (score is not None and score < joint_threshold)
        ):
            continue
        points[record.joint_name] = np.array((record.x_3d, record.y_3d, record.z_3d), dtype=float)
    return points


def _confident_value(
    record: Pose3DRecord,
    axis: str,
    confidence_threshold: float,
    threshold_config: dict[str, object] | None,
) -> float:
    score = record.input_quality_score
    joint_threshold = threshold_for_joint(record.joint_name, confidence_threshold, threshold_config)
    value = getattr(record, axis)
    if value is None or (score is not None and score < joint_threshold):
        return np.nan
    return float(value)


def _remove_jump_outliers(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    jump_threshold_multiplier: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps: list[float] = []
    previous = None
    for values in zip(x_values, y_values, z_values):
        if any(np.isnan(v) for v in values):
            continue
        current = np.array(values, dtype=float)
        if previous is not None:
            steps.append(float(np.linalg.norm(current - previous)))
        previous = current
    if not steps:
        return x_values, y_values, z_values
    median_step = statistics.median([s for s in steps if s > 0] or [0.0])
    if median_step <= 0:
        return x_values, y_values, z_values
    threshold = median_step * jump_threshold_multiplier
    cleaned_x = x_values.copy()
    cleaned_y = y_values.copy()
    cleaned_z = z_values.copy()
    previous = None
    for idx, values in enumerate(zip(cleaned_x, cleaned_y, cleaned_z)):
        if any(np.isnan(v) for v in values):
            continue
        current = np.array(values, dtype=float)
        if previous is not None and float(np.linalg.norm(current - previous)) > threshold:
            cleaned_x[idx] = np.nan
            cleaned_y[idx] = np.nan
            cleaned_z[idx] = np.nan
            continue
        previous = current
    return cleaned_x, cleaned_y, cleaned_z


def _interpolate_short_gaps(values: np.ndarray, max_gap_frames: int) -> np.ndarray:
    if max_gap_frames <= 0:
        return values
    output = values.copy()
    valid_indices = np.flatnonzero(~np.isnan(values))
    if len(valid_indices) < 2:
        return output
    for left, right in zip(valid_indices, valid_indices[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > max_gap_frames:
            continue
        output[left : right + 1] = np.interp(
            np.arange(left, right + 1),
            [left, right],
            [values[left], values[right]],
        )
    return output


def _median_valid_segments(values: np.ndarray, window_length: int) -> np.ndarray:
    if window_length <= 1:
        return values
    if window_length % 2 == 0:
        window_length += 1
    output = values.copy()
    for start, end in _valid_segments(values):
        segment = values[start:end]
        if len(segment) < window_length:
            continue
        radius = window_length // 2
        padded = np.pad(segment, (radius, radius), mode="edge")
        filtered = np.array(
            [np.median(padded[idx : idx + window_length]) for idx in range(len(segment))],
            dtype=float,
        )
        output[start:end] = filtered
    return output


def _savgol_valid_segments(values: np.ndarray, window_length: int, polyorder: int) -> np.ndarray:
    output = values.copy()
    for start, end in _valid_segments(values):
        segment = values[start:end]
        if len(segment) <= polyorder:
            continue
        segment_window = min(window_length, len(segment) if len(segment) % 2 == 1 else len(segment) - 1)
        if segment_window <= polyorder or segment_window < 3:
            continue
        output[start:end] = savgol_filter(segment, segment_window, polyorder, mode="interp")
    return output


def _moving_average_valid_segments(values: np.ndarray, window_length: int) -> np.ndarray:
    if window_length <= 1:
        return values
    if window_length % 2 == 0:
        window_length += 1
    output = values.copy()
    for start, end in _valid_segments(values):
        segment = values[start:end]
        if len(segment) < window_length:
            continue
        kernel = np.ones(window_length, dtype=float) / window_length
        radius = window_length // 2
        padded = np.pad(segment, (radius, radius), mode="edge")
        output[start:end] = np.convolve(padded, kernel, mode="valid")
    return output


def _valid_segments(values: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(values):
        if np.isnan(value):
            if start is not None:
                segments.append((start, index))
                start = None
            continue
        if start is None:
            start = index
    if start is not None:
        segments.append((start, len(values)))
    return segments
