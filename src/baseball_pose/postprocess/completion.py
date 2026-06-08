"""Bone-constrained 2D pose completion for short occlusion gaps."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
import statistics

from baseball_pose.pose.quality import threshold_for_joint
from baseball_pose.pose.schema import PoseRecord, pose_score


LIMB_CHAINS = (
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
)


def complete_pose_records(
    records: list[PoseRecord],
    confidence_threshold: float = 0.5,
    threshold_config: dict[str, object] | None = None,
    max_gap_frames: int = 5,
    max_gap_config: dict[str, object] | None = None,
    imputed_confidence: float = 0.62,
) -> list[PoseRecord]:
    """Fill short missing limb gaps without changing the pose schema.

    The method is deliberately conservative: it only completes elbows, wrists,
    knees, and ankles when neighboring frames and adjacent joints provide enough
    evidence. Imputed records are marked by appending ``+imputed`` to ``backend``.
    """

    if max_gap_frames < 0:
        raise ValueError("max_gap_frames must be non-negative.")
    if imputed_confidence <= 0:
        raise ValueError("imputed_confidence must be positive.")
    if not records:
        return []

    by_key = _records_by_key(records)
    frame_records = _records_by_frame(records)
    segment_lengths = _median_segment_lengths(
        frame_records,
        confidence_threshold=confidence_threshold,
        threshold_config=threshold_config,
    )
    completed: dict[tuple[int, str], PoseRecord] = {}

    for proximal, middle, distal in LIMB_CHAINS:
        _complete_distal_joint(
            by_key=by_key,
            frame_records=frame_records,
            output=completed,
            proximal=middle,
            distal=distal,
            segment_length=segment_lengths.get((middle, distal)),
            confidence_threshold=confidence_threshold,
            threshold_config=threshold_config,
            max_gap_frames=max_gap_frames,
            max_gap_config=max_gap_config,
            imputed_confidence=imputed_confidence,
        )
        _complete_middle_joint(
            by_key=by_key,
            frame_records=frame_records,
            output=completed,
            proximal=proximal,
            middle=middle,
            distal=distal,
            proximal_length=segment_lengths.get((proximal, middle)),
            distal_length=segment_lengths.get((middle, distal)),
            confidence_threshold=confidence_threshold,
            threshold_config=threshold_config,
            max_gap_frames=max_gap_frames,
            max_gap_config=max_gap_config,
            imputed_confidence=imputed_confidence,
        )

    return [
        completed.get((record.frame_index, record.joint_name), record)
        for record in records
    ]


def _records_by_key(records: list[PoseRecord]) -> dict[str, list[PoseRecord]]:
    by_key: dict[str, list[PoseRecord]] = defaultdict(list)
    for record in records:
        by_key[record.joint_name].append(record)
    for key_records in by_key.values():
        key_records.sort(key=lambda item: item.frame_index)
    return by_key


def _records_by_frame(records: list[PoseRecord]) -> dict[int, dict[str, PoseRecord]]:
    by_frame: dict[int, dict[str, PoseRecord]] = defaultdict(dict)
    for record in records:
        by_frame[record.frame_index][record.joint_name] = record
    return by_frame


def _median_segment_lengths(
    frame_records: dict[int, dict[str, PoseRecord]],
    confidence_threshold: float,
    threshold_config: dict[str, object] | None,
) -> dict[tuple[str, str], float]:
    lengths: dict[tuple[str, str], list[float]] = defaultdict(list)
    for records in frame_records.values():
        for proximal, middle, distal in LIMB_CHAINS:
            for first_name, second_name in ((proximal, middle), (middle, distal)):
                first = records.get(first_name)
                second = records.get(second_name)
                if not _usable(first, confidence_threshold, threshold_config):
                    continue
                if not _usable(second, confidence_threshold, threshold_config):
                    continue
                lengths[(first_name, second_name)].append(
                    _distance((first.x, first.y), (second.x, second.y))
                )
    return {
        segment: statistics.median(values)
        for segment, values in lengths.items()
        if values
    }


def _complete_distal_joint(
    by_key: dict[str, list[PoseRecord]],
    frame_records: dict[int, dict[str, PoseRecord]],
    output: dict[tuple[int, str], PoseRecord],
    proximal: str,
    distal: str,
    segment_length: float | None,
    confidence_threshold: float,
    threshold_config: dict[str, object] | None,
    max_gap_frames: int,
    max_gap_config: dict[str, object] | None,
    imputed_confidence: float,
) -> None:
    if segment_length is None or segment_length <= 0:
        return
    joint_records = by_key.get(distal, [])
    for start, end in _missing_runs(joint_records, confidence_threshold, threshold_config):
        gap_length = end - start
        joint_gap = _gap_for_joint(distal, max_gap_frames, max_gap_config)
        if gap_length <= 0 or gap_length > joint_gap:
            continue
        before = _record_at(joint_records, start - 1)
        after = _record_at(joint_records, end)
        if not _usable(before, confidence_threshold, threshold_config):
            continue
        if not _usable(after, confidence_threshold, threshold_config):
            continue
        for offset, frame_index in enumerate(range(start, end), start=1):
            base = _record_at(joint_records, frame_index)
            proximal_record = frame_records.get(frame_index, {}).get(proximal)
            if base is None or not _usable(proximal_record, confidence_threshold, threshold_config):
                continue
            alpha = offset / (gap_length + 1)
            target = _lerp((before.x, before.y), (after.x, after.y), alpha)
            vector = (target[0] - proximal_record.x, target[1] - proximal_record.y)
            if _norm(vector) <= 1e-6:
                vector = _nearest_segment_direction(
                    frame_records,
                    frame_index=frame_index,
                    proximal=proximal,
                    distal=distal,
                    confidence_threshold=confidence_threshold,
                    threshold_config=threshold_config,
                )
            point = _project_from(proximal_record, vector, segment_length)
            output[(frame_index, distal)] = _imputed_record(base, point, imputed_confidence)


def _complete_middle_joint(
    by_key: dict[str, list[PoseRecord]],
    frame_records: dict[int, dict[str, PoseRecord]],
    output: dict[tuple[int, str], PoseRecord],
    proximal: str,
    middle: str,
    distal: str,
    proximal_length: float | None,
    distal_length: float | None,
    confidence_threshold: float,
    threshold_config: dict[str, object] | None,
    max_gap_frames: int,
    max_gap_config: dict[str, object] | None,
    imputed_confidence: float,
) -> None:
    if proximal_length is None or distal_length is None:
        return
    if proximal_length <= 0 or distal_length <= 0:
        return
    joint_records = by_key.get(middle, [])
    for start, end in _missing_runs(joint_records, confidence_threshold, threshold_config):
        gap_length = end - start
        joint_gap = _gap_for_joint(middle, max_gap_frames, max_gap_config)
        if gap_length <= 0 or gap_length > joint_gap:
            continue
        before = _record_at(joint_records, start - 1)
        after = _record_at(joint_records, end)
        if not _usable(before, confidence_threshold, threshold_config):
            continue
        if not _usable(after, confidence_threshold, threshold_config):
            continue
        for offset, frame_index in enumerate(range(start, end), start=1):
            base = _record_at(joint_records, frame_index)
            proximal_record = frame_records.get(frame_index, {}).get(proximal)
            distal_record = frame_records.get(frame_index, {}).get(distal)
            if base is None:
                continue
            if not _usable(proximal_record, confidence_threshold, threshold_config):
                continue
            if not _usable(distal_record, confidence_threshold, threshold_config):
                continue
            alpha = offset / (gap_length + 1)
            expected = _lerp((before.x, before.y), (after.x, after.y), alpha)
            candidates = _circle_intersections(
                (proximal_record.x, proximal_record.y),
                proximal_length,
                (distal_record.x, distal_record.y),
                distal_length,
            )
            if candidates:
                point = min(candidates, key=lambda item: _distance(item, expected))
            else:
                point = _lerp((proximal_record.x, proximal_record.y), (distal_record.x, distal_record.y), 0.5)
            output[(frame_index, middle)] = _imputed_record(base, point, imputed_confidence)


def _missing_runs(
    records: list[PoseRecord],
    confidence_threshold: float,
    threshold_config: dict[str, object] | None,
) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(records):
        if _usable(records[index], confidence_threshold, threshold_config):
            index += 1
            continue
        start = records[index].frame_index
        while index < len(records) and not _usable(records[index], confidence_threshold, threshold_config):
            index += 1
        end = records[index].frame_index if index < len(records) else records[-1].frame_index + 1
        runs.append((start, end))
    return runs


def _record_at(records: list[PoseRecord], frame_index: int) -> PoseRecord | None:
    for record in records:
        if record.frame_index == frame_index:
            return record
    return None


def _usable(
    record: PoseRecord | None,
    confidence_threshold: float,
    threshold_config: dict[str, object] | None,
) -> bool:
    if record is None or record.x is None or record.y is None:
        return False
    score = pose_score(record)
    joint_threshold = threshold_for_joint(record.joint_name, confidence_threshold, threshold_config)
    return score is None or score >= joint_threshold


def _gap_for_joint(
    joint_name: str,
    default_gap: int,
    config: dict[str, object] | None,
) -> int:
    if not config:
        return default_gap
    overrides = config.get("joint_overrides", {})
    if isinstance(overrides, dict) and joint_name in overrides:
        return int(overrides[joint_name])
    if joint_name.endswith("_wrist") or joint_name.endswith("_ankle"):
        return int(config.get("distal", default_gap))
    if joint_name.endswith("_elbow") or joint_name.endswith("_knee"):
        return int(config.get("mid_limb", default_gap))
    return int(config.get("default", default_gap))


def _imputed_record(record: PoseRecord, point: tuple[float, float], confidence: float) -> PoseRecord:
    x_value = min(max(float(point[0]), 0.0), 1.0)
    y_value = min(max(float(point[1]), 0.0), 1.0)
    backend = record.backend if record.backend.endswith("+imputed") else f"{record.backend}+imputed"
    return replace(
        record,
        x=x_value,
        y=y_value,
        visibility=min(float(confidence), 1.0),
        confidence=min(float(confidence), 1.0),
        backend=backend,
    )


def _nearest_segment_direction(
    frame_records: dict[int, dict[str, PoseRecord]],
    frame_index: int,
    proximal: str,
    distal: str,
    confidence_threshold: float,
    threshold_config: dict[str, object] | None,
) -> tuple[float, float]:
    for delta in range(1, 6):
        for candidate_frame in (frame_index - delta, frame_index + delta):
            records = frame_records.get(candidate_frame, {})
            proximal_record = records.get(proximal)
            distal_record = records.get(distal)
            if not _usable(proximal_record, confidence_threshold, threshold_config):
                continue
            if not _usable(distal_record, confidence_threshold, threshold_config):
                continue
            return (distal_record.x - proximal_record.x, distal_record.y - proximal_record.y)
    return (1.0, 0.0)


def _circle_intersections(
    first_center: tuple[float, float],
    first_radius: float,
    second_center: tuple[float, float],
    second_radius: float,
) -> list[tuple[float, float]]:
    dx = second_center[0] - first_center[0]
    dy = second_center[1] - first_center[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return []
    if distance > first_radius + second_radius:
        return []
    if distance < abs(first_radius - second_radius):
        return []
    a_value = (first_radius**2 - second_radius**2 + distance**2) / (2 * distance)
    height_sq = first_radius**2 - a_value**2
    if height_sq < 0:
        return []
    height = math.sqrt(height_sq)
    mid_x = first_center[0] + a_value * dx / distance
    mid_y = first_center[1] + a_value * dy / distance
    rx = -dy * height / distance
    ry = dx * height / distance
    return [(mid_x + rx, mid_y + ry), (mid_x - rx, mid_y - ry)]


def _project_from(
    proximal: PoseRecord,
    vector: tuple[float, float],
    segment_length: float,
) -> tuple[float, float]:
    length = _norm(vector)
    if length <= 1e-9:
        return (proximal.x, proximal.y)
    scale = segment_length / length
    return (proximal.x + vector[0] * scale, proximal.y + vector[1] * scale)


def _lerp(first: tuple[float, float], second: tuple[float, float], alpha: float) -> tuple[float, float]:
    return (first[0] + (second[0] - first[0]) * alpha, first[1] + (second[1] - first[1]) * alpha)


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _norm(vector: tuple[float, float]) -> float:
    return math.hypot(vector[0], vector[1])
