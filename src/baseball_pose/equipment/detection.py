"""Heuristic bat and ball detection from sampled frames."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from baseball_pose.equipment.schema import ObjectTrackRecord
from baseball_pose.io.frame_csv import read_frame_records
from baseball_pose.io.pose_csv import read_pose_records
from baseball_pose.io.video import read_frame
from baseball_pose.pose.quality import threshold_for_joint
from baseball_pose.pose.schema import PoseRecord, pose_score


@dataclass(frozen=True)
class EquipmentTrackingConfig:
    bat_min_line_length_ratio: float = 0.08
    bat_max_line_length_ratio: float = 0.32
    bat_max_line_gap_ratio: float = 0.02
    bat_wrist_prior_weight: float = 0.45
    bat_previous_weight: float = 0.20
    bat_max_handle_distance_ratio: float = 0.10
    bat_max_wrist_spread_ratio: float = 0.18
    bat_max_frame_jump_ratio: float = 0.16
    ball_min_radius_px: int = 3
    ball_max_radius_px: int = 14
    ball_min_brightness: int = 160
    ball_min_saturation: int = 35
    ball_motion_threshold: int = 22
    ball_previous_weight: float = 0.45
    ball_max_frame_jump_ratio: float = 0.16
    pose_confidence_threshold: float = 0.12
    pose_thresholds: dict[str, object] | None = None
    detect_bat: bool = True
    detect_ball: bool = True
    use_wrist_bat_proxy: bool = True


@dataclass(frozen=True)
class _BatCandidate:
    barrel: tuple[float, float]
    handle: tuple[float, float]
    confidence: float


@dataclass(frozen=True)
class _BallCandidate:
    center: tuple[float, float]
    radius_px: float
    confidence: float


def detect_equipment_tracks(
    frames_csv: str | Path,
    clip_id: str,
    condition_id: str,
    pose_csv: str | Path | None = None,
    config: EquipmentTrackingConfig | None = None,
) -> list[ObjectTrackRecord]:
    """Track bat barrel/handle and ball center for one clip condition."""

    cfg = config or EquipmentTrackingConfig()
    frames = read_frame_records(frames_csv)
    pose_by_frame = _pose_by_frame(pose_csv, cfg) if pose_csv is not None and Path(pose_csv).exists() else {}
    records: list[ObjectTrackRecord] = []
    previous_bat: _BatCandidate | None = None
    previous_ball: _BallCandidate | None = None
    previous_gray = None

    for frame in frames:
        image = read_frame(frame.frame_path)
        cv2 = _require_cv2()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        wrist_points = _wrist_points(pose_by_frame.get(frame.frame_index, []), width, height, cfg)

        bat = _detect_bat(image, wrist_points, previous_bat, cfg) if cfg.detect_bat else None
        bat_source = "hough_line_wrist_prior"
        if bat is None and cfg.detect_bat and cfg.use_wrist_bat_proxy:
            bat = _bat_wrist_proxy(wrist_points)
            bat_source = "wrist_proxy"
        if bat is not None:
            previous_bat = bat
            records.append(
                ObjectTrackRecord(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    object_name="bat",
                    x=bat.barrel[0] / width,
                    y=bat.barrel[1] / height,
                    x2=bat.handle[0] / width,
                    y2=bat.handle[1] / height,
                    confidence=bat.confidence,
                    width=width,
                    height=height,
                    source=bat_source,
                )
            )

        ball = _detect_ball(image, previous_gray, previous_ball, cfg) if cfg.detect_ball else None
        if ball is not None:
            previous_ball = ball
            records.append(
                ObjectTrackRecord(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    object_name="ball",
                    x=ball.center[0] / width,
                    y=ball.center[1] / height,
                    x2=None,
                    y2=None,
                    confidence=ball.confidence,
                    width=width,
                    height=height,
                    source="bright_small_blob_temporal",
                )
            )
        previous_gray = gray

    return records


def _detect_bat(
    image,
    wrist_points: list[tuple[float, float]],
    previous: _BatCandidate | None,
    config: EquipmentTrackingConfig,
) -> _BatCandidate | None:
    cv2 = _require_cv2()
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    min_line_length = max(20, int(config.bat_min_line_length_ratio * max(width, height)))
    max_line_gap = max(4, int(config.bat_max_line_gap_ratio * max(width, height)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180,
        threshold=40,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return None

    candidates: list[_BatCandidate] = []
    diagonal = math.hypot(width, height)
    wrist_center = _mean_point(wrist_points)
    if wrist_center is None:
        return None
    if len(wrist_points) >= 2 and _distance(wrist_points[0], wrist_points[1]) > config.bat_max_wrist_spread_ratio * diagonal:
        return None
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(value) for value in line]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_line_length:
            continue
        if length > config.bat_max_line_length_ratio * diagonal:
            continue
        first = (x1, y1)
        second = (x2, y2)
        handle, barrel = _orient_bat(first, second, wrist_center)
        if barrel[1] > handle[1] + 0.18 * height:
            continue
        handle_distance = _distance(handle, wrist_center)
        if handle_distance > config.bat_max_handle_distance_ratio * diagonal:
            continue
        length_score = min(1.0, length / (0.35 * diagonal))
        wrist_distance = _distance_to_segment(wrist_center, first, second)
        wrist_score = 0.65 * max(0.0, 1.0 - handle_distance / (config.bat_max_handle_distance_ratio * diagonal))
        wrist_score += 0.35 * max(0.0, 1.0 - wrist_distance / (0.08 * diagonal))
        previous_score = 0.5
        if previous is not None:
            jump = _distance(barrel, previous.barrel)
            if jump > config.bat_max_frame_jump_ratio * diagonal:
                continue
            previous_score = max(0.0, 1.0 - jump / (config.bat_max_frame_jump_ratio * diagonal))
        confidence = (
            (1.0 - config.bat_wrist_prior_weight - config.bat_previous_weight) * length_score
            + config.bat_wrist_prior_weight * wrist_score
            + config.bat_previous_weight * previous_score
        )
        candidates.append(_BatCandidate(barrel=barrel, handle=handle, confidence=confidence))

    if not candidates:
        return None
    best = max(candidates, key=lambda item: item.confidence)
    return best if best.confidence >= 0.50 else None


def _detect_ball(
    image,
    previous_gray,
    previous: _BallCandidate | None,
    config: EquipmentTrackingConfig,
) -> _BallCandidate | None:
    cv2 = _require_cv2()
    if previous_gray is None:
        return None
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    motion = cv2.absdiff(gray, previous_gray)
    _, motion_mask = cv2.threshold(motion, config.ball_motion_threshold, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, None, iterations=1)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = (0, 0, config.ball_min_brightness)
    upper = (180, config.ball_min_saturation, 255)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_and(mask, motion_mask)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diagonal = math.hypot(width, height)
    candidates: list[_BallCandidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= 0:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < config.ball_min_radius_px or radius > config.ball_max_radius_px:
            continue
        circle_area = math.pi * radius * radius
        circularity = min(1.0, area / circle_area) if circle_area > 0 else 0.0
        if circularity < 0.35:
            continue
        size_score = 1.0 - abs(radius - (config.ball_min_radius_px + config.ball_max_radius_px) / 2) / max(
            1.0,
            (config.ball_max_radius_px - config.ball_min_radius_px) / 2,
        )
        size_score = max(0.0, size_score)
        previous_score = 0.5
        if previous is not None:
            jump = _distance((x, y), previous.center)
            if jump > config.ball_max_frame_jump_ratio * diagonal:
                continue
            previous_score = max(0.0, 1.0 - jump / (config.ball_max_frame_jump_ratio * diagonal))
        confidence = (
            (1.0 - config.ball_previous_weight) * (0.65 * circularity + 0.35 * size_score)
            + config.ball_previous_weight * previous_score
        )
        candidates.append(_BallCandidate(center=(x, y), radius_px=radius, confidence=confidence))
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item.confidence)
    return best if best.confidence >= 0.40 else None


def _bat_wrist_proxy(wrist_points: list[tuple[float, float]]) -> _BatCandidate | None:
    if len(wrist_points) < 2:
        return None
    center = _mean_point(wrist_points)
    if center is None:
        return None
    return _BatCandidate(barrel=center, handle=center, confidence=0.50)


def _pose_by_frame(
    pose_csv: str | Path,
    config: EquipmentTrackingConfig,
) -> dict[int, list[PoseRecord]]:
    by_frame: dict[int, list[PoseRecord]] = {}
    for record in read_pose_records(pose_csv):
        if record.joint_name not in {"left_wrist", "right_wrist"}:
            continue
        score = pose_score(record)
        threshold = threshold_for_joint(record.joint_name, config.pose_confidence_threshold, config.pose_thresholds)
        if record.x is None or record.y is None or (score is not None and score < threshold):
            continue
        by_frame.setdefault(record.frame_index, []).append(record)
    return by_frame


def _wrist_points(
    records: list[PoseRecord],
    width: int,
    height: int,
    config: EquipmentTrackingConfig,
) -> list[tuple[float, float]]:
    points = []
    for record in records:
        if record.x is None or record.y is None:
            continue
        points.append((record.x * width, record.y * height))
    return points


def _orient_bat(
    first: tuple[float, float],
    second: tuple[float, float],
    wrist_center: tuple[float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if wrist_center is None:
        return first, second
    if _distance(first, wrist_center) <= _distance(second, wrist_center):
        return first, second
    return second, first


def _mean_point(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _distance_to_segment(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return _distance(point, first)
    t = max(0.0, min(1.0, ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / length_sq))
    projection = (first[0] + t * dx, first[1] + t * dy)
    return _distance(point, projection)


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for equipment detection. Install project dependencies first."
        ) from exc

    return cv2
