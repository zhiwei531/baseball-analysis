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
    detector_backend: str = "motion"
    yolo_model_path: str | None = None
    yolo_confidence: float = 0.25
    yolo_image_size: int | None = None
    yolo_device: str | None = None
    yolo_bat_class_ids: tuple[int, ...] = (34,)
    yolo_ball_class_ids: tuple[int, ...] = (32,)
    yolo_bat_weight: float = 0.65
    yolo_ball_weight: float = 0.70
    yolo_bat_min_confidence: float = 0.18
    yolo_ball_min_confidence: float = 0.18
    motion_backend: str = "background"
    motion_background_samples: int = 60
    motion_background_threshold: int = 28
    motion_frame_threshold: int = 18
    motion_min_area_px: int = 8
    motion_dilate_iterations: int = 1
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
    ball_previous_weight: float = 0.45
    ball_max_frame_jump_ratio: float = 0.16
    ball_start_anchor_distance_ratio: float = 0.18
    ball_max_below_anchor_ratio: float = 0.20
    ball_max_y_ratio: float = 0.66
    ball_max_area_px: int = 650
    ball_max_aspect_ratio: float = 5.0
    pose_confidence_threshold: float = 0.12
    pose_thresholds: dict[str, object] | None = None
    detect_bat: bool = True
    detect_ball: bool = True
    detect_ball_for_batting: bool = False
    detect_ball_for_pitching: bool = True
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


@dataclass(frozen=True)
class _YoloDetection:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


class _YoloObjectDetector:
    def __init__(self, config: EquipmentTrackingConfig) -> None:
        if config.yolo_model_path is None:
            raise ValueError("equipment_tracking.yolo_model_path is required for YOLO object tracking.")
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ultralytics is required for YOLO equipment detection. "
                "Install ultralytics or use equipment_tracking.detector_backend: motion."
            ) from exc

        self._model = YOLO(config.yolo_model_path)
        self._config = config

    def detect(self, image) -> list[_YoloDetection]:
        kwargs: dict[str, object] = {
            "conf": self._config.yolo_confidence,
            "verbose": False,
        }
        if self._config.yolo_image_size is not None:
            kwargs["imgsz"] = self._config.yolo_image_size
        if self._config.yolo_device is not None:
            kwargs["device"] = self._config.yolo_device
        result = self._model.predict(image, **kwargs)[0]
        detections: list[_YoloDetection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        for cls, conf, xyxy in zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.xyxy.cpu().tolist()):
            detections.append(
                _YoloDetection(
                    class_id=int(cls),
                    confidence=float(conf),
                    xyxy=tuple(float(value) for value in xyxy),
                )
            )
        return detections


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
    background_gray = _build_background_gray(frames, cfg)
    pose_by_frame = _pose_by_frame(pose_csv, cfg) if pose_csv is not None and Path(pose_csv).exists() else {}
    yolo_detector = _create_yolo_detector(cfg)
    records: list[ObjectTrackRecord] = []
    previous_bat: _BatCandidate | None = None
    previous_ball: _BallCandidate | None = None
    previous_gray = None

    for frame in frames:
        image = read_frame(frame.frame_path)
        cv2 = _require_cv2()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        motion_mask = _motion_mask(gray, previous_gray, background_gray, cfg)
        height, width = image.shape[:2]
        wrist_points = _wrist_points(pose_by_frame.get(frame.frame_index, []), width, height, cfg)
        yolo_detections = yolo_detector.detect(image) if yolo_detector is not None else []

        bat = None
        bat_source = "motion_line_wrist_prior"
        if cfg.detect_bat and _uses_yolo(cfg):
            bat = _detect_bat_yolo(yolo_detections, wrist_points, previous_bat, width, height, cfg)
            bat_source = "yolo_baseball_bat"
        if bat is None and cfg.detect_bat and _uses_motion(cfg):
            bat = _detect_bat(image, motion_mask, wrist_points, previous_bat, cfg)
            bat_source = "motion_line_wrist_prior"
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

        ball_anchors = list(wrist_points)
        if bat is not None:
            ball_anchors.extend([bat.barrel, bat.handle])
        ball = None
        ball_source = "motion_blob_temporal"
        if cfg.detect_ball and _uses_yolo(cfg):
            ball = _detect_ball_yolo(yolo_detections, previous_ball, ball_anchors, width, height, cfg)
            ball_source = "yolo_sports_ball"
        if ball is None and cfg.detect_ball and _uses_motion(cfg):
            ball = _detect_ball(image, motion_mask, previous_ball, ball_anchors, cfg)
            ball_source = "motion_blob_temporal"
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
                    source=ball_source,
                )
            )
        previous_gray = gray

    return records


def _create_yolo_detector(config: EquipmentTrackingConfig) -> _YoloObjectDetector | None:
    if not _uses_yolo(config):
        return None
    return _YoloObjectDetector(config)


def _uses_yolo(config: EquipmentTrackingConfig) -> bool:
    return config.detector_backend in {"yolo", "hybrid_yolo_motion"}


def _uses_motion(config: EquipmentTrackingConfig) -> bool:
    return config.detector_backend in {"motion", "hybrid_yolo_motion"}


def _detect_bat_yolo(
    detections: list[_YoloDetection],
    wrist_points: list[tuple[float, float]],
    previous: _BatCandidate | None,
    width: int,
    height: int,
    config: EquipmentTrackingConfig,
) -> _BatCandidate | None:
    wrist_center = _mean_point(wrist_points)
    if wrist_center is None:
        return None
    diagonal = math.hypot(width, height)
    candidates: list[_BatCandidate] = []
    for detection in detections:
        if detection.class_id not in config.yolo_bat_class_ids:
            continue
        if detection.confidence < config.yolo_bat_min_confidence:
            continue
        handle, barrel = _bat_candidate_from_box(detection.xyxy, wrist_center)
        handle_distance = _distance(handle, wrist_center)
        if handle_distance > config.bat_max_handle_distance_ratio * diagonal:
            continue
        previous_score = 0.5
        if previous is not None:
            jump = _distance(barrel, previous.barrel)
            if jump > config.bat_max_frame_jump_ratio * diagonal:
                continue
            previous_score = max(0.0, 1.0 - jump / (config.bat_max_frame_jump_ratio * diagonal))
        wrist_score = max(0.0, 1.0 - handle_distance / (config.bat_max_handle_distance_ratio * diagonal))
        confidence = (
            config.yolo_bat_weight * detection.confidence
            + (1.0 - config.yolo_bat_weight) * (0.65 * wrist_score + 0.35 * previous_score)
        )
        candidates.append(_BatCandidate(barrel=barrel, handle=handle, confidence=confidence))
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item.confidence)
    return best if best.confidence >= 0.35 else None


def _detect_ball_yolo(
    detections: list[_YoloDetection],
    previous: _BallCandidate | None,
    anchors: list[tuple[float, float]],
    width: int,
    height: int,
    config: EquipmentTrackingConfig,
) -> _BallCandidate | None:
    diagonal = math.hypot(width, height)
    candidates: list[_BallCandidate] = []
    for detection in detections:
        if detection.class_id not in config.yolo_ball_class_ids:
            continue
        if detection.confidence < config.yolo_ball_min_confidence:
            continue
        x1, y1, x2, y2 = detection.xyxy
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        radius = max(x2 - x1, y2 - y1) / 2
        if radius < config.ball_min_radius_px or radius > config.ball_max_radius_px * 2.5:
            continue
        if center[1] > config.ball_max_y_ratio * height:
            continue
        if anchors and center[1] > max(anchor[1] for anchor in anchors) + config.ball_max_below_anchor_ratio * height:
            continue
        if previous is None and not _near_any_anchor(
            center,
            anchors,
            config.ball_start_anchor_distance_ratio * diagonal,
        ):
            continue
        previous_score = 0.5
        if previous is not None:
            jump = _distance(center, previous.center)
            if jump > config.ball_max_frame_jump_ratio * diagonal:
                continue
            previous_score = max(0.0, 1.0 - jump / (config.ball_max_frame_jump_ratio * diagonal))
        confidence = config.yolo_ball_weight * detection.confidence + (1.0 - config.yolo_ball_weight) * previous_score
        candidates.append(_BallCandidate(center=center, radius_px=radius, confidence=confidence))
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item.confidence)
    return best if best.confidence >= 0.35 else None


def _bat_candidate_from_box(
    xyxy: tuple[float, float, float, float],
    wrist_center: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    x1, y1, x2, y2 = xyxy
    if (x2 - x1) >= (y2 - y1):
        first = (x1, (y1 + y2) / 2)
        second = (x2, (y1 + y2) / 2)
    else:
        first = ((x1 + x2) / 2, y1)
        second = ((x1 + x2) / 2, y2)
    return _orient_bat(first, second, wrist_center)


def _detect_bat(
    image,
    motion_mask,
    wrist_points: list[tuple[float, float]],
    previous: _BatCandidate | None,
    config: EquipmentTrackingConfig,
) -> _BatCandidate | None:
    cv2 = _require_cv2()
    height, width = image.shape[:2]
    if motion_mask is None:
        return None
    min_line_length = max(20, int(config.bat_min_line_length_ratio * max(width, height)))
    max_line_gap = max(4, int(config.bat_max_line_gap_ratio * max(width, height)))
    moving = cv2.bitwise_and(image, image, mask=motion_mask)
    gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 35, 120)
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
    motion_mask,
    previous: _BallCandidate | None,
    anchors: list[tuple[float, float]],
    config: EquipmentTrackingConfig,
) -> _BallCandidate | None:
    cv2 = _require_cv2()
    if motion_mask is None:
        return None
    height, width = image.shape[:2]
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diagonal = math.hypot(width, height)
    candidates: list[_BallCandidate] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.motion_min_area_px or area > config.ball_max_area_px:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < config.ball_min_radius_px or radius > config.ball_max_radius_px:
            continue
        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > config.ball_max_aspect_ratio:
            continue
        if y > config.ball_max_y_ratio * height:
            continue
        if anchors and y > max(anchor[1] for anchor in anchors) + config.ball_max_below_anchor_ratio * height:
            continue
        if previous is None and not _near_any_anchor(
            (x, y),
            anchors,
            config.ball_start_anchor_distance_ratio * diagonal,
        ):
            continue
        circle_area = math.pi * radius * radius
        circularity = min(1.0, area / circle_area) if circle_area > 0 else 0.0
        if circularity < 0.18:
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
            (1.0 - config.ball_previous_weight) * (0.45 * circularity + 0.35 * size_score + 0.20 * (1.0 / aspect))
            + config.ball_previous_weight * previous_score
        )
        candidates.append(_BallCandidate(center=(x, y), radius_px=radius, confidence=confidence))
    if not candidates:
        return None
    best = max(candidates, key=lambda item: item.confidence)
    return best if best.confidence >= 0.40 else None


def _near_any_anchor(
    point: tuple[float, float],
    anchors: list[tuple[float, float]],
    max_distance: float,
) -> bool:
    if not anchors:
        return False
    return any(_distance(point, anchor) <= max_distance for anchor in anchors)


def _build_background_gray(frame_records, config: EquipmentTrackingConfig):
    if config.motion_backend != "background" or not frame_records:
        return None
    cv2 = _require_cv2()
    sample_count = min(config.motion_background_samples, len(frame_records))
    if sample_count <= 0:
        return None
    if sample_count == len(frame_records):
        sampled = frame_records
    else:
        step = (len(frame_records) - 1) / max(1, sample_count - 1)
        sampled = [frame_records[round(index * step)] for index in range(sample_count)]
    gray_frames = []
    for frame in sampled:
        image = read_frame(frame.frame_path)
        gray_frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    import numpy as np

    return np.median(np.stack(gray_frames, axis=0), axis=0).astype("uint8")


def _motion_mask(
    gray,
    previous_gray,
    background_gray,
    config: EquipmentTrackingConfig,
):
    cv2 = _require_cv2()
    masks = []
    if background_gray is not None:
        bg_delta = cv2.absdiff(gray, background_gray)
        _, bg_mask = cv2.threshold(bg_delta, config.motion_background_threshold, 255, cv2.THRESH_BINARY)
        masks.append(bg_mask)
    if previous_gray is not None:
        frame_delta = cv2.absdiff(gray, previous_gray)
        _, frame_mask = cv2.threshold(frame_delta, config.motion_frame_threshold, 255, cv2.THRESH_BINARY)
        masks.append(frame_mask)
    if not masks:
        return None
    mask = masks[0]
    for extra in masks[1:]:
        mask = cv2.bitwise_and(mask, extra) if background_gray is not None else cv2.bitwise_or(mask, extra)
    if config.motion_dilate_iterations > 0:
        mask = cv2.dilate(mask, None, iterations=config.motion_dilate_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=1)
    return mask


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
