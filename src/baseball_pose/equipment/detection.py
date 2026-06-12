"""Heuristic bat and ball detection from sampled frames."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
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
    yolo_device: str | None = "auto"
    yolo_bat_class_ids: tuple[int, ...] = (34,)
    yolo_ball_class_ids: tuple[int, ...] = (32,)
    yolo_bat_weight: float = 0.65
    yolo_ball_weight: float = 0.70
    yolo_bat_min_confidence: float = 0.18
    yolo_ball_min_confidence: float = 0.18
    yolo_ball_reacquire_confidence: float = 0.55
    yolo_bat_roi_padding_ratio: float = 0.12
    yolo_bat_min_line_length_ratio: float = 0.04
    yolo_bat_max_box_area_ratio: float = 0.22
    yolo_bat_edge_margin_ratio: float = 0.01
    yolo_bat_max_angle_change_deg: float = 75.0
    yolo_bat_allow_box_fallback: bool = False
    interpolate_max_gap_frames: int = 0
    interpolate_confidence_scale: float = 0.75
    use_pose_priors: bool = False
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
    bat_smoothing_window_frames: int = 0
    bat_smoothing_min_segment_frames: int = 3
    bat_smoothing_passes: int = 1
    bat_smoothing_contact_margin_frames: int = 0
    bat_smoothing_contact_original_weight: float = 0.0
    bat_smoothing_fast_angle_threshold_deg: float = 0.0
    bat_smoothing_fast_margin_frames: int = 0
    bat_smoothing_fast_original_weight: float = 0.0
    ball_min_radius_px: int = 3
    ball_max_radius_px: int = 14
    ball_previous_weight: float = 0.45
    ball_max_frame_jump_ratio: float = 0.16
    ball_start_anchor_distance_ratio: float = 0.18
    ball_max_below_anchor_ratio: float = 0.20
    ball_max_y_ratio: float = 0.66
    ball_max_area_px: int = 650
    ball_max_aspect_ratio: float = 5.0
    ball_min_track_length_frames: int = 1
    ball_track_max_gap_frames: int = 2
    pose_confidence_threshold: float = 0.12
    pose_thresholds: dict[str, object] | None = None
    detect_bat: bool = True
    detect_ball: bool = True
    detect_ball_for_batting: bool = False
    detect_ball_for_pitching: bool = True
    use_wrist_bat_proxy: bool = False


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
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
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
        device = _resolve_yolo_device(self._config.yolo_device)
        if device is not None:
            kwargs["device"] = device
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
    pose_by_frame = (
        _pose_by_frame(pose_csv, cfg)
        if cfg.use_pose_priors and pose_csv is not None and Path(pose_csv).exists()
        else {}
    )
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
        bat_source = "motion_line"
        if cfg.detect_bat and _uses_yolo(cfg):
            bat = _detect_bat_yolo(image, yolo_detections, wrist_points, previous_bat, width, height, cfg)
            bat_source = "yolo_bat_roi_line"
        if bat is None and cfg.detect_bat and _uses_motion(cfg):
            bat = _detect_bat(image, motion_mask, wrist_points, previous_bat, cfg)
            bat_source = "motion_line_pose_prior" if wrist_points else "motion_line"
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

    records = _filter_short_object_tracks(
        records,
        object_name="ball",
        min_track_length=_positive_or_default(cfg.ball_min_track_length_frames, 1),
        max_gap_frames=_positive_or_default(cfg.ball_track_max_gap_frames, 2),
    )
    records = _interpolate_object_records(records, frames, cfg)
    return _smooth_bat_records(records, cfg)


def _positive_or_default(value: int, default: int) -> int:
    return value if value > 0 else default


def _create_yolo_detector(config: EquipmentTrackingConfig) -> _YoloObjectDetector | None:
    if not _uses_yolo(config):
        return None
    return _YoloObjectDetector(config)


def _uses_yolo(config: EquipmentTrackingConfig) -> bool:
    return config.detector_backend in {"yolo", "hybrid_yolo_motion"}


def _uses_motion(config: EquipmentTrackingConfig) -> bool:
    return config.detector_backend in {"motion", "hybrid_yolo_motion"}


def _resolve_yolo_device(device: str | None) -> str | None:
    if device in {None, "", "none"}:
        return None
    if device != "auto":
        return device
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _detect_bat_yolo(
    image,
    detections: list[_YoloDetection],
    wrist_points: list[tuple[float, float]],
    previous: _BatCandidate | None,
    width: int,
    height: int,
    config: EquipmentTrackingConfig,
) -> _BatCandidate | None:
    wrist_center = _mean_point(wrist_points)
    diagonal = math.hypot(width, height)
    candidates: list[_BatCandidate] = []
    for detection in detections:
        if detection.class_id not in config.yolo_bat_class_ids:
            continue
        if detection.confidence < config.yolo_bat_min_confidence:
            continue
        if _reject_yolo_bat_box(detection.xyxy, width, height, config):
            continue
        refined = _bat_candidate_from_yolo_roi(image, detection.xyxy, wrist_center, previous, width, height, config)
        if refined is None and config.yolo_bat_allow_box_fallback:
            refined = _bat_candidate_from_box(detection.xyxy, wrist_center, previous)
        if refined is None:
            continue
        handle, barrel = refined
        wrist_score = 0.5
        if wrist_center is not None:
            handle_distance = _distance(handle, wrist_center)
            if handle_distance > config.bat_max_handle_distance_ratio * diagonal:
                continue
            wrist_score = max(0.0, 1.0 - handle_distance / (config.bat_max_handle_distance_ratio * diagonal))
        previous_score = 0.5
        if previous is not None:
            jump = _distance(barrel, previous.barrel)
            if jump > config.bat_max_frame_jump_ratio * diagonal:
                continue
            angle_delta = _angle_delta_deg(_bat_angle(handle, barrel), _bat_angle(previous.handle, previous.barrel))
            if angle_delta > config.yolo_bat_max_angle_change_deg:
                continue
            previous_score = max(0.0, 1.0 - jump / (config.bat_max_frame_jump_ratio * diagonal))
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
        if previous is None and anchors and not _near_any_anchor(
            center,
            anchors,
            config.ball_start_anchor_distance_ratio * diagonal,
        ):
            continue
        previous_score = 0.5
        if previous is not None:
            jump = _distance(center, previous.center)
            if jump > config.ball_max_frame_jump_ratio * diagonal and detection.confidence < config.yolo_ball_reacquire_confidence:
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
    wrist_center: tuple[float, float] | None,
    previous: _BatCandidate | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x1, y1, x2, y2 = xyxy
    if (x2 - x1) >= (y2 - y1):
        first = (x1, (y1 + y2) / 2)
        second = (x2, (y1 + y2) / 2)
    else:
        first = ((x1 + x2) / 2, y1)
        second = ((x1 + x2) / 2, y2)
    return _orient_bat_with_reference(first, second, wrist_center, previous)


def _bat_candidate_from_yolo_roi(
    image,
    xyxy: tuple[float, float, float, float],
    wrist_center: tuple[float, float] | None,
    previous: _BatCandidate | None,
    width: int,
    height: int,
    config: EquipmentTrackingConfig,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if image is None:
        return None
    cv2 = _require_cv2()
    x1, y1, x2, y2 = _expand_box(xyxy, width, height, config.yolo_bat_roi_padding_ratio)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 40, 130)
    diagonal = math.hypot(width, height)
    min_line_length = max(18, int(config.yolo_bat_min_line_length_ratio * diagonal))
    max_line_gap = max(4, int(config.bat_max_line_gap_ratio * diagonal))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=math.pi / 180,
        threshold=24,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return None

    best: tuple[float, tuple[float, float], tuple[float, float]] | None = None
    for line in lines[:, 0, :]:
        lx1, ly1, lx2, ly2 = [float(value) for value in line]
        first = (lx1 + x1, ly1 + y1)
        second = (lx2 + x1, ly2 + y1)
        length = _distance(first, second)
        if length < min_line_length:
            continue
        if length > config.bat_max_line_length_ratio * diagonal:
            continue
        handle, barrel = _orient_bat_with_reference(first, second, wrist_center, previous)
        previous_score = 0.5
        if previous is not None:
            jump = _distance(barrel, previous.barrel)
            if jump > config.bat_max_frame_jump_ratio * diagonal:
                continue
            angle_delta = _angle_delta_deg(_bat_angle(handle, barrel), _bat_angle(previous.handle, previous.barrel))
            if angle_delta > config.yolo_bat_max_angle_change_deg:
                continue
            previous_score = max(0.0, 1.0 - jump / (config.bat_max_frame_jump_ratio * diagonal))
        wrist_score = 0.5
        if wrist_center is not None:
            wrist_score = max(0.0, 1.0 - _distance_to_segment(wrist_center, first, second) / (0.08 * diagonal))
        score = 0.55 * min(1.0, length / (0.25 * diagonal)) + 0.25 * previous_score + 0.20 * wrist_score
        if best is None or score > best[0]:
            best = (score, handle, barrel)
    if best is None:
        return None
    return best[1], best[2]


def _reject_yolo_bat_box(
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    config: EquipmentTrackingConfig,
) -> bool:
    x1, y1, x2, y2 = xyxy
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    if box_width <= 1.0 or box_height <= 1.0:
        return True
    if (box_width * box_height) / max(1.0, width * height) > config.yolo_bat_max_box_area_ratio:
        return True
    margin_x = config.yolo_bat_edge_margin_ratio * width
    margin_y = config.yolo_bat_edge_margin_ratio * height
    touches_left_or_right = x1 <= margin_x or x2 >= width - margin_x
    touches_top_or_bottom = y1 <= margin_y or y2 >= height - margin_y
    return touches_left_or_right and touches_top_or_bottom


def _expand_box(
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    pad = padding_ratio * max(x2 - x1, y2 - y1)
    return (
        max(0, int(math.floor(x1 - pad))),
        max(0, int(math.floor(y1 - pad))),
        min(width, int(math.ceil(x2 + pad))),
        min(height, int(math.ceil(y2 + pad))),
    )


def _interpolate_object_records(
    records: list[ObjectTrackRecord],
    frame_records,
    config: EquipmentTrackingConfig,
) -> list[ObjectTrackRecord]:
    if config.interpolate_max_gap_frames <= 0 or not records:
        return records
    frame_by_index = {frame.frame_index: frame for frame in frame_records}
    records_by_object: dict[str, list[ObjectTrackRecord]] = {}
    for record in records:
        records_by_object.setdefault(record.object_name, []).append(record)

    interpolated = list(records)
    for object_name, object_records in records_by_object.items():
        ordered = sorted(object_records, key=lambda item: item.frame_index)
        for previous, current in zip(ordered, ordered[1:]):
            gap = current.frame_index - previous.frame_index - 1
            if gap <= 0 or gap > config.interpolate_max_gap_frames:
                continue
            if not _can_interpolate(previous, current):
                continue
            for step in range(1, gap + 1):
                frame_index = previous.frame_index + step
                frame = frame_by_index.get(frame_index)
                if frame is None:
                    continue
                ratio = step / (gap + 1)
                confidence = None
                if previous.confidence is not None and current.confidence is not None:
                    confidence = min(previous.confidence, current.confidence) * config.interpolate_confidence_scale
                interpolated.append(
                    ObjectTrackRecord(
                        clip_id=previous.clip_id,
                        condition_id=previous.condition_id,
                        frame_index=frame_index,
                        timestamp_sec=frame.timestamp_sec,
                        object_name=object_name,
                        x=_lerp_optional(previous.x, current.x, ratio),
                        y=_lerp_optional(previous.y, current.y, ratio),
                        x2=_lerp_optional(previous.x2, current.x2, ratio),
                        y2=_lerp_optional(previous.y2, current.y2, ratio),
                        confidence=confidence,
                        width=previous.width if previous.width is not None else current.width,
                        height=previous.height if previous.height is not None else current.height,
                        source="temporal_interpolation",
                    )
                )
    return sorted(interpolated, key=lambda item: (item.frame_index, item.object_name))


def _smooth_bat_records(
    records: list[ObjectTrackRecord],
    config: EquipmentTrackingConfig,
) -> list[ObjectTrackRecord]:
    window = _odd_window(config.bat_smoothing_window_frames)
    if window <= 1:
        return records
    original_bat_by_frame = {
        record.frame_index: record
        for record in records
        if record.object_name == "bat"
    }
    original_weights = _bat_original_blend_weights(records, original_bat_by_frame, config)
    smoothed = records
    for _ in range(max(1, config.bat_smoothing_passes)):
        smoothed = _smooth_bat_records_once(smoothed, config, window)
    return _blend_protected_bat_records(
        smoothed,
        original_bat_by_frame,
        original_weights,
    )


def _smooth_bat_records_once(
    records: list[ObjectTrackRecord],
    config: EquipmentTrackingConfig,
    window: int,
) -> list[ObjectTrackRecord]:
    bat_records = sorted(
        [record for record in records if record.object_name == "bat"],
        key=lambda item: item.frame_index,
    )
    if not bat_records:
        return records

    smoothed_by_frame: dict[int, ObjectTrackRecord] = {}
    current: list[ObjectTrackRecord] = []
    previous_frame: int | None = None
    for record in bat_records:
        if previous_frame is None or record.frame_index - previous_frame <= 1:
            current.append(record)
        else:
            smoothed_by_frame.update(_smooth_bat_segment(current, window, config.bat_smoothing_min_segment_frames))
            current = [record]
        previous_frame = record.frame_index
    smoothed_by_frame.update(_smooth_bat_segment(current, window, config.bat_smoothing_min_segment_frames))

    return sorted(
        [smoothed_by_frame.get(record.frame_index, record) if record.object_name == "bat" else record for record in records],
        key=lambda item: (item.frame_index, item.object_name),
    )


def _bat_original_blend_weights(
    records: list[ObjectTrackRecord],
    original_bat_by_frame: dict[int, ObjectTrackRecord],
    config: EquipmentTrackingConfig,
) -> dict[int, float]:
    weights: dict[int, float] = {}
    for record in records:
        if record.object_name != "ball":
            continue
        _add_protected_frame_weights(
            weights,
            record.frame_index,
            config.bat_smoothing_contact_margin_frames,
            config.bat_smoothing_contact_original_weight,
        )
    if config.bat_smoothing_fast_angle_threshold_deg > 0 and config.bat_smoothing_fast_original_weight > 0:
        bat_records = sorted(original_bat_by_frame.values(), key=lambda item: item.frame_index)
        previous: ObjectTrackRecord | None = None
        for record in bat_records:
            if previous is not None and record.frame_index - previous.frame_index <= 1:
                angle_delta = abs(_angle_delta_deg(_bat_record_angle(previous), _bat_record_angle(record)))
                if angle_delta >= config.bat_smoothing_fast_angle_threshold_deg:
                    _add_protected_frame_weights(
                        weights,
                        previous.frame_index,
                        config.bat_smoothing_fast_margin_frames,
                        config.bat_smoothing_fast_original_weight,
                    )
                    _add_protected_frame_weights(
                        weights,
                        record.frame_index,
                        config.bat_smoothing_fast_margin_frames,
                        config.bat_smoothing_fast_original_weight,
                    )
            previous = record
    return weights


def _add_protected_frame_weights(
    weights: dict[int, float],
    center_frame: int,
    margin_frames: int,
    original_weight: float,
) -> None:
    if original_weight <= 0:
        return
    margin_frames = max(0, margin_frames)
    original_weight = min(1.0, original_weight)
    for frame_index in range(center_frame - margin_frames, center_frame + margin_frames + 1):
        distance = abs(frame_index - center_frame)
        falloff = 1.0 - distance / (margin_frames + 1)
        weights[frame_index] = max(weights.get(frame_index, 0.0), original_weight * falloff)


def _blend_protected_bat_records(
    records: list[ObjectTrackRecord],
    original_bat_by_frame: dict[int, ObjectTrackRecord],
    original_weights: dict[int, float],
) -> list[ObjectTrackRecord]:
    if not original_weights:
        return records
    blended: list[ObjectTrackRecord] = []
    for record in records:
        original = original_bat_by_frame.get(record.frame_index)
        original_weight = min(1.0, original_weights.get(record.frame_index, 0.0))
        if record.object_name != "bat" or original_weight <= 0 or original is None:
            blended.append(record)
            continue
        smoothed_weight = 1.0 - original_weight
        blended.append(
            ObjectTrackRecord(
                clip_id=record.clip_id,
                condition_id=record.condition_id,
                frame_index=record.frame_index,
                timestamp_sec=record.timestamp_sec,
                object_name=record.object_name,
                x=_blend_optional(record.x, original.x, smoothed_weight, original_weight),
                y=_blend_optional(record.y, original.y, smoothed_weight, original_weight),
                x2=_blend_optional(record.x2, original.x2, smoothed_weight, original_weight),
                y2=_blend_optional(record.y2, original.y2, smoothed_weight, original_weight),
                confidence=record.confidence,
                width=record.width,
                height=record.height,
                source=record.source,
            )
        )
    return sorted(blended, key=lambda item: (item.frame_index, item.object_name))


def _bat_record_angle(record: ObjectTrackRecord) -> float:
    if record.x is None or record.y is None or record.x2 is None or record.y2 is None:
        return 0.0
    return math.degrees(math.atan2(record.y - record.y2, record.x - record.x2))


def _blend_optional(
    smoothed: float | None,
    original: float | None,
    smoothed_weight: float,
    original_weight: float,
) -> float | None:
    if smoothed is None or original is None:
        return smoothed if smoothed is not None else original
    return smoothed * smoothed_weight + original * original_weight


def _smooth_bat_segment(
    records: list[ObjectTrackRecord],
    window: int,
    min_segment_frames: int,
) -> dict[int, ObjectTrackRecord]:
    if len(records) < min_segment_frames:
        return {}
    if any(record.x2 is None or record.y2 is None for record in records):
        return _smooth_bat_coordinate_segment(records, window)
    return _smooth_bat_geometry_segment(records, window)


def _smooth_bat_geometry_segment(records: list[ObjectTrackRecord], window: int) -> dict[int, ObjectTrackRecord]:
    centers_x: list[float] = []
    centers_y: list[float] = []
    angles: list[float] = []
    lengths: list[float] = []
    previous_angle: float | None = None
    for record in records:
        assert record.x is not None and record.y is not None and record.x2 is not None and record.y2 is not None
        center_x = (record.x + record.x2) / 2
        center_y = (record.y + record.y2) / 2
        dx = record.x - record.x2
        dy = record.y - record.y2
        angle = math.atan2(dy, dx)
        if previous_angle is not None:
            while angle - previous_angle > math.pi:
                angle -= 2 * math.pi
            while angle - previous_angle < -math.pi:
                angle += 2 * math.pi
        previous_angle = angle
        centers_x.append(center_x)
        centers_y.append(center_y)
        angles.append(angle)
        lengths.append(math.hypot(dx, dy))

    half_window = window // 2
    smoothed: dict[int, ObjectTrackRecord] = {}
    for index, record in enumerate(records):
        start = max(0, index - half_window)
        end = min(len(records), index + half_window + 1)
        target_index = index - start
        neighbors = records[start:end]
        center_x = _smooth_numeric_window(centers_x[start:end], neighbors, target_index)
        center_y = _smooth_numeric_window(centers_y[start:end], neighbors, target_index)
        angle = _smooth_numeric_window(angles[start:end], neighbors, target_index)
        length = _smooth_numeric_window(lengths[start:end], neighbors, target_index)
        half_length = length / 2
        dx = math.cos(angle) * half_length
        dy = math.sin(angle) * half_length
        smoothed[record.frame_index] = ObjectTrackRecord(
            clip_id=record.clip_id,
            condition_id=record.condition_id,
            frame_index=record.frame_index,
            timestamp_sec=record.timestamp_sec,
            object_name=record.object_name,
            x=center_x + dx,
            y=center_y + dy,
            x2=center_x - dx,
            y2=center_y - dy,
            confidence=record.confidence,
            width=record.width,
            height=record.height,
            source=record.source,
        )
    return smoothed


def _smooth_bat_coordinate_segment(records: list[ObjectTrackRecord], window: int) -> dict[int, ObjectTrackRecord]:
    half_window = window // 2
    smoothed: dict[int, ObjectTrackRecord] = {}
    for index, record in enumerate(records):
        start = max(0, index - half_window)
        end = min(len(records), index + half_window + 1)
        neighbors = records[start:end]
        smoothed[record.frame_index] = ObjectTrackRecord(
            clip_id=record.clip_id,
            condition_id=record.condition_id,
            frame_index=record.frame_index,
            timestamp_sec=record.timestamp_sec,
            object_name=record.object_name,
            x=_smooth_optional_coordinate(neighbors, "x", index - start),
            y=_smooth_optional_coordinate(neighbors, "y", index - start),
            x2=_smooth_optional_coordinate(neighbors, "x2", index - start),
            y2=_smooth_optional_coordinate(neighbors, "y2", index - start),
            confidence=record.confidence,
            width=record.width,
            height=record.height,
            source=record.source,
        )
    return smoothed


def _smooth_optional_coordinate(records: list[ObjectTrackRecord], field_name: str, target_index: int) -> float | None:
    values: list[tuple[float, float]] = []
    for index, record in enumerate(records):
        value = getattr(record, field_name)
        if value is None:
            continue
        temporal_weight = 1.0 / (1.0 + abs(index - target_index))
        confidence_weight = max(0.05, record.confidence if record.confidence is not None else 0.5)
        values.append((float(value), temporal_weight * confidence_weight))
    if not values:
        return None
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return values[len(values) // 2][0]
    return sum(value * weight for value, weight in values) / total_weight


def _smooth_numeric_window(values: list[float], records: list[ObjectTrackRecord], target_index: int) -> float:
    weighted_values: list[tuple[float, float]] = []
    for index, (value, record) in enumerate(zip(values, records)):
        temporal_weight = 1.0 / (1.0 + abs(index - target_index))
        confidence_weight = max(0.05, record.confidence if record.confidence is not None else 0.5)
        weighted_values.append((value, temporal_weight * confidence_weight))
    if len(weighted_values) >= 3:
        try:
            import numpy as np

            degree = 2 if len(weighted_values) >= 5 else 1
            xs = np.array([index - target_index for index in range(len(weighted_values))], dtype=float)
            ys = np.array([value for value, _ in weighted_values], dtype=float)
            weights = np.array([weight for _, weight in weighted_values], dtype=float)
            if float(weights.sum()) > 0:
                coefficients = np.polyfit(xs, ys, degree, w=weights)
                return float(np.polyval(coefficients, 0.0))
        except Exception:
            pass
    total_weight = sum(weight for _, weight in weighted_values)
    if total_weight <= 0:
        return values[target_index]
    return sum(value * weight for value, weight in weighted_values) / total_weight


def _odd_window(value: int) -> int:
    if value <= 1:
        return 0
    return value if value % 2 == 1 else value - 1


def _filter_short_object_tracks(
    records: list[ObjectTrackRecord],
    object_name: str,
    min_track_length: int,
    max_gap_frames: int,
) -> list[ObjectTrackRecord]:
    if min_track_length <= 1:
        return records
    target_records = sorted(
        [record for record in records if record.object_name == object_name],
        key=lambda item: item.frame_index,
    )
    keep_ids: set[tuple[str, str, int, str]] = set()
    current: list[ObjectTrackRecord] = []
    previous_frame: int | None = None
    for record in target_records:
        if previous_frame is None or record.frame_index - previous_frame <= max_gap_frames + 1:
            current.append(record)
        else:
            _mark_track_if_long_enough(current, min_track_length, keep_ids)
            current = [record]
        previous_frame = record.frame_index
    _mark_track_if_long_enough(current, min_track_length, keep_ids)
    return [
        record
        for record in records
        if record.object_name != object_name
        or (record.clip_id, record.condition_id, record.frame_index, record.object_name) in keep_ids
    ]


def _mark_track_if_long_enough(
    records: list[ObjectTrackRecord],
    min_track_length: int,
    keep_ids: set[tuple[str, str, int, str]],
) -> None:
    if len(records) < min_track_length:
        return
    for record in records:
        keep_ids.add((record.clip_id, record.condition_id, record.frame_index, record.object_name))


def _can_interpolate(previous: ObjectTrackRecord, current: ObjectTrackRecord) -> bool:
    return (
        previous.object_name == current.object_name
        and previous.x is not None
        and previous.y is not None
        and current.x is not None
        and current.y is not None
    )


def _lerp_optional(first: float | None, second: float | None, ratio: float) -> float | None:
    if first is None or second is None:
        return None
    return first + (second - first) * ratio


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
        if wrist_center is not None and barrel[1] > handle[1] + 0.18 * height:
            continue
        length_score = min(1.0, length / (0.35 * diagonal))
        wrist_score = 0.5
        if wrist_center is not None:
            handle_distance = _distance(handle, wrist_center)
            if handle_distance > config.bat_max_handle_distance_ratio * diagonal:
                continue
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
            + (config.bat_wrist_prior_weight if wrist_center is not None else 0.0) * wrist_score
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
        if previous is None and anchors and not _near_any_anchor(
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


def _orient_bat_with_reference(
    first: tuple[float, float],
    second: tuple[float, float],
    wrist_center: tuple[float, float] | None,
    previous: _BatCandidate | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if wrist_center is not None:
        return _orient_bat(first, second, wrist_center)
    if previous is None:
        return first, second
    same = _distance(first, previous.handle) + _distance(second, previous.barrel)
    flipped = _distance(second, previous.handle) + _distance(first, previous.barrel)
    if same <= flipped:
        return first, second
    return second, first


def _mean_point(points: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))


def _bat_angle(handle: tuple[float, float], barrel: tuple[float, float]) -> float:
    return math.degrees(math.atan2(barrel[1] - handle[1], barrel[0] - handle[0]))


def _angle_delta_deg(first: float, second: float) -> float:
    delta = abs((first - second + 90.0) % 180.0 - 90.0)
    return min(delta, 180.0 - delta)


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
