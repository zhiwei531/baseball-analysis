"""ROI data structures and automatic ROI proposal helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baseball_pose.io.video import FrameRecord, read_frame, write_video_from_frames
from baseball_pose.pose.schema import PoseRecord


@dataclass(frozen=True)
class RoiBox:
    x: float
    y: float
    width: float
    height: float
    coordinate_space: str = "pixel"

    def expanded(self, factor: float) -> "RoiBox":
        dx = self.width * factor / 2
        dy = self.height * factor / 2
        return RoiBox(
            x=self.x - dx,
            y=self.y - dy,
            width=self.width * (1 + factor),
            height=self.height * (1 + factor),
            coordinate_space=self.coordinate_space,
        )

    def clamped(self, image_width: int, image_height: int) -> "RoiBox":
        x = max(0.0, min(float(self.x), float(image_width - 1)))
        y = max(0.0, min(float(self.y), float(image_height - 1)))
        right = max(x + 1.0, min(float(self.x + self.width), float(image_width)))
        bottom = max(y + 1.0, min(float(self.y + self.height), float(image_height)))
        return RoiBox(
            x=x,
            y=y,
            width=right - x,
            height=bottom - y,
            coordinate_space=self.coordinate_space,
        )

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return round(self.x), round(self.y), round(self.width), round(self.height)


@dataclass(frozen=True)
class AutoRoiResult:
    clip_id: str
    condition_id: str
    roi: RoiBox
    source_frame_count: int
    candidate_count: int


def estimate_clip_auto_roi(
    frames: list[FrameRecord],
    expansion: float = 0.35,
    max_frames: int = 60,
    min_area_ratio: float = 0.002,
) -> AutoRoiResult:
    """Estimate one fixed clip-level ROI from motion and edge candidates."""

    if not frames:
        raise ValueError("frames cannot be empty.")

    cv2 = _require_cv2()
    selected_frames = frames[:max_frames]
    previous_gray = None
    candidates: list[RoiBox] = []
    first_image = read_frame(selected_frames[0].frame_path)
    image_height, image_width = first_image.shape[:2]

    for frame in selected_frames:
        image = read_frame(frame.frame_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if previous_gray is None:
            previous_gray = gray
            continue

        motion_mask = _motion_mask(previous_gray, gray)
        edge_mask = _edge_mask(gray)
        combined = _combined_roi_mask(motion_mask, edge_mask)
        candidates.extend(
            _candidate_boxes(
                combined,
                edge_mask,
                image_width=image_width,
                image_height=image_height,
                min_area_ratio=min_area_ratio,
            )
        )
        previous_gray = gray

    roi = aggregate_roi_boxes(candidates, image_width, image_height).expanded(expansion)
    roi = roi.clamped(image_width, image_height)
    return AutoRoiResult(
        clip_id=frames[0].clip_id,
        condition_id="auto_roi_raw",
        roi=roi,
        source_frame_count=len(selected_frames),
        candidate_count=len(candidates),
    )


def aggregate_roi_boxes(
    boxes: list[RoiBox],
    image_width: int,
    image_height: int,
) -> RoiBox:
    """Aggregate candidate boxes into one fixed clip-level ROI."""

    if not boxes:
        return RoiBox(0, 0, image_width, image_height)

    left = min(box.x for box in boxes)
    top = min(box.y for box in boxes)
    right = max(box.x + box.width for box in boxes)
    bottom = max(box.y + box.height for box in boxes)
    return RoiBox(left, top, right - left, bottom - top).clamped(image_width, image_height)


def estimate_pose_prior_roi(
    records: list[PoseRecord],
    image_width: int,
    image_height: int,
    condition_id: str = "auto_roi_pose_prior",
    confidence_threshold: float = 0.5,
    min_joints_per_frame: int = 4,
    horizontal_padding: float = 0.35,
    top_padding: float = 0.45,
    bottom_padding: float = 0.25,
) -> AutoRoiResult:
    """Estimate one clip-level ROI from baseline pose landmarks."""

    boxes = _pose_frame_boxes(
        records,
        image_width=image_width,
        image_height=image_height,
        confidence_threshold=confidence_threshold,
        min_joints_per_frame=min_joints_per_frame,
    )
    if not boxes:
        clip_id = records[0].clip_id if records else "unknown"
        return AutoRoiResult(
            clip_id=clip_id,
            condition_id=condition_id,
            roi=RoiBox(0, 0, image_width, image_height),
            source_frame_count=0,
            candidate_count=0,
        )

    left = _percentile([box.x for box in boxes], 0.10)
    top = _percentile([box.y for box in boxes], 0.05)
    right = _percentile([box.x + box.width for box in boxes], 0.90)
    bottom = _percentile([box.y + box.height for box in boxes], 0.95)
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)

    padded = RoiBox(
        x=left - width * horizontal_padding,
        y=top - height * top_padding,
        width=width * (1 + 2 * horizontal_padding),
        height=height * (1 + top_padding + bottom_padding),
    ).clamped(image_width, image_height)
    return AutoRoiResult(
        clip_id=records[0].clip_id,
        condition_id=condition_id,
        roi=padded,
        source_frame_count=len({record.frame_index for record in records}),
        candidate_count=len(boxes),
    )


def _pose_frame_boxes(
    records: list[PoseRecord],
    image_width: int,
    image_height: int,
    confidence_threshold: float,
    min_joints_per_frame: int,
) -> list[RoiBox]:
    by_frame: dict[int, list[PoseRecord]] = {}
    for record in records:
        if record.x is None or record.y is None:
            continue
        score = record.confidence if record.confidence is not None else record.visibility
        if score is not None and score < confidence_threshold:
            continue
        by_frame.setdefault(record.frame_index, []).append(record)

    boxes: list[RoiBox] = []
    for frame_records in by_frame.values():
        if len(frame_records) < min_joints_per_frame:
            continue
        xs = [record.x * image_width for record in frame_records if record.x is not None]
        ys = [record.y * image_height for record in frame_records if record.y is not None]
        if not xs or not ys:
            continue
        boxes.append(RoiBox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)))
    return boxes


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values cannot be empty.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def crop_to_roi(image, roi: RoiBox):
    x, y, width, height = roi.as_int_tuple()
    return image[y : y + height, x : x + width]


def remap_pose_records_to_full_frame(
    records: list[PoseRecord],
    roi: RoiBox,
    image_width: int,
    image_height: int,
) -> list[PoseRecord]:
    """Map crop-normalized pose coordinates back to full-frame normalized coordinates."""

    remapped: list[PoseRecord] = []
    for record in records:
        x = None if record.x is None else (roi.x + record.x * roi.width) / image_width
        y = None if record.y is None else (roi.y + record.y * roi.height) / image_height
        remapped.append(
            PoseRecord(
                clip_id=record.clip_id,
                condition_id=record.condition_id,
                frame_index=record.frame_index,
                timestamp_sec=record.timestamp_sec,
                joint_name=record.joint_name,
                x=x,
                y=y,
                visibility=record.visibility,
                confidence=record.confidence,
                backend=record.backend,
                inference_time_ms=record.inference_time_ms,
            )
        )
    return remapped


def write_auto_roi_csv(path: str | Path, result: AutoRoiResult) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "clip_id,condition_id,x,y,width,height,coordinate_space,source_frame_count,candidate_count\n"
        f"{result.clip_id},{result.condition_id},{result.roi.x},{result.roi.y},"
        f"{result.roi.width},{result.roi.height},{result.roi.coordinate_space},"
        f"{result.source_frame_count},{result.candidate_count}\n",
        encoding="utf-8",
    )


def write_roi_debug_video(
    frames: list[FrameRecord],
    roi: RoiBox,
    output_path: str | Path,
    fps: float,
) -> None:
    """Write a debug video with the selected fixed ROI box drawn on original frames."""

    cv2 = _require_cv2()
    output_dir = Path(output_path).parent / "frames" / frames[0].clip_id
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_frames: list[Path] = []
    x, y, width, height = roi.as_int_tuple()
    condition_id = frames[0].condition_id

    for frame in frames:
        image = read_frame(frame.frame_path)
        debug = image.copy()
        cv2.rectangle(debug, (x, y), (x + width, y + height), (60, 220, 255), 3)
        label = f"{frame.clip_id} {condition_id} x={x} y={y} w={width} h={height}"
        cv2.rectangle(debug, (12, 12), (12 + min(920, 11 * len(label)), 44), (0, 0, 0), -1)
        cv2.putText(debug, label, (22, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        frame_path = output_dir / frame.frame_path.name
        if not cv2.imwrite(str(frame_path), debug):
            raise RuntimeError(f"Could not write ROI debug frame: {frame_path}")
        debug_frames.append(frame_path)

    write_video_from_frames(debug_frames, output_path, fps=fps)


def _motion_mask(previous_gray, current_gray):
    cv2 = _require_cv2()
    diff = cv2.absdiff(previous_gray, current_gray)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.dilate(mask, kernel, iterations=2)


def _edge_mask(gray):
    cv2 = _require_cv2()
    return cv2.Canny(gray, 60, 160)


def _combined_roi_mask(motion_mask, edge_mask):
    cv2 = _require_cv2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    edge_near_motion = cv2.bitwise_and(edge_mask, cv2.dilate(motion_mask, kernel, iterations=1))
    combined = cv2.bitwise_or(motion_mask, edge_near_motion)
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)


def _candidate_boxes(
    mask,
    edge_mask,
    image_width: int,
    image_height: int,
    min_area_ratio: float,
) -> list[RoiBox]:
    cv2 = _require_cv2()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = image_width * image_height * min_area_ratio
    scored: list[tuple[float, RoiBox]] = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if area < min_area:
            continue
        aspect = width / max(height, 1)
        if aspect < 0.15 or aspect > 5.0:
            continue
        edge_density = cv2.countNonZero(edge_mask[y : y + height, x : x + width]) / max(area, 1)
        score = area * (1.0 + edge_density)
        scored.append((score, RoiBox(x, y, width, height)))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in scored[:3]]


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required for ROI processing.") from exc

    return cv2
