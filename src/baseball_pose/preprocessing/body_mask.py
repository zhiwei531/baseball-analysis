"""Body-prior masks for suppressing nearby non-subject people."""

from __future__ import annotations

from dataclasses import dataclass

from baseball_pose.pose.schema import POSE_CONNECTIONS, PoseRecord
from baseball_pose.preprocessing.roi import RoiBox, crop_to_roi


@dataclass(frozen=True)
class MaskedBodyCrop:
    image: object
    roi: RoiBox
    prior_joint_count: int


def create_body_prior_masked_crop(
    image,
    prior_records: list[PoseRecord],
    image_width: int,
    image_height: int,
    fallback_roi: RoiBox,
    confidence_threshold: float = 0.5,
    padding_ratio: float = 0.55,
    min_width_ratio: float = 0.28,
    min_height_ratio: float = 0.50,
    limb_thickness_ratio: float = 0.045,
    joint_radius_ratio: float = 0.035,
) -> MaskedBodyCrop:
    """Crop around prior body landmarks and black out pixels outside a skeleton-shaped mask."""

    cv2 = _require_cv2()
    points = _confident_points(prior_records, image_width, image_height, confidence_threshold)
    if len(points) < 4:
        roi = fallback_roi
        crop = crop_to_roi(image, roi)
        return MaskedBodyCrop(image=crop, roi=roi, prior_joint_count=len(points))

    roi = _body_roi_from_points(
        points,
        image_width=image_width,
        image_height=image_height,
        padding_ratio=padding_ratio,
        min_width_ratio=min_width_ratio,
        min_height_ratio=min_height_ratio,
    )
    crop = crop_to_roi(image, roi)
    x0, y0, width, height = roi.as_int_tuple()
    crop_points = {
        joint_name: (round(x - x0), round(y - y0))
        for joint_name, (x, y) in points.items()
        if x0 <= x <= x0 + width and y0 <= y <= y0 + height
    }
    if len(crop_points) < 4:
        return MaskedBodyCrop(image=crop, roi=roi, prior_joint_count=len(points))

    mask = _body_mask(
        crop.shape[:2],
        crop_points,
        limb_thickness=max(6, round(min(width, height) * limb_thickness_ratio)),
        joint_radius=max(8, round(min(width, height) * joint_radius_ratio)),
    )
    masked = cv2.bitwise_and(crop, crop, mask=mask)
    return MaskedBodyCrop(image=masked, roi=roi, prior_joint_count=len(points))


def _confident_points(
    records: list[PoseRecord],
    image_width: int,
    image_height: int,
    confidence_threshold: float,
) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    for record in records:
        if record.x is None or record.y is None:
            continue
        score = record.confidence if record.confidence is not None else record.visibility
        if score is not None and score < confidence_threshold:
            continue
        points[record.joint_name] = (record.x * image_width, record.y * image_height)
    return points


def _body_roi_from_points(
    points: dict[str, tuple[float, float]],
    image_width: int,
    image_height: int,
    padding_ratio: float,
    min_width_ratio: float,
    min_height_ratio: float,
) -> RoiBox:
    xs = [point[0] for point in points.values()]
    ys = [point[1] for point in points.values()]
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)
    width = max(right - left, image_width * min_width_ratio)
    height = max(bottom - top, image_height * min_height_ratio)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    padded_width = width * (1 + padding_ratio)
    padded_height = height * (1 + padding_ratio)
    return RoiBox(
        x=center_x - padded_width / 2,
        y=center_y - padded_height / 2,
        width=padded_width,
        height=padded_height,
    ).clamped(image_width, image_height)


def _body_mask(
    shape: tuple[int, int],
    points: dict[str, tuple[int, int]],
    limb_thickness: int,
    joint_radius: int,
):
    cv2 = _require_cv2()
    height, width = shape
    mask = _zeros_mask(height, width)

    for start, end in POSE_CONNECTIONS:
        if start in points and end in points:
            cv2.line(mask, points[start], points[end], 255, limb_thickness, cv2.LINE_AA)

    torso_points = [
        points[joint]
        for joint in ("left_shoulder", "right_shoulder", "right_hip", "left_hip")
        if joint in points
    ]
    if len(torso_points) >= 3:
        cv2.fillConvexPoly(mask, _as_int32_array(torso_points), 255, cv2.LINE_AA)

    for point in points.values():
        cv2.circle(mask, point, joint_radius, 255, -1, cv2.LINE_AA)

    blur_size = max(5, (joint_radius // 2) * 2 + 1)
    return cv2.GaussianBlur(mask, (blur_size, blur_size), 0)


def _zeros_mask(height: int, width: int):
    import numpy as np

    return np.zeros((height, width), dtype=np.uint8)


def _as_int32_array(points: list[tuple[int, int]]):
    import numpy as np

    return np.array(points, dtype=np.int32)


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for body-prior masking. Install project dependencies first."
        ) from exc

    return cv2
