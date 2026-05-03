"""Image-processing subject proposals that do not depend on pose skeletons."""

from __future__ import annotations

from dataclasses import dataclass

from baseball_pose.preprocessing.roi import RoiBox


@dataclass(frozen=True)
class ImageProposal:
    mask: object
    roi: RoiBox
    candidate_count: int


def create_center_motion_grabcut_proposal(
    image,
    previous_image=None,
    background_subtractor=None,
    center_x: float = 0.5,
    center_width_ratio: float = 0.54,
    min_area_ratio: float = 0.006,
    grabcut_iterations: int = 2,
) -> ImageProposal:
    """Create a subject mask from image evidence, center prior, and foreground segmentation."""

    cv2 = _require_cv2()
    import numpy as np

    height, width = image.shape[:2]
    center_mask = _center_band_mask(height, width, center_x, center_width_ratio)
    motion_mask = _motion_mask(image, previous_image)
    foreground_mask = _foreground_mask(image, background_subtractor)
    grabcut_mask = _grabcut_mask(
        image,
        center_mask=center_mask,
        motion_mask=motion_mask,
        foreground_mask=foreground_mask,
        center_x=center_x,
        center_width_ratio=center_width_ratio,
        iterations=grabcut_iterations,
    )
    combined = cv2.bitwise_and(grabcut_mask, center_mask)
    motion_supported = cv2.bitwise_or(motion_mask, foreground_mask)
    supported = cv2.bitwise_or(combined, cv2.bitwise_and(motion_supported, center_mask))
    supported = _clean_mask(supported)

    selected, roi, candidate_count = _select_subject_component(
        supported,
        image_width=width,
        image_height=height,
        center_x=center_x,
        min_area_ratio=min_area_ratio,
    )
    if selected is None:
        selected = _clean_mask(cv2.bitwise_and(grabcut_mask, center_mask))
        roi = _mask_roi(selected, width, height)
        candidate_count = 0

    selected = _smooth_subject_shape(selected)
    if np.count_nonzero(selected) == 0:
        roi_width = width * center_width_ratio
        roi = RoiBox(width * center_x - roi_width / 2, 0, roi_width, height).clamped(width, height)
    else:
        roi = _mask_roi(selected, width, height)
    return ImageProposal(mask=selected, roi=roi, candidate_count=candidate_count)


def draw_image_proposal_overlay(image, proposal: ImageProposal):
    """Draw a pure image-processing proposal mask and ROI on the original frame."""

    cv2 = _require_cv2()
    overlay = image.copy()
    color_mask = _zeros_color(image.shape[0], image.shape[1])
    color_mask[:, :, 1] = proposal.mask
    overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.45, 0)
    contours, _ = cv2.findContours(proposal.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    x, y, width, height = proposal.roi.as_int_tuple()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (40, 40, 255), 2)
    cv2.putText(
        overlay,
        f"image proposal components: {proposal.candidate_count}",
        (x + 8, max(24, y + 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (40, 40, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def apply_image_proposal_mask(image, proposal: ImageProposal):
    """Return a full-frame black canvas containing only proposal pixels."""

    cv2 = _require_cv2()
    return cv2.bitwise_and(image, image, mask=proposal.mask)


def _center_band_mask(height: int, width: int, center_x: float, width_ratio: float):
    import numpy as np

    mask = np.zeros((height, width), dtype=np.uint8)
    band_width = width * width_ratio
    left = max(0, round(width * center_x - band_width / 2))
    right = min(width, round(width * center_x + band_width / 2))
    mask[:, left:right] = 255
    return mask


def _motion_mask(image, previous_image):
    cv2 = _require_cv2()
    import numpy as np

    if previous_image is None:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    previous_gray = cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, previous_gray)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    return _clean_mask(mask)


def _foreground_mask(image, background_subtractor):
    cv2 = _require_cv2()
    import numpy as np

    if background_subtractor is None:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    mask = background_subtractor.apply(image)
    _, mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
    return _clean_mask(mask)


def _grabcut_mask(
    image,
    center_mask,
    motion_mask,
    foreground_mask,
    center_x: float,
    center_width_ratio: float,
    iterations: int,
):
    cv2 = _require_cv2()
    import numpy as np

    height, width = image.shape[:2]
    mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    mask[center_mask > 0] = cv2.GC_PR_BGD

    center_fg = _ellipse_mask(height, width, center_x, center_width_ratio * 0.50)
    evidence_fg = cv2.bitwise_and(cv2.bitwise_or(motion_mask, foreground_mask), center_mask)
    mask[center_fg > 0] = cv2.GC_PR_FGD
    mask[evidence_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return center_fg
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")


def _ellipse_mask(height: int, width: int, center_x: float, width_ratio: float):
    cv2 = _require_cv2()
    import numpy as np

    mask = np.zeros((height, width), dtype=np.uint8)
    center = (round(width * center_x), round(height * 0.53))
    axes = (round(width * width_ratio / 2), round(height * 0.43))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    return mask


def _clean_mask(mask):
    cv2 = _require_cv2()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.dilate(mask, kernel, iterations=1)


def _select_subject_component(
    mask,
    image_width: int,
    image_height: int,
    center_x: float,
    min_area_ratio: float,
):
    cv2 = _require_cv2()
    import numpy as np

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = image_width * image_height * min_area_ratio
    scored = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if height < image_height * 0.20:
            continue
        center_distance = abs((x + width / 2) / image_width - center_x)
        vertical_bonus = height / image_height
        score = area / (1 + 4 * center_distance) + area * vertical_bonus
        scored.append((score, contour))
    if not scored:
        return None, RoiBox(0, 0, image_width, image_height), len(contours)

    _, best_contour = max(scored, key=lambda item: item[0])
    selected = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(selected, [best_contour], -1, 255, -1)
    return selected, _mask_roi(selected, image_width, image_height), len(scored)


def _smooth_subject_shape(mask):
    cv2 = _require_cv2()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _mask_roi(mask, image_width: int, image_height: int) -> RoiBox:
    cv2 = _require_cv2()

    x, y, width, height = cv2.boundingRect(mask)
    if width == 0 or height == 0:
        return RoiBox(0, 0, image_width, image_height)
    return RoiBox(x, y, width, height).expanded(0.15).clamped(image_width, image_height)


def _zeros_color(height: int, width: int):
    import numpy as np

    return np.zeros((height, width, 3), dtype=np.uint8)


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for image proposals. Install project dependencies first."
        ) from exc

    return cv2
