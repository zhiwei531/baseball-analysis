"""Image-processing subject proposals that do not depend on pose skeletons."""

from __future__ import annotations

from dataclasses import dataclass

from baseball_pose.preprocessing.roi import RoiBox


@dataclass(frozen=True)
class ImageProposal:
    mask: object
    roi: RoiBox
    candidate_count: int
    center_x: float = 0.5
    center_width_ratio: float = 0.54


@dataclass
class ImageProposalTracker:
    """Track the subject-centered proposal prior across frames."""

    initial_center_x: float = 0.5
    center_x: float = 0.5
    max_offset: float = 0.18
    max_step: float = 0.02
    smoothing: float = 0.45
    previous_area_ratio: float | None = None

    def update(self, proposal: ImageProposal) -> None:
        cv2 = _require_cv2()
        height, width = proposal.mask.shape[:2]
        area_ratio = cv2.countNonZero(proposal.mask) / max(width * height, 1)
        if area_ratio <= 0:
            return
        if self.previous_area_ratio is not None:
            area_change = area_ratio / max(self.previous_area_ratio, 1e-6)
            if area_change > 2.8 or area_change < 0.35:
                return

        measured_center = _estimate_tracking_center_x(
            proposal.mask,
            current_center_x=self.center_x,
            fallback_center_x=proposal.center_x,
        )
        max_left = self.initial_center_x - self.max_offset
        max_right = self.initial_center_x + self.max_offset
        measured_center = max(max_left, min(max_right, measured_center))
        delta = max(-self.max_step, min(self.max_step, measured_center - self.center_x))
        self.center_x = max(max_left, min(max_right, self.center_x + delta * self.smoothing))
        self.previous_area_ratio = area_ratio


def create_center_motion_grabcut_proposal(
    image,
    previous_image=None,
    previous_mask=None,
    background_subtractor=None,
    center_x: float = 0.5,
    center_width_ratio: float = 0.54,
    min_area_ratio: float = 0.006,
    grabcut_iterations: int = 2,
    processing_scale: float = 1.0,
    vertical_body_width_ratio: float = 0.22,
) -> ImageProposal:
    """Create a subject mask from image evidence, center prior, and foreground segmentation."""

    cv2 = _require_cv2()
    import numpy as np

    if not 0 < processing_scale <= 1:
        raise ValueError("processing_scale must be in (0, 1].")
    if processing_scale < 1:
        small_size = (
            max(1, round(image.shape[1] * processing_scale)),
            max(1, round(image.shape[0] * processing_scale)),
        )
        small_image = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)
        small_previous = (
            None
            if previous_image is None
            else cv2.resize(previous_image, small_size, interpolation=cv2.INTER_AREA)
        )
        small_previous_mask = (
            None
            if previous_mask is None
            else cv2.resize(previous_mask, small_size, interpolation=cv2.INTER_NEAREST)
        )
        small_proposal = create_center_motion_grabcut_proposal(
            image=small_image,
            previous_image=small_previous,
            previous_mask=small_previous_mask,
            background_subtractor=background_subtractor,
            center_x=center_x,
            center_width_ratio=center_width_ratio,
            min_area_ratio=min_area_ratio,
            grabcut_iterations=grabcut_iterations,
            processing_scale=1.0,
            vertical_body_width_ratio=vertical_body_width_ratio,
        )
        mask = cv2.resize(
            small_proposal.mask,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        roi = RoiBox(
            small_proposal.roi.x / processing_scale,
            small_proposal.roi.y / processing_scale,
            small_proposal.roi.width / processing_scale,
            small_proposal.roi.height / processing_scale,
        ).clamped(image.shape[1], image.shape[0])
        return ImageProposal(
            mask=mask,
            roi=roi,
            candidate_count=small_proposal.candidate_count,
            center_x=center_x,
            center_width_ratio=center_width_ratio,
        )

    enhanced = _enhance_subject_contrast(image)
    previous_enhanced = None if previous_image is None else _enhance_subject_contrast(previous_image)
    height, width = image.shape[:2]
    center_mask = _center_band_mask(height, width, center_x, center_width_ratio)
    motion_mask = _motion_mask(enhanced, previous_enhanced)
    foreground_mask = _foreground_mask(enhanced, background_subtractor)
    grabcut_mask = _grabcut_mask(
        enhanced,
        center_mask=center_mask,
        motion_mask=motion_mask,
        foreground_mask=foreground_mask,
        previous_mask=previous_mask,
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
        previous_mask=previous_mask,
    )
    if selected is None:
        selected = _clean_mask(cv2.bitwise_and(grabcut_mask, center_mask))
        roi = _mask_roi(selected, width, height)
        candidate_count = 0

    selected = _keep_center_vertical_body_region(
        selected,
        center_x=center_x,
        image_width=width,
        image_height=height,
        body_width_ratio=vertical_body_width_ratio,
    )
    selected = _smooth_subject_shape(selected)
    selected = _keep_center_vertical_body_region(
        selected,
        center_x=center_x,
        image_width=width,
        image_height=height,
        body_width_ratio=vertical_body_width_ratio,
    )
    selected = _grabcut_refine_center_body(
        enhanced,
        selected,
        center_x=center_x,
        body_width_ratio=vertical_body_width_ratio,
        iterations=grabcut_iterations,
    )
    selected = _temporal_stabilize_mask(selected, previous_mask)
    if np.count_nonzero(selected) == 0:
        roi_width = width * center_width_ratio
        roi = RoiBox(width * center_x - roi_width / 2, 0, roi_width, height).clamped(width, height)
    else:
        roi = _mask_roi(selected, width, height)
    return ImageProposal(
        mask=selected,
        roi=roi,
        candidate_count=candidate_count,
        center_x=center_x,
        center_width_ratio=center_width_ratio,
    )


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
    _, mask = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)
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
    previous_mask,
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
    if previous_mask is not None:
        previous_core = cv2.erode(
            previous_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        previous_support = cv2.dilate(
            previous_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
        )
        mask[previous_support > 0] = cv2.GC_PR_FGD
        mask[previous_core > 0] = cv2.GC_FGD

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


def _enhance_subject_contrast(image):
    cv2 = _require_cv2()

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    enhanced = cv2.cvtColor(
        cv2.merge((enhanced_lightness, a_channel, b_channel)),
        cv2.COLOR_LAB2BGR,
    )
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    return cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)


def _select_subject_component(
    mask,
    image_width: int,
    image_height: int,
    center_x: float,
    min_area_ratio: float,
    previous_mask,
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
        overlap_bonus = 0.0
        if previous_mask is not None:
            component_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(component_mask, [contour], -1, 255, -1)
            overlap = np.count_nonzero(cv2.bitwise_and(component_mask, previous_mask))
            overlap_bonus = 3.0 * overlap
        score = area / (1 + 6 * center_distance) + area * vertical_bonus + overlap_bonus
        scored.append((score, contour))
    if not scored:
        return None, RoiBox(0, 0, image_width, image_height), len(contours)

    _, best_contour = max(scored, key=lambda item: item[0])
    selected = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(selected, [best_contour], -1, 255, -1)
    return selected, _mask_roi(selected, image_width, image_height), len(scored)


def _estimate_tracking_center_x(mask, current_center_x: float, fallback_center_x: float) -> float:
    cv2 = _require_cv2()

    height, width = mask.shape[:2]
    if cv2.countNonZero(mask) == 0:
        return fallback_center_x

    tracking_band = _center_band_mask(height, width, current_center_x, 0.42)
    tracked_pixels = cv2.bitwise_and(mask, tracking_band)
    if cv2.countNonZero(tracked_pixels) == 0:
        tracked_pixels = mask

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, round(width * 0.025)), max(15, round(height * 0.16))),
    )
    vertical_core = cv2.morphologyEx(tracked_pixels, cv2.MORPH_OPEN, vertical_kernel)
    if cv2.countNonZero(vertical_core) > 0:
        return _mask_center_x(vertical_core) / width
    return _mask_center_x(tracked_pixels) / width


def _keep_center_vertical_body_region(
    mask,
    center_x: float,
    image_width: int,
    image_height: int,
    body_width_ratio: float,
):
    cv2 = _require_cv2()
    import numpy as np

    if cv2.countNonZero(mask) == 0:
        return mask

    guide_width = max(3, round(image_width * body_width_ratio))
    guide_left = max(0, round(image_width * center_x - guide_width / 2))
    guide_right = min(image_width, round(image_width * center_x + guide_width / 2))

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, round(image_width * 0.025)), max(15, round(image_height * 0.18))),
    )
    vertical_core = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vertical_kernel)
    if cv2.countNonZero(vertical_core) == 0:
        vertical_core = cv2.bitwise_and(mask, _center_band_mask(image_height, image_width, center_x, body_width_ratio))

    contours, _ = cv2.findContours(vertical_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.bitwise_and(mask, _center_band_mask(image_height, image_width, center_x, body_width_ratio))

    best_contour = None
    best_score = -1.0
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if height < image_height * 0.18:
            continue
        component = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 255, -1)
        guide_overlap = cv2.countNonZero(component[:, guide_left:guide_right])
        if guide_overlap == 0:
            continue
        center_distance = abs((x + width / 2) / image_width - center_x)
        score = guide_overlap * (1 + height / image_height) / (1 + 8 * center_distance)
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return cv2.bitwise_and(mask, _center_band_mask(image_height, image_width, center_x, body_width_ratio))

    seed = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(seed, [best_contour], -1, 255, -1)
    seed_center = _mask_center_x(seed[:, guide_left:guide_right], offset_x=guide_left) / image_width
    support_band = _body_envelope_mask(image_height, image_width, seed_center, body_width_ratio)
    seed = cv2.bitwise_and(seed, support_band)

    grown = seed.copy()
    grow_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    constrained_mask = cv2.bitwise_and(mask, support_band)
    for _ in range(8):
        expanded = cv2.dilate(grown, grow_kernel, iterations=1)
        expanded = cv2.bitwise_and(expanded, constrained_mask)
        if np.array_equal(expanded, grown):
            break
        grown = expanded

    if cv2.countNonZero(grown) == 0:
        return cv2.bitwise_and(mask, support_band)
    return grown


def _mask_center_x(mask, offset_x: int = 0) -> float:
    cv2 = _require_cv2()

    if cv2.countNonZero(mask) == 0:
        x, _, width, _ = cv2.boundingRect(mask)
        return offset_x + x + width / 2
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] == 0:
        x, _, width, _ = cv2.boundingRect(mask)
        return offset_x + x + width / 2
    return offset_x + moments["m10"] / moments["m00"]


def _body_envelope_mask(height: int, width: int, center_x: float, body_width_ratio: float):
    import numpy as np

    envelope = np.zeros((height, width), dtype=np.uint8)
    center_px = width * center_x
    upper_left = width * max(0.13, body_width_ratio * 0.72)
    upper_right = width * max(0.16, body_width_ratio * 0.82)
    lower_left = width * max(0.075, body_width_ratio * 0.42)
    lower_right = width * max(0.14, body_width_ratio * 0.70)
    transition_start = height * 0.34
    transition_end = height * 0.58

    for y in range(height):
        if y <= transition_start:
            weight = 0.0
        elif y >= transition_end:
            weight = 1.0
        else:
            weight = (y - transition_start) / max(transition_end - transition_start, 1)
        left_extent = upper_left * (1 - weight) + lower_left * weight
        right_extent = upper_right * (1 - weight) + lower_right * weight
        left = max(0, round(center_px - left_extent))
        right = min(width, round(center_px + right_extent))
        envelope[y, left:right] = 255

    return envelope


def _grabcut_refine_center_body(
    image,
    mask,
    center_x: float,
    body_width_ratio: float,
    iterations: int,
):
    cv2 = _require_cv2()
    import numpy as np

    if cv2.countNonZero(mask) == 0:
        return mask

    height, width = mask.shape[:2]
    guide = _center_band_mask(height, width, center_x, max(0.08, body_width_ratio * 0.55))
    guide_pixels = cv2.bitwise_and(mask, guide)
    if cv2.countNonZero(guide_pixels) == 0:
        return mask

    seed_center = _mask_center_x(guide_pixels) / width
    seed_envelope = _body_envelope_mask(height, width, seed_center, body_width_ratio * 0.72)
    foreground_seed = cv2.bitwise_and(mask, seed_envelope)
    foreground_seed = cv2.morphologyEx(
        foreground_seed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 17)),
    )
    if cv2.countNonZero(foreground_seed) == 0:
        return mask

    grabcut_labels = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    grabcut_labels[mask > 0] = cv2.GC_PR_BGD
    probable_foreground = cv2.dilate(
        foreground_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 31)),
        iterations=1,
    )
    grabcut_labels[cv2.bitwise_and(probable_foreground, mask) > 0] = cv2.GC_PR_FGD
    sure_foreground = cv2.erode(
        foreground_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 11)),
        iterations=1,
    )
    grabcut_labels[sure_foreground > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            image,
            grabcut_labels,
            None,
            bgd_model,
            fgd_model,
            max(1, iterations),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return mask

    refined = np.where(
        (grabcut_labels == cv2.GC_FGD) | (grabcut_labels == cv2.GC_PR_FGD),
        255,
        0,
    ).astype("uint8")
    refined = cv2.bitwise_and(refined, mask)
    if cv2.countNonZero(refined) < cv2.countNonZero(foreground_seed) * 0.85:
        return mask
    return _clean_refined_body_mask(refined)


def _clean_refined_body_mask(mask):
    cv2 = _require_cv2()

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    body_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, body_kernel)


def _smooth_subject_shape(mask):
    cv2 = _require_cv2()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _temporal_stabilize_mask(mask, previous_mask):
    if previous_mask is None:
        return mask

    cv2 = _require_cv2()
    blended = cv2.addWeighted(mask, 0.72, previous_mask, 0.28, 0)
    _, stabilized = cv2.threshold(blended, 120, 255, cv2.THRESH_BINARY)
    support = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=1)
    stabilized = cv2.bitwise_and(stabilized, support)
    if cv2.countNonZero(stabilized) == 0:
        return mask
    return stabilized


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
