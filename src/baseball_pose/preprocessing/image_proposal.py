"""Image-processing subject proposals that do not depend on pose skeletons."""

from __future__ import annotations

from dataclasses import dataclass, field

from baseball_pose.preprocessing.roi import RoiBox


@dataclass(frozen=True)
class ImageProposal:
    mask: object
    roi: RoiBox
    candidate_count: int
    center_x: float = 0.5
    center_width_ratio: float = 0.54
    subject_center_x: float | None = None


@dataclass
class ImageProposalTracker:
    """Track the subject-centered proposal prior across frames."""

    initial_center_x: float = 0.5
    initial_width_ratio: float = 0.62
    center_x: float = 0.5
    center_width_ratio: float = 0.62
    max_offset: float = 0.12
    max_center_step: float = 0.015
    max_width_step: float = 0.025
    center_smoothing: float = 0.55
    width_smoothing: float = 0.45
    min_width_ratio: float = 0.56
    max_width_ratio: float = 0.72
    warmup_frames: int = 90
    frame_count: int = 0
    previous_area_ratio: float | None = None
    warmup_center_samples: list[float] = field(default_factory=list)
    warmup_width_samples: list[float] = field(default_factory=list)
    previous_gray: object | None = field(default=None, repr=False)
    tracking_points: object | None = field(default=None, repr=False)
    min_tracking_points: int = 12
    feature_band_width_ratio: float = 0.24

    def update(self, proposal: ImageProposal, image=None) -> None:
        cv2 = _require_cv2()
        height, width = proposal.mask.shape[:2]
        area_ratio = cv2.countNonZero(proposal.mask) / max(width * height, 1)
        if area_ratio <= 0:
            return
        measured_center = (
            self._track_feature_center_x(image, proposal)
            if image is not None
            else None
        )
        if measured_center is None:
            measured_center = (
                proposal.subject_center_x
                if proposal.subject_center_x is not None
                else _estimate_subject_core_center_x(proposal.mask, self.center_x)
            )
        measured_width_ratio = _proposal_search_width_ratio(
            proposal.roi.width / max(width, 1),
            min_width_ratio=self.min_width_ratio,
            max_width_ratio=self.max_width_ratio,
        )

        if self.frame_count < self.warmup_frames:
            self.warmup_center_samples.append(measured_center)
            self.warmup_width_samples.append(measured_width_ratio)
            self.frame_count += 1
            self.previous_area_ratio = area_ratio
            return

        if self.previous_area_ratio is not None:
            area_change = area_ratio / max(self.previous_area_ratio, 1e-6)
            if area_change > 2.8 or area_change < 0.35:
                return

        self._move_toward(measured_center, measured_width_ratio)
        self.frame_count += 1
        self.previous_area_ratio = area_ratio

    def _move_toward(self, measured_center: float, measured_width_ratio: float) -> None:
        max_left = self.initial_center_x - self.max_offset
        max_right = self.initial_center_x + self.max_offset
        measured_center = max(max_left, min(max_right, measured_center))
        center_delta = max(-self.max_center_step, min(self.max_center_step, measured_center - self.center_x))
        self.center_x = max(
            max_left,
            min(max_right, self.center_x + center_delta * self.center_smoothing),
        )
        width_delta = max(
            -self.max_width_step,
            min(self.max_width_step, measured_width_ratio - self.center_width_ratio),
        )
        self.center_width_ratio = max(
            self.min_width_ratio,
            min(
                self.max_width_ratio,
                self.center_width_ratio + width_delta * self.width_smoothing,
            ),
        )

    def _track_feature_center_x(self, image, proposal: ImageProposal) -> float | None:
        cv2 = _require_cv2()
        import numpy as np

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        measured_center = None
        if self.previous_gray is not None and self.tracking_points is not None:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray,
                gray,
                self.tracking_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            if next_points is not None and status is not None:
                good_points = next_points[status.reshape(-1) == 1]
                if len(good_points) >= self.min_tracking_points:
                    xs = good_points[:, 0, 0]
                    measured_center = float(np.median(xs) / max(width, 1))
                    self.tracking_points = good_points.reshape(-1, 1, 2)
                else:
                    self.tracking_points = None

        if self.tracking_points is None or len(self.tracking_points) < self.min_tracking_points:
            self.tracking_points = _good_features_in_subject_core(
                gray,
                proposal.mask,
                center_x=measured_center if measured_center is not None else self.center_x,
                width_ratio=self.feature_band_width_ratio,
            )

        self.previous_gray = gray
        return measured_center


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
    lower_body_width_ratio: float | None = None,
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
            lower_body_width_ratio=lower_body_width_ratio,
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
        subject_center_x = _estimate_subject_core_center_x(mask, center_x)
        return ImageProposal(
            mask=mask,
            roi=roi,
            candidate_count=small_proposal.candidate_count,
            center_x=center_x,
            center_width_ratio=center_width_ratio,
            subject_center_x=subject_center_x,
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
        lower_body_width_ratio=lower_body_width_ratio,
    )
    selected = _smooth_subject_shape(selected)
    selected = _keep_center_vertical_body_region(
        selected,
        center_x=center_x,
        image_width=width,
        image_height=height,
        body_width_ratio=vertical_body_width_ratio,
        lower_body_width_ratio=lower_body_width_ratio,
    )
    selected = _grabcut_refine_center_body(
        enhanced,
        selected,
        center_x=center_x,
        body_width_ratio=vertical_body_width_ratio,
        lower_body_width_ratio=lower_body_width_ratio,
        iterations=grabcut_iterations,
    )
    selected = _temporal_stabilize_mask(selected, previous_mask)
    if np.count_nonzero(selected) == 0:
        roi_width = width * center_width_ratio
        roi = RoiBox(width * center_x - roi_width / 2, 0, roi_width, height).clamped(width, height)
        subject_center_x = center_x
    else:
        roi = _mask_roi(selected, width, height)
        subject_center_x = _estimate_subject_core_center_x(selected, center_x)
    return ImageProposal(
        mask=selected,
        roi=roi,
        candidate_count=candidate_count,
        center_x=center_x,
        center_width_ratio=center_width_ratio,
        subject_center_x=subject_center_x,
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
    center_px = round(image.shape[1] * proposal.center_x)
    band_width = round(image.shape[1] * proposal.center_width_ratio)
    band_left = max(0, center_px - band_width // 2)
    band_right = min(image.shape[1] - 1, center_px + band_width // 2)
    cv2.line(overlay, (center_px, 0), (center_px, image.shape[0] - 1), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(overlay, (band_left, 0), (band_left, image.shape[0] - 1), (255, 180, 0), 1, cv2.LINE_AA)
    cv2.line(overlay, (band_right, 0), (band_right, image.shape[0] - 1), (255, 180, 0), 1, cv2.LINE_AA)
    x, y, width, height = proposal.roi.as_int_tuple()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (40, 40, 255), 2)
    cv2.putText(
        overlay,
        (
            "image proposal components: "
            f"{proposal.candidate_count} center_x={proposal.center_x:.3f} "
            f"width={proposal.center_width_ratio:.3f} "
            f"subject_x={(proposal.subject_center_x if proposal.subject_center_x is not None else proposal.center_x):.3f}"
        ),
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


def _proposal_roi_center_x(roi: RoiBox, image_width: int) -> float:
    return (roi.x + roi.width / 2) / max(image_width, 1)


def _estimate_subject_core_center_x(mask, previous_center_x: float) -> float:
    cv2 = _require_cv2()
    import numpy as np

    height, width = mask.shape[:2]
    if cv2.countNonZero(mask) == 0:
        return previous_center_x

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, round(width * 0.018)), max(17, round(height * 0.18))),
    )
    vertical_core = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if cv2.countNonZero(vertical_core) == 0:
        vertical_core = mask

    contours, _ = cv2.findContours(vertical_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _mask_center_x(mask) / width

    best_contour = None
    best_score = -1.0
    for contour in contours:
        x, y, contour_width, contour_height = cv2.boundingRect(contour)
        if contour_height < height * 0.18:
            continue
        component = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(component, [contour], -1, 255, -1)
        area = cv2.countNonZero(component)
        center = (x + contour_width / 2) / width
        center_distance = abs(center - previous_center_x)
        vertical_extent = contour_height / height
        compactness = contour_height / max(contour_width, 1)
        score = area * (1 + vertical_extent) * min(compactness, 6.0) / (1 + 12 * center_distance)
        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        return _mask_center_x(mask) / width

    component = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(component, [best_contour], -1, 255, -1)
    return _mask_center_x(component) / width


def _good_features_in_subject_core(gray, mask, center_x: float, width_ratio: float):
    cv2 = _require_cv2()

    height, width = mask.shape[:2]
    feature_mask = cv2.bitwise_and(mask, _center_band_mask(height, width, center_x, width_ratio))
    feature_mask = cv2.erode(
        feature_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    )
    if cv2.countNonZero(feature_mask) == 0:
        return None
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=80,
        qualityLevel=0.01,
        minDistance=7,
        mask=feature_mask,
        blockSize=7,
    )


def _proposal_search_width_ratio(
    roi_width_ratio: float,
    min_width_ratio: float,
    max_width_ratio: float,
) -> float:
    target = roi_width_ratio * 1.35 + 0.12
    return max(min_width_ratio, min(max_width_ratio, target))


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("values cannot be empty.")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _keep_center_vertical_body_region(
    mask,
    center_x: float,
    image_width: int,
    image_height: int,
    body_width_ratio: float,
    lower_body_width_ratio: float | None = None,
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
    support_band = _body_envelope_mask(
        image_height,
        image_width,
        seed_center,
        body_width_ratio,
        lower_body_width_ratio=lower_body_width_ratio,
    )
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


def _body_envelope_mask(
    height: int,
    width: int,
    center_x: float,
    body_width_ratio: float,
    lower_body_width_ratio: float | None = None,
):
    import numpy as np

    envelope = np.zeros((height, width), dtype=np.uint8)
    center_px = width * center_x
    lower_ratio = body_width_ratio if lower_body_width_ratio is None else lower_body_width_ratio
    upper_left = width * max(0.13, body_width_ratio * 0.72)
    upper_right = width * max(0.16, body_width_ratio * 0.82)
    lower_left = width * max(0.075, lower_ratio * 0.48)
    lower_right = width * max(0.14, lower_ratio * 0.82)
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
    lower_body_width_ratio: float | None,
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
    seed_envelope = _body_envelope_mask(
        height,
        width,
        seed_center,
        body_width_ratio * 0.72,
        lower_body_width_ratio=(
            None
            if lower_body_width_ratio is None
            else lower_body_width_ratio * 0.86
        ),
    )
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
