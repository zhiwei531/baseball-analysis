"""Configuration helpers for image-proposal ROI conditions."""

from __future__ import annotations

from typing import Any


def image_proposal_roi_config(
    raw_config: dict[str, Any],
    clip_id: str,
    condition_id: str = "image_center_motion_grabcut_pose",
) -> dict[str, Any]:
    """Return ROI config with optional clip-specific overrides applied."""

    base_config = (
        raw_config.get("conditions", {})
        .get(condition_id, {})
        .get("roi", {})
    )
    merged = {
        key: value
        for key, value in base_config.items()
        if key != "clip_overrides"
    }
    clip_overrides = base_config.get("clip_overrides", {})
    if isinstance(clip_overrides, dict):
        override = clip_overrides.get(clip_id, {})
        if isinstance(override, dict):
            merged.update(override)
    return merged
