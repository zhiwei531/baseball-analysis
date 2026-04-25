"""Transform composition helpers."""

from __future__ import annotations

from typing import Any

from baseball_pose.preprocessing.enhance import apply_transform


def apply_transforms(image: Any, transforms: list[dict[str, Any]]) -> Any:
    result = image
    for transform in transforms:
        result = apply_transform(result, transform)
    return result
