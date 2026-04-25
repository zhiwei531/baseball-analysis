"""Image enhancement function boundaries."""

from __future__ import annotations

from typing import Any


def apply_transform(image: Any, transform: dict[str, Any]) -> Any:
    """Apply one configured image transform.

    Concrete OpenCV operations will be added after the ingestion baseline is in place.
    """

    name = transform.get("name")
    if name not in {"clahe", "denoise", "sharpen", "deblur"}:
        raise ValueError(f"Unsupported transform: {name}")
    raise NotImplementedError(f"Transform not implemented yet: {name}")
