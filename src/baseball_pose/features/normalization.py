"""Coordinate and metric normalization helpers."""

from __future__ import annotations


def normalize_distance(distance: float, scale: float) -> float:
    if scale <= 0:
        raise ValueError("scale must be positive.")
    return distance / scale
