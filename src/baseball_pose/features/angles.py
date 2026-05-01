"""2D joint-angle helpers."""

from __future__ import annotations

import math


Point = tuple[float, float]


def angle_degrees(a: Point, b: Point, c: Point) -> float:
    """Compute angle ABC in degrees."""

    bax = a[0] - b[0]
    bay = a[1] - b[1]
    bcx = c[0] - b[0]
    bcy = c[1] - b[1]
    dot = bax * bcx + bay * bcy
    norm_ab = math.hypot(bax, bay)
    norm_cb = math.hypot(bcx, bcy)
    if norm_ab == 0 or norm_cb == 0:
        raise ValueError("Cannot compute angle with duplicate points.")
    cosine = max(-1.0, min(1.0, dot / (norm_ab * norm_cb)))
    return math.degrees(math.acos(cosine))


def segment_angle_degrees(a: Point, b: Point) -> float:
    """Compute a 2D segment orientation in image coordinates, using y-up degrees."""

    dx = b[0] - a[0]
    dy = -(b[1] - a[1])
    if dx == 0 and dy == 0:
        raise ValueError("Cannot compute orientation for duplicate points.")
    return math.degrees(math.atan2(dy, dx))


def signed_angle_delta_degrees(start: float, end: float) -> float:
    """Return the signed shortest angular change from start to end."""

    return (end - start + 180) % 360 - 180
