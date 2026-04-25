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
