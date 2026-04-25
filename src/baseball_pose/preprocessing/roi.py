"""Manual ROI data structures and coordinate helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoiBox:
    x: float
    y: float
    width: float
    height: float
    coordinate_space: str = "pixel"

    def expanded(self, factor: float) -> "RoiBox":
        dx = self.width * factor / 2
        dy = self.height * factor / 2
        return RoiBox(
            x=self.x - dx,
            y=self.y - dy,
            width=self.width * (1 + factor),
            height=self.height * (1 + factor),
            coordinate_space=self.coordinate_space,
        )
