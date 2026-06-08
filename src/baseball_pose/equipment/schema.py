"""Schema for tracked baseball equipment objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectTrackRecord:
    clip_id: str
    condition_id: str
    frame_index: int
    timestamp_sec: float
    object_name: str
    x: float | None
    y: float | None
    x2: float | None
    y2: float | None
    confidence: float | None
    width: int | None
    height: int | None
    source: str


@dataclass(frozen=True)
class ObjectFeatureRow:
    clip_id: str
    condition_id: str
    frame_index: int
    timestamp_sec: float
    bat_barrel_x: float | None
    bat_barrel_y: float | None
    bat_handle_x: float | None
    bat_handle_y: float | None
    bat_speed_px_s: float | None
    bat_speed_norm_s: float | None
    bat_angle_deg: float | None
    ball_x: float | None
    ball_y: float | None
    ball_speed_px_s: float | None
    ball_speed_norm_s: float | None
