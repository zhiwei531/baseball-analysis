"""Manual event-anchor structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventAnchor:
    clip_id: str
    event_name: str
    frame_index: int
    notes: str = ""
