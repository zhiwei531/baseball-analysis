"""Diagnostic artifact naming helpers."""

from __future__ import annotations


def diagnostic_frame_name(clip_id: str, condition_id: str, frame_index: int) -> str:
    return f"{clip_id}__{condition_id}__frame_{frame_index:06d}.png"
