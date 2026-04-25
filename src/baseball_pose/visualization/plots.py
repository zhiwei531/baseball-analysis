"""Plotting boundaries for report figures."""

from __future__ import annotations


def figure_name(clip_id: str, condition_id: str, figure_type: str) -> str:
    return f"{clip_id}__{condition_id}__{figure_type}.png"
