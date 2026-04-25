"""Run one clip through one condition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClipRunRequest:
    clip_id: str
    condition_id: str
    config_path: str


def run_clip(request: ClipRunRequest) -> None:
    raise NotImplementedError("Clip execution will be implemented after stage modules are ready.")
