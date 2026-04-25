"""Run an experiment matrix."""

from __future__ import annotations

from baseball_pose.config import RuntimeConfig


def planned_runs(config: RuntimeConfig) -> list[tuple[str, str]]:
    return [(clip_id, condition_id) for clip_id in config.clip_ids for condition_id in config.condition_ids]
