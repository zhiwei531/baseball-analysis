"""Condition comparison records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricRecord:
    clip_id: str
    condition_id: str
    metric_name: str
    joint_group: str
    value: float
    aggregation: str
    notes: str = ""
