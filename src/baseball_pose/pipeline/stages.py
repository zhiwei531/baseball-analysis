"""Pipeline stage names."""

from __future__ import annotations

from enum import StrEnum


class PipelineStage(StrEnum):
    LOAD_METADATA = "load_metadata"
    SAMPLE_FRAMES = "sample_frames"
    APPLY_ROI = "apply_roi"
    APPLY_PREPROCESSING = "apply_preprocessing"
    ESTIMATE_POSE = "estimate_pose"
    POSTPROCESS_POSE = "postprocess_pose"
    EXTRACT_FEATURES = "extract_features"
    EVALUATE_METRICS = "evaluate_metrics"
    WRITE_VISUALIZATIONS = "write_visualizations"
