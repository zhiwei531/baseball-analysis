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
    ESTIMATE_POSE_3D = "estimate_pose_3d"
    EXTRACT_FEATURES = "extract_features"
    EXTRACT_FEATURES_3D = "extract_features_3d"
    EVALUATE_METRICS = "evaluate_metrics"
    WRITE_VISUALIZATIONS = "write_visualizations"
    WRITE_VISUALIZATIONS_3D = "write_visualizations_3d"
