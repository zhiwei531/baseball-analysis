from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from baseball_pose.pipeline.report_summary import build_report_summary


def test_build_report_summary_maps_available_partial_and_unavailable_metrics() -> None:
    reference_path = Path(__file__).resolve().parents[1] / "data/metadata/llm_biomechanics_reference.yaml"
    with reference_path.open("r", encoding="utf-8") as handle:
        reference = yaml.safe_load(handle)
    reference["_reference_path"] = str(reference_path)

    rows = [
        {
            "clip_id": "pitching_1",
            "condition_id": "image_center_motion_grabcut_pose_smooth",
            "frame_index": "0",
            "timestamp_sec": "0.0",
            "left_elbow_angle": "100.0",
            "right_elbow_angle": "90.0",
            "left_shoulder_angle": "120.0",
            "right_shoulder_angle": "110.0",
            "left_knee_angle": "170.0",
            "right_knee_angle": "160.0",
            "trunk_tilt_deg": "2.0",
            "pelvis_rotation_deg": "10.0",
            "shoulder_rotation_deg": "20.0",
            "hip_shoulder_separation_deg": "30.0",
            "pelvis_rotation_velocity_deg_s": "500.0",
            "trunk_rotation_velocity_deg_s": "900.0",
            "center_of_mass_x": "0.40",
            "center_of_mass_y": "0.60",
            "left_knee_extension_from_start_deg": "0.0",
            "right_knee_extension_from_start_deg": "0.0",
            "left_knee_angular_velocity_deg_s": "0.0",
            "right_knee_angular_velocity_deg_s": "0.0",
            "left_wrist_x": "0.50",
            "left_wrist_y": "0.20",
            "right_wrist_x": "0.45",
            "right_wrist_y": "0.18",
            "left_wrist_speed": "0.0",
            "right_wrist_speed": "0.0",
            "hand_speed_proxy": "20.0",
        },
        {
            "clip_id": "pitching_1",
            "condition_id": "image_center_motion_grabcut_pose_smooth",
            "frame_index": "1",
            "timestamp_sec": "0.03",
            "left_elbow_angle": "105.0",
            "right_elbow_angle": "92.0",
            "left_shoulder_angle": "125.0",
            "right_shoulder_angle": "112.0",
            "left_knee_angle": "168.0",
            "right_knee_angle": "155.0",
            "trunk_tilt_deg": "3.0",
            "pelvis_rotation_deg": "12.0",
            "shoulder_rotation_deg": "24.0",
            "hip_shoulder_separation_deg": "35.0",
            "pelvis_rotation_velocity_deg_s": "620.0",
            "trunk_rotation_velocity_deg_s": "980.0",
            "center_of_mass_x": "0.42",
            "center_of_mass_y": "0.61",
            "left_knee_extension_from_start_deg": "2.0",
            "right_knee_extension_from_start_deg": "5.0",
            "left_knee_angular_velocity_deg_s": "10.0",
            "right_knee_angular_velocity_deg_s": "15.0",
            "left_wrist_x": "0.51",
            "left_wrist_y": "0.21",
            "right_wrist_x": "0.44",
            "right_wrist_y": "0.17",
            "left_wrist_speed": "10.0",
            "right_wrist_speed": "12.0",
            "hand_speed_proxy": "25.0",
        },
    ]

    summary = build_report_summary(
        clip_id="pitching_1",
        condition_id="image_center_motion_grabcut_pose_smooth",
        action_type="pitching",
        rows=rows,
        reference=reference,
        athlete_group="high_school_pitcher",
        source_path=Path("data_full/processed/features/pitching_1/image_center_motion_grabcut_pose_smooth.csv"),
    )

    mapping = summary["standard_metric_mapping"]

    assert mapping["peak_trunk_rotation_velocity_deg_s"]["status"] == "available"
    assert mapping["peak_trunk_rotation_velocity_deg_s"]["observed_value"] == pytest.approx(976.0)
    assert mapping["hip_shoulder_separation_foot_contact_deg"]["status"] == "partial"
    assert mapping["lead_knee_flexion_foot_contact_deg"]["status"] == "unavailable"
    assert summary["derived_metric_summaries"]["left_knee_flexion_deg"]["max"] == 12.0
    assert summary["llm_ready_summary"]["highlights"]
