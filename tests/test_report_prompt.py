from __future__ import annotations

from baseball_pose.pipeline.report_prompt import _build_prompt_payload, _build_prompt_text


def test_prompt_payload_keeps_selected_metrics_and_constraints() -> None:
    summary = {
        "clip_id": "pitching_1",
        "condition_id": "cond_a",
        "action_type": "pitching",
        "athlete_group": "high_school_pitcher",
        "measurement_source": "2d_pose_video",
        "frame_count": 100,
        "standard_metric_mapping": {
            "peak_trunk_rotation_velocity_deg_s": {
                "status": "available",
                "observed_value": 920.0,
                "observed_summary_key": "p95_abs",
                "observed_summary": {"p95_abs": 920.0},
                "comparison": {"mode": "mean_sd", "band": "within_1sd", "mean": 959.0},
            },
            "peak_pelvis_rotation_velocity_deg_s": {
                "status": "available",
                "observed_value": 610.0,
                "observed_summary_key": "p95_abs",
                "observed_summary": {"p95_abs": 610.0},
                "comparison": {"mode": "mean_sd", "band": "within_1sd", "mean": 596.0},
            },
            "hip_shoulder_separation_foot_contact_deg": {
                "status": "partial",
                "observed_value": 48.0,
                "observed_summary_key": "p95_abs",
                "observed_summary": {"p95_abs": 48.0},
                "note": "Proxy only.",
            },
            "observed_left_knee_flexion_deg": {
                "status": "available",
                "observed_summary": {"max": 12.0},
            },
            "observed_hand_speed_proxy": {
                "status": "available",
                "observed_summary": {"max": 1.2},
            },
            "lead_knee_flexion_foot_contact_deg": {
                "status": "unavailable",
            },
        },
        "llm_ready_summary": {
            "limitations": ["Event timing is not available."],
            "forbidden_claims": ["Do not diagnose injury."],
            "preferred_phrasing": ["lower than common age-matched reference"],
        },
    }

    payload = _build_prompt_payload(summary)
    prompt_text = _build_prompt_text(payload)

    assert payload["clip_context"]["clip_id"] == "pitching_1"
    assert len(payload["selected_metrics"]) >= 3
    assert "Do not diagnose injury." in payload["report_constraints"]["must_not_do"]
    assert "Input JSON" in prompt_text
