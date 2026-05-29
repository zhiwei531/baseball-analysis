"""Build LLM-ready summaries by mapping feature CSV outputs to reference metrics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only before dependencies are installed.
    yaml = None

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.feature_csv import read_feature_rows
from baseball_pose.io.metadata import load_clips
from baseball_pose.io.paths import feature_path, report_summary_path
from baseball_pose.pipeline.report_window import filter_rows_to_action_window


AVAILABLE_MEASUREMENT_SOURCE = "2d_pose_video"
DEFAULT_REFERENCE_PATH = Path("data/metadata/llm_biomechanics_reference.yaml")
PUBLIC_METRIC_METADATA = {
    "peak_trunk_rotation_velocity_deg_s": {
        "label_cn": "躯干旋转速度代理值",
        "label_en": "Trunk rotation speed proxy",
        "unit": "deg/s",
        "audience_explanation_cn": "反映上半身转动速度，适合用于同一位运动员前后对比。",
    },
    "peak_pelvis_rotation_velocity_deg_s": {
        "label_cn": "骨盆旋转速度代理值",
        "label_en": "Pelvis rotation speed proxy",
        "unit": "deg/s",
        "audience_explanation_cn": "反映下半身带动动作的速度，适合观察动力链是否积极。",
    },
    "hip_shoulder_separation_foot_contact_deg": {
        "label_cn": "髋肩分离代理值",
        "label_en": "Hip-shoulder separation proxy",
        "unit": "deg",
        "audience_explanation_cn": "反映髋部和肩部错开的程度，但当前仍是整段视频代理值。",
    },
    "observed_left_knee_flexion_deg": {
        "label_cn": "左膝弯曲深度",
        "label_en": "Left knee bend depth",
        "unit": "deg",
        "audience_explanation_cn": "数值越大，说明视频中观察到的膝盖弯曲越明显。",
    },
    "observed_right_knee_flexion_deg": {
        "label_cn": "右膝弯曲深度",
        "label_en": "Right knee bend depth",
        "unit": "deg",
        "audience_explanation_cn": "可与左侧一起看，判断下肢动作是否接近。",
    },
    "observed_hand_speed_proxy": {
        "label_cn": "双手速度代理值",
        "label_en": "Hand speed proxy",
        "unit": "px/s",
        "audience_explanation_cn": "反映手部末端移动快慢，适合做同一设备条件下的前后比较。",
    },
}


@dataclass(frozen=True)
class ReportSummaryResult:
    clip_id: str
    condition_id: str
    summary_path: Path
    frame_count: int
    available_metric_count: int
    partial_metric_count: int
    unavailable_metric_count: int


def build_report_summaries(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str] | None = None,
    athlete_group: str | None = None,
    reference_path: str | Path = DEFAULT_REFERENCE_PATH,
) -> list[ReportSummaryResult]:
    """Build structured JSON summaries for LLM prompting."""

    reference = _read_yaml(Path(reference_path))
    clips = {clip.clip_id: clip for clip in load_clips(config.clips_file)}
    condition_ids = conditions if conditions is not None else config.condition_ids
    results: list[ReportSummaryResult] = []

    for clip_id in clip_ids:
        clip = clips.get(clip_id)
        if clip is None:
            continue
        for condition_id in condition_ids:
            source_path = feature_path(config.data_dir, clip_id, condition_id)
            if not source_path.exists():
                continue
            rows = read_feature_rows(source_path)
            summary = build_report_summary(
                clip_id=clip_id,
                condition_id=condition_id,
                action_type=clip.action_type,
                rows=rows,
                reference=reference,
                athlete_group=athlete_group,
                source_path=source_path,
            )
            output_path = report_summary_path(config.data_dir, clip_id, condition_id)
            _write_json(output_path, summary)
            mapping = summary["standard_metric_mapping"]
            results.append(
                ReportSummaryResult(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    summary_path=output_path,
                    frame_count=int(summary["frame_count"]),
                    available_metric_count=_count_status(mapping, "available"),
                    partial_metric_count=_count_status(mapping, "partial"),
                    unavailable_metric_count=_count_status(mapping, "unavailable"),
                )
            )

    return results


def build_report_summary(
    clip_id: str,
    condition_id: str,
    action_type: str,
    rows: list[dict[str, str]],
    reference: dict[str, Any],
    athlete_group: str | None,
    source_path: str | Path,
) -> dict[str, Any]:
    """Return one LLM-ready summary payload."""

    parsed_rows = [_parse_feature_row(row) for row in rows]
    action_rows, action_window = filter_rows_to_action_window(
        parsed_rows,
        action_type=action_type,
        expanded=(action_type == "batting"),
    )
    window_rows = action_rows if action_rows else parsed_rows
    feature_stats = _summarize_feature_rows(window_rows)
    derived_stats = _summarize_derived_metrics(window_rows)
    metric_mapping = _map_standard_metrics(
        feature_stats=feature_stats,
        derived_stats=derived_stats,
        reference=reference,
        action_type=action_type,
        athlete_group=athlete_group,
    )
    highlights = _build_highlights(metric_mapping)
    limitations = _build_limitations(metric_mapping, action_type)
    selected_reference_sources = _collect_reference_sources(reference, metric_mapping)
    injury_prevention_context = _build_injury_prevention_context(reference, action_type, metric_mapping)
    output_metric_inventory = _build_output_metric_inventory(feature_stats, derived_stats)
    public_metric_cards = _build_public_metric_cards(metric_mapping, reference.get("source_index", {}))

    return {
        "clip_id": clip_id,
        "condition_id": condition_id,
        "action_type": action_type,
        "athlete_group": athlete_group or "unspecified",
        "measurement_source": AVAILABLE_MEASUREMENT_SOURCE,
        "frame_count": len(window_rows),
        "total_frame_count": len(parsed_rows),
        "action_window": (
            {
                "start_frame": action_window.start_frame,
                "end_frame": action_window.end_frame,
                "start_time_sec": action_window.start_time_sec,
                "end_time_sec": action_window.end_time_sec,
                "peak_frame": action_window.peak_frame,
                "frame_count": action_window.frame_count,
            }
            if action_window is not None
            else None
        ),
        "source_feature_csv": str(source_path),
        "reference_file": str(reference.get("_reference_path", "")),
        "feature_field_summaries": feature_stats,
        "derived_metric_summaries": derived_stats,
        "standard_metric_mapping": metric_mapping,
        "selected_reference_sources": selected_reference_sources,
        "injury_prevention_context": injury_prevention_context,
        "output_metric_inventory": output_metric_inventory,
        "public_metric_cards": public_metric_cards,
        "llm_ready_summary": {
            "highlights": highlights,
            "limitations": limitations,
            "preferred_phrasing": reference.get("interpretation_rules", {}).get(
                "preferred_phrasing",
                [],
            ),
            "forbidden_claims": reference.get("interpretation_rules", {}).get(
                "forbidden_claims",
                [],
            ),
        },
    }


def _map_standard_metrics(
    feature_stats: dict[str, Any],
    derived_stats: dict[str, Any],
    reference: dict[str, Any],
    action_type: str,
    athlete_group: str | None,
) -> dict[str, dict[str, Any]]:
    measurements = {
        "peak_pelvis_rotation_velocity_deg_s": _available_measurement(
            feature_stats,
            field_name="pelvis_rotation_velocity_deg_s",
            summary_key="p95_abs",
            comparison_mode="cohort_mean_sd",
            metric_ref=_reference_metric(reference, "video_kinematics_pitching", "peak_pelvis_rotation_velocity_deg_s"),
            athlete_group=athlete_group,
        ),
        "peak_trunk_rotation_velocity_deg_s": _available_measurement(
            feature_stats,
            field_name="trunk_rotation_velocity_deg_s",
            summary_key="p95_abs",
            comparison_mode="cohort_mean_sd",
            metric_ref=_reference_metric(reference, "video_kinematics_pitching", "peak_trunk_rotation_velocity_deg_s"),
            athlete_group=athlete_group,
        ),
        "hip_shoulder_separation_foot_contact_deg": _partial_measurement(
            feature_stats,
            field_name="hip_shoulder_separation_deg",
            observed_key="p95_abs",
            metric_ref=_reference_metric(reference, "video_kinematics_pitching", "hip_shoulder_separation_foot_contact_deg"),
            note=(
                "Current pipeline summarizes full-clip separation and does not detect foot contact, "
                "so this is only a partial proxy for the foot-contact reference."
            ),
        ),
        "lead_knee_flexion_foot_contact_deg": _unavailable_measurement(
            reference,
            "video_kinematics_pitching",
            "lead_knee_flexion_foot_contact_deg",
            "Current feature CSV stores left/right knee angles, but it does not identify lead side or foot-contact timing.",
        ),
        "lead_knee_flexion_ball_release_deg": _unavailable_measurement(
            reference,
            "video_kinematics_pitching",
            "lead_knee_flexion_ball_release_deg",
            "Current feature CSV stores left/right knee angles, but it does not identify lead side or ball-release timing.",
        ),
        "stride_length_pct_height": _unavailable_measurement(
            reference,
            "video_kinematics_pitching",
            "stride_length_pct_height",
            "Stride length is not currently extracted from the feature CSV.",
        ),
        "elbow_flexion_foot_contact_deg": _unavailable_measurement(
            reference,
            "video_kinematics_pitching",
            "elbow_flexion_foot_contact_deg",
            "Current feature CSV stores left/right elbow angles, but it does not detect foot contact timing.",
        ),
        "shoulder_abduction_foot_contact_deg": _unavailable_measurement(
            reference,
            "video_kinematics_pitching",
            "shoulder_abduction_foot_contact_deg",
            "Shoulder abduction at foot contact is not currently represented in the feature CSV.",
        ),
        "shoulder_horizontal_abduction_foot_contact_deg": _unavailable_measurement(
            reference,
            "video_kinematics_pitching",
            "shoulder_horizontal_abduction_foot_contact_deg",
            "Shoulder horizontal abduction at foot contact is not currently represented in the feature CSV.",
        ),
        "shoulder_external_rotation_foot_contact_deg": _unavailable_measurement(
            reference,
            "video_kinematics_pitching",
            "shoulder_external_rotation_foot_contact_deg",
            "Shoulder external rotation at foot contact is not currently represented in the feature CSV.",
        ),
        "max_shoulder_external_rotation_deg": _unavailable_measurement(
            reference,
            "video_kinematics_pitching",
            "max_shoulder_external_rotation_deg",
            "Current feature CSV does not include shoulder external rotation as a validated measurement.",
        ),
        "shoulder_internal_rotation_90_deg": _unavailable_measurement(
            reference,
            "clinical_rom",
            "shoulder_internal_rotation_90_deg",
            "Clinical passive shoulder ROM is not available from pose-only feature CSV outputs.",
        ),
        "shoulder_external_rotation_90_deg": _unavailable_measurement(
            reference,
            "clinical_rom",
            "shoulder_external_rotation_90_deg",
            "Clinical passive shoulder ROM is not available from pose-only feature CSV outputs.",
        ),
        "shoulder_total_arc_deg": _unavailable_measurement(
            reference,
            "clinical_rom",
            "shoulder_total_arc_deg",
            "Shoulder total arc requires passive IR and ER measurements rather than pose-only video outputs.",
        ),
        "shoulder_clinical_gird_deg": _unavailable_measurement(
            reference,
            "clinical_rom",
            "shoulder_clinical_gird_deg",
            "GIRD requires passive side-to-side shoulder internal rotation measurements.",
        ),
        "shoulder_total_rotational_motion_deficit_deg": _unavailable_measurement(
            reference,
            "clinical_rom",
            "shoulder_total_rotational_motion_deficit_deg",
            "Total rotational motion deficit requires passive shoulder ROM measurements.",
        ),
        "elbow_extension_side_difference_deg": _unavailable_measurement(
            reference,
            "clinical_rom",
            "elbow_extension_side_difference_deg",
            "Clinical elbow extension difference is not available from pose-only feature CSV outputs.",
        ),
        "elbow_flexion_side_difference_deg": _unavailable_measurement(
            reference,
            "clinical_rom",
            "elbow_flexion_side_difference_deg",
            "Clinical elbow flexion difference is not available from pose-only feature CSV outputs.",
        ),
        "elbow_flexion_extension_arc_side_difference_deg": _unavailable_measurement(
            reference,
            "clinical_rom",
            "elbow_flexion_extension_arc_side_difference_deg",
            "Clinical elbow flexion-extension arc is not available from pose-only feature CSV outputs.",
        ),
        "hip_external_rotation_90_flexed_deg": _unavailable_measurement(
            reference,
            "clinical_rom",
            "hip_external_rotation_90_flexed_deg",
            "Clinical passive hip ROM is not available from pose-only feature CSV outputs.",
        ),
        "hip_internal_rotation_90_flexed_deg": _unavailable_measurement(
            reference,
            "clinical_rom",
            "hip_internal_rotation_90_flexed_deg",
            "Clinical passive hip ROM is not available from pose-only feature CSV outputs.",
        ),
        "hip_total_rotation_arc_deg": _unavailable_measurement(
            reference,
            "clinical_rom",
            "hip_total_rotation_arc_deg",
            "Hip total rotation arc requires passive IR and ER measurements rather than pose-only video outputs.",
        ),
        "trunk_rotation_deg": _partial_measurement(
            feature_stats,
            field_name="shoulder_rotation_deg",
            observed_key="range",
            metric_ref=_reference_metric(reference, "clinical_rom", "trunk_rotation_deg"),
            note=(
                "Current feature CSV measures dynamic shoulder-line rotation across the clip, "
                "not clinical passive trunk rotation with a fixed pelvis."
            ),
        ),
    }

    if action_type != "pitching":
        for metric_name, mapped in list(measurements.items()):
            if mapped["status"] == "available":
                mapped["note"] = (
                    f"{mapped.get('note', '').strip()} This reference was derived from pitching literature; "
                    "use cautiously for batting."
                ).strip()
            elif mapped["status"] == "partial":
                mapped["note"] = (
                    f"{mapped.get('note', '').strip()} Reference interpretation is pitching-specific."
                ).strip()

    measurements["observed_left_knee_flexion_deg"] = _observation_only(derived_stats, "left_knee_flexion_deg")
    measurements["observed_right_knee_flexion_deg"] = _observation_only(derived_stats, "right_knee_flexion_deg")
    measurements["observed_hand_speed_proxy"] = _observation_only(feature_stats, "hand_speed_proxy")
    return measurements


def _observation_only(summary_group: dict[str, Any], metric_name: str) -> dict[str, Any]:
    summary = summary_group.get(metric_name)
    if not summary:
        return {
            "status": "unavailable",
            "match_type": "observation_only",
            "reason": "Metric not found in current feature summary.",
        }
    return {
        "status": "available",
        "match_type": "observation_only",
        "observed_summary": summary,
        "reason": "Observed from current feature CSV but not matched to a standard reference metric.",
    }


def _available_measurement(
    feature_stats: dict[str, Any],
    field_name: str,
    summary_key: str,
    comparison_mode: str,
    metric_ref: dict[str, Any] | None,
    athlete_group: str | None,
) -> dict[str, Any]:
    summary = feature_stats.get(field_name)
    if not summary or summary.get("valid_count", 0) == 0:
        return {
            "status": "unavailable",
            "match_type": "exact",
            "reason": f"{field_name} is missing from the current feature CSV summary.",
        }
    observed_value = summary.get(summary_key)
    reference_context = _resolve_reference_context(metric_ref, comparison_mode, athlete_group)
    return {
        "status": "available",
        "match_type": "exact",
        "observed_value": observed_value,
        "observed_summary_key": summary_key,
        "observed_summary": summary,
        "reference_context": reference_context,
        "comparison": _compare_to_reference(observed_value, reference_context),
        "source_field": field_name,
    }


def _partial_measurement(
    feature_stats: dict[str, Any],
    field_name: str,
    observed_key: str,
    metric_ref: dict[str, Any] | None,
    note: str,
) -> dict[str, Any]:
    summary = feature_stats.get(field_name)
    if not summary or summary.get("valid_count", 0) == 0:
        return {
            "status": "unavailable",
            "match_type": "partial",
            "reason": f"{field_name} is missing from the current feature CSV summary.",
        }
    return {
        "status": "partial",
        "match_type": "proxy",
        "observed_value": summary.get(observed_key),
        "observed_summary_key": observed_key,
        "observed_summary": summary,
        "reference_context": metric_ref,
        "source_field": field_name,
        "note": note,
    }


def _unavailable_measurement(
    reference: dict[str, Any],
    section_name: str,
    metric_name: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "match_type": "none",
        "reference_context": _reference_metric(reference, section_name, metric_name),
        "reason": reason,
    }


def _reference_metric(reference: dict[str, Any], section_name: str, metric_name: str) -> dict[str, Any] | None:
    return reference.get("metrics", {}).get(section_name, {}).get(metric_name)


def _resolve_reference_context(
    metric_ref: dict[str, Any] | None,
    comparison_mode: str,
    athlete_group: str | None,
) -> dict[str, Any]:
    if metric_ref is None:
        return {}
    if comparison_mode == "cohort_mean_sd":
        cohort_reference = metric_ref.get("cohort_reference")
        if cohort_reference:
            first_key = next(iter(cohort_reference))
            return {
                "kind": "cohort_mean_sd",
                "cohort_name": first_key,
                "source_ids": metric_ref.get("source_ids", []),
                **cohort_reference[first_key],
            }
    if athlete_group:
        cohort = metric_ref.get("cohorts", {}).get(athlete_group)
        if cohort:
            return {
                "kind": "athlete_group",
                "athlete_group": athlete_group,
                "source_ids": metric_ref.get("source_ids", []),
                **cohort,
            }
    return metric_ref


def _compare_to_reference(observed_value: float | None, reference_context: dict[str, Any]) -> dict[str, Any] | None:
    if observed_value is None:
        return None
    if "mean_sd" in reference_context:
        mean_value, std_value = reference_context["mean_sd"]
        return _sd_band_assessment(observed_value, mean_value, std_value)
    if "target_reference" in reference_context and "mean" in reference_context["target_reference"]:
        target = float(reference_context["target_reference"]["mean"])
        return {
            "mode": "target_reference",
            "target": target,
            "delta": observed_value - target,
        }
    return None


def _sd_band_assessment(observed_value: float, mean_value: float, std_value: float) -> dict[str, Any]:
    lower = mean_value - std_value
    upper = mean_value + std_value
    if observed_value < lower:
        band = "below_1sd"
    elif observed_value > upper:
        band = "above_1sd"
    else:
        band = "within_1sd"
    return {
        "mode": "mean_sd",
        "mean": mean_value,
        "std": std_value,
        "lower_1sd": lower,
        "upper_1sd": upper,
        "band": band,
        "delta_from_mean": observed_value - mean_value,
    }


def _build_highlights(metric_mapping: dict[str, dict[str, Any]]) -> list[str]:
    highlights: list[str] = []
    trunk = metric_mapping.get("peak_trunk_rotation_velocity_deg_s", {})
    if trunk.get("status") == "available":
        band = trunk.get("comparison", {}).get("band")
        value = trunk.get("observed_value")
        if band and value is not None:
            highlights.append(
                f"Peak trunk rotation velocity is {value:.1f} deg/s and falls {band.replace('_', ' ')} relative to the available pitching reference."
            )
    pelvis = metric_mapping.get("peak_pelvis_rotation_velocity_deg_s", {})
    if pelvis.get("status") == "available":
        band = pelvis.get("comparison", {}).get("band")
        value = pelvis.get("observed_value")
        if band and value is not None:
            highlights.append(
                f"Peak pelvis rotation velocity is {value:.1f} deg/s and falls {band.replace('_', ' ')} relative to the available pitching reference."
            )
    separation = metric_mapping.get("hip_shoulder_separation_foot_contact_deg", {})
    if separation.get("status") == "partial":
        value = separation.get("observed_value")
        if value is not None:
            highlights.append(
                f"Clip-level peak hip-shoulder separation proxy is {value:.1f} deg, but it is not a true foot-contact event measurement."
            )
    return highlights


def _build_limitations(metric_mapping: dict[str, dict[str, Any]], action_type: str) -> list[str]:
    limitations = [
        "Current summary is built from 2D pose-derived feature CSV outputs.",
        "Clinical passive ROM metrics remain unavailable unless separate physical screening data are provided.",
    ]
    if action_type == "pitching":
        limitations.append(
            "Event-specific pitching metrics such as foot contact and ball release remain unavailable until event detection is added."
        )
    return limitations


def _build_output_metric_inventory(
    feature_stats: dict[str, Any],
    derived_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    inventory = [
        {
            "category_cn": "关节点坐标",
            "category_en": "Joint coordinates",
            "examples": ["left_wrist_x", "left_wrist_y", "right_wrist_x", "right_wrist_y", "center_of_mass_x", "center_of_mass_y"],
            "plain_language_cn": "描述手部和身体中心在画面中的位置变化。",
        },
        {
            "category_cn": "关节角度",
            "category_en": "Joint angles",
            "examples": ["left_elbow_angle", "right_elbow_angle", "left_shoulder_angle", "right_shoulder_angle", "left_knee_angle", "right_knee_angle"],
            "plain_language_cn": "描述肘、肩、膝在动作中弯曲或打开的程度。",
        },
        {
            "category_cn": "旋转与姿态",
            "category_en": "Rotation and posture",
            "examples": ["trunk_tilt_deg", "pelvis_rotation_deg", "shoulder_rotation_deg", "hip_shoulder_separation_deg"],
            "plain_language_cn": "描述上半身倾斜、骨盆旋转以及髋肩之间的错开程度。",
        },
        {
            "category_cn": "速度指标",
            "category_en": "Velocity metrics",
            "examples": ["pelvis_rotation_velocity_deg_s", "trunk_rotation_velocity_deg_s", "left_wrist_speed", "right_wrist_speed", "hand_speed_proxy"],
            "plain_language_cn": "描述身体旋转和手部移动的快慢。",
        },
        {
            "category_cn": "左右侧与下肢指标",
            "category_en": "Lower-body and side-to-side metrics",
            "examples": ["left_knee_flexion_deg", "right_knee_flexion_deg", "left_knee_extension_from_start_deg", "right_knee_extension_from_start_deg"],
            "plain_language_cn": "描述左右膝弯曲深度，以及下肢动作是否接近。",
        },
    ]
    available_keys = set(feature_stats) | set(derived_stats)
    return [
        {
            **item,
            "available_examples": [example for example in item["examples"] if example in available_keys],
        }
        for item in inventory
    ]


def _build_public_metric_cards(
    metric_mapping: dict[str, dict[str, Any]],
    source_index: dict[str, Any],
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for metric_name, metadata in PUBLIC_METRIC_METADATA.items():
        metric = metric_mapping.get(metric_name, {})
        if metric.get("status") == "unavailable":
            continue
        reference_context = metric.get("reference_context")
        source_ids = []
        if isinstance(reference_context, dict):
            source_ids = list(reference_context.get("source_ids", []))
        cards.append(
            {
                "metric_name": metric_name,
                "label_cn": metadata["label_cn"],
                "label_en": metadata["label_en"],
                "unit": metadata["unit"],
                "status": metric.get("status"),
                "observed_value": metric.get("observed_value"),
                "observed_summary": metric.get("observed_summary"),
                "comparison": metric.get("comparison"),
                "note": metric.get("note"),
                "audience_explanation_cn": metadata["audience_explanation_cn"],
                "reference_short_names": [
                    source_index[source_id].get("short_name", source_id)
                    for source_id in source_ids
                    if source_id in source_index
                ],
            }
        )
    return cards


def _collect_reference_sources(
    reference: dict[str, Any],
    metric_mapping: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    source_index = reference.get("source_index", {})
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for metric in metric_mapping.values():
        context = metric.get("reference_context")
        if not isinstance(context, dict):
            continue
        for source_id in context.get("source_ids", []):
            if source_id in source_index and source_id not in seen:
                seen.add(source_id)
                ordered_ids.append(source_id)
    return [
        {
            "source_id": source_id,
            "short_name": source_index[source_id].get("short_name", source_id),
            "citation": source_index[source_id].get("citation", ""),
            "focus": source_index[source_id].get("focus", ""),
            "urls": source_index[source_id].get("urls", []),
            "local_note_path": source_index[source_id].get("local_note_path"),
        }
        for source_id in ordered_ids
    ]


def _build_injury_prevention_context(
    reference: dict[str, Any],
    action_type: str,
    metric_mapping: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    source_index = reference.get("source_index", {})
    trunk = metric_mapping.get("peak_trunk_rotation_velocity_deg_s", {})
    pelvis = metric_mapping.get("peak_pelvis_rotation_velocity_deg_s", {})
    separation = metric_mapping.get("hip_shoulder_separation_foot_contact_deg", {})

    notes: list[dict[str, Any]] = []
    if trunk.get("status") == "available" or pelvis.get("status") == "available":
        notes.append(
            _injury_note(
                source_index,
                title="Monitor trunk and pelvis sequencing",
                guidance=(
                    "When trunk and pelvis rotation proxies are consistently low, use coaching to improve "
                    "timing and force transfer before simply asking the athlete to swing harder."
                ),
                rationale=(
                    "Published baseball biomechanics reviews link pelvis and trunk rotational contribution "
                    "to performance output, but this 2D pipeline cannot diagnose injury from rotation speed alone."
                ),
                source_ids=["orishimo_2023", "diffendaffer_2023"],
                action_type=action_type,
            )
        )
    if separation.get("status") in {"available", "partial"}:
        notes.append(
            _injury_note(
                source_index,
                title="Treat separation as a coordination checkpoint, not a diagnosis",
                guidance=(
                    "Hip-shoulder separation can be useful for coaching rotational timing, but this project uses "
                    "a full-clip proxy rather than a true event-timed measurement."
                ),
                rationale=(
                    "The literature reference is pitching-oriented, so the safest use here is movement coaching and "
                    "repeat-video monitoring rather than injury labeling."
                ),
                source_ids=["orishimo_2023", "diffendaffer_2023"],
                action_type=action_type,
            )
        )
    notes.append(
        _injury_note(
            source_index,
            title="Pair video with periodic ROM screening",
            guidance=(
                "If the athlete also pitches or reports arm discomfort, combine the video report with simple "
                "shoulder and elbow range-of-motion screening done by a qualified clinician or athletic trainer."
            ),
            rationale=(
                "Important injury-prevention markers such as shoulder internal rotation deficit, total arc deficit, "
                "and elbow extension loss are clinical measurements and are not recoverable from this 2D video alone."
            ),
            source_ids=["paul_2025", "wilk_2011", "wright_2006"],
            action_type=action_type,
        )
    )
    notes.append(
        _injury_note(
            source_index,
            title="Screen hip mobility if lower-body rotation looks limited",
            guidance=(
                "When video repeatedly suggests limited lower-body contribution, add a simple hip rotation screen "
                "instead of assuming the issue is only technique."
            ),
            rationale=(
                "Prospective and correlation studies in baseball pitchers report links between hip mobility and "
                "throwing mechanics or shoulder-elbow injury risk."
            ),
            source_ids=["hamano_2020", "robb_2010"],
            action_type=action_type,
        )
    )
    return notes


def _injury_note(
    source_index: dict[str, Any],
    title: str,
    guidance: str,
    rationale: str,
    source_ids: list[str],
    action_type: str,
) -> dict[str, Any]:
    return {
        "title": title,
        "guidance": guidance,
        "rationale": rationale,
        "scope_note": (
            "Evidence base is strongest for pitching and overhead baseball populations; "
            "apply cautiously to batting-only interpretation."
            if action_type != "pitching"
            else "Evidence base is pitching-focused and is aligned with the analyzed action."
        ),
        "sources": [
            {
                "source_id": source_id,
                "short_name": source_index.get(source_id, {}).get("short_name", source_id),
                "citation": source_index.get(source_id, {}).get("citation", ""),
                "urls": source_index.get(source_id, {}).get("urls", []),
            }
            for source_id in source_ids
            if source_id in source_index
        ],
    }


def _summarize_derived_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    left_knee_flexion = [_optional_knee_flexion(row.get("left_knee_angle")) for row in rows]
    right_knee_flexion = [_optional_knee_flexion(row.get("right_knee_angle")) for row in rows]
    return {
        "left_knee_flexion_deg": _numeric_summary(left_knee_flexion),
        "right_knee_flexion_deg": _numeric_summary(right_knee_flexion),
    }


def _summarize_feature_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = {
        key
        for row in rows
        for key, value in row.items()
        if key not in {"clip_id", "condition_id"} and isinstance(value, (float, int))
    }
    summaries: dict[str, Any] = {}
    for field_name in sorted(numeric_fields):
        values = [row[field_name] for row in rows if isinstance(row.get(field_name), (float, int))]
        summaries[field_name] = _numeric_summary(values)
    return summaries


def _numeric_summary(values: list[float | None]) -> dict[str, Any]:
    filtered = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not filtered:
        return {"valid_count": 0}
    peak_abs = max(filtered, key=lambda value: abs(value))
    sorted_values = sorted(filtered)
    abs_sorted_values = sorted(abs(value) for value in filtered)
    return {
        "valid_count": len(filtered),
        "mean": mean(filtered),
        "median": median(filtered),
        "min": min(filtered),
        "max": max(filtered),
        "range": max(filtered) - min(filtered),
        "peak_abs": abs(peak_abs),
        "p05": _percentile(sorted_values, 0.05),
        "p95": _percentile(sorted_values, 0.95),
        "p95_abs": _percentile(abs_sorted_values, 0.95),
    }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _optional_knee_flexion(angle: float | None) -> float | None:
    if angle is None:
        return None
    return 180.0 - angle


def _parse_feature_row(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in row.items():
        if value == "":
            parsed[key] = None
            continue
        if key in {"clip_id", "condition_id"}:
            parsed[key] = value
            continue
        try:
            numeric = float(value)
        except ValueError:
            parsed[key] = value
            continue
        if key == "frame_index":
            parsed[key] = int(numeric)
        else:
            parsed[key] = numeric
    return parsed


def _count_status(mapping: dict[str, dict[str, Any]], status: str) -> int:
    return sum(1 for item in mapping.values() if item.get("status") == status)


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ModuleNotFoundError("pyyaml is required for report summary generation.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Reference YAML must be a mapping: {path}")
    data["_reference_path"] = str(path)
    return data


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
