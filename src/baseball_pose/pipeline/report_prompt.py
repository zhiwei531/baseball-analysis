"""Generate LLM-ready prompt packages from report summary JSON files."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.paths import report_prompt_dir, report_summary_path


@dataclass(frozen=True)
class ReportPromptResult:
    clip_id: str
    condition_id: str
    prompt_path: Path
    payload_path: Path
    draft_report_path: Path


def build_report_prompts(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str] | None = None,
) -> list[ReportPromptResult]:
    """Generate prompt.txt, prompt_payload.json, and draft_report.md artifacts."""

    condition_ids = conditions if conditions is not None else config.condition_ids
    results: list[ReportPromptResult] = []

    for clip_id in clip_ids:
        for condition_id in condition_ids:
            summary_path = report_summary_path(config.data_dir, clip_id, condition_id)
            if not summary_path.exists():
                continue
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            payload = _build_prompt_payload(summary)
            prompt_text = _build_prompt_text(payload)
            draft_report = _build_draft_report(payload)

            output_dir = report_prompt_dir(config.data_dir, clip_id, condition_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            payload_path = output_dir / "prompt_payload.json"
            prompt_path = output_dir / "prompt.txt"
            draft_path = output_dir / "draft_report.md"

            _write_json(payload_path, payload)
            prompt_path.write_text(prompt_text, encoding="utf-8")
            draft_path.write_text(draft_report, encoding="utf-8")

            results.append(
                ReportPromptResult(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    prompt_path=prompt_path,
                    payload_path=payload_path,
                    draft_report_path=draft_path,
                )
            )

    return results


def _build_prompt_payload(summary: dict[str, Any]) -> dict[str, Any]:
    metric_mapping = summary.get("standard_metric_mapping", {})
    selected_metrics = _select_metrics(metric_mapping)
    return {
        "task": "面向中国家长和教练，生成一份通俗、谨慎、可用于沟通的棒球动作分析摘要。",
        "clip_context": {
            "clip_id": summary["clip_id"],
            "condition_id": summary["condition_id"],
            "action_type": summary["action_type"],
            "athlete_group": summary["athlete_group"],
            "measurement_source": summary["measurement_source"],
            "frame_count": summary["frame_count"],
            "total_frame_count": summary.get("total_frame_count"),
            "action_window": summary.get("action_window"),
        },
        "report_constraints": {
            "audience": "中国家长、基层教练、没有生物力学背景的学生",
            "tone": "通俗、谨慎、非诊断性、建设性",
            "length_target_words": [180, 260],
            "must_do": [
                "先说明这次动作中最明显的优点或已经做得比较好的地方。",
                "再指出1到2个值得训练中重点关注的地方。",
                "区分直接观察结果与不确定、代理指标、缺失指标。",
                "明确说明这份结论来自2D姿态视频分析，而不是医学诊断或实验室动作捕捉。",
            ],
            "must_not_do": summary.get("llm_ready_summary", {}).get("forbidden_claims", []),
        },
        "output_metric_inventory": summary.get("output_metric_inventory", []),
        "public_metric_cards": summary.get("public_metric_cards", []),
        "selected_metrics": selected_metrics,
        "selected_reference_sources": summary.get("selected_reference_sources", []),
        "known_limitations": summary.get("llm_ready_summary", {}).get("limitations", []),
        "injury_prevention_context": summary.get("injury_prevention_context", []),
        "preferred_phrasing": summary.get("llm_ready_summary", {}).get("preferred_phrasing", []),
        "ready_to_quote_summary": _build_ready_to_quote_summary(selected_metrics),
    }


def _select_metrics(metric_mapping: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected_names = (
        "peak_trunk_rotation_velocity_deg_s",
        "peak_pelvis_rotation_velocity_deg_s",
        "hip_shoulder_separation_foot_contact_deg",
        "observed_left_knee_flexion_deg",
        "observed_right_knee_flexion_deg",
        "observed_hand_speed_proxy",
    )
    selected: list[dict[str, Any]] = []
    for name in selected_names:
        item = metric_mapping.get(name)
        if not item or item.get("status") == "unavailable":
            continue
        selected.append(
            {
                "metric_name": name,
                "label_cn": _metric_label(name, "cn"),
                "label_en": _metric_label(name, "en"),
                "unit": _metric_unit(name),
                "status": item.get("status"),
                "observed_value": item.get("observed_value"),
                "observed_summary_key": item.get("observed_summary_key"),
                "observed_summary": item.get("observed_summary"),
                "comparison": item.get("comparison"),
                "reference_context": item.get("reference_context"),
                "note": item.get("note"),
                "reason": item.get("reason"),
                "audience_explanation_cn": _metric_explanation(name),
            }
        )
    return selected


def _build_ready_to_quote_summary(selected_metrics: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for metric in selected_metrics:
        name = metric["metric_name"]
        status = metric["status"]
        observed_value = metric.get("observed_value")
        if observed_value is not None:
            if name == "peak_trunk_rotation_velocity_deg_s":
                lines.append(
                    f"躯干旋转速度代理指标（p95 absolute）约为 {observed_value:.1f} deg/s。"
                )
            elif name == "peak_pelvis_rotation_velocity_deg_s":
                lines.append(
                    f"骨盆旋转速度代理指标（p95 absolute）约为 {observed_value:.1f} deg/s。"
                )
            elif name == "hip_shoulder_separation_foot_contact_deg":
                lines.append(
                    f"髋肩分离代理指标（p95 absolute）约为 {observed_value:.1f} deg。"
                )
        comparison = metric.get("comparison")
        if comparison and comparison.get("mode") == "mean_sd":
            lines.append(
                f"{name} 相对于参考均值 {comparison['mean']:.1f}，属于 {comparison['band'].replace('_', ' ')}。"
            )
        if status == "partial" and metric.get("note"):
            lines.append(metric["note"])
    return lines


def _build_prompt_text(payload: dict[str, Any]) -> str:
    json_block = json.dumps(payload, indent=2, ensure_ascii=True)
    return (
        "你是一名棒球生物力学写作助手。\n"
        "请面向中国家长和教练，输出一段中文 markdown 报告正文。\n"
        "严格遵守以下规则：\n"
        "1. 必须使用下面给出的 markdown 标题。\n"
        "2. 全文控制在220到320个中文字符左右，尽量精炼。\n"
        "3. 在“动作优点”部分，先写1到2个已经观察到的优点。\n"
        "4. 在“训练重点”部分，写1到2个值得教练继续关注的重点。\n"
        "5. 在“伤病预防提示”部分，只写风险管理、监测建议和补充筛查建议，不要做诊断。\n"
        "6. 区分直接观察结果与不确定、代理指标或缺失指标。\n"
        "7. 如果某个指标是代理指标或不是事件点精确测量，要用通俗中文说明。\n"
        "8. “伤病预防提示”里至少提到一个文献短名，例如 Orishimo 2023。\n"
        "9. 最后给出1条具体、可执行的下一步建议。\n"
        "10. 可以引用 output_metric_inventory 中的项目名称，帮助非专业观众知道这个模型到底测出了哪些东西。\n\n"
        "输出格式：\n"
        "## 动作优点\n"
        "<一小段>\n\n"
        "## 训练重点\n"
        "<一小段>\n\n"
        "## 伤病预防提示\n"
        "<一小段>\n\n"
        "输入 JSON：\n"
        f"{json_block}\n"
    )


def _build_draft_report(payload: dict[str, Any]) -> str:
    context = payload["clip_context"]
    metrics = payload["selected_metrics"]
    strengths: list[str] = []
    cautions: list[str] = []

    for metric in metrics:
        name = metric["metric_name"]
        comparison = metric.get("comparison") or {}
        band = comparison.get("band")
        value = metric.get("observed_value")
        if name == "peak_trunk_rotation_velocity_deg_s" and value is not None:
            if band == "within_1sd":
                strengths.append(
                    f"Trunk rotation speed looked close to the available pitching reference ({value:.1f} deg/s robust proxy)."
                )
            elif band == "above_1sd":
                strengths.append(
                    f"Trunk rotation speed appeared high relative to the available reference ({value:.1f} deg/s robust proxy)."
                )
            elif band == "below_1sd":
                cautions.append(
                    f"Trunk rotation speed appeared lower than the available reference ({value:.1f} deg/s robust proxy)."
                )
        elif name == "peak_pelvis_rotation_velocity_deg_s" and value is not None:
            if band in {"within_1sd", "above_1sd"}:
                strengths.append(
                    f"Pelvis rotation speed was at least within the broad reference band ({value:.1f} deg/s robust proxy)."
                )
            elif band == "below_1sd":
                cautions.append(
                    f"Pelvis rotation speed was below the reference band ({value:.1f} deg/s robust proxy)."
                )
        elif name == "hip_shoulder_separation_foot_contact_deg" and value is not None:
            cautions.append(
                f"Trunk-pelvis separation was only estimated as a clip-level proxy ({value:.1f} deg), so event-specific interpretation remains limited."
            )

    if not strengths:
        strengths.append("The clip provided enough pose data to recover several rotation and knee-motion trends across the movement.")
    if not cautions:
        cautions.append("Several clinically important or event-specific metrics are still missing because the current system does not yet detect exact pitching events.")

    return (
        f"# Draft Report: {context['clip_id']} / {context['condition_id']}\n\n"
        f"This {context['action_type']} clip was summarized from {context['measurement_source']} data over {context['frame_count']} analyzed frames. "
        f"A positive sign is that {' '.join(strengths[:2])} "
        f"At the same time, {' '.join(cautions[:2])} "
        "This summary should be read as a movement-quality description rather than a medical conclusion. "
        "Injury-prevention context should be limited to monitoring, coaching, and referral for screening when symptoms or repeated limitations appear. "
        "A practical next step is to combine this video summary with event detection or simple clinical screening so the same athlete can be reviewed with more specific timing-based metrics.\n"
    )


def _metric_label(metric_name: str, language: str) -> str:
    labels = {
        "peak_trunk_rotation_velocity_deg_s": {"cn": "躯干旋转速度代理值", "en": "Trunk rotation speed proxy"},
        "peak_pelvis_rotation_velocity_deg_s": {"cn": "骨盆旋转速度代理值", "en": "Pelvis rotation speed proxy"},
        "hip_shoulder_separation_foot_contact_deg": {"cn": "髋肩分离代理值", "en": "Hip-shoulder separation proxy"},
        "observed_left_knee_flexion_deg": {"cn": "左膝弯曲深度", "en": "Left knee bend depth"},
        "observed_right_knee_flexion_deg": {"cn": "右膝弯曲深度", "en": "Right knee bend depth"},
        "observed_hand_speed_proxy": {"cn": "双手速度代理值", "en": "Hand speed proxy"},
    }
    return labels.get(metric_name, {}).get(language, metric_name)


def _metric_unit(metric_name: str) -> str:
    if "velocity" in metric_name:
        return "deg/s"
    if "separation" in metric_name or "flexion" in metric_name:
        return "deg"
    if "hand_speed" in metric_name:
        return "px/s"
    return ""


def _metric_explanation(metric_name: str) -> str:
    explanations = {
        "peak_trunk_rotation_velocity_deg_s": "描述上半身转动快慢，适合同一位运动员前后对比。",
        "peak_pelvis_rotation_velocity_deg_s": "描述下半身转动快慢，用来观察动力链是否积极。",
        "hip_shoulder_separation_foot_contact_deg": "描述髋部和肩部是否能形成错开，但当前仍是整段视频代理值。",
        "observed_left_knee_flexion_deg": "描述左膝在动作中弯曲得是否明显。",
        "observed_right_knee_flexion_deg": "描述右膝在动作中弯曲得是否明显。",
        "observed_hand_speed_proxy": "描述手部末端移动快慢，适合做同设备条件下的重复比较。",
    }
    return explanations.get(metric_name, "")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
