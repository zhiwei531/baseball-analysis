"""Call an OpenAI-compatible LLM endpoint to generate final report text."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request

from baseball_pose.config import RuntimeConfig
from baseball_pose.io.feature_csv import read_feature_rows
from baseball_pose.io.paths import feature_path, overlay_frame_dir, pose_path, report_llm_dir, report_prompt_dir
from baseball_pose.io.pose_csv import read_pose_records
from baseball_pose.pipeline.report_window import frame_indices_in_action_window
from baseball_pose.pose.schema import pose_score


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1"


@dataclass(frozen=True)
class LlmReportResult:
    clip_id: str
    condition_id: str
    report_path: Path
    request_path: Path
    response_path: Path
    model: str


def generate_llm_reports(
    clip_ids: list[str],
    config: RuntimeConfig,
    conditions: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key_env: str = "OPENAI_API_KEY",
    temperature: float = 0.2,
    timeout_sec: float = 120.0,
) -> list[LlmReportResult]:
    """Read prompt packages, call the LLM, and persist report artifacts."""

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env} is not set.")

    condition_ids = conditions if conditions is not None else config.condition_ids
    results: list[LlmReportResult] = []

    for clip_id in clip_ids:
        for condition_id in condition_ids:
            prompt_dir = report_prompt_dir(config.data_dir, clip_id, condition_id)
            prompt_path = prompt_dir / "prompt.txt"
            payload_path = prompt_dir / "prompt_payload.json"
            if not prompt_path.exists() or not payload_path.exists():
                continue

            prompt_text = prompt_path.read_text(encoding="utf-8")
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            request_body = _build_chat_request(prompt_text, model=model, temperature=temperature)
            response_body = _post_chat_completion(
                request_body=request_body,
                api_key=api_key,
                base_url=base_url,
                timeout_sec=timeout_sec,
            )
            llm_text = _extract_report_text(response_body)
            report_text = _build_final_report(
                clip_id=clip_id,
                condition_id=condition_id,
                config=config,
                payload=payload,
                llm_text=llm_text,
            )

            output_dir = report_llm_dir(config.data_dir, clip_id, condition_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            request_path = output_dir / "llm_request.json"
            response_path = output_dir / "llm_response.json"
            report_path = output_dir / "report.md"
            metadata_path = output_dir / "report_metadata.json"

            _write_json(request_path, request_body)
            _write_json(response_path, response_body)
            report_path.write_text(report_text.rstrip() + "\n", encoding="utf-8")
            _write_json(
                metadata_path,
                {
                    "clip_id": clip_id,
                    "condition_id": condition_id,
                    "model": model,
                    "base_url": base_url,
                    "api_key_env": api_key_env,
                    "prompt_path": str(prompt_path),
                    "payload_path": str(payload_path),
                    "payload_preview": payload.get("clip_context", {}),
                },
            )

            results.append(
                LlmReportResult(
                    clip_id=clip_id,
                    condition_id=condition_id,
                    report_path=report_path,
                    request_path=request_path,
                    response_path=response_path,
                    model=model,
                )
            )

    return results


def _build_chat_request(prompt_text: str, model: str, temperature: float) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是一名严谨的棒球动作分析写作助手。"
                    "必须严格按照提示词输出中文，语言要谨慎、清晰、非诊断性。"
                ),
            },
            {
                "role": "user",
                "content": prompt_text,
            },
        ],
    }


def _post_chat_completion(
    request_body: dict[str, Any],
    api_key: str,
    base_url: str,
    timeout_sec: float,
) -> dict[str, Any]:
    endpoint = _normalize_base_url(base_url) + "/chat/completions"
    payload = json.dumps(request_body).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout_sec) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:  # pragma: no cover - exercised only with live API failures.
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {error_body}") from exc
    except error.URLError as exc:  # pragma: no cover - exercised only with live API failures.
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise RuntimeError("LLM response was not a JSON object.")
    return decoded


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return normalized + "/v1"


def _extract_report_text(response_body: dict[str, Any]) -> str:
    choices = response_body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response missing choices.")
    first_choice = choices[0]
    message = first_choice.get("message", {})
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        text_chunks = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        ]
        combined = "\n".join(chunk for chunk in text_chunks if chunk)
        if combined.strip():
            return combined.strip()
    raise RuntimeError("Could not extract text content from LLM response.")


def _build_final_report(
    clip_id: str,
    condition_id: str,
    config: RuntimeConfig,
    payload: dict[str, Any],
    llm_text: str,
) -> str:
    context = payload.get("clip_context", {})
    visuals = _resolve_visuals(
        config,
        clip_id,
        condition_id,
        action_type=str(context.get("action_type", "batting")),
    )
    metrics = payload.get("selected_metrics", [])
    metric_inventory = payload.get("output_metric_inventory", [])
    public_cards = payload.get("public_metric_cards", [])
    references = payload.get("selected_reference_sources", [])
    reference_lookup = {
        str(reference.get("source_id")): str(reference.get("short_name", reference.get("source_id", "")))
        for reference in references
        if reference.get("source_id")
    }
    injury_notes = payload.get("injury_prevention_context", [])

    lines = [
        f"# Baseball Motion Report: {clip_id} / {condition_id}",
        "",
        "## Clip Context",
        f"- Action: {context.get('action_type', 'unknown')}",
        f"- Measurement source: {context.get('measurement_source', 'unknown')}",
        f"- Action-window frames analyzed: {context.get('frame_count', 'unknown')}",
        "",
        "## Key Visuals",
    ]
    total_frame_count = context.get("total_frame_count")
    if total_frame_count is not None:
        lines.insert(5, f"- Total clip frames: {total_frame_count}")
    action_window = context.get("action_window")
    if isinstance(action_window, dict) and action_window.get("start_frame") is not None:
        lines.insert(
            6,
            "- Action window: "
            f"frames {action_window.get('start_frame')} to {action_window.get('end_frame')} "
            f"({float(action_window.get('start_time_sec', 0.0)):.2f}s to {float(action_window.get('end_time_sec', 0.0)):.2f}s)",
        )

    if visuals.get("overlay_frame"):
        lines.extend(
            [
                "Representative skeleton overlay frame:",
                f"![Representative overlay]({visuals['overlay_frame']})",
                "",
            ]
        )
    if visuals.get("dashboard_figure"):
        lines.extend(
            [
                "Parent-facing movement dashboard:",
                f"![Movement dashboard]({visuals['dashboard_figure']})",
                "",
            ]
        )
    if visuals.get("knee_figure"):
        lines.extend(
            [
                "Knee bend summary:",
                f"![Knee bend summary]({visuals['knee_figure']})",
                "",
            ]
        )
    if visuals.get("side_balance_figure"):
        lines.extend(
            [
                "Side-to-side comparison:",
                f"![Side-to-side summary]({visuals['side_balance_figure']})",
                "",
            ]
        )
    if visuals.get("overlay_video"):
        lines.append(f"Overlay video: `{visuals['overlay_video']}`")
        lines.append("")

    lines.extend(
        [
            "## Plain-Language Interpretation",
            llm_text.strip(),
            "",
            "## 模型输出指标清单",
        ]
    )
    for item in metric_inventory:
        available_examples = item.get("available_examples", [])
        example_text = ", ".join(f"`{example}`" for example in available_examples) if available_examples else "当前片段中暂无稳定字段示例。"
        lines.append(
            f"- **{item.get('category_en', 'Metric category')} / {item.get('category_cn', '')}**: "
            f"{item.get('plain_language_cn', '')} 字段示例：{example_text}"
        )
    lines.append("")

    lines.extend(
        [
            "## 面向家长与教练的关键指标",
            "| 指标 | 本次结果 | 面向家长与教练的解释 | 文献支持 |",
            "|---|---:|---|---|",
        ]
    )
    for card in public_cards:
        lines.append(_public_metric_card_row(card))
    lines.append("")

    lines.extend(
        [
            "## 关键指标明细",
            "| 指标 | 本次结果 | 解读 | 依据来源 |",
            "|---|---:|---|---|",
        ]
    )
    for metric in metrics:
        lines.append(_metric_table_row(metric, reference_lookup))
    lines.append("")

    lines.extend(
        [
            "## 伤病预防提示",
        ]
    )
    for note in injury_notes:
        lines.extend(_render_injury_note(note))
    lines.append("")

    lines.extend(
        [
            "## 局限性说明",
        ]
    )
    for limitation in payload.get("known_limitations", []):
        lines.append(f"- {limitation}")
    lines.append("- 当前对击球动作的部分解释仍借用了投球研究中的文献依据，因此更适合作为训练沟通材料，而不适合作为临床推断。")
    lines.append("")

    lines.extend(
        [
            "## 参考文献",
        ]
    )
    for reference in references:
        lines.append(_render_reference(reference))

    return "\n".join(lines).rstrip() + "\n"


def _resolve_visuals(
    config: RuntimeConfig,
    clip_id: str,
    condition_id: str,
    action_type: str,
) -> dict[str, str]:
    output_root = config.output_dir.resolve()
    visuals: dict[str, str] = {}
    dashboard = output_root / "figures" / f"{clip_id}__movement_quality_dashboard.png"
    knee = output_root / "figures" / f"{clip_id}__knee_balance_summary.png"
    side_balance = output_root / "figures" / f"{clip_id}__side_to_side_summary.png"
    overlay_video = output_root / "overlays" / f"{clip_id}__{condition_id}.mp4"
    if dashboard.exists():
        visuals["dashboard_figure"] = str(dashboard)
    if knee.exists():
        visuals["knee_figure"] = str(knee)
    if side_balance.exists():
        visuals["side_balance_figure"] = str(side_balance)
    if overlay_video.exists():
        visuals["overlay_video"] = str(overlay_video)
    overlay_frame = _pick_overlay_frame(config, clip_id, condition_id, action_type)
    if overlay_frame:
        visuals["overlay_frame"] = overlay_frame
    return visuals


def _pick_overlay_frame(
    config: RuntimeConfig,
    clip_id: str,
    condition_id: str,
    action_type: str,
) -> str | None:
    frame_root = overlay_frame_dir(config.output_dir, clip_id, condition_id)
    if not frame_root.exists():
        return None
    feature_csv = feature_path(config.data_dir, clip_id, condition_id)
    candidate_index: int | None = None
    action_frames: set[int] = set()
    if feature_csv.exists():
        rows = read_feature_rows(feature_csv)
        action_frames = frame_indices_in_action_window(rows, action_type=action_type)
    pose_csv = pose_path(config.data_dir, clip_id, condition_id)
    if pose_csv.exists():
        candidate_index = _best_stable_action_frame(
            read_pose_records(pose_csv),
            action_frames=action_frames,
        )
    frame_candidates = sorted(frame_root.glob("*.png"))
    if not frame_candidates:
        return None
    if candidate_index is not None:
        preferred = frame_root / f"{clip_id}__{condition_id}__frame_{candidate_index:06d}.png"
        if preferred.exists():
            return str(preferred.resolve())
    return str(frame_candidates[len(frame_candidates) // 2].resolve())


def _best_stable_action_frame(records: list[Any], action_frames: set[int]) -> int | None:
    by_frame: dict[int, list[Any]] = {}
    for record in records:
        by_frame.setdefault(record.frame_index, []).append(record)
    candidate_frames = sorted(action_frames) if action_frames else sorted(by_frame)
    best_frame: int | None = None
    best_score = -1.0
    for frame_index in candidate_frames:
        frame_records = by_frame.get(frame_index, [])
        if not frame_records:
            continue
        stable_joint_count = 0
        stable_score_sum = 0.0
        core_joint_count = 0
        for record in frame_records:
            score = pose_score(record)
            if record.x is None or record.y is None or score is None:
                continue
            if score >= 0.55:
                stable_joint_count += 1
                stable_score_sum += score
                if record.joint_name in {
                    "left_shoulder",
                    "right_shoulder",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                }:
                    core_joint_count += 1
        frame_score = stable_joint_count * 10.0 + core_joint_count * 3.0 + stable_score_sum
        if frame_score > best_score:
            best_score = frame_score
            best_frame = frame_index
    return best_frame


def _metric_table_row(metric: dict[str, Any], reference_lookup: dict[str, str]) -> str:
    name = metric.get("metric_name", "unknown")
    status = metric.get("status", "unknown")
    observed_value = metric.get("observed_value")
    summary = metric.get("observed_summary", {})
    comparison = metric.get("comparison") or {}
    reference_text = _metric_reference_names(metric, reference_lookup)

    if observed_value is None and isinstance(summary, dict):
        if summary.get("p95_abs") is not None:
            observed_text = f"{float(summary['p95_abs']):.1f}（P95 绝对值）"
        elif summary.get("mean") is not None:
            observed_text = f"{float(summary['mean']):.1f}（均值）"
        else:
            observed_text = "不适用"
    elif observed_value is None:
        observed_text = "不适用"
    elif "velocity" in name:
        observed_text = f"{float(observed_value):.1f} 度/秒"
    elif "separation" in name or "flexion" in name:
        observed_text = f"{float(observed_value):.1f} 度"
    else:
        observed_text = f"{float(observed_value):.3f}"

    if status == "partial":
        interpretation = _zh_note(str(metric.get("note") or "Partial proxy only."))
    elif comparison.get("mode") == "mean_sd":
        band = str(comparison.get("band", "unknown")).replace("_", " ")
        interpretation = f"{_zh_band(band)}。"
    elif metric.get("reason"):
        interpretation = _zh_reason(str(metric["reason"]))
    else:
        interpretation = "直接来自当前特征表的观察结果。"

    return f"| `{name}` | {observed_text} | {interpretation} | {reference_text} |"


def _metric_reference_names(metric: dict[str, Any], reference_lookup: dict[str, str]) -> str:
    context = metric.get("reference_context")
    if not isinstance(context, dict):
        return "仅为当前项目直接输出"
    source_ids = context.get("source_ids", [])
    if not source_ids:
        note = metric.get("note")
        return _zh_note(str(note)) if note else "仅为当前项目直接输出"
    return ", ".join(reference_lookup.get(str(source_id), str(source_id)) for source_id in source_ids)


def _public_metric_card_row(card: dict[str, Any]) -> str:
    label = str(card.get("label_cn") or card.get("label_en") or card.get("metric_name", "metric"))
    observed_value = card.get("observed_value")
    summary = card.get("observed_summary") if isinstance(card.get("observed_summary"), dict) else {}
    unit = str(card.get("unit", ""))
    if observed_value is None and summary.get("p95_abs") is not None:
        observed_text = f"{float(summary['p95_abs']):.1f} {unit}".strip()
    elif observed_value is None and summary.get("mean") is not None:
        observed_text = f"{float(summary['mean']):.1f} {unit}".strip()
    elif observed_value is None:
        observed_text = "不适用"
    elif unit:
        observed_text = f"{float(observed_value):.1f} {unit}"
    else:
        observed_text = f"{float(observed_value):.1f}"
    explanation = str(card.get("audience_explanation_cn", ""))
    raw_note = card.get("note")
    note = "" if raw_note in {None, "None"} else str(raw_note).strip()
    if note:
        explanation = f"{explanation} 注意：{_zh_note(note)}".strip()
    sources = card.get("reference_short_names", [])
    source_text = ", ".join(str(source) for source in sources) if sources else "仅基于当前视频片段"
    return f"| {label} | {observed_text} | {explanation} | {source_text} |"


def _render_injury_note(note: dict[str, Any]) -> list[str]:
    sources = note.get("sources", [])
    source_names = ", ".join(str(source.get("short_name", "")) for source in sources if source.get("short_name"))
    guidance = note.get("guidance", "")
    rationale = note.get("rationale", "")
    scope_note = note.get("scope_note", "")
    title = note.get("title", "Injury prevention note")
    lines = [f"- **{_zh_injury_title(str(title))}**：{_zh_injury_text(str(guidance))}"]
    if rationale:
        lines.append(f"  解释依据：{_zh_injury_text(str(rationale))}")
    if scope_note:
        lines.append(f"  适用范围说明：{_zh_injury_text(str(scope_note))}")
    if source_names:
        lines.append(f"  参考文献：{source_names}。")
    return lines


def _render_reference(reference: dict[str, Any]) -> str:
    short_name = reference.get("short_name", reference.get("source_id", "source"))
    citation = reference.get("citation", "")
    urls = reference.get("urls", [])
    if urls:
        return f"- **{short_name}**：{citation} 来源：{urls[0]}"
    return f"- **{short_name}**：{citation}"


def _zh_band(text: str) -> str:
    mapping = {
        "below 1sd": "低于现有参考区间的 1 个标准差范围",
        "within 1sd": "位于现有参考区间的 1 个标准差范围内",
        "above 1sd": "高于现有参考区间的 1 个标准差范围",
    }
    return mapping.get(text, text)


def _zh_reason(text: str) -> str:
    mapping = {
        "Observed from current feature CSV but not matched to a standard reference metric.": "当前可以从本次特征表直接观察到该指标，但还没有匹配到稳定可比的标准参考值。",
        "Current feature CSV stores left/right knee angles, but it does not identify lead side or foot-contact timing.": "当前特征表有左右膝角度，但还不能自动识别前导腿，也不能自动定位脚落地时刻。",
        "No stable example fields in this clip.": "当前片段中暂无稳定字段示例。",
    }
    return mapping.get(text, text)


def _zh_note(text: str) -> str:
    mapping = {
        "This reference was derived from pitching literature; use cautiously for batting.": "该参考值主要来自投球研究，用于击球动作时需要谨慎解释。",
        "Current pipeline summarizes full-clip separation and does not detect foot contact, so this is only a partial proxy for the foot-contact reference. Reference interpretation is pitching-specific.": "当前流程只统计整段视频中的髋肩分离变化，不能自动定位脚落地时刻，因此这里只能作为脚落地分离角的部分代理指标；参考解释也主要来自投球研究。",
        "Partial proxy only.": "当前只能作为部分代理指标使用。",
        "Current clip only": "仅基于当前视频片段",
        "Current project output only": "仅为当前项目直接输出",
    }
    return mapping.get(text, text)


def _zh_injury_title(text: str) -> str:
    mapping = {
        "Guidance": "训练建议",
        "Injury prevention note": "伤病预防提示",
        "Monitor trunk and pelvis sequencing": "持续关注躯干与骨盆的发力顺序",
        "Treat separation as a coordination checkpoint, not a diagnosis": "应把髋肩分离看作动作协调的检查点，而不是诊断结论",
        "Pair video with periodic ROM screening": "建议将视频分析与定期关节活动度筛查结合使用",
        "Screen hip mobility if lower-body rotation looks limited": "如果下肢旋转看起来长期受限，建议进一步做髋关节活动度筛查",
    }
    return mapping.get(text, text)


def _zh_injury_text(text: str) -> str:
    mapping = {
        "Monitor trunk and pelvis sequencing. When trunk and pelvis rotation proxies are consistently low, use coaching to improve timing and force transfer before simply asking the athlete to swing harder.": "持续关注躯干与骨盆的发力顺序。如果躯干和骨盆旋转代理指标长期偏低，应该先通过训练改善发力时序和力量传递，而不是只要求运动员更用力挥棒。",
        "Published baseball biomechanics reviews link pelvis and trunk rotational contribution to performance output, but this 2D pipeline cannot diagnose injury from rotation speed alone.": "已有棒球生物力学综述指出，骨盆和躯干的旋转贡献与动作表现有关，但当前这套 2D 流程不能仅凭旋转速度就判断伤病风险。",
        "Evidence base is strongest for pitching and overhead baseball populations; apply cautiously to batting-only interpretation.": "现有证据主要来自投球或过头用力的棒球人群，因此用于纯击球动作解释时需要谨慎。",
        "Treat separation as a coordination checkpoint, not a diagnosis. Hip-shoulder separation can be useful for coaching rotational timing, but this project uses a full-clip proxy rather than a true event-timed measurement.": "应把髋肩分离看作动作协调的检查点，而不是诊断结论。髋肩分离对指导旋转时序有帮助，但本项目使用的是整段视频代理指标，不是真正按关键事件精确测得的数值。",
        "The literature reference is pitching-oriented, so the safest use here is movement coaching and repeat-video monitoring rather than injury labeling.": "相关文献参考主要偏向投球研究，因此这里最稳妥的用途是动作训练指导和重复视频监测，而不是给出伤病标签。",
        "Pair video with periodic ROM screening. If the athlete also pitches or reports arm discomfort, combine the video report with simple shoulder and elbow range-of-motion screening done by a qualified clinician or athletic trainer.": "建议将视频分析与定期关节活动度筛查结合使用。如果运动员同时参与投球，或已经出现手臂不适，应把视频报告与由合格临床医生或运动防护人员完成的肩肘活动度筛查结合起来。",
        "Important injury-prevention markers such as shoulder internal rotation deficit, total arc deficit, and elbow extension loss are clinical measurements and are not recoverable from this 2D video alone.": "一些重要的伤病预防指标，例如肩内旋不足、总活动弧度不足和肘伸展受限，属于临床测量项目，无法仅凭这段 2D 视频直接恢复出来。",
        "Screen hip mobility if lower-body rotation looks limited. When video repeatedly suggests limited lower-body contribution, add a simple hip rotation screen instead of assuming the issue is only technique.": "如果下肢旋转看起来长期受限，建议进一步做髋关节活动度筛查。当视频多次提示下肢贡献不足时，不应直接假设问题只来自技术动作，还应检查髋部旋转能力。",
        "Prospective and correlation studies in baseball pitchers report links between hip mobility and throwing mechanics or shoulder-elbow injury risk.": "在棒球投手人群中的前瞻性研究和相关性研究都提示，髋关节活动度与投掷力学表现以及肩肘伤病风险之间存在联系。",
        "When trunk and pelvis rotation proxies are consistently low, use coaching to improve timing and force transfer before simply asking the athlete to swing harder.": "如果躯干和骨盆旋转代理指标长期偏低，应该先通过训练改善发力时序和力量传递，而不是只要求运动员更用力挥棒。",
        "Hip-shoulder separation can be useful for coaching rotational timing, but this project uses a full-clip proxy rather than a true event-timed measurement.": "髋肩分离对指导旋转时序有帮助，但本项目使用的是整段视频代理指标，不是真正按关键事件精确测得的数值。",
        "If the athlete also pitches or reports arm discomfort, combine the video report with simple shoulder and elbow range-of-motion screening done by a qualified clinician or athletic trainer.": "如果运动员同时参与投球，或已经出现手臂不适，应把视频报告与由合格临床医生或运动防护人员完成的肩肘活动度筛查结合起来。",
        "When video repeatedly suggests limited lower-body contribution, add a simple hip rotation screen instead of assuming the issue is only technique.": "当视频多次提示下肢贡献不足时，不应直接假设问题只来自技术动作，还应检查髋部旋转能力。",
    }
    return mapping.get(text, text)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
