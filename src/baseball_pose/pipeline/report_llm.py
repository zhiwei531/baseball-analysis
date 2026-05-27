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
            "## Output Metrics Produced By This Model",
        ]
    )
    for item in metric_inventory:
        available_examples = item.get("available_examples", [])
        example_text = ", ".join(f"`{example}`" for example in available_examples) if available_examples else "No stable example fields in this clip."
        lines.append(
            f"- **{item.get('category_en', 'Metric category')} / {item.get('category_cn', '')}**: "
            f"{item.get('plain_language_cn', '')} Example fields: {example_text}"
        )
    lines.append("")

    lines.extend(
        [
            "## Public-Facing Key Metrics",
            "| Metric | Observed value | Meaning for parents and coaches | Literature support |",
            "|---|---:|---|---|",
        ]
    )
    for card in public_cards:
        lines.append(_public_metric_card_row(card))
    lines.append("")

    lines.extend(
        [
            "## Key Metrics",
            "| Metric | Observed value | Interpretation | Evidence source |",
            "|---|---:|---|---|",
        ]
    )
    for metric in metrics:
        lines.append(_metric_table_row(metric, reference_lookup))
    lines.append("")

    lines.extend(
        [
            "## Injury Prevention Considerations",
        ]
    )
    for note in injury_notes:
        lines.extend(_render_injury_note(note))
    lines.append("")

    lines.extend(
        [
            "## Scope And Limits",
        ]
    )
    for limitation in payload.get("known_limitations", []):
        lines.append(f"- {limitation}")
    lines.append("- Batting interpretation is still borrowing part of its evidence base from pitching literature, so coaching use is stronger than clinical inference.")
    lines.append("")

    lines.extend(
        [
            "## Reference Sources",
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
            observed_text = f"{float(summary['p95_abs']):.1f} (p95 abs)"
        elif summary.get("mean") is not None:
            observed_text = f"{float(summary['mean']):.1f} (mean)"
        else:
            observed_text = "n/a"
    elif observed_value is None:
        observed_text = "n/a"
    elif "velocity" in name:
        observed_text = f"{float(observed_value):.1f} deg/s"
    elif "separation" in name or "flexion" in name:
        observed_text = f"{float(observed_value):.1f} deg"
    else:
        observed_text = f"{float(observed_value):.3f}"

    if status == "partial":
        interpretation = metric.get("note") or "Partial proxy only."
    elif comparison.get("mode") == "mean_sd":
        band = str(comparison.get("band", "unknown")).replace("_", " ")
        interpretation = f"{band} vs available reference."
    elif metric.get("reason"):
        interpretation = str(metric["reason"])
    else:
        interpretation = "Observed directly from the current feature CSV."

    return f"| `{name}` | {observed_text} | {interpretation} | {reference_text} |"


def _metric_reference_names(metric: dict[str, Any], reference_lookup: dict[str, str]) -> str:
    context = metric.get("reference_context")
    if not isinstance(context, dict):
        return "Current project output only"
    source_ids = context.get("source_ids", [])
    if not source_ids:
        note = metric.get("note")
        return str(note) if note else "Current project output only"
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
        observed_text = "n/a"
    elif unit:
        observed_text = f"{float(observed_value):.1f} {unit}"
    else:
        observed_text = f"{float(observed_value):.1f}"
    explanation = str(card.get("audience_explanation_cn", ""))
    raw_note = card.get("note")
    note = "" if raw_note in {None, "None"} else str(raw_note).strip()
    if note:
        explanation = f"{explanation} 注意：{note}".strip()
    sources = card.get("reference_short_names", [])
    source_text = ", ".join(str(source) for source in sources) if sources else "Current clip only"
    return f"| {label} | {observed_text} | {explanation} | {source_text} |"


def _render_injury_note(note: dict[str, Any]) -> list[str]:
    sources = note.get("sources", [])
    source_names = ", ".join(str(source.get("short_name", "")) for source in sources if source.get("short_name"))
    guidance = note.get("guidance", "")
    rationale = note.get("rationale", "")
    scope_note = note.get("scope_note", "")
    title = note.get("title", "Injury prevention note")
    lines = [f"- **{title}**: {guidance}"]
    if rationale:
        lines.append(f"  Rationale: {rationale}")
    if scope_note:
        lines.append(f"  Scope: {scope_note}")
    if source_names:
        lines.append(f"  Literature: {source_names}.")
    return lines


def _render_reference(reference: dict[str, Any]) -> str:
    short_name = reference.get("short_name", reference.get("source_id", "source"))
    citation = reference.get("citation", "")
    urls = reference.get("urls", [])
    if urls:
        return f"- **{short_name}**: {citation} Source: {urls[0]}"
    return f"- **{short_name}**: {citation}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
