from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
import sys
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from baseball_pose.io.pose_csv import read_pose_records
from baseball_pose.pipeline.report_window import frame_indices_in_action_window
from baseball_pose.pose.schema import pose_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a polished PDF report from existing baseball analysis artifacts.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    clip_id = args.clip_id
    condition = args.condition
    output_path = Path(args.out).resolve()

    payload_path = project_root / "data_full" / "processed" / "metrics" / "report_prompts" / clip_id / condition / "prompt_payload.json"
    report_md_path = project_root / "data_full" / "processed" / "metrics" / "report_llm" / clip_id / condition / "report.md"
    if not payload_path.exists():
        raise FileNotFoundError(payload_path)
    if not report_md_path.exists():
        raise FileNotFoundError(report_md_path)

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    report_md = report_md_path.read_text(encoding="utf-8")
    font_name = _register_cjk_font()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.70 * inch,
        leftMargin=0.70 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.65 * inch,
        title=f"Baseball Motion Report - {clip_id}",
        author="baseball-analysis",
    )

    styles = _build_styles(font_name)
    story: list[Any] = []
    _build_cover(story, styles, payload, clip_id, condition)
    _build_visuals(story, styles, project_root, clip_id, condition, payload)
    _build_llm_sections(story, styles, report_md)
    _build_metric_comparison(story, styles, payload)
    _build_injury_prevention(story, styles, payload)
    _build_scope_and_sources(story, styles, payload)

    doc.build(story, onFirstPage=_draw_page, onLaterPages=_draw_page)
    print(output_path)


def _build_styles(font_name: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "TitleCustom",
            parent=base["Title"],
            fontName=font_name,
            fontSize=21,
            leading=26,
            textColor=colors.HexColor("#123B5D"),
            spaceAfter=14,
            alignment=TA_LEFT,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCustom",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=10.2,
            leading=14,
            textColor=colors.HexColor("#4B6072"),
            spaceAfter=6,
        ),
        "section": ParagraphStyle(
            "SectionCustom",
            parent=base["Heading1"],
            fontName=font_name,
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#123B5D"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "subsection": ParagraphStyle(
            "SubsectionCustom",
            parent=base["Heading2"],
            fontName=font_name,
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#1E4668"),
            spaceBefore=5,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyCustom",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=9.3,
            leading=13.6,
            textColor=colors.HexColor("#1C1C1C"),
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "SmallCustom",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=8.2,
            leading=11.2,
            textColor=colors.HexColor("#495867"),
            spaceAfter=3,
        ),
        "metric_cell": ParagraphStyle(
            "MetricCell",
            parent=base["BodyText"],
            fontName=font_name,
            fontSize=7.8,
            leading=9.4,
            textColor=colors.HexColor("#1F2933"),
        ),
    }
    return styles


def _build_cover(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
    clip_id: str,
    condition: str,
) -> None:
    context = payload.get("clip_context", {})
    story.append(Paragraph("棒球动作分析报告", styles["title"]))
    story.append(Paragraph(f"视频编号：<b>{clip_id}</b> &nbsp;&nbsp; 处理条件：<b>{condition}</b>", styles["subtitle"]))
    story.append(
        Paragraph(
            f"动作类型：<b>{_action_type_cn(str(context.get('action_type', 'unknown')))}</b> &nbsp;&nbsp; "
            f"动作窗口帧数：<b>{context.get('frame_count', 'unknown')}</b> &nbsp;&nbsp; "
            f"数据来源：<b>{_measurement_source_cn(str(context.get('measurement_source', 'unknown')))}</b>",
            styles["subtitle"],
        )
    )
    if context.get("total_frame_count"):
        story.append(
            Paragraph(
                f"原始整段视频帧数：<b>{context.get('total_frame_count')}</b>",
                styles["subtitle"],
            )
        )
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        Paragraph(
            "本报告整合了 2D 姿态视频分析得到的动作指标、自动生成的图表、文献参考区间，以及面向中国家长和教练的通俗解释。",
            styles["body"],
        )
    )
    story.append(Spacer(1, 0.10 * inch))


def _build_visuals(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    project_root: Path,
    clip_id: str,
    condition: str,
    payload: dict[str, Any],
) -> None:
    story.append(Paragraph("可视化证据", styles["section"]))
    overlay_frame = _pick_overlay_frame(
        project_root,
        clip_id,
        condition,
        action_type=str(payload.get("clip_context", {}).get("action_type", "batting")),
    )
    dashboard = project_root / "outputs_full" / "figures" / f"{clip_id}__movement_quality_dashboard.png"
    knee = project_root / "outputs_full" / "figures" / f"{clip_id}__knee_balance_summary.png"
    side = project_root / "outputs_full" / "figures" / f"{clip_id}__side_to_side_summary.png"

    images = [
        overlay_frame,
        dashboard if dashboard.exists() else None,
        knee if knee.exists() else None,
        side if side.exists() else None,
    ]
    rows = []
    for row_start in (0, 2):
        row = []
        for image_path in images[row_start:row_start + 2]:
            if image_path and Path(image_path).exists():
                row.append(_scaled_image(Path(image_path), 3.25 * inch, 2.35 * inch))
            else:
                row.append(Paragraph("Image unavailable", styles["small"]))
        rows.append(row)

    visuals = Table(rows, colWidths=[3.35 * inch, 3.35 * inch])
    visuals.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D3DD")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FBFD")),
            ]
        )
    )
    story.append(visuals)
    story.append(
        Paragraph(
            "左上：代表性骨架叠加帧。右上：面向家长和教练的动作指标卡。左下：膝盖弯曲深度条形图。右下：左右侧对比图。",
            styles["small"],
        )
    )
    story.append(Spacer(1, 0.10 * inch))
    _build_output_metric_inventory(story, styles, payload)


def _build_output_metric_inventory(
    story: list[Any],
    styles: dict[str, ParagraphStyle],
    payload: dict[str, Any],
) -> None:
    story.append(Paragraph("模型输出了哪些指标", styles["section"]))
    story.append(
        Paragraph(
            "这部分不是评价好坏，而是直接告诉家长和教练：模型到底测到了哪些维度。这样在展示后续图表和文字报告时，观众能更容易理解每个结论来自哪里。",
            styles["body"],
        )
    )
    for item in payload.get("output_metric_inventory", []):
        category = item.get("category_cn", "")
        description = item.get("plain_language_cn", "")
        examples = ", ".join(item.get("available_examples", []))
        story.append(
            Paragraph(
                f"• <b>{_escape(str(category))}</b>：{_escape(str(description))} "
                f"本次视频中可直接调用的字段示例包括 { _escape(examples) if examples else '暂无稳定字段'}。",
                styles["body"],
            )
        )
    story.append(Spacer(1, 0.06 * inch))


def _build_llm_sections(story: list[Any], styles: dict[str, ParagraphStyle], report_md: str) -> None:
    story.append(Paragraph("面向教练与家长的解读", styles["section"]))
    sections = _parse_markdown_sections(report_md)
    for title in ("动作优点", "训练重点", "伤病预防提示", "Summary", "Coaching Priorities", "Injury Prevention Context"):
        body = sections.get(title)
        if body:
            story.append(Paragraph(_section_title_cn(title), styles["subsection"]))
            story.append(Paragraph(_escape(body), styles["body"]))
    story.append(Spacer(1, 0.06 * inch))


def _build_metric_comparison(story: list[Any], styles: dict[str, ParagraphStyle], payload: dict[str, Any]) -> None:
    story.append(Paragraph("与标准指标的详细对比", styles["section"]))
    story.append(
        Paragraph(
            "下表把“直接观察结果”和“文献参考解释”分开呈现。只要存在文献参考，报告就会同时给出本次观察值、参考区间、与参考均值的差值，以及该参考是否主要来自投球研究的说明。",
            styles["body"],
        )
    )

    header = [
        Paragraph("<b>指标</b>", styles["metric_cell"]),
        Paragraph("<b>本次结果</b>", styles["metric_cell"]),
        Paragraph("<b>参考区间</b>", styles["metric_cell"]),
        Paragraph("<b>差值</b>", styles["metric_cell"]),
        Paragraph("<b>解读</b>", styles["metric_cell"]),
        Paragraph("<b>文献 / 注意事项</b>", styles["metric_cell"]),
    ]
    rows = [header]
    for metric in payload.get("selected_metrics", []):
        rows.append(_metric_row(metric, styles["metric_cell"]))

    table = Table(
        rows,
        colWidths=[1.40 * inch, 0.95 * inch, 1.20 * inch, 0.88 * inch, 1.35 * inch, 1.42 * inch],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123B5D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CCD7E0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.10 * inch))

    for metric in payload.get("selected_metrics", []):
        note = _metric_detail_paragraph(metric)
        if note:
            story.append(Paragraph(note, styles["small"]))

    story.append(Spacer(1, 0.08 * inch))


def _build_injury_prevention(story: list[Any], styles: dict[str, ParagraphStyle], payload: dict[str, Any]) -> None:
    story.append(Paragraph("伤病预防与后续监测建议", styles["section"]))
    for item in payload.get("injury_prevention_context", []):
        title = _escape(str(item.get("title", "Guidance")))
        guidance = _escape(str(item.get("guidance", "")))
        rationale = _escape(str(item.get("rationale", "")))
        scope_note = _escape(str(item.get("scope_note", "")))
        sources = ", ".join(str(source.get("short_name", "")) for source in item.get("sources", []))
        story.append(Paragraph(f"<b>{title}.</b> {guidance}", styles["body"]))
        if rationale:
            story.append(Paragraph(f"解释依据：{rationale}", styles["small"]))
        if scope_note:
            story.append(Paragraph(f"适用范围说明：{scope_note}", styles["small"]))
        if sources:
            story.append(Paragraph(f"参考文献：{sources}。", styles["small"]))
        story.append(Spacer(1, 0.04 * inch))


def _build_scope_and_sources(story: list[Any], styles: dict[str, ParagraphStyle], payload: dict[str, Any]) -> None:
    story.append(Paragraph("局限性与参考文献", styles["section"]))
    for limitation in payload.get("known_limitations", []):
        story.append(Paragraph(f"• {_escape(_limitation_cn(str(limitation)))}", styles["body"]))
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph("本报告引用的主要文献", styles["subsection"]))
    for reference in payload.get("selected_reference_sources", []):
        short_name = _escape(str(reference.get("short_name", reference.get("source_id", ""))))
        citation = _escape(str(reference.get("citation", "")).strip())
        focus = _escape(str(reference.get("focus", "")))
        line = f"<b>{short_name}</b>: {citation}"
        if focus:
            line += f" 关注重点：{focus}。"
        story.append(Paragraph(line, styles["small"]))


def _metric_row(metric: dict[str, Any], style: ParagraphStyle) -> list[Any]:
    name = _humanize_metric_name(str(metric.get("metric_name", "unknown")))
    observed_value = _format_observed(metric)
    reference_text, gap_text, interpretation, evidence = _comparison_text(metric)
    return [
        Paragraph(_escape(name), style),
        Paragraph(_escape(observed_value), style),
        Paragraph(_escape(reference_text), style),
        Paragraph(_escape(gap_text), style),
        Paragraph(_escape(interpretation), style),
        Paragraph(_escape(evidence), style),
    ]


def _comparison_text(metric: dict[str, Any]) -> tuple[str, str, str, str]:
    comparison = metric.get("comparison") or {}
    reference_context = metric.get("reference_context") or {}
    note = str(metric.get("note", "") or "")
    if comparison.get("mode") == "mean_sd":
        mean = float(comparison["mean"])
        std = float(comparison["std"])
        band = str(comparison.get("band", "unknown")).replace("_", " ")
        delta = float(comparison.get("delta_from_mean", 0.0))
        reference = f"{mean:.1f} ± {std:.1f}"
        gap = f"{delta:+.1f}"
        source_names = ", ".join(_source_short_names(reference_context))
        evidence = source_names or _note_cn(note) or "主要参考投球文献"
        if note:
            evidence = f"{evidence}；{_note_cn(note)}"
        return reference, gap, _band_cn(band), evidence
    if metric.get("status") == "partial":
        source_names = ", ".join(_source_short_names(reference_context))
        return "代理指标", "n/a", "部分代理", source_names or _note_cn(note) or "当前指标不是精确事件点测量"
    if metric.get("reason"):
        return "暂无匹配标准", "n/a", "仅直接观察", _reason_cn(str(metric["reason"]))
    return "暂无匹配标准", "n/a", "直接观察", note or "当前项目直接输出"


def _metric_detail_paragraph(metric: dict[str, Any]) -> str:
    name = _humanize_metric_name(str(metric.get("metric_name", "unknown")))
    comparison = metric.get("comparison") or {}
    note = str(metric.get("note", "") or "")
    if comparison.get("mode") == "mean_sd":
        mean = float(comparison["mean"])
        std = float(comparison["std"])
        lower = float(comparison["lower_1sd"])
        upper = float(comparison["upper_1sd"])
        delta = float(comparison.get("delta_from_mean", 0.0))
        band = str(comparison.get("band", "unknown")).replace("_", " ")
        observed = _format_observed(metric)
        text = (
            f"<b>{_escape(name)}：</b>本次结果为 {observed}；参考区间为 {lower:.1f} 到 {upper:.1f} "
            f"（均值 {mean:.1f}，标准差 {std:.1f}）；与参考均值相比差值为 {delta:+.1f}。"
            f"按照该参考，本次结果属于 <b>{_escape(_band_cn(band))}</b>。"
        )
        if note:
            text += f" 注意：{_escape(note)}"
        return text
    if metric.get("status") == "partial":
        return f"<b>{_escape(name)}：</b>{_escape(note or '当前流程中，这个指标只能作为部分代理指标使用。')}"
    reason = metric.get("reason")
    if reason:
        return f"<b>{_escape(name)}：</b>{_escape(_reason_cn(str(reason)))}"
    return ""


def _format_observed(metric: dict[str, Any]) -> str:
    observed_value = metric.get("observed_value")
    name = str(metric.get("metric_name", ""))
    summary = metric.get("observed_summary") or {}
    if observed_value is None:
        if isinstance(summary, dict) and summary.get("p95_abs") is not None:
            return f"{float(summary['p95_abs']):.1f} (p95 abs)"
        if isinstance(summary, dict) and summary.get("mean") is not None:
            return f"{float(summary['mean']):.1f} (mean)"
        return "n/a"
    value = float(observed_value)
    if "velocity" in name:
        return f"{value:.1f} deg/s"
    if "separation" in name or "flexion" in name:
        return f"{value:.1f} deg"
    return f"{value:.3f}"


def _source_short_names(reference_context: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for source_id in reference_context.get("source_ids", []):
        if source_id == "orishimo_2023":
            names.append("Orishimo 2023")
        elif source_id == "diffendaffer_2023":
            names.append("Diffendaffer 2023")
        elif source_id == "hiramoto_2019":
            names.append("Hiramoto 2019")
        elif source_id == "paul_2025":
            names.append("Paul 2025")
        elif source_id == "wilk_2011":
            names.append("Wilk 2011")
        elif source_id == "wright_2006":
            names.append("Wright 2006")
        elif source_id == "hamano_2020":
            names.append("Hamano 2020")
        elif source_id == "robb_2010":
            names.append("Robb 2010")
    return names


def _pick_overlay_frame(project_root: Path, clip_id: str, condition: str, action_type: str) -> Path | None:
    frame_dir = project_root / "outputs_full" / "overlays" / "frames" / clip_id / condition
    if not frame_dir.exists():
        return None
    feature_csv = project_root / "data_full" / "processed" / "features" / clip_id / f"{condition}.csv"
    preferred_index: int | None = None
    action_frames: set[int] = set()
    if feature_csv.exists():
        import csv
        with feature_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        action_frames = frame_indices_in_action_window(rows, action_type=action_type)
    pose_csv = project_root / "data_full" / "processed" / "poses" / clip_id / f"{condition}.csv"
    if pose_csv.exists():
        preferred_index = _best_stable_action_frame(read_pose_records(pose_csv), action_frames)
    if preferred_index is not None:
        preferred = frame_dir / f"{clip_id}__{condition}__frame_{preferred_index:06d}.png"
        if preferred.exists():
            return preferred
    candidates = sorted(frame_dir.glob("*.png"))
    return candidates[len(candidates) // 2] if candidates else None


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


def _scaled_image(path: Path, width: float, max_height: float) -> Image:
    image = Image(str(path))
    scale = min(width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return image


def _parse_markdown_sections(report_md: str) -> dict[str, str]:
    matches = re.findall(r"##\s+([^\n]+)\n(.*?)(?=\n##\s+|\Z)", report_md, flags=re.S)
    sections: dict[str, str] = {}
    for title, body in matches:
        cleaned = " ".join(line.strip() for line in body.strip().splitlines() if line.strip() and not line.strip().startswith("!"))
        sections[title.strip()] = cleaned
    return sections


def _humanize_metric_name(name: str) -> str:
    mapping = {
        "peak_trunk_rotation_velocity_deg_s": "Peak trunk rotation velocity",
        "peak_pelvis_rotation_velocity_deg_s": "Peak pelvis rotation velocity",
        "hip_shoulder_separation_foot_contact_deg": "Hip-shoulder separation proxy",
        "observed_left_knee_flexion_deg": "Left knee flexion",
        "observed_right_knee_flexion_deg": "Right knee flexion",
        "observed_hand_speed_proxy": "Hand speed proxy",
    }
    cn_mapping = {
        "Peak trunk rotation velocity": "躯干峰值旋转速度",
        "Peak pelvis rotation velocity": "骨盆峰值旋转速度",
        "Hip-shoulder separation proxy": "髋肩分离代理指标",
        "Left knee flexion": "左膝屈曲",
        "Right knee flexion": "右膝屈曲",
        "Hand speed proxy": "手部速度代理指标",
    }
    return cn_mapping.get(mapping.get(name, name.replace("_", " ")), name.replace("_", " "))


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def _draw_page(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D8E2EA"))
    canvas.setLineWidth(0.7)
    canvas.line(doc.leftMargin, doc.height + doc.topMargin + 0.16 * inch, letter[0] - doc.rightMargin, doc.height + doc.topMargin + 0.16 * inch)
    font_name = _register_cjk_font()
    canvas.setFont(font_name, 8)
    canvas.setFillColor(colors.HexColor("#6B7C8D"))
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.45 * inch, f"第 {doc.page} 页")
    canvas.restoreState()


def _register_cjk_font() -> str:
    candidates = [
        ("/Library/Fonts/Arial Unicode.ttf", "ArialUnicode"),
        ("/System/Library/Fonts/STHeiti Medium.ttc", "STHeitiMedium"),
        ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "ArialUnicodeSupplemental"),
    ]
    for path, font_name in candidates:
        if Path(path).exists():
            if font_name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(font_name, path))
            return font_name
    return "Helvetica"


def _section_title_cn(title: str) -> str:
    mapping = {
        "Summary": "动作优点",
        "Coaching Priorities": "训练重点",
        "Injury Prevention Context": "伤病预防提示",
    }
    return mapping.get(title, title)


def _action_type_cn(value: str) -> str:
    return {"batting": "击球", "pitching": "投球"}.get(value, value)


def _band_cn(value: str) -> str:
    mapping = {
        "below 1sd": "低于参考区间",
        "within 1sd": "位于参考区间内",
        "above 1sd": "高于参考区间",
    }
    return mapping.get(value, value)


def _reason_cn(text: str) -> str:
    mapping = {
        "Observed from current feature CSV but not matched to a standard reference metric.": "当前可以从本次特征表直接观察到该指标，但还没有匹配到稳定可比的标准参考值。",
        "Current feature CSV stores left/right knee angles, but it does not identify lead side or foot-contact timing.": "当前特征表有左右膝角度，但还不能自动识别前导腿，也不能自动定位脚落地时刻。",
    }
    return mapping.get(text, text)


def _limitation_cn(text: str) -> str:
    mapping = {
        "Current summary is built from 2D pose-derived feature CSV outputs.": "当前结论基于 2D 姿态视频特征，不是三维动作捕捉结果。",
        "Clinical passive ROM metrics remain unavailable unless separate physical screening data are provided.": "除非额外提供体格筛查或关节活动度测量数据，否则无法得到临床被动活动度指标。",
    }
    return mapping.get(text, text)


def _note_cn(text: str) -> str:
    mapping = {
        "This reference was derived from pitching literature; use cautiously for batting.": "该参考值主要来自投球研究，用于击球动作时需要谨慎解释。",
        "Current pipeline summarizes full-clip separation and does not detect foot contact, so this is only a partial proxy for the foot-contact reference. Reference interpretation is pitching-specific.": "当前流程只统计整段视频中的髋肩分离变化，不能自动定位脚落地时刻，因此这里只能作为脚落地分离角的部分代理指标；参考解释也主要来自投球研究。",
    }
    return mapping.get(text, text)


def _measurement_source_cn(text: str) -> str:
    mapping = {
        "2d_pose_video": "2D 姿态视频分析",
    }
    return mapping.get(text, text)


if __name__ == "__main__":
    main()
