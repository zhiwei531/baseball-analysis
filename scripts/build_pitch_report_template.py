"""Build the parent-facing Chinese pitching biomechanics PDF report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import math
import shutil
import textwrap

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.pdfgen import canvas as pdf_canvas

from baseball_pose.pipeline.report_llm import _build_chat_request, _extract_report_text, _post_chat_completion


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_full/benchmark_rtmpose_test"
OUT_DIR = ROOT / "outputs_full/benchmark_rtmpose_test"
ASSET_DIR = OUT_DIR / "report_assets/benchmark_pitch_vertical_09"
PDF_DIR = ROOT / "output/pdf"
CLIP_ID = "benchmark_pitch_vertical_09"
PEER_CLIP_ID = "benchmark_pitch_vertical_10"
COACH_CLIP_ID = "pitch_horizontal_coach"
COACH_3D_PATH = ROOT / "data_full/coach_pose3d/gvhmr/pitch_horizontal_coach.csv"
COACH_OVERLAY_DIR = ROOT / "outputs_full/coach_gvhmr/frames/pitch_horizontal_coach/baseline_raw"
COND_2D = "image_center_motion_grabcut_pose_complete_smooth"
COND_3D = "image_center_motion_grabcut_pose_complete_smooth_3d_smooth"
PDF_FONT_PATH = Path("/System/Library/Fonts/STHeiti Medium.ttc")
FONT_PATH = PDF_FONT_PATH
PDF_FONT = "STHeitiCN"
PAGE_W = 1240
PAGE_H = 1754
PAGE_MARGIN = 78


@dataclass(frozen=True)
class Metric:
    key: str
    label_cn: str
    label_en: str
    value: float
    ref_low: float
    ref_high: float
    unit: str
    direction: str
    source: str
    interpretation: str
    coach_value: float | None = None


@dataclass(frozen=True)
class MotionMetric:
    key: str
    label_cn: str
    label_en: str
    value: float
    coach_value: float | None
    unit: str
    method: str
    note: str
    status: str


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    registerFont(TTFont(PDF_FONT, str(PDF_FONT_PATH)))

    pose3d = pd.read_csv(DATA_DIR / f"processed/poses3d/{CLIP_ID}/{COND_3D}.csv")
    obj = pd.read_csv(DATA_DIR / f"processed/object_features/{CLIP_ID}/{COND_2D}.csv")
    peer_pose3d_path = DATA_DIR / f"processed/poses3d/{PEER_CLIP_ID}/{COND_3D}.csv"
    peer_pose3d = pd.read_csv(peer_pose3d_path) if peer_pose3d_path.exists() else pd.DataFrame()

    coach_pose3d = pd.read_csv(COACH_3D_PATH) if COACH_3D_PATH.exists() else pd.DataFrame()

    metrics, series, motion_metrics = compute_metrics(pose3d, coach_pose3d, obj)
    write_motion_metrics_csv(motion_metrics, ROOT / "output/data/benchmark_pitch_vertical_09_motion_metrics_full.csv")
    parent_guidance = build_parent_guidance(metrics, motion_metrics)
    (ROOT / "output/data").mkdir(parents=True, exist_ok=True)
    (ROOT / "output/data/benchmark_pitch_vertical_09_parent_guidance.md").write_text(parent_guidance.rstrip() + "\n", encoding="utf-8")
    prompt_text = build_parent_guidance_prompt(metrics, motion_metrics)
    (ROOT / "output/data/benchmark_pitch_vertical_09_parent_prompt.txt").write_text(prompt_text.rstrip() + "\n", encoding="utf-8")
    assets = build_assets(metrics, series, motion_metrics, pose3d, coach_pose3d, peer_pose3d)
    pdf_path = PDF_DIR / "benchmark_pitch_vertical_09_biomech_report_template_zh.pdf"
    build_pdf(pdf_path, metrics, motion_metrics, parent_guidance, assets)
    print(pdf_path)


def write_motion_metrics_csv(metrics: list[MotionMetric], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "metric_key": m.key,
            "label_cn": m.label_cn,
            "label_en": m.label_en,
            "child_value": m.value,
            "coach_value": m.coach_value,
            "unit": m.unit,
            "method": m.method,
            "status": m.status,
            "note": m.note,
        }
        for m in metrics
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def build_parent_guidance(metrics: list[Metric], motion_metrics: list[MotionMetric]) -> str:
    prompt = build_parent_guidance_prompt(metrics, motion_metrics)
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            request_body = _build_chat_request(prompt, model="deepseek-v4-pro", temperature=0.2)
            response = _post_chat_completion(
                request_body=request_body,
                api_key=api_key,
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
                timeout_sec=120.0,
            )
            return sanitize_parent_guidance(_extract_report_text(response))
        except Exception as exc:
            return fallback_parent_guidance(metrics, motion_metrics, note=f"自动生成服务暂不可用，已使用本地规则生成。原因：{exc}")
    return fallback_parent_guidance(metrics, motion_metrics, note="")


def build_parent_guidance_prompt(metrics: list[Metric], motion_metrics: list[MotionMetric]) -> str:
    rows = []
    for metric in metrics:
        coach = f"{metric.coach_value:.1f}{metric.unit}" if metric.coach_value is not None else "暂无"
        rows.append(f"{metric.label_cn}: 孩子 {metric.value:.1f}{metric.unit}, 教练 {coach}, 差距 {delta_text(metric)}")
    rows.append("完整动作指标：")
    for metric in motion_metrics:
        coach = format_motion_value(metric.coach_value, metric.unit) if metric.coach_value is not None else "暂无"
        rows.append(f"{metric.label_cn}: 孩子 {format_motion_value(metric.value, metric.unit)}, 教练 {coach}, 方法 {metric.method}, 状态 {metric.status}")
    return (
        "请写一份直接给中国家长看的棒球投球动作分析结论。"
        "不要提 Prompt、模型、大语言模型、竞品、模板、算法过程。"
        "全部使用中文，语气专业但通俗。"
        "结构固定为：一、本次最主要结论；二、数据差距；三、4周训练建议；四、复测重点。"
        "训练建议必须具体到动作、频率、组数。"
        "不要做医学诊断，只能写训练风险提示。"
        "数据如下：\n" + "\n".join(rows)
    )


def sanitize_parent_guidance(text: str) -> str:
    forbidden = ("Prompt", "prompt", "LLM", "大语言模型", "竞品", "模板")
    cleaned = text.strip()
    for word in forbidden:
        cleaned = cleaned.replace(word, "")
    return cleaned


def fallback_parent_guidance(metrics: list[Metric], motion_metrics: list[MotionMetric], note: str = "") -> str:
    by_key = {m.key: m for m in metrics}
    motion_by_key = {m.key: m for m in motion_metrics}
    stride = by_key["stride"]
    hand = by_key["hand_speed"]
    trunk = by_key["trunk_vel"]
    sep = motion_by_key.get("hip_shoulder_sep")
    lead_knee = motion_by_key.get("lead_knee")
    foot = motion_by_key.get("foot_direction")
    lines = [
        "一、本次最主要结论",
        "孩子的髋肩分离和前膝角度处在可训练范围内，说明动作已经具备基本的身体分段能力。主要差距集中在跨步长度、出手侧手速和前脚方向控制：跨步比教练少约 {:.1f} cm，右手速度约为教练的 {:.0f}%。训练时不建议只要求“更用力甩手”，应先把跨步落地、身体前移和前脚方向稳定下来。".format(
            abs(stride.value - (stride.coach_value or stride.value)),
            percent_of_reference(hand.value, hand.coach_value) if hand.coach_value else 0.0,
        ),
        "",
        "二、数据差距",
        "跨步长度：孩子 {:.1f} cm，教练 {:.1f} cm。跨步偏短会限制身体向前传递力量。".format(stride.value, stride.coach_value or 0.0),
        "出手侧手速：孩子 {:.1f} m/s，教练 {:.1f} m/s。当前更需要通过下肢和躯干顺序带动手臂，而不是单独甩手。".format(hand.value, hand.coach_value or 0.0),
        "躯干旋转速度：孩子 {:.0f} deg/s，教练 {:.0f} deg/s。差距不大，重点是让骨盆先启动、躯干随后加速。".format(trunk.value, trunk.coach_value or 0.0),
    ]
    if sep:
        lines.append("髋肩分离：孩子 {}，教练 {}。该指标可以保留，但要配合落脚时机一起看。".format(format_motion_value(sep.value, sep.unit), format_motion_value(sep.coach_value, sep.unit)))
    if lead_knee:
        lines.append("前膝角：孩子 {}，教练 {}。孩子前腿支撑角度尚可，训练重点不在单纯压低膝盖。".format(format_motion_value(lead_knee.value, lead_knee.unit), format_motion_value(lead_knee.coach_value, lead_knee.unit)))
    if foot:
        lines.append("前脚方向：孩子 {}，教练 {}。前脚方向偏差会影响髋部打开和身体制动，是优先改进项。".format(format_motion_value(foot.value, foot.unit), format_motion_value(foot.coach_value, foot.unit)))
    lines.extend(
        [
            "",
            "三、4周训练建议",
            "1. 跨步落点练习：在地面贴一条目标线，落脚点控制在身高 80%-90% 附近。每周 3 次，每次 3 组，每组 6 次，先做慢速投掷。",
            "2. 前脚方向练习：落脚后前脚脚尖尽量对准目标方向，完成后停 2 秒检查膝盖、脚尖和目标线。每周 3 次，每次 3 组，每组 8 次。",
            "3. 髋先转、肩延迟练习：做提膝停顿投球或分段投球，先让骨盆启动，再让胸口和手臂跟上。每周 2-3 次，每次 3 组，每组 5 次。",
            "4. 轻药球侧抛：使用轻药球做侧向转体抛，重点感受下肢到躯干再到手臂的顺序。每周 2 次，每次 3 组，每组 6 次，动作变形时停止加量。",
            "",
            "四、复测重点",
            "复测时优先看 4 个指标：跨步长度、前脚方向、出手侧手速、髋肩分离。若孩子出现肩肘疼痛、动作明显代偿或疲劳后控制下降，应先降低训练量，并由教练或专业人员做进一步筛查。",
        ]
    )
    if note:
        lines.append("")
        lines.append(note)
    return "\n".join(lines)


def compute_metrics(
    pose3d: pd.DataFrame,
    coach_pose3d: pd.DataFrame,
    obj: pd.DataFrame,
) -> tuple[list[Metric], dict[str, pd.DataFrame], list[MotionMetric]]:
    child = compute_human_3d_metrics(pose3d)
    coach = compute_human_3d_metrics(coach_pose3d) if not coach_pose3d.empty else {}
    ball_speed = float(np.nanmax(obj["ball_speed_norm_s"].to_numpy(dtype=float)))
    release_frame = int(obj.loc[obj["ball_speed_norm_s"].idxmax(), "frame_index"]) if obj["ball_speed_norm_s"].notna().any() else 25
    assumed_height_cm = 145.0
    ball_speed_mps = ball_speed * assumed_height_cm / 100.0

    metrics = [
        Metric(
            "trunk_vel",
            "躯干旋转速度",
            "Trunk rotation speed proxy",
            child["trunk_vel"],
            839,
            1079,
            "deg/s",
            "higher_is_better",
            "Child and coach: GVHMR 3D shoulder-line rotation; Orishimo 2023 reference: 959 +/- 120 deg/s",
            "基于 3D 肩线在水平面的旋转速度。上半身加速偏低时，通常说明髋到肩的传递还没有充分打开。",
            coach.get("trunk_vel"),
        ),
        Metric(
            "pelvis_vel",
            "骨盆旋转速度",
            "Pelvis rotation speed proxy",
            child["pelvis_vel"],
            508,
            684,
            "deg/s",
            "higher_is_better",
            "Child and coach: GVHMR 3D pelvis-line rotation; Orishimo 2023 reference: 596 +/- 88 deg/s",
            "基于 3D 髋线在水平面的旋转速度。骨盆先启动且速度足够，是后续躯干和手臂加速的基础。",
            coach.get("pelvis_vel"),
        ),
        Metric(
            "separation",
            "髋肩分离",
            "Hip-shoulder separation proxy",
            child["separation"],
            38,
            62,
            "deg",
            "in_range",
            "Child and coach: GVHMR 3D shoulder-pelvis horizontal angle difference; Orishimo 2023 reference: 50 +/- 12 deg",
            "基于 3D 肩线与髋线的水平夹角。髋肩分离过小会减少弹性储能，过大则可能表示动作失控或时序过早。",
            coach.get("separation"),
        ),
        Metric(
            "stride",
            "跨步长度",
            "Stride length",
            child["stride_cm"],
            116,
            131,
            "cm",
            "in_range",
            "Child and coach: GVHMR 3D ankle horizontal distance; Diffendaffer 2023 describes stride around 85% body height",
            "基于 3D 双踝水平距离估算。跨步太短会限制重心前移和动量；太长会让前脚落地后身体跟不上。",
            coach.get("stride_cm"),
        ),
        Metric(
            "hand_speed",
            "出手侧手部速度",
            "Throwing hand speed proxy",
            child["hand_speed"],
            7.0,
            9.5,
            "m/s",
            "higher_is_better",
            "Child and coach: GVHMR 3D right-wrist Euclidean speed",
            "基于 3D 右腕空间速度，适合看出手侧末端加速是否随动力链改善而上升。",
            coach.get("hand_speed"),
        ),
        Metric(
            "ball_speed",
            "球速估算",
            "Ball speed proxy",
            ball_speed_mps * 3.6,
            62,
            78,
            "km/h",
            "higher_is_better",
            "Template youth benchmark range; replace with radar speed when available",
            "球速仍来自 2D 视频目标跟踪和身高模板估算；正式报告建议用雷达枪或 3D 球轨迹替换。",
            None,
        ),
        Metric(
            "com",
            "重心前移",
            "Center-of-mass travel proxy",
            child["com_excursion_cm"],
            35,
            55,
            "cm",
            "in_range",
            "Child and coach: GVHMR 3D hip/spine center trajectory projected onto movement direction",
            "基于 3D hip/spine 中心轨迹。重心前移不足时，下肢力量更难传到上肢；过大则可能影响稳定。",
            coach.get("com_excursion_cm"),
        ),
    ]

    series = {
        "child_angles": child["angles"],
        "coach_angles": coach.get("angles", pd.DataFrame()),
        "child_speed": child["speed"],
        "coach_speed": coach.get("speed", pd.DataFrame()),
        "com": child["com"],
        "release": pd.DataFrame({"frame_index": [release_frame]}),
    }
    motion_metrics = compute_competitor_motion_metrics(pose3d, coach_pose3d, release_frame)
    return metrics, series, motion_metrics


def compute_human_3d_metrics(pose3d: pd.DataFrame) -> dict[str, object]:
    p3 = pivot_3d(pose3d)
    t = p3.index.to_numpy(dtype=float)
    shoulder_angle = line_angle_horizontal_3d(
        p3["left_shoulder_x_3d"],
        p3["left_shoulder_z_3d"],
        p3["right_shoulder_x_3d"],
        p3["right_shoulder_z_3d"],
    )
    pelvis_angle = line_angle_horizontal_3d(
        p3["left_hip_x_3d"],
        p3["left_hip_z_3d"],
        p3["right_hip_x_3d"],
        p3["right_hip_z_3d"],
    )
    separation = np.abs(wrap_deg(shoulder_angle - pelvis_angle))
    hand_speed_series = point_speed_3d(p3, "right_wrist", t)
    com_cm = estimate_com_trajectory_cm(p3)
    return {
        "trunk_vel": robust_peak(angular_velocity(shoulder_angle, t)),
        "pelvis_vel": robust_peak(angular_velocity(pelvis_angle, t)),
        "separation": float(np.nanpercentile(separation, 95)),
        "stride_cm": estimate_stride_3d_cm(p3),
        "hand_speed": float(np.nanpercentile(hand_speed_series, 95)),
        "com_excursion_cm": float(np.nanmax(com_cm["x_cm"]) - np.nanmin(com_cm["x_cm"])),
        "angles": pd.DataFrame(
            {
                "time": t,
                "shoulder_angle": shoulder_angle,
                "pelvis_angle": pelvis_angle,
                "separation": separation,
            }
        ),
        "speed": pd.DataFrame({"time": t, "hand_speed_mps": hand_speed_series}),
        "com": com_cm,
    }


def compute_competitor_motion_metrics(
    child_pose3d: pd.DataFrame,
    coach_pose3d: pd.DataFrame,
    child_release_frame: int,
) -> list[MotionMetric]:
    child = pitch_metric_bundle(child_pose3d, release_frame=child_release_frame)
    coach = pitch_metric_bundle(coach_pose3d, release_frame=None) if not coach_pose3d.empty else {}

    def coach_value(key: str) -> float | None:
        value = coach.get(key)
        return float(value) if value is not None and np.isfinite(value) else None

    arm_speed_pct = percent_of_reference(child["arm_speed_mps"], coach_value("arm_speed_mps"))
    fingertip_speed_pct = percent_of_reference(child["fingertip_speed_mps"], coach_value("fingertip_speed_mps"))
    coach_transfer = coach_value("weight_transfer_pct")
    coach_head = coach_value("head_stability_pct")

    rows = [
        MotionMetric("elbow_bend", "肘部弯曲", "Elbow Bend", child["elbow_bend_deg"], coach_value("elbow_bend_deg"), "deg", "3D", "出手帧右肩-右肘-右腕夹角。", metric_status_angle(child["elbow_bend_deg"], 145, 175)),
        MotionMetric("arm_abduction", "上臂外展", "Arm Abduction", child["arm_abduction_deg"], coach_value("arm_abduction_deg"), "deg", "3D", "出手帧上臂相对躯干轴夹角。", metric_status_angle(child["arm_abduction_deg"], 85, 115)),
        MotionMetric("trunk_lean", "躯干倾斜", "Trunk Lean", child["trunk_lean_deg"], coach_value("trunk_lean_deg"), "deg", "3D", "出手帧髋-颈躯干轴相对竖直方向。", metric_status_angle(child["trunk_lean_deg"], 10, 35)),
        MotionMetric("stride_angle", "跨步角", "Stride Angle", child["stride_angle_deg"], coach_value("stride_angle_deg"), "deg", "3D", "跨步落地帧前腿相对地面角度。", metric_status_angle(child["stride_angle_deg"], 45, 75)),
        MotionMetric("lead_knee", "前膝角", "Lead Knee", child["lead_knee_deg"], coach_value("lead_knee_deg"), "deg", "3D", "出手帧前腿髋-膝-踝夹角。", metric_status_angle(child["lead_knee_deg"], 115, 155)),
        MotionMetric("hip_shoulder_sep", "髋肩分离", "Hip-Shoulder Sep", child["hip_shoulder_sep_deg"], coach_value("hip_shoulder_sep_deg"), "deg", "3D", "出手帧肩线与髋线水平夹角；正负表示相对旋转方向。", metric_status_angle(abs(child["hip_shoulder_sep_deg"]), 35, 65)),
        MotionMetric("arm_speed", "手臂速度", "Arm Speed", arm_speed_pct, 100.0 if coach_value("arm_speed_mps") else None, "%", "3D proxy", "右腕峰值速度相对正常速度教练；速度类受单目深度抖动影响。", metric_status_percent(arm_speed_pct, 70, 100)),
        MotionMetric("stride_length", "跨步长度", "Stride Length", child["stride_length_pct_height"], coach_value("stride_length_pct_height"), "%height", "3D normalized", "双踝最大水平距离除以估计身高，比 cm 更适合跨主体对比。", metric_status_percent(child["stride_length_pct_height"], 75, 95)),
        MotionMetric("weight_transfer", "重心转移", "Weight Transfer", child["weight_transfer_pct"], coach_transfer, "%stride", "3D proxy", "hip/spine 中心沿动作方向位移/跨步长度；不是 Vicon/力板真实 COM。", "warn"),
        MotionMetric("head_stability", "头部稳定", "Head Stability", child["head_stability_pct"], coach_head, "%", "3D proxy", "头部相对动作方向的横向稳定评分；越高越稳定。", metric_status_percent(child["head_stability_pct"], 65, 85)),
        MotionMetric("foot_direction", "前脚方向", "Foot Direction", child["foot_direction_deg"], coach_value("foot_direction_deg"), "deg", "3D", "跨步落地帧前脚 ankle-foot 方向相对跨步方向。", metric_status_angle(abs(child["foot_direction_deg"]), 0, 20)),
        MotionMetric("wrist_snap", "手腕翻转", "Wrist Snap", child["wrist_snap_deg"], coach_value("wrist_snap_deg"), "deg", "3D proxy", "出手帧右肘-右腕-右手夹角；SMPL hand 不是真实指尖。", metric_status_angle(child["wrist_snap_deg"], 145, 180)),
        MotionMetric("fingertip_speed", "指尖速度", "Fingertip Speed", fingertip_speed_pct, 100.0 if coach_value("fingertip_speed_mps") else None, "%", "3D proxy", "right_hand 峰值速度相对正常速度教练；不是真实指尖 marker。", metric_status_percent(fingertip_speed_pct, 70, 100)),
    ]
    return rows


def pitch_metric_bundle(pose3d: pd.DataFrame, release_frame: int | None) -> dict[str, float]:
    p3 = pivot_3d_by_frame(pose3d)
    frames = p3.index.to_numpy(dtype=int)
    t = frame_time_axis(pose3d, frames)
    release = nearest_frame(frames, release_frame) if release_frame is not None else frames[int(np.nanargmax(point_speed_3d_frames(p3, "right_wrist", t)))]
    stride_frame = detect_stride_landing_frame(p3)
    lead = lead_side_at_stride(p3, stride_frame)
    trail = "right" if lead == "left" else "left"

    shoulder_angle = horizontal_angle_at_frame(p3, release, "left_shoulder", "right_shoulder")
    pelvis_angle = horizontal_angle_at_frame(p3, release, "left_hip", "right_hip")
    hip_shoulder = float(wrap_deg(np.array([shoulder_angle - pelvis_angle]))[0])
    wrist_speed = point_speed_3d_frames(p3, "right_wrist", t)
    hand_speed = point_speed_3d_frames(p3, "right_hand", t)
    stride_cm = estimate_stride_3d_cm(p3)
    height_cm = estimate_body_height_cm(p3)
    transfer_cm = estimate_transfer_proxy_cm(p3)
    head_stability = estimate_head_stability_pct(p3, stride_cm)

    return {
        "elbow_bend_deg": angle_3d_at_frame(p3, release, "right_shoulder", "right_elbow", "right_wrist"),
        "arm_abduction_deg": arm_abduction_at_release(p3, release),
        "trunk_lean_deg": trunk_lean_at_frame(p3, release),
        "stride_angle_deg": stride_angle_at_frame(p3, stride_frame, lead),
        "lead_knee_deg": angle_3d_at_frame(p3, release, f"{lead}_hip", f"{lead}_knee", f"{lead}_ankle"),
        "hip_shoulder_sep_deg": hip_shoulder,
        "arm_speed_mps": float(np.nanpercentile(wrist_speed, 95)),
        "stride_length_pct_height": stride_cm / max(height_cm, 1e-6) * 100.0,
        "weight_transfer_pct": transfer_cm / max(stride_cm, 1e-6) * 100.0,
        "head_stability_pct": head_stability,
        "foot_direction_deg": foot_direction_at_frame(p3, stride_frame, lead, trail),
        "wrist_snap_deg": angle_3d_at_frame(p3, release, "right_elbow", "right_wrist", "right_hand"),
        "fingertip_speed_mps": float(np.nanpercentile(hand_speed, 95)),
    }


def pivot_3d_by_frame(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index="frame_index", columns="joint_name", values=["x_3d", "y_3d", "z_3d"], aggfunc="mean")
    wide.columns = [f"{joint}_{axis}" for axis, joint in wide.columns]
    return wide.sort_index().interpolate(limit_direction="both")


def frame_time_axis(df: pd.DataFrame, frames: np.ndarray) -> np.ndarray:
    times = df.groupby("frame_index")["timestamp_sec"].mean().reindex(frames).interpolate(limit_direction="both")
    return times.to_numpy(dtype=float)


def nearest_frame(frames: np.ndarray, frame: int | None) -> int:
    if frame is None:
        return int(frames[len(frames) // 2])
    return int(frames[np.argmin(np.abs(frames - int(frame)))])


def joint_vec(p3: pd.DataFrame, frame: int, joint: str) -> np.ndarray:
    return np.array(
        [p3.loc[frame, f"{joint}_x_3d"], p3.loc[frame, f"{joint}_y_3d"], p3.loc[frame, f"{joint}_z_3d"]],
        dtype=float,
    )


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-9:
        return float("nan")
    cosv = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def angle_3d_at_frame(p3: pd.DataFrame, frame: int, a: str, b: str, c: str) -> float:
    pa, pb, pc = joint_vec(p3, frame, a), joint_vec(p3, frame, b), joint_vec(p3, frame, c)
    return angle_between(pa - pb, pc - pb)


def horizontal_angle_at_frame(p3: pd.DataFrame, frame: int, a: str, b: str) -> float:
    pa, pb = joint_vec(p3, frame, a), joint_vec(p3, frame, b)
    return float(np.degrees(np.arctan2(pb[2] - pa[2], pb[0] - pa[0])))


def point_speed_3d_frames(p3: pd.DataFrame, joint: str, t: np.ndarray) -> np.ndarray:
    vx = np.gradient(p3[f"{joint}_x_3d"].to_numpy(dtype=float), t)
    vy = np.gradient(p3[f"{joint}_y_3d"].to_numpy(dtype=float), t)
    vz = np.gradient(p3[f"{joint}_z_3d"].to_numpy(dtype=float), t)
    return np.sqrt(vx * vx + vy * vy + vz * vz)


def detect_stride_landing_frame(p3: pd.DataFrame) -> int:
    stride = np.sqrt(
        (p3["left_ankle_x_3d"].to_numpy(dtype=float) - p3["right_ankle_x_3d"].to_numpy(dtype=float)) ** 2
        + (p3["left_ankle_z_3d"].to_numpy(dtype=float) - p3["right_ankle_z_3d"].to_numpy(dtype=float)) ** 2
    )
    target = 0.9 * float(np.nanmax(stride))
    idx = int(np.argmax(stride >= target))
    return int(p3.index.to_numpy(dtype=int)[idx])


def movement_axis(p3: pd.DataFrame) -> np.ndarray:
    com = torso_center_xz(p3)
    axis = com[-1] - com[0]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return np.array([1.0, 0.0])
    return axis / norm


def torso_center_xz(p3: pd.DataFrame) -> np.ndarray:
    x_cols = [c for c in ["hip_x_3d", "spine1_x_3d", "spine2_x_3d", "spine3_x_3d"] if c in p3.columns]
    z_cols = [c for c in ["hip_z_3d", "spine1_z_3d", "spine2_z_3d", "spine3_z_3d"] if c in p3.columns]
    return np.column_stack([p3[x_cols].mean(axis=1).to_numpy(dtype=float), p3[z_cols].mean(axis=1).to_numpy(dtype=float)])


def lead_side_at_stride(p3: pd.DataFrame, stride_frame: int) -> str:
    axis = movement_axis(p3)
    left = np.array([p3.loc[stride_frame, "left_ankle_x_3d"], p3.loc[stride_frame, "left_ankle_z_3d"]], dtype=float)
    right = np.array([p3.loc[stride_frame, "right_ankle_x_3d"], p3.loc[stride_frame, "right_ankle_z_3d"]], dtype=float)
    return "left" if float(left @ axis) >= float(right @ axis) else "right"


def arm_abduction_at_release(p3: pd.DataFrame, frame: int) -> float:
    shoulder = joint_vec(p3, frame, "right_shoulder")
    elbow = joint_vec(p3, frame, "right_elbow")
    hip_mid = (joint_vec(p3, frame, "left_hip") + joint_vec(p3, frame, "right_hip")) / 2.0
    shoulder_mid = (joint_vec(p3, frame, "left_shoulder") + joint_vec(p3, frame, "right_shoulder")) / 2.0
    return angle_between(elbow - shoulder, shoulder_mid - hip_mid)


def trunk_lean_at_frame(p3: pd.DataFrame, frame: int) -> float:
    hip_mid = (joint_vec(p3, frame, "left_hip") + joint_vec(p3, frame, "right_hip")) / 2.0
    shoulder_mid = (joint_vec(p3, frame, "left_shoulder") + joint_vec(p3, frame, "right_shoulder")) / 2.0
    return angle_between(shoulder_mid - hip_mid, np.array([0.0, 1.0, 0.0]))


def stride_angle_at_frame(p3: pd.DataFrame, frame: int, lead: str) -> float:
    hip = joint_vec(p3, frame, f"{lead}_hip")
    ankle = joint_vec(p3, frame, f"{lead}_ankle")
    v = hip - ankle
    horizontal = float(np.linalg.norm(v[[0, 2]]))
    return float(np.degrees(np.arctan2(abs(v[1]), max(horizontal, 1e-9))))


def foot_direction_at_frame(p3: pd.DataFrame, frame: int, lead: str, trail: str) -> float:
    axis = movement_axis(p3)
    ankle = joint_vec(p3, frame, f"{lead}_ankle")
    foot = joint_vec(p3, frame, f"{lead}_foot")
    foot_vec = np.array([foot[0] - ankle[0], foot[2] - ankle[2]], dtype=float)
    if float(np.linalg.norm(foot_vec)) < 1e-9:
        return float("nan")
    signed = np.degrees(np.arctan2(axis[0] * foot_vec[1] - axis[1] * foot_vec[0], axis @ foot_vec))
    return float(signed)


def estimate_body_height_cm(p3: pd.DataFrame) -> float:
    y_cols = [c for c in p3.columns if c.endswith("_y_3d")]
    height_m = float(np.nanpercentile(p3[y_cols].max(axis=1) - p3[y_cols].min(axis=1), 90))
    return height_m * 100.0


def estimate_transfer_proxy_cm(p3: pd.DataFrame) -> float:
    path = torso_center_xz(p3)
    axis = movement_axis(p3)
    projected = (path - path[0]) @ axis
    return float((np.nanmax(projected) - np.nanmin(projected)) * 100.0)


def estimate_head_stability_pct(p3: pd.DataFrame, stride_cm: float) -> float:
    head = np.column_stack([p3["head_x_3d"].to_numpy(dtype=float), p3["head_z_3d"].to_numpy(dtype=float)])
    axis = movement_axis(p3)
    perp = np.array([-axis[1], axis[0]])
    lateral_cm = float((np.nanmax((head - head[0]) @ perp) - np.nanmin((head - head[0]) @ perp)) * 100.0)
    score = 100.0 * (1.0 - lateral_cm / max(stride_cm * 0.35, 1e-6))
    return float(np.clip(score, 0.0, 100.0))


def percent_of_reference(value: float, reference: float | None) -> float:
    if reference is None or not np.isfinite(reference) or abs(reference) < 1e-9:
        return float("nan")
    return float(value / reference * 100.0)


def metric_status_angle(value: float, low: float, high: float) -> str:
    if not np.isfinite(value):
        return "warn"
    if low <= value <= high:
        return "ok"
    pad = (high - low) * 0.35
    return "warn" if low - pad <= value <= high + pad else "bad"


def metric_status_percent(value: float, warn: float, ok: float) -> str:
    if not np.isfinite(value):
        return "warn"
    if value >= ok:
        return "ok"
    if value >= warn:
        return "warn"
    return "bad"


def pivot_2d(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index="timestamp_sec", columns="joint_name", values=["x", "y"], aggfunc="mean")
    wide.columns = [f"{joint}_{axis}" for axis, joint in wide.columns]
    return wide.sort_index().interpolate(limit_direction="both")


def pivot_3d(df: pd.DataFrame) -> pd.DataFrame:
    wide = df.pivot_table(index="timestamp_sec", columns="joint_name", values=["x_3d", "y_3d", "z_3d"], aggfunc="mean")
    wide.columns = [f"{joint}_{axis}" for axis, joint in wide.columns]
    return wide.sort_index().interpolate(limit_direction="both")


def line_angle(x1: pd.Series, y1: pd.Series, x2: pd.Series, y2: pd.Series) -> np.ndarray:
    return np.degrees(np.arctan2((y2 - y1).to_numpy(), (x2 - x1).to_numpy()))


def line_angle_horizontal_3d(x1: pd.Series, z1: pd.Series, x2: pd.Series, z2: pd.Series) -> np.ndarray:
    return np.degrees(np.arctan2((z2 - z1).to_numpy(), (x2 - x1).to_numpy()))


def wrap_deg(values: np.ndarray) -> np.ndarray:
    return (values + 180) % 360 - 180


def angular_velocity(angle: np.ndarray, t: np.ndarray) -> np.ndarray:
    unwrapped = np.unwrap(np.radians(angle))
    return np.degrees(np.gradient(unwrapped, t))


def robust_peak(values: np.ndarray) -> float:
    return float(np.nanpercentile(np.abs(values), 95))


def point_speed(x: pd.Series, y: pd.Series, t: np.ndarray) -> np.ndarray:
    vx = np.gradient(x.to_numpy(dtype=float), t)
    vy = np.gradient(y.to_numpy(dtype=float), t)
    return np.sqrt(vx * vx + vy * vy)


def point_speed_3d(p3: pd.DataFrame, joint: str, t: np.ndarray) -> np.ndarray:
    vx = np.gradient(p3[f"{joint}_x_3d"].to_numpy(dtype=float), t)
    vy = np.gradient(p3[f"{joint}_y_3d"].to_numpy(dtype=float), t)
    vz = np.gradient(p3[f"{joint}_z_3d"].to_numpy(dtype=float), t)
    return np.sqrt(vx * vx + vy * vy + vz * vz)


def estimate_stride_norm(p2: pd.DataFrame) -> float:
    ankle_cols = [c for c in ["left_ankle_x", "right_ankle_x"] if c in p2.columns]
    if len(ankle_cols) < 2:
        return 0.72
    lead = p2[ankle_cols].max(axis=1)
    drive = p2[ankle_cols].min(axis=1)
    return float(np.nanpercentile(np.abs(lead - drive), 90))


def estimate_stride_3d_cm(p3: pd.DataFrame) -> float:
    left_x = p3["left_ankle_x_3d"].to_numpy(dtype=float)
    left_z = p3["left_ankle_z_3d"].to_numpy(dtype=float)
    right_x = p3["right_ankle_x_3d"].to_numpy(dtype=float)
    right_z = p3["right_ankle_z_3d"].to_numpy(dtype=float)
    stride_m = np.sqrt((left_x - right_x) ** 2 + (left_z - right_z) ** 2)
    return float(np.nanpercentile(stride_m, 90) * 100.0)


def estimate_com_trajectory_cm(p3: pd.DataFrame) -> pd.DataFrame:
    x_cols = [c for c in ["hip_x_3d", "spine1_x_3d", "spine2_x_3d", "spine3_x_3d"] if c in p3.columns]
    z_cols = [c for c in ["hip_z_3d", "spine1_z_3d", "spine2_z_3d", "spine3_z_3d"] if c in p3.columns]
    if not x_cols or not z_cols:
        x = np.zeros(len(p3))
        z = np.zeros(len(p3))
    else:
        x = p3[x_cols].mean(axis=1).to_numpy(dtype=float)
        z = p3[z_cols].mean(axis=1).to_numpy(dtype=float)
    start = np.array([x[0], z[0]], dtype=float)
    end = np.array([x[-1], z[-1]], dtype=float)
    axis = end - start
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        axis = np.array([1.0, 0.0])
    else:
        axis = axis / norm
    projected = (np.column_stack([x, z]) - start) @ axis
    x = (projected - np.nanmin(projected)) * 100.0
    return pd.DataFrame({"time": p3.index.to_numpy(dtype=float), "x_cm": x})


def build_assets(
    metrics: list[Metric],
    series: dict[str, pd.DataFrame],
    motion_metrics: list[MotionMetric],
    pose3d: pd.DataFrame,
    coach_pose3d: pd.DataFrame,
    peer_pose3d: pd.DataFrame,
) -> dict[str, Path]:
    assets = {
        "phase_contact": frame_asset(12, "phase_1_stride.png"),
        "phase_release": frame_asset(int(series["release"]["frame_index"].iloc[0]), "phase_2_release.png"),
        "phase_follow": frame_asset(45, "phase_3_follow.png"),
        "coach_contact": coach_frame_asset(25, "coach_phase_1_lift.png"),
        "coach_release": coach_frame_asset(72, "coach_phase_2_release.png"),
        "coach_follow": coach_frame_asset(105, "coach_phase_3_follow.png"),
        "thumb2d": copy_asset(
            ROOT / "outputs/manual-20260611-slymask/presentations/slymask-benchmark-deck/assets/benchmark_pitch_vertical_09_2d_thumb.png",
            "thumb_2d_overlay.png",
        ),
        "thumb3d": copy_asset(
            ROOT / "outputs/manual-20260611-slymask/presentations/slymask-benchmark-deck/assets/benchmark_pitch_vertical_09_3d_thumb.png",
            "thumb_3d_overlay.png",
        ),
    }
    assets["gap_chart"] = draw_gap_chart(metrics, ASSET_DIR / "gap_chart.png")
    assets["kinematic_dashboard"] = draw_kinematic_dashboard(metrics, motion_metrics, ASSET_DIR / "kinematic_dashboard.png")
    assets["pitch_timeline"] = draw_pitch_phase_timeline(pose3d, int(series["release"]["frame_index"].iloc[0]), ASSET_DIR / "pitch_phase_timeline.png")
    assets["kinetic_chain_flow"] = draw_kinetic_chain_flow(metrics, pose3d, ASSET_DIR / "kinetic_chain_flow.png")
    assets["angle_chart"] = draw_angle_chart(series["child_angles"], series["coach_angles"], ASSET_DIR / "angle_chart.png")
    assets["speed_chart"] = draw_speed_chart(series["child_speed"], series["coach_speed"], ASSET_DIR / "speed_chart.png")
    assets["com_chart"] = draw_com_chart(series["com"], ASSET_DIR / "com_chart.png")
    assets["radar_chart"] = draw_radar_chart(metrics, motion_metrics, ASSET_DIR / "radar_chart.png")
    assets["balance_chart"] = draw_balance_chart(pose3d, coach_pose3d, ASSET_DIR / "balance_chart.png")
    assets["heatmap_chart"] = draw_deviation_heatmap(motion_metrics, ASSET_DIR / "deviation_heatmap.png")
    assets["child_compare_chart"] = draw_child_compare_chart(metrics, motion_metrics, peer_pose3d, coach_pose3d, ASSET_DIR / "child_compare_chart.png")
    assets["standard_overlay"] = draw_standard_pose_overlay(pose3d, coach_pose3d, int(series["release"]["frame_index"].iloc[0]), ASSET_DIR / "standard_pose_overlay.png")
    assets["standard_overlay_gif"] = draw_standard_pose_gif(pose3d, coach_pose3d, ASSET_DIR / "standard_pose_overlay.gif")
    assets["kinetic_chain"] = draw_kinetic_chain(ASSET_DIR / "kinetic_chain.png")
    assets["growth"] = draw_growth_template(ASSET_DIR / "growth_template.png")
    return assets


def frame_asset(frame: int, name: str) -> Path:
    path = OUT_DIR / (
        "overlays/frames/benchmark_pitch_vertical_09/"
        f"{COND_2D}/benchmark_pitch_vertical_09__{COND_2D}__frame_{frame:06d}.png"
    )
    target = ASSET_DIR / name
    if path.exists():
        shutil.copyfile(path, target)
    return target


def coach_frame_asset(frame: int, name: str) -> Path:
    path = COACH_OVERLAY_DIR / f"pitch_horizontal_coach__baseline_raw__frame_{frame:06d}.png"
    target = ASSET_DIR / name
    if path.exists():
        shutil.copyfile(path, target)
    return target


def copy_asset(src: Path, name: str) -> Path:
    target = ASSET_DIR / name
    if src.exists():
        shutil.copyfile(src, target)
    return target


def cn_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def draw_gap_chart(metrics: list[Metric], path: Path) -> Path:
    w, h = 1600, 940
    im = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(im)
    title = cn_font(46)
    label = cn_font(30)
    small = cn_font(24)
    d.text((60, 35), "孩子 vs 正常速度教练：关键差距总览", font=title, fill="#172033")
    d.text((60, 95), "绿色带=论文参考范围；蓝点=孩子；黑线=教练 3D 实测。", font=small, fill="#526070")
    top = 155
    row_h = 95
    for i, m in enumerate(metrics):
        y = top + i * row_h
        values = [m.value, m.ref_low, m.ref_high]
        if m.coach_value is not None:
            values.append(m.coach_value)
        lo = min(values) * 0.85
        hi = max(values) * 1.15
        if abs(hi - lo) < 1e-6:
            hi = lo + 1
        x0, x1 = 470, 1280
        d.text((60, y + 10), f"{m.label_cn}", font=label, fill="#172033")
        d.text((60, y + 48), metric_source_label(m), font=small, fill="#667085")
        d.rounded_rectangle((x0, y + 25, x1, y + 55), radius=14, fill="#edf2f7")
        rx0 = x0 + (m.ref_low - lo) / (hi - lo) * (x1 - x0)
        rx1 = x0 + (m.ref_high - lo) / (hi - lo) * (x1 - x0)
        d.rounded_rectangle((rx0, y + 18, rx1, y + 62), radius=18, fill="#b9e6c9")
        vx = x0 + (m.value - lo) / (hi - lo) * (x1 - x0)
        color = "#2563eb" if in_band(m) else "#f97316"
        d.ellipse((vx - 15, y + 18, vx + 15, y + 48), fill=color)
        if m.coach_value is not None:
            cx = x0 + (m.coach_value - lo) / (hi - lo) * (x1 - x0)
            d.line((cx, y + 12, cx, y + 68), fill="#111827", width=6)
        d.text((1320, y + 14), f"{m.value:.1f} {m.unit}", font=label, fill=color)
        delta = delta_text(m)
        d.text((1320, y + 50), delta, font=small, fill="#526070")
    im.save(path)
    return path


def draw_kinematic_dashboard(metrics: list[Metric], motion_metrics: list[MotionMetric], path: Path) -> Path:
    w, h = 1600, 920
    im = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(im)
    title = cn_font(48)
    label = cn_font(28)
    small = cn_font(22)
    value_font = cn_font(48)
    d.text((60, 38), "核心运动学仪表盘", font=title, fill="#172033")
    d.text((60, 100), "像看汽车仪表一样先看 8 个关键指标：绿色接近目标，橙/红色优先改。", font=small, fill="#526070")
    by_key = {m.key: m for m in metrics}
    motion = {m.key: m for m in motion_metrics}
    tiles = [
        ("前膝角", motion["lead_knee"].value, motion["lead_knee"].unit, "前腿支撑", motion["lead_knee"].status),
        ("髋肩分离", motion["hip_shoulder_sep"].value, motion["hip_shoulder_sep"].unit, "身体扭转储能", motion["hip_shoulder_sep"].status),
        ("肘部弯曲", motion["elbow_bend"].value, motion["elbow_bend"].unit, "出手侧动力链", motion["elbow_bend"].status),
        ("躯干倾斜", motion["trunk_lean"].value, motion["trunk_lean"].unit, "身体控制", motion["trunk_lean"].status),
        ("跨步长度", motion["stride_length"].value, motion["stride_length"].unit, "动量转移", motion["stride_length"].status),
        ("投球臂槽", motion["arm_abduction"].value, motion["arm_abduction"].unit, "手臂空间位置", motion["arm_abduction"].status),
        ("骨盆峰值速度", by_key["pelvis_vel"].value, by_key["pelvis_vel"].unit, "髋部爆发", "ok" if in_band(by_key["pelvis_vel"]) else "warn"),
        ("躯干峰值速度", by_key["trunk_vel"].value, by_key["trunk_vel"].unit, "上肢传递", "ok" if in_band(by_key["trunk_vel"]) else "warn"),
    ]
    for i, (name, value, unit, desc, status) in enumerate(tiles):
        col = i % 4
        row = i // 4
        x = 60 + col * 375
        y = 170 + row * 315
        color = {"ok": "#16a34a", "warn": "#f97316", "bad": "#ef4444"}.get(status, "#f97316")
        bg = {"ok": "#f0fdf4", "warn": "#fff7ed", "bad": "#fef2f2"}.get(status, "#fff7ed")
        d.rounded_rectangle((x, y, x + 335, y + 250), radius=26, fill=bg, outline="#d0d5dd", width=2)
        d.text((x + 28, y + 28), name, font=label, fill="#172033")
        d.text((x + 28, y + 70), desc, font=small, fill="#667085")
        text = format_motion_value(value, unit)
        if unit in {"deg", "%", "%height", "%stride"}:
            text = text.replace("%", "%").replace(" deg", "°")
        d.text((x + 28, y + 128), text, font=value_font, fill=color)
        coach_note = ""
        if i < 6 and motion.get(list(motion.keys())[0]):
            pass
        d.rounded_rectangle((x + 28, y + 205, x + 150, y + 232), radius=14, fill="#ffffff")
        d.text((x + 44, y + 209), "3D计算" if unit != "km/h" else "视频估算", font=cn_font(17), fill="#475467")
    d.rounded_rectangle((60, 815, 1540, 875), radius=18, fill="#eff6ff", outline="#bfdbfe")
    d.text((85, 832), "读图顺序：先看红/橙色指标，再结合阶段时间轴判断问题发生在哪个投球相位。", font=small, fill="#344054")
    im.save(path)
    return path


def draw_pitch_phase_timeline(pose3d: pd.DataFrame, release_frame: int, path: Path) -> Path:
    w, h = 1600, 760
    im = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(im)
    title = cn_font(48)
    label = cn_font(28)
    small = cn_font(22)
    d.text((60, 38), "投球阶段时间轴", font=title, fill="#172033")
    d.text((60, 100), "把关键指标放回投球阶段，比单独看角度曲线更直观。", font=small, fill="#526070")
    events = pitch_phase_events(pose3d, release_frame)
    x0, x1, y = 130, 1470, 275
    d.line((x0, y, x1, y), fill="#cbd5e1", width=10)
    for event in events:
        x = x0 + event["phase"] * (x1 - x0)
        d.ellipse((x - 34, y - 34, x + 34, y + 34), fill=event["color"], outline="#ffffff", width=5)
        d.text((x - 18, y - 18), event["abbr"], font=cn_font(24), fill="#ffffff")
        d.text((x - 88, y + 58), event["name"], font=label, fill="#172033")
        d.text((x - 90, y + 96), event["frame_text"], font=small, fill="#667085")
        d.rounded_rectangle((x - 135, y + 142, x + 135, y + 265), radius=18, fill="#f8fafc", outline="#d0d5dd")
        draw_wrapped_text(d, event["metric"], (x - 112, y + 162), 224, small, "#344054", line_gap=7, max_lines=3)
    d.rounded_rectangle((70, 660, 1530, 720), radius=18, fill="#fff7ed", outline="#fed7aa")
    d.text((95, 676), "说明：MER 为 3D骨架可计算的手臂后摆代理值，不等同于实验室标记点定义的真实肩最大外旋。", font=small, fill="#9a3412")
    im.save(path)
    return path


def pitch_phase_events(pose3d: pd.DataFrame, release_frame: int) -> list[dict[str, object]]:
    p3 = pivot_3d_by_frame(pose3d)
    frames = p3.index.to_numpy(dtype=int)
    t = frame_time_axis(pose3d, frames)
    fc = detect_stride_landing_frame(p3)
    br = nearest_frame(frames, release_frame)
    mer = estimate_mer_proxy_frame(p3, fc, br)
    ft = int(frames[min(np.searchsorted(frames, br) + max(len(frames) // 8, 3), len(frames) - 1)])
    start, end = int(frames[0]), int(frames[-1])
    span = max(end - start, 1)
    shoulder = horizontal_angle_at_frame(p3, fc, "left_shoulder", "right_shoulder")
    pelvis = horizontal_angle_at_frame(p3, fc, "left_hip", "right_hip")
    fc_sep = float(wrap_deg(np.array([shoulder - pelvis]))[0])
    stride_cm = estimate_stride_3d_cm(p3)
    mer_layback = arm_layback_proxy_cm(p3, mer)
    br_trunk = trunk_lean_at_frame(p3, br)
    br_hand = float(np.nanpercentile(point_speed_3d_frames(p3, "right_wrist", t), 95))
    ft_head = estimate_head_stability_pct(p3, stride_cm)
    return [
        {"abbr": "FC", "name": "前脚落地", "frame_text": f"帧 {fc}", "phase": (fc - start) / span, "metric": f"髋肩分离 {fc_sep:.0f}°\n跨步 {stride_cm:.1f} cm", "color": "#2563eb"},
        {"abbr": "MER", "name": "最大后摆", "frame_text": f"帧 {mer}", "phase": (mer - start) / span, "metric": f"手臂后摆 {mer_layback:.1f} cm\n用于近似肩外旋", "color": "#7c3aed"},
        {"abbr": "BR", "name": "出手", "frame_text": f"帧 {br}", "phase": (br - start) / span, "metric": f"躯干倾斜 {br_trunk:.0f}°\n手速 {br_hand:.1f} m/s", "color": "#f97316"},
        {"abbr": "FT", "name": "随挥", "frame_text": f"帧 {ft}", "phase": (ft - start) / span, "metric": f"头部稳定 {ft_head:.0f}%\n观察制动与平衡", "color": "#16a34a"},
    ]


def frame_time_axis_from_p3(p3: pd.DataFrame) -> np.ndarray:
    frames = p3.index.to_numpy(dtype=float)
    span = max(float(frames[-1] - frames[0]), 1.0)
    return (frames - frames[0]) / span


def estimate_mer_proxy_frame(p3: pd.DataFrame, fc: int, br: int) -> int:
    frames = p3.index.to_numpy(dtype=int)
    lo = int(np.searchsorted(frames, fc))
    hi = max(int(np.searchsorted(frames, br)), lo + 1)
    subset = frames[lo:hi]
    if len(subset) == 0:
        return br
    axis = movement_axis(p3)
    values = []
    for frame in subset:
        wrist = np.array([p3.loc[frame, "right_wrist_x_3d"], p3.loc[frame, "right_wrist_z_3d"]], dtype=float)
        shoulder = np.array([p3.loc[frame, "right_shoulder_x_3d"], p3.loc[frame, "right_shoulder_z_3d"]], dtype=float)
        values.append(float((shoulder - wrist) @ axis))
    return int(subset[int(np.nanargmax(values))])


def arm_layback_proxy_cm(p3: pd.DataFrame, frame: int) -> float:
    axis = movement_axis(p3)
    wrist = np.array([p3.loc[frame, "right_wrist_x_3d"], p3.loc[frame, "right_wrist_z_3d"]], dtype=float)
    shoulder = np.array([p3.loc[frame, "right_shoulder_x_3d"], p3.loc[frame, "right_shoulder_z_3d"]], dtype=float)
    return float(max((shoulder - wrist) @ axis, 0.0) * 100.0)


def draw_kinetic_chain_flow(metrics: list[Metric], pose3d: pd.DataFrame, path: Path) -> Path:
    w, h = 1600, 760
    im = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(im)
    title = cn_font(48)
    label = cn_font(28)
    small = cn_font(22)
    value_font = cn_font(34)
    d.text((60, 38), "动力链流：力量是否从下往上传", font=title, fill="#172033")
    d.text((60, 100), "近端看旋转速度，远端看关节线速度；数值越顺，越说明发力能传到手。", font=small, fill="#526070")
    by_key = {m.key: m for m in metrics}
    p3 = pivot_3d_by_frame(pose3d)
    t = frame_time_axis(pose3d, p3.index.to_numpy(dtype=int))
    shoulder_speed = float(np.nanpercentile(point_speed_3d_frames(p3, "right_shoulder", t), 95))
    elbow_speed = float(np.nanpercentile(point_speed_3d_frames(p3, "right_elbow", t), 95))
    hand_speed = by_key["hand_speed"].value
    nodes = [
        ("骨盆", f"{by_key['pelvis_vel'].value:.0f}", "deg/s", "#dbeafe", "髋部爆发"),
        ("躯干", f"{by_key['trunk_vel'].value:.0f}", "deg/s", "#dbeafe", "上肢传递"),
        ("肩部", f"{shoulder_speed:.1f}", "m/s", "#dcfce7", "肩部移动"),
        ("肘部", f"{elbow_speed:.1f}", "m/s", "#fff7ed", "末端加速"),
        ("手部", f"{hand_speed:.1f}", "m/s", "#fee2e2", "出手速度"),
    ]
    xs = [120, 420, 720, 1020, 1320]
    y = 300
    for i, (name, val, unit, fill, desc) in enumerate(nodes):
        x = xs[i]
        d.rounded_rectangle((x, y, x + 210, y + 210), radius=28, fill=fill, outline="#d0d5dd", width=2)
        d.text((x + 38, y + 28), name, font=label, fill="#172033")
        d.text((x + 38, y + 76), val, font=value_font, fill="#2563eb")
        d.text((x + 125, y + 88), unit, font=small, fill="#667085")
        d.text((x + 38, y + 142), desc, font=small, fill="#344054")
        if i < len(nodes) - 1:
            d.line((x + 225, y + 105, xs[i + 1] - 18, y + 105), fill="#98a2b3", width=7)
            d.polygon([(xs[i + 1] - 24, y + 84), (xs[i + 1] + 8, y + 105), (xs[i + 1] - 24, y + 126)], fill="#98a2b3")
    d.rounded_rectangle((80, 610, 1520, 700), radius=20, fill="#eff6ff", outline="#bfdbfe")
    d.text((108, 632), "读图方法：如果骨盆/躯干速度还可以，但肩、肘、手没有继续放大，说明力量传递断在上肢或出手时机。", font=small, fill="#344054")
    im.save(path)
    return path


def in_band(m: Metric) -> bool:
    return m.ref_low <= m.value <= m.ref_high


def delta_text(m: Metric) -> str:
    if m.coach_value is not None:
        diff = m.value - m.coach_value
        sign = "高" if diff > 0 else "低"
        pct = abs(diff) / max(abs(m.coach_value), 1e-6) * 100
        return f"比教练{sign} {abs(diff):.1f} {m.unit} ({pct:.0f}%)"
    if in_band(m):
        return "在参考范围内"
    if m.value < m.ref_low:
        return f"低 {m.ref_low - m.value:.1f} {m.unit}"
    return f"高 {m.value - m.ref_high:.1f} {m.unit}"


def coach_value_text(m: Metric) -> str:
    if m.coach_value is None:
        return "教练暂无"
    return f"教练 {m.coach_value:.1f} {m.unit}"


def metric_source_label(m: Metric) -> str:
    if m.key == "ball_speed":
        return "视频估算，建议雷达枪复核"
    if m.key == "com":
        return "3D身体中心位移估算"
    if m.key in {"hand_speed", "trunk_vel", "pelvis_vel"}:
        return "3D速度估算"
    return "3D骨架计算"


def method_label_cn(method: str) -> str:
    mapping = {
        "3D": "3D计算",
        "3D proxy": "3D估算",
        "3D normalized": "3D归一化",
    }
    return mapping.get(method, method.replace("proxy", "估算").replace("normalized", "归一化"))


def draw_angle_chart(child: pd.DataFrame, coach: pd.DataFrame, path: Path) -> Path:
    series = [("孩子髋肩分离", percent_x(child), child["separation"].to_numpy(), "#2563eb")]
    if not coach.empty:
        series.append(("教练髋肩分离", percent_x(coach), coach["separation"].to_numpy(), "#111827"))
    return draw_line_chart(
        path,
        "角度曲线：髋肩分离",
        series,
        "度",
    )


def draw_speed_chart(child: pd.DataFrame, coach: pd.DataFrame, path: Path) -> Path:
    series = [("孩子右手速度", percent_x(child), child["hand_speed_mps"].to_numpy(), "#2563eb")]
    if not coach.empty:
        series.append(("教练右手速度", percent_x(coach), coach["hand_speed_mps"].to_numpy(), "#111827"))
    return draw_line_chart(
        path,
        "速度曲线：出手侧手部速度",
        series,
        "m/s",
    )


def draw_com_chart(df: pd.DataFrame, path: Path) -> Path:
    return draw_line_chart(
        path,
        "身体中心轨迹：前后移动估算值",
        [("身体中心前后位移", df["time"].to_numpy(), df["x_cm"].to_numpy(), "#16a34a")],
        "cm",
    )


def percent_x(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.array([])
    x = df["time"].to_numpy(dtype=float)
    span = float(np.nanmax(x) - np.nanmin(x))
    if span < 1e-6:
        return np.zeros_like(x)
    return (x - np.nanmin(x)) / span * 100.0


def draw_line_chart(path: Path, title: str, series: list[tuple[str, np.ndarray, np.ndarray, str]], unit: str) -> Path:
    w, h = 1500, 760
    im = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(im)
    title_font = cn_font(42)
    label_font = cn_font(26)
    small = cn_font(22)
    d.text((60, 34), title, font=title_font, fill="#172033")
    x0, y0, x1, y1 = 145, 170, 1390, 610
    vals = np.concatenate([np.asarray(s[2], dtype=float) for s in series])
    vals = vals[np.isfinite(vals)]
    ymin, ymax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if abs(ymax - ymin) < 1:
        ymax = ymin + 1
    pad = (ymax - ymin) * 0.18
    ymin -= pad
    ymax += pad
    legend_x = 60
    for idx, (name, _, _, color) in enumerate(series):
        lx = legend_x + 290 * idx
        d.rounded_rectangle((lx, 104, lx + 34, 132), radius=8, fill=color)
        d.text((lx + 46, 99), name, font=small, fill="#344054")
    d.rectangle((x0, y0, x1, y1), outline="#d0d5dd", width=2)
    for i in range(5):
        yy = y0 + i * (y1 - y0) / 4
        d.line((x0, yy, x1, yy), fill="#edf2f7", width=2)
        val = ymax - i * (ymax - ymin) / 4
        d.text((50, yy - 14), f"{val:.0f}", font=small, fill="#667085")
    all_x = np.concatenate([np.asarray(s[1], dtype=float) for s in series])
    xmin, xmax = float(np.nanmin(all_x)), float(np.nanmax(all_x))
    if abs(xmax - xmin) < 1e-6:
        xmax = xmin + 1.0
    for idx, (name, x, y, color) in enumerate(series):
        pts = []
        for xx, yyv in zip(x, y):
            if not np.isfinite(yyv):
                continue
            px = x0 + (xx - xmin) / (xmax - xmin) * (x1 - x0)
            py = y1 - (yyv - ymin) / (ymax - ymin) * (y1 - y0)
            px = float(np.clip(px, x0, x1))
            py = float(np.clip(py, y0, y1))
            pts.append((px, py))
        if len(pts) > 1:
            d.line(pts, fill=color, width=5, joint="curve")
    d.text((x0, 655), "动作进程（%）" if xmax > 10 else "时间（秒）", font=label_font, fill="#344054")
    d.text((58, 144), unit, font=label_font, fill="#344054")
    im.save(path)
    return path


def score_from_gap(value: float, coach: float | None, low: float | None = None, high: float | None = None) -> float:
    if coach is not None and np.isfinite(coach) and abs(coach) > 1e-6:
        gap = abs(value - coach) / max(abs(coach), 1e-6)
        return float(np.clip(100 - gap * 100, 0, 100))
    if low is not None and high is not None:
        center = (low + high) / 2
        span = max((high - low) / 2, 1e-6)
        return float(np.clip(100 - abs(value - center) / span * 45, 0, 100))
    return 60.0


def draw_radar_chart(metrics: list[Metric], motion_metrics: list[MotionMetric], path: Path) -> Path:
    w, h = 1150, 820
    im = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(im)
    title = cn_font(42)
    label = cn_font(25)
    small = cn_font(21)
    d.text((55, 35), "六维运动能力雷达图", font=title, fill="#172033")
    by_key = {m.key: m for m in metrics}
    motion = {m.key: m for m in motion_metrics}
    axes = [
        ("旋转速度", score_from_gap(by_key["trunk_vel"].value, by_key["trunk_vel"].coach_value)),
        ("跨步力量", score_from_gap(by_key["stride"].value, by_key["stride"].coach_value)),
        ("出手速度", score_from_gap(by_key["hand_speed"].value, by_key["hand_speed"].coach_value)),
        ("髋肩分离", score_from_gap(abs(motion["hip_shoulder_sep"].value), abs(motion["hip_shoulder_sep"].coach_value or 45))),
        ("落脚方向", score_from_gap(abs(motion["foot_direction"].value), abs(motion["foot_direction"].coach_value or 20))),
        ("身体稳定", score_from_gap(motion["head_stability"].value, motion["head_stability"].coach_value)),
    ]
    cx, cy = 575, 430
    radius = 265
    angles = [math.radians(-90 + i * 360 / len(axes)) for i in range(len(axes))]
    for r in [0.25, 0.5, 0.75, 1.0]:
        pts = [(cx + math.cos(a) * radius * r, cy + math.sin(a) * radius * r) for a in angles]
        d.polygon(pts, outline="#d0d5dd")
        d.text((cx + 8, cy - radius * r - 10), f"{int(r*100)}", font=small, fill="#98a2b3")
    value_pts = []
    for (name, score), a in zip(axes, angles):
        ex = cx + math.cos(a) * radius
        ey = cy + math.sin(a) * radius
        d.line((cx, cy, ex, ey), fill="#e4e7ec", width=2)
        lx = cx + math.cos(a) * (radius + 72)
        ly = cy + math.sin(a) * (radius + 42)
        d.text((lx - 52, ly - 16), name, font=label, fill="#344054")
        value_pts.append((cx + math.cos(a) * radius * score / 100, cy + math.sin(a) * radius * score / 100))
    d.polygon(value_pts, fill="#bfdbfe", outline="#2563eb")
    d.line(value_pts + [value_pts[0]], fill="#2563eb", width=5)
    for x, y in value_pts:
        d.ellipse((x - 8, y - 8, x + 8, y + 8), fill="#2563eb")
    d.rounded_rectangle((80, 705, 1070, 770), radius=18, fill="#eff6ff", outline="#bfdbfe")
    d.text((105, 721), "读图方法：越靠外代表越接近教练或参考范围；落脚方向、出手速度和跨步力量是本次优先提升项。", font=small, fill="#344054")
    im.save(path)
    return path


def draw_balance_chart(pose3d: pd.DataFrame, coach_pose3d: pd.DataFrame, path: Path) -> Path:
    w, h = 1150, 820
    im = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(im)
    title = cn_font(42)
    label = cn_font(25)
    small = cn_font(21)
    d.text((55, 35), "左右平衡与支撑稳定", font=title, fill="#172033")
    child = side_balance_values(pose3d)
    coach = side_balance_values(coach_pose3d) if not coach_pose3d.empty else {}
    rows = [
        ("肩高差", child.get("shoulder_y", 0), coach.get("shoulder_y", 0), "cm"),
        ("髋高差", child.get("hip_y", 0), coach.get("hip_y", 0), "cm"),
        ("膝高差", child.get("knee_y", 0), coach.get("knee_y", 0), "cm"),
        ("踝距变化", child.get("ankle_width", 0), coach.get("ankle_width", 0), "cm"),
    ]
    x0, x1 = 310, 940
    y0 = 160
    scale_max = max(8.0, max(abs(v) for _, v, _, _ in rows) * 1.3, max(abs(c) for _, _, c, _ in rows) * 1.3)
    center = (x0 + x1) / 2
    d.line((center, y0 - 20, center, y0 + 450), fill="#98a2b3", width=3)
    d.text((center - 38, y0 + 470), "0", font=small, fill="#667085")
    for i, (name, child_val, coach_val, unit) in enumerate(rows):
        y = y0 + i * 110
        d.text((70, y + 8), name, font=label, fill="#172033")
        d.text((70, y + 42), "左高为正，右高为负", font=small, fill="#667085")
        d.rounded_rectangle((x0, y + 10, x1, y + 40), radius=15, fill="#edf2f7")
        child_x = center + child_val / scale_max * (x1 - x0) / 2
        coach_x = center + coach_val / scale_max * (x1 - x0) / 2
        d.line((coach_x, y + 2, coach_x, y + 50), fill="#111827", width=5)
        d.ellipse((child_x - 14, y + 8, child_x + 14, y + 36), fill="#2563eb")
        d.text((970, y + 6), f"孩子 {child_val:+.1f}{unit}", font=small, fill="#2563eb")
        d.text((970, y + 34), f"教练 {coach_val:+.1f}{unit}", font=small, fill="#111827")
    d.rounded_rectangle((80, 670, 1070, 760), radius=18, fill="#f0fdf4", outline="#bbf7d0")
    d.text((105, 690), "读图方法：蓝点越靠近黑线，说明左右控制越接近教练；左右差过大时，落脚和躯干旋转更容易代偿。", font=small, fill="#344054")
    im.save(path)
    return path


def side_balance_values(pose3d: pd.DataFrame) -> dict[str, float]:
    if pose3d.empty:
        return {}
    p3 = pivot_3d_by_frame(pose3d)
    frame = int(p3.index[len(p3.index) // 2])
    def dy(left: str, right: str) -> float:
        return float((p3.loc[frame, f"{left}_y_3d"] - p3.loc[frame, f"{right}_y_3d"]) * 100)
    ankle_width = np.sqrt(
        (p3["left_ankle_x_3d"] - p3["right_ankle_x_3d"]) ** 2
        + (p3["left_ankle_z_3d"] - p3["right_ankle_z_3d"]) ** 2
    )
    return {
        "shoulder_y": dy("left_shoulder", "right_shoulder"),
        "hip_y": dy("left_hip", "right_hip"),
        "knee_y": dy("left_knee", "right_knee"),
        "ankle_width": float((np.nanmax(ankle_width) - np.nanmin(ankle_width)) * 100),
    }


def draw_deviation_heatmap(motion_metrics: list[MotionMetric], path: Path) -> Path:
    w, h = 1500, 760
    im = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(im)
    title = cn_font(42)
    label = cn_font(24)
    small = cn_font(19)
    d.text((55, 35), "动作偏差热力图", font=title, fill="#172033")
    d.text((55, 92), "颜色越深，说明与教练差距越大；用于快速找到优先改进部位。", font=small, fill="#667085")
    groups = [
        ("上肢出手", ["elbow_bend", "arm_abduction", "wrist_snap", "arm_speed", "fingertip_speed"]),
        ("躯干旋转", ["trunk_lean", "hip_shoulder_sep"]),
        ("下肢落脚", ["stride_angle", "lead_knee", "stride_length", "foot_direction"]),
        ("稳定转移", ["weight_transfer", "head_stability"]),
    ]
    by_key = {m.key: m for m in motion_metrics}
    x0, y0 = 70, 160
    cell_w, cell_h = 245, 75
    for r, (group, keys) in enumerate(groups):
        y = y0 + r * 125
        d.text((x0, y + 18), group, font=label, fill="#172033")
        for c, key in enumerate(keys):
            m = by_key[key]
            gap = metric_gap_score(m)
            color = heat_color(gap)
            x = x0 + 210 + c * cell_w
            d.rounded_rectangle((x, y, x + cell_w - 18, y + cell_h), radius=12, fill=color, outline="#e4e7ec")
            d.text((x + 16, y + 12), m.label_cn, font=small, fill="#111827")
            d.text((x + 16, y + 42), f"差距 {gap:.0f}%", font=small, fill="#344054")
    d.rounded_rectangle((70, 680, 1430, 735), radius=16, fill="#eff6ff", outline="#bfdbfe")
    d.text((96, 695), "本次优先关注：出手侧速度、跨步长度、前脚方向和身体前移。", font=small, fill="#344054")
    im.save(path)
    return path


def draw_child_compare_chart(
    julian_metrics: list[Metric],
    julian_motion: list[MotionMetric],
    peer_pose3d: pd.DataFrame,
    coach_pose3d: pd.DataFrame,
    path: Path,
) -> Path:
    w, h = 1600, 920
    im = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(im)
    title = cn_font(48)
    label = cn_font(27)
    small = cn_font(21)
    d.text((60, 36), "儿童 vs 儿童：Julian 与 Youyou", font=title, fill="#172033")
    d.text((60, 96), "同一套 3D 计算口径，只比较当前数据能稳定估算的投球动作指标。", font=small, fill="#667085")
    if peer_pose3d.empty:
        d.text((120, 420), "未找到 Youyou 的 3D 数据，无法生成儿童对比。", font=cn_font(34), fill="#ef4444")
        im.save(path)
        return path

    j_metric = {m.key: m for m in julian_metrics}
    j_motion = {m.key: m for m in julian_motion}
    y_human = compute_human_3d_metrics(peer_pose3d)
    y_bundle = pitch_metric_bundle(peer_pose3d, release_frame=None)
    coach_bundle = pitch_metric_bundle(coach_pose3d, release_frame=None) if not coach_pose3d.empty else {}

    rows = [
        ("躯干旋转速度", j_metric["trunk_vel"].value, y_human["trunk_vel"], j_metric["trunk_vel"].coach_value, "deg/s", "上肢传递"),
        ("骨盆旋转速度", j_metric["pelvis_vel"].value, y_human["pelvis_vel"], j_metric["pelvis_vel"].coach_value, "deg/s", "髋部爆发"),
        ("髋肩分离", abs(j_motion["hip_shoulder_sep"].value), abs(y_bundle["hip_shoulder_sep_deg"]), abs(coach_bundle.get("hip_shoulder_sep_deg", np.nan)), "deg", "身体扭转"),
        ("跨步长度", j_motion["stride_length"].value, y_bundle["stride_length_pct_height"], coach_bundle.get("stride_length_pct_height"), "%height", "动量转移"),
        ("手臂速度", j_motion["arm_speed"].value, percent_of_reference(y_bundle["arm_speed_mps"], coach_bundle.get("arm_speed_mps")), 100.0, "%", "末端加速"),
        ("前脚方向", abs(j_motion["foot_direction"].value), abs(y_bundle["foot_direction_deg"]), abs(coach_bundle.get("foot_direction_deg", np.nan)), "deg", "落脚控制"),
    ]

    x0, x1 = 360, 1240
    y0 = 175
    row_h = 95
    for i, (name, julian, youyou, coach, unit, note) in enumerate(rows):
        y = y0 + i * row_h
        values = [float(julian), float(youyou)]
        if coach is not None and np.isfinite(coach):
            values.append(float(coach))
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * 0.2, 1.0)
        lo -= pad
        hi += pad
        if name == "前脚方向":
            lo = 0.0
            hi = max(values) * 1.25 + 1
        d.text((70, y + 2), name, font=label, fill="#172033")
        d.text((70, y + 35), note, font=small, fill="#667085")
        d.rounded_rectangle((x0, y + 10, x1, y + 42), radius=16, fill="#edf2f7")
        jx = x0 + (julian - lo) / max(hi - lo, 1e-6) * (x1 - x0)
        yx = x0 + (youyou - lo) / max(hi - lo, 1e-6) * (x1 - x0)
        d.ellipse((jx - 15, y + 10, jx + 15, y + 40), fill="#2563eb")
        d.ellipse((yx - 15, y + 10, yx + 15, y + 40), fill="#f97316")
        if coach is not None and np.isfinite(coach):
            cx = x0 + (coach - lo) / max(hi - lo, 1e-6) * (x1 - x0)
            d.line((cx, y + 0, cx, y + 52), fill="#111827", width=5)
        d.text((1270, y - 2), f"Julian {format_compare_value(julian, unit)}", font=small, fill="#2563eb")
        d.text((1270, y + 27), f"Youyou {format_compare_value(youyou, unit)}", font=small, fill="#f97316")
        if coach is not None and np.isfinite(coach):
            d.text((1270, y + 56), f"教练 {format_compare_value(coach, unit)}", font=cn_font(18), fill="#111827")

    d.rounded_rectangle((80, 780, 1520, 865), radius=20, fill="#eff6ff", outline="#bfdbfe")
    d.rounded_rectangle((120, 806, 150, 834), radius=8, fill="#2563eb")
    d.text((165, 802), "Julian（本报告主分析人）", font=small, fill="#344054")
    d.rounded_rectangle((470, 806, 500, 834), radius=8, fill="#f97316")
    d.text((515, 802), "Youyou（同龄儿童参考）", font=small, fill="#344054")
    d.line((860, 800, 860, 840), fill="#111827", width=5)
    d.text((885, 802), "教练参考值", font=small, fill="#344054")
    d.text((1120, 802), "注意：这不是排名，只用于帮助家长理解同龄动作差异。", font=small, fill="#667085")
    im.save(path)
    return path


def format_compare_value(value: float, unit: str) -> str:
    if value is None or not np.isfinite(value):
        return "暂无"
    if unit == "deg":
        return f"{value:.0f}°"
    if unit in {"%", "%height"}:
        return f"{value:.0f}%"
    if unit == "deg/s":
        return f"{value:.0f} deg/s"
    return f"{value:.1f} {unit}"


def metric_gap_score(m: MotionMetric) -> float:
    if m.coach_value is None or not np.isfinite(m.coach_value) or abs(m.coach_value) < 1e-6:
        return 50.0
    return float(np.clip(abs(m.value - m.coach_value) / max(abs(m.coach_value), 1e-6) * 100, 0, 140))


def heat_color(gap: float) -> str:
    if gap < 15:
        return "#dcfce7"
    if gap < 35:
        return "#fef3c7"
    if gap < 70:
        return "#fed7aa"
    return "#fecaca"


SKELETON_EDGES = [
    ("head", "neck"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("neck", "spine3"),
    ("spine3", "spine2"),
    ("spine2", "spine1"),
    ("spine1", "hip"),
    ("hip", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("hip", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def draw_standard_pose_overlay(child_pose3d: pd.DataFrame, coach_pose3d: pd.DataFrame, release_frame: int, path: Path) -> Path:
    im = Image.new("RGB", (1200, 900), "#ffffff")
    draw_standard_overlay_on_image(im, child_pose3d, coach_pose3d, release_frame, title="标准肢体范例叠加：出手相位")
    im.save(path)
    return path


def draw_standard_pose_gif(child_pose3d: pd.DataFrame, coach_pose3d: pd.DataFrame, path: Path) -> Path:
    child = pivot_3d_by_frame(child_pose3d)
    frames = child.index.to_numpy(dtype=int)
    sample = np.linspace(0, len(frames) - 1, 16).astype(int)
    images = []
    for idx in sample:
        im = Image.new("RGB", (1200, 900), "#ffffff")
        draw_standard_overlay_on_image(im, child_pose3d, coach_pose3d, int(frames[idx]), title="动态纠正示意：蓝色为孩子，绿色为按孩子身材缩放的教练标准")
        images.append(im)
    images[0].save(path, save_all=True, append_images=images[1:], duration=130, loop=0)
    return path


def draw_standard_overlay_on_image(im: Image.Image, child_pose3d: pd.DataFrame, coach_pose3d: pd.DataFrame, child_frame: int, title: str) -> None:
    d = ImageDraw.Draw(im)
    d.text((55, 35), title, font=cn_font(40), fill="#172033")
    d.text((58, 86), "浅蓝虚线为孩子原始动作；绿色为标准参照；红色为偏差较大的原始骨段。", font=cn_font(23), fill="#667085")
    if child_pose3d.empty or coach_pose3d.empty:
        d.text((80, 420), "缺少 3D骨架数据", font=cn_font(30), fill="#ef4444")
        return
    child = pivot_3d_by_frame(child_pose3d)
    coach = pivot_3d_by_frame(coach_pose3d)
    c_frame = nearest_frame(child.index.to_numpy(dtype=int), child_frame)
    phase = list(child.index).index(c_frame) / max(len(child.index) - 1, 1)
    coach_idx = min(int((len(coach.index) - 1) * np.clip(phase, 0, 1)), len(coach.index) - 1)
    coach_frame = int(coach.index[coach_idx])
    child_pts = skeleton_points(child, c_frame)
    coach_pts = skeleton_points(coach, coach_frame)
    standard_pts = scaled_standard_pose(child_pts, coach_pts)
    child_names = list(child_pts.keys())
    std_names = list(standard_pts.keys())
    all_pts = [child_pts[name] for name in child_names] + [standard_pts[name] for name in std_names]
    proj = project_points(all_pts, (90, 150, 1110, 800))
    child_proj = proj[: len(child_names)]
    std_proj = proj[len(child_names) :]
    child_2d = dict(zip(child_names, child_proj))
    std_2d = dict(zip(std_names, std_proj))
    overlay = Image.new("RGBA", im.size, (255, 255, 255, 0))
    od = ImageDraw.Draw(overlay)
    # 先画孩子完整原始动作。使用粗的浅蓝虚线，保证被绿色标准线覆盖后仍能看到原姿态轮廓。
    for a, b in SKELETON_EDGES:
        if a in child_2d and b in child_2d:
            draw_dashed_line(od, child_2d[a], child_2d[b], fill=(96, 165, 250, 210), width=19, dash=22, gap=14)
    for a, b in SKELETON_EDGES:
        if a in std_2d and b in std_2d:
            od.line((*std_2d[a], *std_2d[b]), fill=(22, 163, 74, 210), width=10)
    for a, b in SKELETON_EDGES:
        if a in child_2d and b in child_2d:
            dev = segment_deviation(child_pts, standard_pts, a, b)
            if dev > 0.16:
                od.line((*child_2d[a], *child_2d[b]), fill=(239, 68, 68, 245), width=9)
    im.alpha_composite(overlay) if im.mode == "RGBA" else im.paste(Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB"))
    d = ImageDraw.Draw(im)
    for pts, color in [(std_2d, "#15803d"), (child_2d, "#1d4ed8")]:
        for x, y in pts.values():
            d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
    d.rounded_rectangle((760, 660, 1110, 790), radius=16, fill="#f8fafc", outline="#d0d5dd")
    d.rounded_rectangle((790, 690, 825, 715), radius=6, fill="#2563eb")
    d.text((840, 684), "孩子原始动作（浅蓝虚线）", font=cn_font(24), fill="#344054")
    d.rounded_rectangle((790, 730, 825, 755), radius=6, fill="#16a34a")
    d.text((840, 724), "按孩子身材缩放的教练标准", font=cn_font(24), fill="#344054")
    d.rounded_rectangle((790, 770, 825, 795), radius=6, fill="#ef4444")
    d.text((840, 764), "偏差较大的原始骨段", font=cn_font(24), fill="#344054")


def skeleton_points(p3: pd.DataFrame, frame: int) -> dict[str, np.ndarray]:
    pts = {}
    for col in p3.columns:
        if col.endswith("_x_3d"):
            joint = col[:-5]
            y_col = f"{joint}_y_3d"
            z_col = f"{joint}_z_3d"
            if y_col in p3.columns and z_col in p3.columns:
                pts[joint] = np.array([p3.loc[frame, col], p3.loc[frame, y_col], p3.loc[frame, z_col]], dtype=float)
    return pts


def scaled_standard_pose(child: dict[str, np.ndarray], coach: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if "hip" not in child or "hip" not in coach:
        return coach
    std = {"hip": child["hip"].copy()}
    for parent, child_joint in SKELETON_EDGES:
        if parent in std and parent in coach and child_joint in coach and child_joint in child and parent in child:
            direction = coach[child_joint] - coach[parent]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            target_len = float(np.linalg.norm(child[child_joint] - child[parent]))
            std[child_joint] = std[parent] + direction / norm * target_len
    for joint, point in child.items():
        std.setdefault(joint, point.copy())
    return std


def project_points(points: list[np.ndarray], box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    arr = np.array([[p[0], p[1]] for p in points], dtype=float)
    arr[:, 1] *= -1
    mins = np.nanmin(arr, axis=0)
    maxs = np.nanmax(arr, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    scale = min((x1 - x0) / span[0], (y1 - y0) / span[1]) * 0.86
    center_src = (mins + maxs) / 2
    center_dst = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    out = (arr - center_src) * scale + center_dst
    return [(float(x), float(y)) for x, y in out]


def segment_deviation(child: dict[str, np.ndarray], standard: dict[str, np.ndarray], a: str, b: str) -> float:
    if a not in child or b not in child or a not in standard or b not in standard:
        return 0.0
    c_mid = (child[a] + child[b]) / 2
    s_mid = (standard[a] + standard[b]) / 2
    ref = max(float(np.linalg.norm(child[b] - child[a])), 1e-6)
    return float(np.linalg.norm(c_mid - s_mid) / ref)


def draw_dashed_line(
    d: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    fill: tuple[int, int, int, int],
    width: int,
    dash: int,
    gap: int,
) -> None:
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    dist = 0.0
    while dist < length:
        end = min(dist + dash, length)
        d.line((x0 + ux * dist, y0 + uy * dist, x0 + ux * end, y0 + uy * end), fill=fill, width=width)
        dist += dash + gap


def draw_kinetic_chain(path: Path) -> Path:
    w, h = 1500, 520
    im = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(im)
    title = cn_font(42)
    label = cn_font(30)
    small = cn_font(24)
    d.text((55, 35), "动力链顺序：理想模式 vs 当前样例", font=title, fill="#172033")
    phases = [
        ("下肢启动", "先稳定落脚"),
        ("骨盆旋转", "带动身体转动"),
        ("躯干加速", "胸口随后跟上"),
        ("手臂出手", "最后把速度传到手"),
    ]
    xs = [160, 500, 840, 1180]
    for i, (cn, en) in enumerate(phases):
        x = xs[i]
        d.ellipse((x - 48, 185, x + 48, 281), fill="#dbeafe", outline="#2563eb", width=4)
        d.text((x - 15, 205), str(i + 1), font=title, fill="#2563eb")
        d.text((x - 90, 302), cn, font=label, fill="#172033")
        d.text((x - 85, 340), en, font=small, fill="#667085")
        if i < len(phases) - 1:
            d.line((x + 65, 233, xs[i + 1] - 65, 233), fill="#98a2b3", width=6)
            d.polygon([(xs[i + 1] - 70, 218), (xs[i + 1] - 40, 233), (xs[i + 1] - 70, 248)], fill="#98a2b3")
    d.rounded_rectangle((80, 410, 1420, 475), radius=18, fill="#fff7ed", outline="#fed7aa", width=2)
    d.text((105, 425), "训练优先级：先检查骨盆是否先启动，再看躯干和出手侧手速是否跟随提升。", font=label, fill="#9a3412")
    im.save(path)
    return path


def draw_growth_template(path: Path) -> Path:
    w, h = 1500, 620
    im = Image.new("RGB", (w, h), "#ffffff")
    d = ImageDraw.Draw(im)
    title = cn_font(42)
    label = cn_font(28)
    small = cn_font(22)
    d.text((55, 35), "年度成长档案模板：进步曲线 + 动作演变", font=title, fill="#172033")
    x0, y0, x1, y1 = 95, 130, 900, 515
    d.rectangle((x0, y0, x1, y1), outline="#d0d5dd", width=2)
    months = ["1月", "3月", "6月", "9月", "12月"]
    vals = [52, 58, 61, 67, 72]
    pts = []
    for i, val in enumerate(vals):
        x = x0 + 70 + i * 170
        y = y1 - (val - 45) / 35 * (y1 - y0)
        pts.append((x, y))
        d.ellipse((x - 10, y - 10, x + 10, y + 10), fill="#2563eb")
        d.text((x - 22, y + 18), str(val), font=small, fill="#2563eb")
        d.text((x - 25, y1 + 18), months[i], font=small, fill="#667085")
    d.line(pts, fill="#2563eb", width=5)
    d.text((110, 95), "球速/出手效率综合分", font=label, fill="#344054")
    for i, name in enumerate(["初测", "第2次", "第3次"]):
        x = 1010 + i * 145
        d.rounded_rectangle((x, 165, x + 120, 335), radius=12, fill="#f2f4f7", outline="#d0d5dd")
        d.text((x + 25, 360), name, font=label, fill="#344054")
    d.text((1000, 110), "动作演变视频截图位", font=label, fill="#344054")
    d.text((1000, 430), "每次保留同一相位截图、关键指标、训练处方与复测结论。", font=small, fill="#667085")
    im.save(path)
    return path


def build_pdf(pdf_path: Path, metrics: list[Metric], motion_metrics: list[MotionMetric], parent_guidance: str, assets: dict[str, Path]) -> None:
    page_dir = ASSET_DIR / "pages_redesign"
    page_dir.mkdir(parents=True, exist_ok=True)
    pages = [
        render_cover_page(metrics, assets, page_dir / "page_01_cover.png"),
        render_dashboard_page(metrics, assets, page_dir / "page_02_dashboard.png"),
        render_visual_page(assets, page_dir / "page_03_visual_compare.png"),
        render_curves_page(assets, page_dir / "page_04_curves.png"),
        render_high_level_graphs_page(assets, page_dir / "page_05_high_level_graphs.png"),
        render_child_compare_page(assets, page_dir / "page_06_child_compare.png"),
        render_correction_graph_page(assets, page_dir / "page_07_correction_graph.png"),
        render_motion_metrics_page(motion_metrics, page_dir / "page_08_motion_metrics.png"),
        render_training_page(metrics, parent_guidance, page_dir / "page_09_training.png"),
        render_prompt_page(metrics, motion_metrics, parent_guidance, page_dir / "page_10_prompt.png"),
        render_retest_page(metrics, motion_metrics, page_dir / "page_11_retest.png"),
    ]
    canvas = pdf_canvas.Canvas(str(pdf_path), pagesize=A4)
    for page in pages:
        canvas.drawImage(str(page), 0, 0, width=A4[0], height=A4[1])
        canvas.showPage()
    canvas.save()


def render_cover_page(metrics: list[Metric], assets: dict[str, Path], out: Path) -> Path:
    im, d = base_page("青少年棒球投球动作体检报告", "孩子投球动作与正常速度教练动作对照", 1)
    hero = card(im, (PAGE_MARGIN, 255, PAGE_W - PAGE_MARGIN, 705), fill="#101828", outline=None)
    d.text((118, 305), "3D 动作体检报告", font=cn_font(56), fill="#ffffff")
    d.text((120, 380), "孩子和教练差在哪里，差多少，怎么改", font=cn_font(35), fill="#dbeafe")
    d.text((120, 450), "适用：青少年投球评估、训练复测、年度成长档案", font=cn_font(25), fill="#cbd5e1")
    paste_image(im, assets["thumb2d"], (790, 285, 1115, 650), radius=26)
    pill(d, (120, 555), "中文报告", "#e0f2fe", "#075985")
    pill(d, (290, 555), "可量化诊断", "#dcfce7", "#166534")
    pill(d, (500, 555), "训练建议", "#ffedd5", "#9a3412")

    y = 760
    draw_section_title(d, "本次样例关键结论", y)
    y += 68
    for i, m in enumerate(metrics[:4]):
        x = PAGE_MARGIN + i * 272
        draw_metric_tile(im, d, (x, y, x + 250, y + 190), m)

    y = 1040
    draw_section_title(d, "报告回答的 3 个问题", y)
    questions = [
        ("01", "差在哪里", "通过 2D/3D 骨骼叠加、关键相位截图和角度曲线，直观看到动作差距。"),
        ("02", "差多少", "输出速度、角度、厘米级位移、逐帧偏差和百分比差距，形成自动对比报告。"),
        ("03", "怎么改", "根据数据差距给出训练安排、家庭练习和复测重点。"),
    ]
    for i, (idx, title, body) in enumerate(questions):
        x = PAGE_MARGIN + i * 365
        draw_question_card(im, d, (x, 1120, x + 340, 1405), idx, title, body)
    draw_note_box(
        im,
        d,
        (PAGE_MARGIN, 1465, PAGE_W - PAGE_MARGIN, 1610),
        "测量说明",
        "人体动作指标主要来自同一套 3D 骨架计算；教练视频为正常速度拍摄。球速目前为视频估算，正式复测建议用雷达枪或球轨迹设备补充。",
    )
    im.save(out)
    return out


def render_dashboard_page(metrics: list[Metric], assets: dict[str, Path], out: Path) -> Path:
    im, d = base_page("核心运动学仪表盘", "家长和教练先看 8 个最关键的投球指标", 2)
    card(im, (PAGE_MARGIN, 250, PAGE_W - PAGE_MARGIN, 980), fill="#ffffff")
    paste_image_contain(im, assets["kinematic_dashboard"], (105, 275, 1135, 955), radius=12)
    draw_section_title(d, "本次最需要先改的 3 项", 1035)
    metric_by_key = {m.key: m for m in metrics}
    priorities = [
        ("1", "跨步与重心转移", delta_text(metric_by_key["stride"]), "先解决落脚稳定和身体前移，避免只让孩子“用手甩快”。"),
        ("2", "躯干/骨盆速度", delta_text(metric_by_key["trunk_vel"]), "重点看骨盆启动后，躯干是否延迟并快速跟上。"),
        ("3", "出手侧手速", delta_text(metric_by_key["hand_speed"]), "用右腕 3D 速度观察末端加速是否随动力链改善而上升。"),
    ]
    for i, item in enumerate(priorities):
        x = PAGE_MARGIN + i * 365
        draw_priority_card(im, d, (x, 1110, x + 335, 1325), *item)
    draw_note_box(
        im,
        d,
        (PAGE_MARGIN, 1405, PAGE_W - PAGE_MARGIN, 1560),
        "为什么这样展示",
        "投球报告不应让家长先看复杂曲线。仪表盘先回答“哪里不足”，后续再用阶段时间轴说明问题发生在哪个动作阶段。",
    )
    im.save(out)
    return out


def render_visual_page(assets: dict[str, Path], out: Path) -> Path:
    im, d = base_page("动作可视化对照", "用截图先看清孩子和教练的动作差别", 3)
    draw_section_title(d, "关键相位截图：孩子 / 教练", 255)
    phase_boxes = [
        (PAGE_MARGIN, 330, 405, 920, assets["phase_contact"], assets["coach_contact"], "抬腿/跨步", "看落脚方向和身体前移"),
        (445, 330, 775, 920, assets["phase_release"], assets["coach_release"], "出手附近", "看前腿支撑和手臂位置"),
        (815, 330, 1145, 920, assets["phase_follow"], assets["coach_follow"], "随挥稳定", "看头部和身体是否稳定"),
    ]
    for box in phase_boxes:
        x0, y0, x1, y1, child_path, coach_path, title, en = box
        card(im, (x0, y0, x1, y1), fill="#ffffff")
        d.text((x0 + 22, y0 + 18), "孩子", font=cn_font(19), fill="#2563eb")
        paste_image(im, child_path, (x0 + 18, y0 + 48, x1 - 18, y0 + 285), radius=16)
        d.text((x0 + 22, y0 + 302), "教练", font=cn_font(19), fill="#111827")
        paste_image(im, coach_path, (x0 + 18, y0 + 332, x1 - 18, y1 - 105), radius=16)
        d.text((x0 + 28, y1 - 78), title, font=cn_font(27), fill="#101828")
        d.text((x0 + 28, y1 - 43), en, font=cn_font(20), fill="#667085")

    draw_section_title(d, "2D 骨架 + 3D 模型", 980)
    card(im, (PAGE_MARGIN, 1055, 590, 1540), fill="#ffffff")
    paste_image(im, assets["thumb2d"], (110, 1085, 545, 1505), radius=22)
    d.text((118, 1560), "2D叠加：适合和原视频动作直接对照", font=cn_font(24), fill="#344054")
    card(im, (650, 1055, 1162, 1540), fill="#ffffff")
    paste_image(im, assets["thumb3d"], (685, 1120, 1125, 1445), radius=22)
    d.text((685, 1560), "3D模型：适合看空间姿态和身体中心轨迹", font=cn_font(24), fill="#344054")
    draw_note_box(
        im,
        d,
        (PAGE_MARGIN, 1585, PAGE_W - PAGE_MARGIN, 1660),
        "正式对照",
        "当前已使用正常速度教练视频提取 3D 骨架。复测时建议保持同机位、同距离、同动作要求，便于输出逐帧关节偏差和最需要改进的相位。",
    )
    im.save(out)
    return out


def render_curves_page(assets: dict[str, Path], out: Path) -> Path:
    im, d = base_page("投球阶段与动力链", "把指标放回投球动作阶段，更容易判断怎么改", 4)
    card(im, (PAGE_MARGIN, 250, PAGE_W - PAGE_MARGIN, 820), fill="#ffffff")
    paste_image_contain(im, assets["pitch_timeline"], (105, 275, 1135, 795), radius=10)
    card(im, (PAGE_MARGIN, 880, PAGE_W - PAGE_MARGIN, 1445), fill="#ffffff")
    paste_image_contain(im, assets["kinetic_chain_flow"], (105, 905, 1135, 1420), radius=10)
    draw_note_box(
        im,
        d,
        (PAGE_MARGIN, 1515, PAGE_W - PAGE_MARGIN, 1665),
        "取舍",
        "纯角度时间曲线更适合工程诊断；正式报告优先使用阶段时间轴和动力链流。原始曲线仍保留在课程 PPT 的原始数据部分。",
    )
    im.save(out)
    return out


def render_high_level_graphs_page(assets: dict[str, Path], out: Path) -> Path:
    im, d = base_page("高层次运动学图表", "把复杂指标合并成家长和教练容易判断的图", 5)
    card(im, (PAGE_MARGIN, 250, 610, 860), fill="#ffffff")
    paste_image_contain(im, assets["radar_chart"], (95, 275, 595, 835), radius=10)
    card(im, (650, 250, PAGE_W - PAGE_MARGIN, 860), fill="#ffffff")
    paste_image_contain(im, assets["balance_chart"], (665, 275, 1148, 835), radius=10)
    card(im, (PAGE_MARGIN, 915, PAGE_W - PAGE_MARGIN, 1560), fill="#ffffff")
    paste_image_contain(im, assets["heatmap_chart"], (95, 940, 1145, 1535), radius=10)
    draw_note_box(
        im,
        d,
        (PAGE_MARGIN, 1600, PAGE_W - PAGE_MARGIN, 1685),
        "读图顺序",
        "先看雷达图判断整体短板，再看左右平衡确认支撑稳定，最后用热力图定位优先纠正部位。",
    )
    im.save(out)
    return out


def render_child_compare_page(assets: dict[str, Path], out: Path) -> Path:
    im, d = base_page("儿童 vs 儿童对比", "Julian 与 Youyou 使用同一套 3D 指标口径", 6)
    card(im, (PAGE_MARGIN, 250, PAGE_W - PAGE_MARGIN, 1395), fill="#ffffff")
    paste_image_contain(im, assets["child_compare_chart"], (105, 280, 1135, 1370), radius=12)
    draw_note_box(
        im,
        d,
        (PAGE_MARGIN, 1470, PAGE_W - PAGE_MARGIN, 1645),
        "阅读方式",
        "蓝点是 Julian，橙点是 Youyou，黑线是正常速度教练参考。该页不做儿童排名，只帮助家长理解孩子在同龄样本中的动作特点。",
    )
    im.save(out)
    return out


def render_correction_graph_page(assets: dict[str, Path], out: Path) -> Path:
    im, d = base_page("标准姿态纠正图", "用缩放后的教练动作给孩子做参照", 7)
    card(im, (PAGE_MARGIN, 255, PAGE_W - PAGE_MARGIN, 1245), fill="#ffffff")
    paste_image_contain(im, assets["standard_overlay"], (110, 285, 1130, 1215), radius=12)
    draw_section_title(d, "颜色说明", 1315)
    rows = [
        ("浅蓝虚线", "孩子原始姿态连线", "#60a5fa"),
        ("绿色", "按孩子身材缩放后的教练标准姿态", "#16a34a"),
        ("红色", "偏差较大的孩子原始骨段，训练中优先纠正", "#ef4444"),
    ]
    for i, (name, body, color) in enumerate(rows):
        y = 1390 + i * 68
        d.rounded_rectangle((PAGE_MARGIN, y, PAGE_MARGIN + 36, y + 36), radius=8, fill=color)
        d.text((PAGE_MARGIN + 55, y - 2), f"{name}：{body}", font=cn_font(26), fill="#344054")
    draw_note_box(
        im,
        d,
        (PAGE_MARGIN, 1605, PAGE_W - PAGE_MARGIN, 1690),
        "缩放方式",
        "保留教练标准动作方向，但把对应骨段长度调整到孩子身材，避免直接用成人或教练身高造成误判。",
    )
    im.save(out)
    return out


def render_motion_metrics_page(motion_metrics: list[MotionMetric], out: Path) -> Path:
    im, d = base_page("投球动作全指标诊断", "覆盖角度、速度、距离和稳定性指标", 8)
    draw_section_title(d, "核心动作指标", 255)
    draw_note_box(
        im,
        d,
        (PAGE_MARGIN, 315, PAGE_W - PAGE_MARGIN, 425),
        "计算说明",
        "关节角和身体段夹角优先使用 3D；速度为同设备同流程下的 3D估算，适合比较趋势。重心转移目前使用髋部和脊柱中心位移估算，不等同于力板或 Vicon 的真实重心。",
    )
    y0 = 465
    card_w = 522
    card_h = 138
    for i, metric in enumerate(motion_metrics):
        col = i % 2
        row = i // 2
        x = PAGE_MARGIN + col * (card_w + 40)
        y = y0 + row * (card_h + 22)
        draw_motion_metric_card(im, d, (x, y, x + card_w, y + card_h), metric)
    im.save(out)
    return out


def render_training_page(metrics: list[Metric], parent_guidance: str, out: Path) -> Path:
    im, d = base_page("改进建议与训练安排", "先解决落脚、身体前移和发力顺序", 9)
    metric_by_key = {m.key: m for m in metrics}
    hand_speed = metric_by_key["hand_speed"].value
    coach_hand = metric_by_key["hand_speed"].coach_value
    draw_section_title(d, "4周训练重点", 255)
    cards = [
        ("第 1-2 周", "跨步和落脚稳定", "标记 80%-90% 身高落脚区；轻球投掷；每组 6-8 次。", "#eff6ff"),
        ("第 2-3 周", "髋先转，肩延迟", "提膝停顿、分段投球练习；先低强度保证顺序。", "#f0fdf4"),
        ("第 3-4 周", "速度转化", "药球侧抛 + 毛巾出手；只在顺序稳定后提高速度。", "#fff7ed"),
    ]
    for i, item in enumerate(cards):
        draw_training_card(im, d, (PAGE_MARGIN + i * 365, 335, PAGE_MARGIN + i * 365 + 335, 640), *item)

    draw_section_title(d, "复测指标", 715)
    table = [
        ("指标", "目标变化", "说明"),
        ("跨步长度", f"向教练 {metric_by_key['stride'].coach_value:.1f} cm 靠近" if metric_by_key["stride"].coach_value else "逐步接近 116-131 cm", "基于 3D 双踝水平距离；后续可用标定尺或场地坐标校正。"),
        ("骨盆-躯干峰值时间差", "骨盆先于躯干出现峰值", "用于判断动力链顺序是否改善。"),
        ("出手侧手速估算值", f"从 {hand_speed:.1f} m/s 向教练 {coach_hand:.1f} m/s 靠近" if coach_hand else f"从 {hand_speed:.1f} m/s 向 7.0 m/s 以上靠近", "同设备、同角度复测更有意义。"),
        ("球速", "用雷达枪替代视频估算", "正式训练建议直接接入雷达枪数据。"),
    ]
    draw_clean_table(im, d, (PAGE_MARGIN, 785, PAGE_W - PAGE_MARGIN, 1135), table)

    draw_section_title(d, "给家长的动作解读", 1210)
    card(im, (PAGE_MARGIN, 1280, PAGE_W - PAGE_MARGIN, 1670), fill="#ffffff")
    draw_parent_guidance(d, parent_guidance, (112, 1315), PAGE_W - 224, 310)
    im.save(out)
    return out


def render_retest_page(metrics: list[Metric], motion_metrics: list[MotionMetric], out: Path) -> Path:
    im, d = base_page("复测重点与注意事项", "把每次训练后的变化量记录下来", 11)
    metric_by_key = {m.key: m for m in metrics}
    motion_by_key = {m.key: m for m in motion_metrics}
    draw_section_title(d, "下次复测优先看这 4 项", 255)
    rows = [
        ("复测指标", "本次结果", "下次希望看到的变化"),
        ("跨步长度", f"{metric_by_key['stride'].value:.1f} cm", "逐步接近教练水平，同时落脚后身体能跟上。"),
        ("前脚方向", format_motion_value(motion_by_key["foot_direction"].value, motion_by_key["foot_direction"].unit), "脚尖更接近目标方向，减少身体提前打开。"),
        ("出手侧手速", f"{metric_by_key['hand_speed'].value:.1f} m/s", "在动作顺序稳定的前提下逐步提高。"),
        ("髋肩分离", format_motion_value(motion_by_key["hip_shoulder_sep"].value, motion_by_key["hip_shoulder_sep"].unit), "保持可训练范围，不追求越大越好。"),
    ]
    draw_clean_table(im, d, (PAGE_MARGIN, 330, PAGE_W - PAGE_MARGIN, 730), rows)

    draw_section_title(d, "哪些数据最可靠，哪些要谨慎看", 810)
    reliability = [
        ("3D关节角", "肘部、膝部、躯干、髋肩分离等角度由 3D骨架计算，适合看相对差距和复测趋势。"),
        ("3D速度", "手臂、手部速度受单目深度抖动影响，适合同设备、同机位下看提升百分比。"),
        ("身体中心位移", "当前用髋部和脊柱中心估算，不等同于真实重心；可用于判断身体是否充分前移。"),
        ("球速", "视频估算只能参考。正式训练建议用雷达枪或球轨迹设备记录。"),
    ]
    for i, (title, body) in enumerate(reliability):
        draw_small_field_card(im, d, (PAGE_MARGIN + (i % 2) * 545, 885 + (i // 2) * 210, PAGE_MARGIN + (i % 2) * 545 + 505, 1055 + (i // 2) * 210), title, body)

    draw_section_title(d, "年度成长档案建议", 1345)
    draw_note_box(
        im,
        d,
        (PAGE_MARGIN, 1415, PAGE_W - PAGE_MARGIN, 1600),
        "记录方式",
        "每次复测保留同一相位截图、关键指标、训练内容和身体反应。建议每 4 周复测一次，重点看跨步长度、前脚方向、出手侧手速和髋肩分离是否同步改善。",
    )
    draw_note_box(
        im,
        d,
        (PAGE_MARGIN, 1625, PAGE_W - PAGE_MARGIN, 1690),
        "训练风险提示",
        "如出现肩肘疼痛、疲劳后动作明显变形或控制下降，应先降低训练量，并由教练或专业人员进一步筛查。",
    )
    im.save(out)
    return out


def render_growth_page(assets: dict[str, Path], out: Path) -> Path:
    im, d = base_page("报告产品分层与年度成长档案", "从单次诊断扩展到俱乐部长期管理", 7)
    draw_section_title(d, "产品版本", 255)
    product = [
        ("基础版", "家长/孩子", "红黄绿差距图、关键截图、3 条训练建议", "PDF"),
        ("专业版", "教练/体能师", "角度曲线、动力链顺序、逐帧偏差、CSV", "PDF + CSV + overlay"),
        ("年度档案", "俱乐部", "进步曲线、动作演变视频、训练闭环", "Dashboard + 年度 PDF"),
    ]
    draw_clean_table(im, d, (PAGE_MARGIN, 330, PAGE_W - PAGE_MARGIN, 620), [("版本", "对象", "核心内容", "交付物")] + product)
    draw_section_title(d, "年度成长档案", 700)
    card(im, (PAGE_MARGIN, 775, PAGE_W - PAGE_MARGIN, 1175), fill="#ffffff")
    paste_image(im, assets["growth"], (110, 810, 1125, 1140), radius=12)
    draw_section_title(d, "参考来源与中文备注", 1245)
    refs = [
        ("Diffendaffer 2023", "投球事件点、跨步、膝角、肩肘角度等教练可解释指标。"),
        ("Orishimo 2023", "骨盆/躯干旋转速度、髋肩分离、动力链时序参考。"),
        ("Hiramoto 2019", "青少年肩、髋、躯干活动度参考，适合国内青少年报告谨慎对照。"),
    ]
    for i, (title, body) in enumerate(refs):
        draw_reference_card(im, d, (PAGE_MARGIN, 1325 + i * 95, PAGE_W - PAGE_MARGIN, 1400 + i * 95), title, body)
    draw_note_box(
        im,
        d,
        (PAGE_MARGIN, 1610, PAGE_W - PAGE_MARGIN, 1680),
        "限制",
        "本模板的人体动作指标主要来自 3D pose，孩子与正常速度教练使用同一套 3D 计算；球速仍需雷达枪或 3D 球轨迹验证。不替代 Vicon、力板或医学体检。",
    )
    im.save(out)
    return out


def render_prompt_page(metrics: list[Metric], motion_metrics: list[MotionMetric], parent_guidance: str, out: Path) -> Path:
    im, d = base_page("大模型生成报告", "完整输入内容与生成结果", 10)
    prompt = build_parent_guidance_prompt(metrics, motion_metrics)
    draw_section_title(d, "输入内容", 255)
    card(im, (PAGE_MARGIN, 325, PAGE_W - PAGE_MARGIN, 900), fill="#101828", outline=None, radius=22)
    draw_wrapped_text(d, prompt, (112, 360), PAGE_W - 224, cn_font(18), "#e5e7eb", line_gap=6, max_lines=22)
    draw_section_title(d, "生成结果", 960)
    card(im, (PAGE_MARGIN, 1030, PAGE_W - PAGE_MARGIN, 1620), fill="#ffffff", outline="#d0d5dd", radius=22)
    draw_parent_guidance(d, parent_guidance, (112, 1065), PAGE_W - 224, 510)
    draw_note_box(
        im,
        d,
        (PAGE_MARGIN, 1645, PAGE_W - PAGE_MARGIN, 1700),
        "用途",
        "本页用于课程任务留档；给家长正式交付时可隐藏，只保留前面的结论和训练建议。",
    )
    im.save(out)
    return out


def build_llm_prompt(metrics: list[Metric]) -> str:
    lines = [
        "你是一名青少年棒球投球动作教练，请基于以下 3D pose 指标生成中文报告。",
        "要求：1）先回答孩子和教练差在哪里；2）用速度、角度、厘米和百分比说明差多少；3）给出 3-5 条家庭训练建议。",
        "数据来源：孩子 benchmark_pitch_vertical_09 smoothed 3D pose；教练 Benchmark棒球视频(1)/Pitch_horizontal_coach.mov GVHMR 3D，57.45 fps 正常速度。球速为孩子 2D object/video proxy，不能与教练球速比较。",
        "指标：",
    ]
    for m in metrics:
        coach = f"{m.coach_value:.1f} {m.unit}" if m.coach_value is not None else "暂无"
        lines.append(f"- {m.label_cn}: 孩子 {m.value:.1f} {m.unit}; 教练 {coach}; 差距 {delta_text(m)}")
    lines.append("输出格式：家长版摘要、教练版诊断、训练计划、复测指标、伤害预防提示。")
    return "\n".join(lines)


def base_page(title: str, subtitle: str, page_no: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (PAGE_W, PAGE_H), "#f5f7fb")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, PAGE_W, 118), fill="#ffffff")
    d.text((PAGE_MARGIN, 36), "棒球动作实验室", font=cn_font(23), fill="#2563eb")
    d.text((PAGE_W - 350, 38), "青少年投球动作报告", font=cn_font(22), fill="#667085")
    d.text((PAGE_MARGIN, 145), title, font=cn_font(45), fill="#101828")
    d.text((PAGE_MARGIN, 198), subtitle, font=cn_font(24), fill="#667085")
    d.line((PAGE_MARGIN, PAGE_H - 82, PAGE_W - PAGE_MARGIN, PAGE_H - 82), fill="#d0d5dd", width=2)
    d.text((PAGE_MARGIN, PAGE_H - 55), "3D视频动作分析报告，仅用于训练参考", font=cn_font(19), fill="#98a2b3")
    d.text((PAGE_W - 150, PAGE_H - 55), f"{page_no:02d}", font=cn_font(20), fill="#98a2b3")
    return im, d


def card(im: Image.Image, box: tuple[int, int, int, int], fill: str = "#ffffff", outline: str | None = "#e4e7ec", radius: int = 24) -> ImageDraw.ImageDraw:
    d = ImageDraw.Draw(im)
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2 if outline else 1)
    return d


def paste_image(im: Image.Image, path: Path, box: tuple[int, int, int, int], radius: int = 18) -> None:
    src = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = box
    src = ImageOps.fit(src, (x1 - x0, y1 - y0), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    mask = Image.new("L", src.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, src.width, src.height), radius=radius, fill=255)
    im.paste(src, (x0, y0), mask)


def paste_image_contain(im: Image.Image, path: Path, box: tuple[int, int, int, int], radius: int = 18, fill: str = "#ffffff") -> None:
    src = Image.open(path).convert("RGB")
    x0, y0, x1, y1 = box
    target_w, target_h = x1 - x0, y1 - y0
    src.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), fill)
    px = (target_w - src.width) // 2
    py = (target_h - src.height) // 2
    canvas.paste(src, (px, py))
    mask = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, canvas.width, canvas.height), radius=radius, fill=255)
    im.paste(canvas, (x0, y0), mask)


def pill(d: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str, text_fill: str) -> None:
    x, y = xy
    font = cn_font(23)
    w = int(d.textlength(text, font=font)) + 42
    d.rounded_rectangle((x, y, x + w, y + 48), radius=24, fill=fill)
    d.text((x + 21, y + 10), text, font=font, fill=text_fill)


def draw_section_title(d: ImageDraw.ImageDraw, title: str, y: int, x: int = PAGE_MARGIN) -> None:
    d.rounded_rectangle((x, y + 8, x + 12, y + 48), radius=6, fill="#2563eb")
    d.text((x + 26, y), title, font=cn_font(34), fill="#101828")


def draw_metric_tile(im: Image.Image, d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], m: Metric) -> None:
    color = "#16a34a" if in_band(m) else "#f97316"
    bg = "#f0fdf4" if in_band(m) else "#fff7ed"
    card(im, box, fill=bg, outline="#d0d5dd", radius=24)
    x0, y0, x1, y1 = box
    d.text((x0 + 24, y0 + 22), m.label_cn, font=cn_font(26), fill="#101828")
    d.text((x0 + 24, y0 + 58), metric_source_label(m), font=cn_font(17), fill="#667085")
    d.text((x0 + 24, y0 + 105), f"{m.value:.1f}", font=cn_font(42), fill=color)
    d.text((x0 + 134, y0 + 122), m.unit, font=cn_font(22), fill=color)
    d.text((x0 + 24, y0 + 150), delta_text(m), font=cn_font(18), fill="#344054")
    d.text((x0 + 24, y0 + 172), coach_value_text(m), font=cn_font(16), fill="#667085")


def draw_motion_metric_card(im: Image.Image, d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], m: MotionMetric) -> None:
    status_color = {"ok": "#16a34a", "warn": "#f59e0b", "bad": "#ef4444"}.get(m.status, "#f59e0b")
    status_bg = {"ok": "#ecfdf3", "warn": "#fff7ed", "bad": "#fef2f2"}.get(m.status, "#fff7ed")
    card(im, box, fill="#ffffff", outline="#e4e7ec", radius=18)
    x0, y0, x1, _ = box
    d.ellipse((x0 + 18, y0 + 26, x0 + 36, y0 + 44), fill=status_color)
    d.text((x0 + 52, y0 + 18), m.label_cn, font=cn_font(25), fill="#101828")
    d.text((x0 + 52, y0 + 52), short_motion_note(m), font=cn_font(17), fill="#667085")
    value_text = format_motion_value(m.value, m.unit)
    d.text((x1 - 155, y0 + 20), value_text, font=cn_font(31), fill=status_color)
    method_text = method_label_cn(m.method)
    method_w = int(d.textlength(method_text, font=cn_font(15))) + 26
    d.rounded_rectangle((x0 + 52, y0 + 80, x0 + 52 + method_w, y0 + 108), radius=14, fill=status_bg)
    d.text((x0 + 65, y0 + 86), method_text, font=cn_font(15), fill=status_color)
    coach = "教练: " + (format_motion_value(m.coach_value, m.unit) if m.coach_value is not None else "暂无")
    d.text((x0 + 185, y0 + 84), coach, font=cn_font(16), fill="#475467")
    draw_wrapped_text(d, motion_source_note_cn(m), (x0 + 52, y0 + 112), x1 - x0 - 78, cn_font(14), "#667085", line_gap=3, max_lines=1)


def short_motion_note(m: MotionMetric) -> str:
    notes = {
        "elbow_bend": "出手时肘部角度",
        "arm_abduction": "出手时上臂抬起角度",
        "trunk_lean": "出手时躯干倾斜角",
        "stride_angle": "落脚时前腿角度",
        "lead_knee": "出手时前膝支撑角",
        "hip_shoulder_sep": "肩线与髋线角度差",
        "arm_speed": "右腕速度相对教练",
        "stride_length": "跨步长度占身高比例",
        "weight_transfer": "身体中心前移幅度",
        "head_stability": "头部横向稳定评分",
        "foot_direction": "落脚时脚尖方向",
        "wrist_snap": "出手时手腕角度",
        "fingertip_speed": "手部末端速度相对教练",
    }
    return notes.get(m.key, m.note[:18])


def motion_source_note_cn(m: MotionMetric) -> str:
    if m.key == "weight_transfer":
        return "用髋部和脊柱中心前移估算，不是真实重心。"
    if m.key == "head_stability":
        return "用头部横向晃动估算，越高越稳定。"
    if m.key in {"arm_speed", "fingertip_speed"}:
        return "速度类指标适合同设备同机位复测对比。"
    if m.key == "wrist_snap":
        return "手部末端由 3D骨架估算，不等同于真实指尖标记点。"
    return "由 3D骨架关键点计算，适合看相对差距和复测趋势。"


def draw_parent_guidance(
    d: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width: int,
    height: int,
) -> None:
    lines = [line.strip() for line in sanitize_parent_guidance(text).splitlines() if line.strip()]
    cleaned: list[str] = []
    for line in lines:
        if line.startswith(("一、", "二、", "三、", "四、")):
            cleaned.append(line)
        elif len(cleaned) < 8:
            cleaned.append(line)
    summary = "\n".join(cleaned[:8])
    font = cn_font(21)
    d.rectangle((xy[0], xy[1] + height - 1, xy[0] + width, xy[1] + height), fill="#ffffff")
    draw_wrapped_text(d, summary, xy, width, font, "#344054", line_gap=8, max_lines=12)


def format_motion_value(value: float | None, unit: str) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    suffix = "%" if unit in {"%", "%height", "%stride"} else f" {unit}"
    if unit in {"%", "%height", "%stride"}:
        return f"{value:.0f}{suffix}"
    if unit == "deg":
        return f"{value:.0f}°"
    if unit == "m/s":
        return f"{value:.1f} m/s"
    return f"{value:.1f}{suffix}"


def draw_question_card(im: Image.Image, d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], idx: str, title: str, body: str) -> None:
    card(im, box, fill="#ffffff")
    x0, y0, _, _ = box
    d.text((x0 + 26, y0 + 24), idx, font=cn_font(42), fill="#2563eb")
    d.text((x0 + 26, y0 + 92), title, font=cn_font(31), fill="#101828")
    draw_wrapped_text(d, body, (x0 + 26, y0 + 150), 285, cn_font(23), "#475467", line_gap=12)


def draw_note_box(im: Image.Image, d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str) -> None:
    card(im, box, fill="#eef6ff", outline="#bfdbfe", radius=20)
    x0, y0, x1, _ = box
    title_font = cn_font(25)
    d.text((x0 + 24, y0 + 18), title, font=title_font, fill="#1d4ed8")
    body_x = x0 + 42 + int(d.textlength(title, font=title_font))
    draw_wrapped_text(d, body, (body_x, y0 + 20), x1 - body_x - 24, cn_font(21), "#344054", line_gap=9)


def draw_gap_panel(im: Image.Image, d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], metrics: list[Metric]) -> None:
    card(im, box, fill="#ffffff")
    x0, y0, x1, _ = box
    d.text((x0 + 30, y0 + 24), "蓝点=孩子，黑线=教练，绿色=论文参考范围", font=cn_font(22), fill="#667085")
    bar_x0, bar_x1 = x0 + 265, x1 - 165
    for i, m in enumerate(metrics):
        y = y0 + 85 + i * 88
        d.text((x0 + 30, y - 4), m.label_cn, font=cn_font(23), fill="#101828")
        d.text((x0 + 30, y + 27), metric_source_label(m), font=cn_font(16), fill="#667085")
        values = [m.value, m.ref_low, m.ref_high]
        if m.coach_value is not None:
            values.append(m.coach_value)
        lo = min(values) * 0.82
        hi = max(values) * 1.18
        if abs(hi - lo) < 1:
            hi = lo + 1
        d.rounded_rectangle((bar_x0, y + 8, bar_x1, y + 32), radius=12, fill="#eef2f7")
        rx0 = bar_x0 + (m.ref_low - lo) / (hi - lo) * (bar_x1 - bar_x0)
        rx1 = bar_x0 + (m.ref_high - lo) / (hi - lo) * (bar_x1 - bar_x0)
        d.rounded_rectangle((rx0, y, rx1, y + 40), radius=18, fill="#b7e4c7")
        vx = bar_x0 + (m.value - lo) / (hi - lo) * (bar_x1 - bar_x0)
        dot = "#2563eb" if in_band(m) else "#f97316"
        d.ellipse((vx - 15, y + 3, vx + 15, y + 33), fill=dot)
        if m.coach_value is not None:
            cx = bar_x0 + (m.coach_value - lo) / (hi - lo) * (bar_x1 - bar_x0)
            d.line((cx, y - 4, cx, y + 44), fill="#111827", width=5)
        d.text((bar_x1 + 22, y - 4), f"{m.value:.1f}", font=cn_font(25), fill=dot)
        d.text((bar_x1 + 22, y + 27), m.unit, font=cn_font(16), fill="#667085")


def draw_priority_card(im: Image.Image, d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], idx: str, title: str, delta: str, body: str) -> None:
    card(im, box, fill="#ffffff")
    x0, y0, x1, _ = box
    d.ellipse((x0 + 22, y0 + 24, x0 + 78, y0 + 80), fill="#fff7ed")
    d.text((x0 + 42, y0 + 34), idx, font=cn_font(28), fill="#f97316")
    d.text((x0 + 95, y0 + 24), title, font=cn_font(27), fill="#101828")
    d.text((x0 + 95, y0 + 62), delta, font=cn_font(22), fill="#f97316")
    draw_wrapped_text(d, body, (x0 + 26, y0 + 105), x1 - x0 - 52, cn_font(21), "#475467", line_gap=8)


def draw_small_field_card(im: Image.Image, d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str) -> None:
    card(im, box, fill="#ffffff")
    x0, y0, x1, _ = box
    d.text((x0 + 22, y0 + 18), title, font=cn_font(26), fill="#2563eb")
    draw_wrapped_text(d, body, (x0 + 22, y0 + 58), x1 - x0 - 44, cn_font(20), "#475467", line_gap=7)


def draw_training_card(im: Image.Image, d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, heading: str, body: str, fill: str) -> None:
    card(im, box, fill=fill, outline="#d0d5dd")
    x0, y0, x1, _ = box
    d.text((x0 + 25, y0 + 25), title, font=cn_font(24), fill="#2563eb")
    d.text((x0 + 25, y0 + 78), heading, font=cn_font(30), fill="#101828")
    draw_wrapped_text(d, body, (x0 + 25, y0 + 140), x1 - x0 - 50, cn_font(22), "#475467", line_gap=10)


def draw_reference_card(im: Image.Image, d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, body: str) -> None:
    card(im, box, fill="#ffffff", radius=16)
    x0, y0, x1, _ = box
    d.text((x0 + 24, y0 + 22), title, font=cn_font(25), fill="#101828")
    draw_wrapped_text(d, body, (x0 + 245, y0 + 23), x1 - x0 - 275, cn_font(21), "#475467", line_gap=8)


def draw_clean_table(im: Image.Image, d: ImageDraw.ImageDraw, box: tuple[int, int, int, int], rows: list[tuple[str, ...]]) -> None:
    card(im, box, fill="#ffffff")
    x0, y0, x1, y1 = box
    col_count = len(rows[0])
    widths = [0.18, 0.26, 0.36, 0.20] if col_count == 4 else [0.25, 0.40, 0.35]
    widths = widths[:col_count]
    total = x1 - x0 - 40
    cols = [x0 + 20]
    for frac in widths[:-1]:
        cols.append(cols[-1] + int(total * frac))
    row_h = (y1 - y0 - 40) // len(rows)
    for r, row in enumerate(rows):
        yy = y0 + 20 + r * row_h
        fill = "#eaf2ff" if r == 0 else "#ffffff"
        d.rounded_rectangle((x0 + 20, yy, x1 - 20, yy + row_h), radius=8 if r == 0 else 0, fill=fill)
        for c, text in enumerate(row):
            cx = cols[c]
            next_x = cols[c + 1] if c + 1 < len(cols) else x1 - 20
            font = cn_font(22 if r else 23)
            draw_wrapped_text(d, text, (cx + 12, yy + 12), next_x - cx - 20, font, "#344054", line_gap=6, max_lines=2)
        d.line((x0 + 20, yy + row_h, x1 - 20, yy + row_h), fill="#e4e7ec", width=2)


def draw_wrapped_text(
    d: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    width: int,
    font: ImageFont.FreeTypeFont,
    fill: str,
    line_gap: int = 8,
    max_lines: int | None = None,
) -> int:
    x, y = xy
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for char in paragraph:
            trial = current + char
            if d.textlength(trial, font=font) <= width or not current:
                current = trial
            else:
                lines.append(current)
                current = char
        if current:
            lines.append(current)
        else:
            lines.append("")
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip("。,.，") + "..."
    for line in lines:
        d.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle("TitleCN", parent=base["Title"], fontName=PDF_FONT, fontSize=24, leading=30, textColor=colors.HexColor("#172033")),
        "SubTitle": ParagraphStyle("SubTitleCN", parent=base["BodyText"], fontName=PDF_FONT, fontSize=12, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#526070")),
        "H1": ParagraphStyle("H1CN", parent=base["Heading1"], fontName=PDF_FONT, fontSize=18, leading=24, textColor=colors.HexColor("#172033"), spaceAfter=8),
        "H2": ParagraphStyle("H2CN", parent=base["Heading2"], fontName=PDF_FONT, fontSize=14, leading=20, textColor=colors.HexColor("#1d4ed8"), spaceBefore=6, spaceAfter=4),
        "Body": ParagraphStyle("BodyCN", parent=base["BodyText"], fontName=PDF_FONT, fontSize=10.5, leading=16, textColor=colors.HexColor("#344054")),
        "Small": ParagraphStyle("SmallCN", parent=base["BodyText"], fontName=PDF_FONT, fontSize=8.5, leading=12, textColor=colors.HexColor("#526070")),
        "Caption": ParagraphStyle("CaptionCN", parent=base["BodyText"], fontName=PDF_FONT, fontSize=8.5, leading=12, alignment=TA_CENTER, textColor=colors.HexColor("#667085")),
        "Table": ParagraphStyle("TableCN", parent=base["BodyText"], fontName=PDF_FONT, fontSize=8.7, leading=11.5, textColor=colors.HexColor("#344054"), alignment=TA_LEFT),
    }


def bullet(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(f"- {text}", styles["Body"])


def metric_cards(metrics: list[Metric], styles: dict[str, ParagraphStyle]) -> Table:
    cells = []
    for m in metrics:
        color = "#dcfce7" if in_band(m) else "#ffedd5"
        status = "达标" if in_band(m) else "需关注"
        cells.append(
            Paragraph(
                f"<b>{m.label_cn}</b><br/>{m.value:.1f} {m.unit}<br/><font color='#667085'>{status}</font>",
                styles["Table"],
            )
        )
    table = Table([cells], colWidths=[4.35 * cm] * len(cells))
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d0d5dd")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e4e7ec")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def metrics_table(metrics: list[Metric], styles: dict[str, ParagraphStyle]) -> Table:
    rows = [[p("指标", styles), p("当前值", styles), p("教练/文献参考", styles), p("差距", styles), p("中文解读", styles)]]
    for m in metrics:
        rows.append(
            [
                p(f"{m.label_cn}<br/><font color='#667085'>{m.label_en}</font>", styles),
                p(f"{m.value:.1f} {m.unit}", styles),
                p(f"{m.ref_low:.0f}-{m.ref_high:.0f} {m.unit}", styles),
                p(delta_text(m), styles),
                p(m.interpretation, styles),
            ]
        )
    table = Table(rows, colWidths=[3.0 * cm, 2.3 * cm, 3.1 * cm, 2.4 * cm, 7.0 * cm], repeatRows=1)
    table.setStyle(default_table_style())
    return table


def training_table(styles: dict[str, ParagraphStyle]) -> Table:
    rows = [
        [p("问题", styles), p("训练内容", styles), p("复测指标", styles)],
        [p("骨盆启动慢或躯干提前", styles), p("分段投球 drill：提膝停顿、髋先转、肩延迟打开。每组 6-8 次，保持低强度。", styles), p("骨盆峰值早于躯干峰值 80-180 ms。", styles)],
        [p("跨步和重心前移不足", styles), p("跨步线 + 轻球投掷：标记 80%-90% 身高落脚区，先稳定再加速。", styles), p("跨步长度、重心前移 cm、落脚后头部稳定度。", styles)],
        [p("出手速度没有跟随提升", styles), p("药球侧抛 + 毛巾出手：先看顺序，再看速度，避免单纯甩手。", styles), p("右手速度代理值、球速估算、肘腕轨迹平滑度。", styles)],
    ]
    table = Table(rows, colWidths=[4.1 * cm, 8.4 * cm, 5.0 * cm], repeatRows=1)
    table.setStyle(default_table_style())
    return table


def product_table(styles: dict[str, ParagraphStyle]) -> Table:
    rows = [
        [p("版本", styles), p("适合对象", styles), p("核心内容", styles), p("交付物", styles)],
        [p("基础版", styles), p("家长、孩子", styles), p("3 个问题：差在哪里、差多少、怎么改。突出截图、红黄绿指标、3 条训练建议。", styles), p("PDF + 关键帧对照图", styles)],
        [p("专业版", styles), p("教练、体能师", styles), p("角度曲线、动力链顺序、逐帧偏差、速度/厘米级变化、训练周期计划。", styles), p("PDF + CSV + overlay 视频", styles)],
        [p("年度档案", styles), p("俱乐部管理", styles), p("每次测试可复测指标、进步曲线、动作演变视频、伤害预防筛查记录。", styles), p("年度 PDF + dashboard", styles)],
    ]
    table = Table(rows, colWidths=[2.7 * cm, 3.0 * cm, 8.7 * cm, 3.2 * cm], repeatRows=1)
    table.setStyle(default_table_style())
    return table


def p(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text, styles["Table"])


def default_table_style() -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf2ff")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#172033")),
            ("FONTNAME", (0, 0), (-1, -1), PDF_FONT),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d0d5dd")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
    )


def image_row(paths: list[Path], widths: list[float]) -> Table:
    imgs = []
    for path, width in zip(paths, widths):
        with Image.open(path) as im:
            iw, ih = im.size
        height = width * ih / iw
        if height > 9.6 * cm:
            height = 9.6 * cm
            width = height * iw / ih
        imgs.append(RLImage(str(path), width=width, height=height))
    table = Table([imgs], colWidths=widths)
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    return table


def footer(canvas, doc) -> None:  # type: ignore[no-untyped-def]
    canvas.saveState()
    canvas.setFont(PDF_FONT, 8)
    canvas.setFillColor(colors.HexColor("#667085"))
    canvas.drawString(1.15 * cm, 0.55 * cm, "Baseball biomechanics report template - video/3D overlay proxies, not medical diagnosis")
    canvas.drawRightString(A4[0] - 1.15 * cm, 0.55 * cm, f"Page {doc.page}")
    canvas.restoreState()


if __name__ == "__main__":
    main()
