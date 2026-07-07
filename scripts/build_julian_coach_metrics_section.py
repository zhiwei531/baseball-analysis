from __future__ import annotations

import argparse
import csv
import html
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = ROOT / "reports" / "vicon_2026_julian_coach" / "batting_dashboard_metrics.csv"
DEFAULT_OUT = ROOT / "reports" / "vicon_2026_julian_coach" / "julian_coach_metrics_section.html"
DEFAULT_PEERS = ROOT / "outputs" / "batting_metrics_excel" / "all_players"
DEFAULT_PEER_METRICS_GLOB = "vicon_2026_*/batting_dashboard_metrics.csv"


UNIT_CN = {
    "deg": "°",
    "deg/s": "°/s",
    "km/h": "km/h",
    "mm": "mm",
    "height_ratio": "身高比",
    "0-100 risk": "风险分",
    "0-100 score": "分",
}


BACKEND_ORDER = [
    "ready_com_height_ratio",
    "ready_rear_hip_flexion_deg",
    "ready_rear_knee_flexion_deg",
    "ready_hip_shoulder_separation_deg",
    "ready_bat_tilt_deg",
    "ready_hand_height_ratio",
    "contact_bat_speed_kmh",
    "contact_attack_angle_deg",
    "contact_pelvis_rotation_open_deg",
    "contact_torso_rotation_open_deg",
    "contact_front_knee_flexion_deg",
    "ready_to_contact_head_displacement_mm",
    "coach_high_com_risk_index",
    "coach_rear_elbow_height_diff_mm",
    "coach_bat_loading_angle_to_catcher_deg",
    "coach_rollover_forearm_roll_velocity_deg_s",
    "coach_hitting_zone_stability_score",
]


FRONT_METRICS = [
    ("Ready Position", "平衡", [("ready_com_height_ratio", 0.6), ("ready_to_contact_head_displacement_mm", 0.4)], "ready_com_height_ratio"),
    ("Ready Position", "下肢加载", [("ready_rear_hip_flexion_deg", 0.5), ("ready_rear_knee_flexion_deg", 0.5)], "ready_rear_hip_flexion_deg"),
    ("Ready Position", "躯干蓄力", [("ready_hip_shoulder_separation_deg", 1.0)], "ready_hip_shoulder_separation_deg"),
    ("Ready Position", "球棒准备", [("ready_bat_tilt_deg", 0.55), ("ready_hand_height_ratio", 0.45)], "ready_bat_tilt_deg"),
    ("Contact Position", "球棒效率", [("contact_bat_speed_kmh", 1.0)], "contact_bat_speed_kmh"),
    ("Contact Position", "挥棒轨迹", [("contact_attack_angle_deg", 1.0)], "contact_attack_angle_deg"),
    ("Contact Position", "下半身姿态", [("contact_pelvis_rotation_open_deg", 1.0)], "contact_pelvis_rotation_open_deg"),
    ("Contact Position", "上半身姿态", [("contact_torso_rotation_open_deg", 1.0)], "contact_torso_rotation_open_deg"),
    ("Contact Position", "支撑能力", [("contact_front_knee_flexion_deg", 1.0)], "contact_front_knee_flexion_deg"),
    ("Contact Position", "稳定性", [("ready_to_contact_head_displacement_mm", 1.0)], "ready_to_contact_head_displacement_mm"),
    ("专项问题", "重心偏高", [("coach_high_com_risk_index", 1.0)], "coach_high_com_risk_index"),
    ("专项问题", "掉肘", [("coach_rear_elbow_height_diff_mm", 1.0)], "coach_rear_elbow_height_diff_mm"),
    ("专项问题", "引棒不足", [("coach_bat_loading_angle_to_catcher_deg", 1.0)], "coach_bat_loading_angle_to_catcher_deg"),
    ("专项问题", "翻腕", [("coach_rollover_forearm_roll_velocity_deg_s", 1.0)], "coach_rollover_forearm_roll_velocity_deg_s"),
]


FRONT_EXPLANATIONS = {
    "平衡": "准备姿态和启动过程中的身体控制会直接影响看球稳定性。",
    "下肢加载": "后腿和髋部是否提前进入可发力姿态，会影响后续挥棒力量。",
    "躯干蓄力": "上、下半身能否形成适当的扭转，是身体储存力量的重要环节。",
    "球棒准备": "球棒和双手是否处在合适位置，会影响启动空间和加速距离。",
    "球棒效率": "击球瞬间球棒速度越稳定，越说明身体力量传递得顺畅。",
    "挥棒轨迹": "球棒进入击球区的方式会影响击球容错和击球质量。",
    "下半身姿态": "下半身打开和支撑顺序决定力量能否顺利传到上半身。",
    "上半身姿态": "上半身稳定打开有助于维持击球点、视线和旋转传递。",
    "支撑能力": "前腿落地后的支撑质量，会影响身体制动和力量释放。",
    "稳定性": "从准备到击球的身体稳定性，会直接影响看球和击球准确性。",
    "重心偏高": "准备姿态如果站得过高，通常会影响下肢蓄力和启动速度。",
    "掉肘": "后肘位置会影响引棒空间和挥棒平面是否稳定。",
    "引棒不足": "引棒空间不足时，后续加速距离会变短，挥棒力量不容易完全释放。",
    "翻腕": "手腕过早翻转会影响击球面稳定，容易降低击球质量。",
}


BACKEND_EN = {
    "ready_com_height_ratio": "Ready Body Height",
    "ready_rear_hip_flexion_deg": "Rear Hip Load",
    "ready_rear_knee_flexion_deg": "Rear Knee Load",
    "ready_hip_shoulder_separation_deg": "Hip-shoulder Separation Angle",
    "ready_bat_tilt_deg": "Bat Angle at Ready",
    "ready_hand_height_ratio": "Hand Height",
    "contact_bat_speed_kmh": "Bat Speed",
    "contact_attack_angle_deg": "Attack Angle",
    "contact_pelvis_rotation_open_deg": "Pelvis Rotation",
    "contact_torso_rotation_open_deg": "Torso Rotation",
    "contact_front_knee_flexion_deg": "Front Knee Support",
    "ready_to_contact_head_displacement_mm": "Head Stability",
    "coach_high_com_risk_index": "High Center of Mass",
    "coach_rear_elbow_height_diff_mm": "Dropped Rear Elbow",
    "coach_bat_loading_angle_to_catcher_deg": "Bat Load",
    "coach_rollover_forearm_roll_velocity_deg_s": "Early Wrist Roll",
    "coach_hitting_zone_stability_score": "Hitting Zone Stability",
}


EXPLANATIONS_EN = {
    "ready_com_height_ratio": "The ready stance is close to the coach example, and the next priority is keeping that athletic height without standing up during the move to contact.",
    "ready_rear_hip_flexion_deg": "The rear hip is already loading well. Training should keep the player feeling pressure in the back leg before the swing starts.",
    "ready_rear_knee_flexion_deg": "The rear knee is in a useful athletic position. Keeping this flexion stable will help the player start with better rhythm and balance.",
    "ready_hip_shoulder_separation_deg": "The upper body is following the lower body too quickly. Holding the shoulders back a little longer will help store more rotational power.",
    "ready_bat_tilt_deg": "The bat is prepared in a position that still limits loading space. A cleaner bat angle will give the player a smoother acceleration path.",
    "ready_hand_height_ratio": "The hands can be held a little more usefully before launch. Better hand height helps protect rear-elbow space and keeps the swing path cleaner.",
    "contact_bat_speed_kmh": "Bat speed can improve as the lower body, trunk, and arms connect more smoothly, with the body leading before the hands accelerate.",
    "contact_attack_angle_deg": "The bat path through the hitting zone can become more repeatable. A steadier entry gives the player more room for timing differences.",
    "contact_pelvis_rotation_open_deg": "The hips are opening actively, but the timing still needs to match the front-side support so lower-body force can move upward.",
    "contact_torso_rotation_open_deg": "The trunk is active at contact. The next step is keeping chest direction and vision stable while rotating.",
    "contact_front_knee_flexion_deg": "Front-side support needs to become stronger. A firmer front leg helps the body brake and release rotational force into the bat.",
    "ready_to_contact_head_displacement_mm": "The head and upper body can stay quieter during the swing. Better stability helps the eyes track the ball and repeat the contact point.",
    "coach_high_com_risk_index": "Body height is controlled well in the ready stance. The player should keep the knees and hips flexed under game pressure.",
    "coach_rear_elbow_height_diff_mm": "The rear elbow is acceptable, but the habit needs to become more consistent so the player keeps loading room before launch.",
    "coach_bat_loading_angle_to_catcher_deg": "The player has basic loading space. A more complete load will give the bat a longer and smoother acceleration path.",
    "coach_rollover_forearm_roll_velocity_deg_s": "The wrist tends to roll a little early near contact. Keeping the barrel face through the ball will improve contact quality.",
    "coach_hitting_zone_stability_score": "The bat can stay available through the hitting zone a little longer. This will help the player make more solid contact on timing variations.",
}


FRONT_FEEDBACK = {
    "平衡": "该球员准备时身体控制总体不错，但启动到击球过程中还要继续减少头部和身体的多余晃动。这样有助于稳定看球视线，让挥棒更容易找到准确的击球点。",
    "下肢加载": "该球员后侧腿已经能较好地进入蓄力姿态，这是一个很好的基础。后续训练可以继续保持后腿承重和髋部坐入的感觉，把下半身力量更稳定地送到挥棒里。",
    "躯干蓄力": "目前下半身启动后，上半身跟进得较快，身体储存力量的时间略短。建议训练中多做髋部先启动、肩膀稍微保留的练习，让身体旋转力量有更多时间传到球棒。",
    "球棒准备": "球棒和双手已经有基本准备位置，但引棒空间还可以更充分。训练时要注意双手高度、后肘空间和球棒角度，让启动时球棒有更顺畅的加速距离。",
    "球棒效率": "击球瞬间的球棒速度还有提升空间，主要需要让下肢、髋部、躯干和手臂衔接得更顺。训练重点不是单纯加快手，而是让身体先带动，再把力量传到球棒。",
    "挥棒轨迹": "该球员的挥棒进入击球区时，球棒路径还可以更稳定。建议通过固定击球区挥棒、T 座不同高度击球等练习，减少过度下砍或过度上挑，让球棒在好球带里停留更久。",
    "下半身姿态": "击球时下半身打开和支撑顺序还不够理想，部分力量没有充分向上半身传递。训练中可以重点练习前脚落地后髋部继续旋转，避免身体过早被手臂带走。",
    "上半身姿态": "上半身在击球阶段表现比较积极，能够参与旋转和完成挥棒。接下来要继续保持胸口方向和视线稳定，避免为了发力而出现上身后仰或提前拉开。",
    "支撑能力": "前腿落地后的支撑还需要加强。前腿如果不能稳定制动，身体旋转力量就容易散掉，建议增加前脚落地定住、前膝稳定和髋部继续转动的分解练习。",
    "稳定性": "从准备到击球过程中，该球员的身体移动幅度还可以再控制。头部和上身越稳定，眼睛越容易跟住球，击球点也会更稳定。",
    "重心偏高": "该球员准备姿态的重心控制较好，没有明显站得过高的问题。后续要继续保持膝髋微屈、后腿有承重的准备状态，避免比赛中因为紧张而站直。",
    "掉肘": "该球员后肘位置整体可以接受，但还需要稳定成习惯。训练中保持后肘有空间、双手不过早掉低，可以让引棒更顺，也能减少挥棒平面被压低的情况。",
    "引棒不足": "该球员的引棒已经具备基本空间，但仍可以做得更完整。建议让球棒在启动前有更充分的向后加载，给后续加速留出更长、更顺的发力距离。",
    "翻腕": "该球员击球附近手腕有偏早翻转的倾向，这会影响击球面稳定。训练时应强调手掌控制球棒面、延长穿过球的感觉，先把球打扎实，再追求更快的挥棒。",
}


FRONT_FEEDBACK_EN = {
    "平衡": "The player shows generally solid body control in the ready position, but should continue reducing extra head and body movement during the move into contact. A steadier body helps keep the eyes on the ball and makes it easier to find a consistent contact point.",
    "下肢加载": "The player's back side already loads well, which gives a strong foundation. Future training should keep reinforcing back-leg pressure and hip loading so lower-body power can transfer more consistently into the swing.",
    "躯干蓄力": "At the moment, the player's upper body follows the lower-body start a little quickly, so the body has less time to store rotational power. Training should emphasize the hips starting first while the shoulders stay back slightly, giving the rotation more time to move into the bat.",
    "球棒准备": "The bat and hands are in a basic ready position, but the player can still create more loading space. In training, attention should stay on hand height, rear-elbow room, and bat angle so the bat has a smoother path to accelerate.",
    "球棒效率": "Bat speed at contact still has room to improve, mainly by making the lower body, hips, trunk, and arms connect more smoothly. The priority is not simply swinging the hands faster, but letting the body lead and then transferring that force into the bat.",
    "挥棒轨迹": "The player's bat path through the hitting zone can become more stable. Tee work at different heights and controlled zone-swing drills can help reduce excessive chopping or lifting, keeping the barrel in the strike zone longer.",
    "下半身姿态": "At contact, the lower-body opening and support sequence are not yet ideal, so some force is not fully transferred upward. Training should focus on letting the hips continue rotating after the front foot lands, instead of allowing the arms to take over too early.",
    "上半身姿态": "The player's upper body is active through contact and contributes to the swing. The next step is to keep the chest direction and vision stable, avoiding leaning back or opening too early when trying to generate power.",
    "支撑能力": "Front-leg support after landing needs to become stronger. If the front leg cannot brace well, rotational force can leak away, so drills should include front-foot landing control, front-knee stability, and continued hip rotation.",
    "稳定性": "The player can still control body movement better from the ready position to contact. The steadier the head and upper body are, the easier it is to track the ball and repeat the contact point.",
    "重心偏高": "The player controls body height well in the ready position and does not show a clear issue of standing too tall. The player should continue keeping the knees and hips slightly flexed with weight loaded into the back leg, especially when game pressure rises.",
    "掉肘": "The player's rear-elbow position is generally acceptable, but it still needs to become a stable habit. Keeping space around the rear elbow and preventing the hands from dropping too early will support a smoother load and a cleaner swing plane.",
    "引棒不足": "The player already has basic loading space, but the load can still become more complete. The bat should load farther back before launch, giving the player a longer and smoother acceleration path.",
    "翻腕": "The player shows a tendency for the wrist to roll a little early around contact, which can affect barrel stability. Training should emphasize controlling the barrel face with the hands and extending through the ball, prioritizing solid contact before chasing more bat speed.",
}


PEER_COLORS = ["#2563eb", "#16a34a", "#f97316", "#a855f7", "#dc2626", "#0891b2", "#ca8a04", "#db2777", "#475569"]


def anonymous_peer_label(index: int) -> str:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < len(alphabet):
        return f"球员{alphabet[index]}"
    return f"球员{index + 1}"


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def linear_score(value: float, low: float, high: float, higher_is_better: bool = True) -> float:
    if high == low:
        return 50.0
    ratio = (value - low) / (high - low)
    if not higher_is_better:
        ratio = 1.0 - ratio
    return clamp(ratio * 100.0)


def target_score(value: float, target: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 50.0
    return clamp(100.0 - abs(value - target) / tolerance * 100.0)


def safe_float(value: str | None) -> float | None:
    x = num(value)
    if x is None or not math.isfinite(x):
        return None
    return x


PEER_SCORE_SOURCES = {
    "平衡": "其他球员使用 ready_com_height_ratio + ready_to_contact_head_displacement_mm",
    "下肢加载": "其他球员使用 ready_rear_hip_flexion_deg + ready_rear_knee_flexion_deg",
    "躯干蓄力": "其他球员使用 ready_hip_shoulder_separation_deg",
    "球棒准备": "其他球员使用 ready_bat_tilt_deg + ready_hand_height_ratio",
    "球棒效率": "其他球员使用 contact_bat_speed_kmh",
    "挥棒轨迹": "其他球员使用 contact_attack_angle_deg",
    "下半身姿态": "其他球员使用 contact_pelvis_rotation_open_deg",
    "上半身姿态": "其他球员使用 contact_torso_rotation_open_deg",
    "支撑能力": "其他球员使用 contact_front_knee_flexion_deg",
    "稳定性": "其他球员使用 ready_to_contact_head_displacement_mm",
    "重心偏高": "其他球员使用 coach_high_com_risk_index",
    "掉肘": "其他球员使用 coach_rear_elbow_height_diff_mm",
    "引棒不足": "其他球员使用 coach_bat_loading_angle_to_catcher_deg",
    "翻腕": "其他球员使用 coach_rollover_forearm_roll_velocity_deg_s",
}


PEER_AXIS_KEYS = {
    "平衡": "ready_com_height_ratio",
    "下肢加载": "ready_rear_hip_flexion_deg",
    "躯干蓄力": "ready_hip_shoulder_separation_deg",
    "球棒准备": "ready_bat_tilt_deg",
    "球棒效率": "contact_bat_speed_kmh",
    "挥棒轨迹": "contact_attack_angle_deg",
    "下半身姿态": "contact_pelvis_rotation_open_deg",
    "上半身姿态": "contact_torso_rotation_open_deg",
    "支撑能力": "contact_front_knee_flexion_deg",
    "稳定性": "ready_to_contact_head_displacement_mm",
    "重心偏高": "coach_high_com_risk_index",
    "掉肘": "coach_rear_elbow_height_diff_mm",
    "引棒不足": "coach_bat_loading_angle_to_catcher_deg",
    "翻腕": "coach_rollover_forearm_roll_velocity_deg_s",
}


BACKEND_FIELD_KEYS = {
    "重心高度": "ready_com_height_ratio",
    "后髋屈曲角": "ready_rear_hip_flexion_deg",
    "后膝屈曲角": "ready_rear_knee_flexion_deg",
    "髋肩分离角": "ready_hip_shoulder_separation_deg",
    "球棒倾角": "ready_bat_tilt_deg",
    "握棒手高度": "ready_hand_height_ratio",
    "球棒速度": "contact_bat_speed_kmh",
    "挥棒路径角": "contact_attack_angle_deg",
    "骨盆旋转角": "contact_pelvis_rotation_open_deg",
    "躯干旋转角": "contact_torso_rotation_open_deg",
    "前膝屈曲角": "contact_front_knee_flexion_deg",
    "头部位移": "ready_to_contact_head_displacement_mm",
    "重心偏高指数": "coach_high_com_risk_index",
    "后肘高度差（掉肘）": "coach_rear_elbow_height_diff_mm",
    "球棒加载角（引棒不足）": "coach_bat_loading_angle_to_catcher_deg",
    "手腕翻转角速度（翻腕）": "coach_rollover_forearm_roll_velocity_deg_s",
    "击球区稳定性": "coach_hitting_zone_stability_score",
}


XLSX_UNIT_KEYS = {
    "%身高": "height_ratio",
    "风险分": "0-100 risk",
    "分": "0-100 score",
}


METRIC_ILLUSTRATIONS = {
    "平衡": "ready_balance_annotated.png",
    "下肢加载": "ready_lower_body_load_annotated.png",
    "躯干蓄力": "ready_torso_coil_annotated.png",
    "球棒准备": "ready_bat_readiness_annotated.png",
    "球棒效率": "contact_bat_efficiency_annotated.png",
    "挥棒轨迹": "contact_swing_path_annotated.png",
    "下半身姿态": "contact_lower_body_posture_annotated.png",
    "上半身姿态": "contact_upper_body_posture_annotated.png",
    "支撑能力": "contact_front_leg_support_annotated.png",
    "稳定性": "contact_stability_annotated.png",
    "重心偏高": "issue_high_center_of_mass_annotated.png",
    "掉肘": "issue_dropped_rear_elbow_annotated.png",
    "引棒不足": "issue_insufficient_bat_load_annotated.png",
    "翻腕": "issue_early_wrist_roll_annotated.png",
}


BACKEND_ILLUSTRATION_NAMES = {
    "ready_com_height_ratio": "平衡",
    "ready_rear_hip_flexion_deg": "下肢加载",
    "ready_rear_knee_flexion_deg": "下肢加载",
    "ready_hip_shoulder_separation_deg": "躯干蓄力",
    "ready_bat_tilt_deg": "球棒准备",
    "ready_hand_height_ratio": "球棒准备",
    "contact_bat_speed_kmh": "球棒效率",
    "contact_attack_angle_deg": "挥棒轨迹",
    "contact_pelvis_rotation_open_deg": "下半身姿态",
    "contact_torso_rotation_open_deg": "上半身姿态",
    "contact_front_knee_flexion_deg": "支撑能力",
    "ready_to_contact_head_displacement_mm": "稳定性",
    "coach_high_com_risk_index": "重心偏高",
    "coach_rear_elbow_height_diff_mm": "掉肘",
    "coach_bat_loading_angle_to_catcher_deg": "引棒不足",
    "coach_rollover_forearm_roll_velocity_deg_s": "翻腕",
    "coach_hitting_zone_stability_score": "挥棒轨迹",
}


ISSUE_BACKEND_KEYS = {
    "coach_high_com_risk_index",
    "coach_rear_elbow_height_diff_mm",
    "coach_bat_loading_angle_to_catcher_deg",
    "coach_rollover_forearm_roll_velocity_deg_s",
}


EXPLANATIONS = {
    "ready_com_height_ratio": "准备姿态整体接近教练示范，说明孩子已经有较好的站姿基础。后续要注意启动时不要突然站高，继续保持膝髋微屈，让下半身随时能发力。",
    "ready_rear_hip_flexion_deg": "后侧髋部已经能够较好地坐入，这是形成下半身力量的好基础。训练中要继续保持后腿承重感，让启动更稳、更有弹性。",
    "ready_rear_knee_flexion_deg": "后膝保持在比较有运动感的位置，有利于启动时快速发力。接下来重点是把这个姿态稳定成习惯，避免比赛中因为紧张而站直。",
    "ready_hip_shoulder_separation_deg": "下半身启动后，上半身跟进得较快，身体储存力量的时间略短。建议多练髋部先走、肩膀稍微保留，让旋转力量更充分地传到球棒。",
    "ready_bat_tilt_deg": "球棒准备角度还有调整空间，目前容易让引棒和加速路线变短。训练时要让双手、后肘和球棒形成更舒服的启动空间。",
    "ready_hand_height_ratio": "握棒手位置还可以更稳定一些。双手保持在合适高度，能给后肘留出空间，也能让球棒启动时更顺。",
    "contact_bat_speed_kmh": "击球附近的球棒速度还有提升空间，重点不是单纯加快手，而是让下肢、躯干和手臂的发力衔接更顺。",
    "contact_attack_angle_deg": "球棒进入击球区的路线还可以更稳定。路径稳定后，孩子面对不同高低和时机的球，会更容易把球打扎实。",
    "contact_pelvis_rotation_open_deg": "髋部打开比较积极，但还需要和前侧支撑配合得更好。前脚落地后髋部继续带动身体，力量才不容易提前散掉。",
    "contact_torso_rotation_open_deg": "上半身在击球阶段参与得比较主动，这是积极的一面。后续要继续保持胸口方向和视线稳定，避免为了发力而过早拉开。",
    "contact_front_knee_flexion_deg": "前腿支撑还有提升空间。前侧如果不能稳定顶住，身体旋转力量就不容易完整传到球棒，建议加强前脚落地定住和前膝稳定练习。",
    "ready_to_contact_head_displacement_mm": "从准备到击球，头部和上身移动还可以再安静一些。身体越稳，眼睛越容易跟住球，击球点也越容易重复。",
    "coach_high_com_risk_index": "准备姿态的重心控制较好，没有明显站得过高的问题。继续保持膝髋微屈、后腿有承重的准备状态即可。",
    "coach_rear_elbow_height_diff_mm": "后肘位置整体可以接受，但需要更稳定地形成习惯。保持后肘有空间，可以让引棒更顺，也能减少挥棒平面被压低。",
    "coach_bat_loading_angle_to_catcher_deg": "孩子已经具备基本引棒空间，后续可以做得更完整。启动前让球棒有更充分的向后加载，会给加速留出更长、更顺的距离。",
    "coach_rollover_forearm_roll_velocity_deg_s": "击球附近有偏早翻腕的倾向，容易让球棒面不够稳定。训练时要强调手掌控制球棒面，延长穿过球的感觉。",
    "coach_hitting_zone_stability_score": "球棒在好球带里的停留还可以更稳定。让球棒更久地穿过击球区，有助于提高扎实击球的机会。",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def col_to_index(cell_ref: str) -> int:
    letters = re.sub(r"\d", "", cell_ref)
    value = 0
    for char in letters:
        value = value * 26 + (ord(char.upper()) - 64)
    return value - 1


def read_xlsx_rows(path: Path, sheet_name: str) -> list[list[object]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rels
        }
        sheet_path = None
        rel_key = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            if sheet.attrib.get("name") == sheet_name:
                target = rel_targets[sheet.attrib[rel_key]]
                target = target.lstrip("/")
                sheet_path = target if target.startswith("xl/") else "xl/" + target
                break
        if not sheet_path:
            return []

        sheet_root = ET.fromstring(zf.read(sheet_path))
        rows: list[list[object]] = []
        for row in sheet_root.findall(".//m:sheetData/m:row", ns):
            cells: list[object] = []
            for cell in row.findall("m:c", ns):
                idx = col_to_index(cell.attrib.get("r", "A1"))
                while len(cells) <= idx:
                    cells.append(None)
                cell_type = cell.attrib.get("t")
                value_node = cell.find("m:v", ns)
                inline_node = cell.find("m:is/m:t", ns)
                value: object = ""
                if cell_type == "s" and value_node is not None:
                    value = shared[int(value_node.text or 0)]
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = inline_node.text or ""
                elif value_node is not None:
                    raw = value_node.text or ""
                    try:
                        value = float(raw)
                        if value.is_integer():
                            value = int(value)
                    except ValueError:
                        value = raw
                cells[idx] = value
            rows.append(cells)
        return rows


def cell_at(row: list[object], idx: int) -> object:
    return row[idx] if idx < len(row) else None


def athlete_from_xlsx(path: Path) -> str:
    for row in read_xlsx_rows(path, "说明"):
        if cell_at(row, 0) == "数据来源":
            source = str(cell_at(row, 1) or "")
            parts = source.split("/")
            if len(parts) >= 2:
                return parts[1]
    return path.name.replace("_batting_report_metrics.xlsx", "")


def xlsx_metric_record(path: Path) -> dict[str, object]:
    athlete = athlete_from_xlsx(path)
    rows_by_key: dict[str, dict[str, str]] = {}
    for row in read_xlsx_rows(path, "报告指标"):
        backend = cell_at(row, 2)
        backend_name = normalize_metric_name(backend)
        key = BACKEND_FIELD_KEYS.get(backend_name)
        value = safe_float(str(cell_at(row, 3))) if cell_at(row, 3) not in (None, "") else None
        if not key or value is None:
            continue
        unit = str(cell_at(row, 4) or "")
        if unit == "%身高":
            value = value / 100.0
        rows_by_key[key] = {
            "metric_key": key,
            "metric_name_zh": backend_name,
            "value": str(value),
            "unit": XLSX_UNIT_KEYS.get(unit, unit),
        }
    return {"name": athlete, "rows": rows_by_key}


def csv_peer_metric_records() -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for csv_path in sorted((ROOT / "reports").glob(DEFAULT_PEER_METRICS_GLOB)):
        for row in read_csv(csv_path):
            name = row.get("sample_name") or row.get("athlete") or ""
            key = row.get("metric_key") or ""
            value = row.get("value")
            if not name or name == "coach" or key not in BACKEND_ORDER or value in (None, ""):
                continue
            record = records.setdefault(name, {"name": name, "rows": {}})
            metric_rows = record["rows"]
            if not isinstance(metric_rows, dict):
                continue
            metric_rows[key] = {
                "metric_key": key,
                "metric_name_zh": row.get("metric_name_zh") or key,
                "value": str(value),
                "unit": row.get("unit") or "",
            }
    return records


def supplement_peer_records_from_csv(records: list[dict[str, object]]) -> list[dict[str, object]]:
    csv_records = csv_peer_metric_records()
    records_by_name = {str(record.get("name") or ""): record for record in records}
    for name, csv_record in csv_records.items():
        record = records_by_name.get(name)
        if record is None:
            record = {"name": name, "rows": {}}
            records.append(record)
            records_by_name[name] = record
        rows = record.get("rows")
        csv_rows = csv_record.get("rows")
        if not isinstance(rows, dict) or not isinstance(csv_rows, dict):
            continue
        for key, metric_row in csv_rows.items():
            rows.setdefault(key, metric_row)
    return records


def read_peer_metrics(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return list(csv_peer_metric_records().values())
    if path.is_dir():
        files = sorted(p for p in path.glob("*_batting_report_metrics.xlsx") if not p.name.startswith("._"))
    else:
        files = [path]
    records = []
    for file_path in files:
        record = xlsx_metric_record(file_path)
        if record["name"] == "coach":
            continue
        if record["rows"]:
            records.append(record)
    return supplement_peer_records_from_csv(records)


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def num(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def unit_cn(unit: str | None) -> str:
    return UNIT_CN.get(unit or "", unit or "")


def fmt(value: str | float | None, unit: str | None) -> str:
    x = num(str(value)) if value is not None else None
    if x is None:
        return "暂无"
    if unit == "height_ratio":
        text = f"{x:.3f}"
    elif unit in {"0-100 risk", "0-100 score"}:
        text = f"{x:.1f}"
    elif abs(x) >= 100:
        text = f"{x:.0f}"
    elif abs(x) >= 10:
        text = f"{x:.1f}"
    else:
        text = f"{x:.2f}"
    label = unit_cn(unit)
    return f"{text}{label}" if label in {"度", "毫米", "分"} else f"{text} {label}".strip()


def delta_text(julian: dict[str, str], coach: dict[str, str] | None) -> str:
    jv = num(julian.get("value"))
    cv = num(coach.get("value")) if coach else None
    unit = julian.get("unit", "")
    if jv is None or cv is None:
        return "暂无对照"
    diff = jv - cv
    sign = "+" if diff > 0 else ""
    return f"{sign}{fmt(diff, unit)}"


def status_for(metric_key: str, julian: dict[str, str], coach: dict[str, str] | None) -> tuple[str, str]:
    jv = num(julian.get("value"))
    cv = num(coach.get("value")) if coach else None
    if jv is None or cv is None:
        return "良好", "review"
    diff = abs(jv - cv)
    scale = max(abs(cv), 1.0)
    ratio = diff / scale
    if metric_key in {"coach_high_com_risk_index", "coach_rollover_forearm_roll_velocity_deg_s", "ready_to_contact_head_displacement_mm"}:
        return ("优秀", "good") if jv <= cv else ("待提高", "risk")
    if metric_key == "coach_hitting_zone_stability_score":
        return ("优秀", "good") if jv >= cv else ("待提高", "risk")
    if ratio <= 0.12:
        return "优秀", "good"
    if ratio <= 0.30:
        return "良好", "review"
    return "待提高", "risk"


def score_for_status(klass: str) -> float:
    return {"good": 100.0, "review": 70.0, "risk": 40.0}.get(klass, 60.0)


def front_metric_score(
    front_metric: tuple[str, str, list[tuple[str, float]], str],
    julian_rows: dict[str, dict[str, str]],
    coach_rows: dict[str, dict[str, str]],
) -> tuple[float | None, str, str]:
    _, _, components, _ = front_metric
    weighted_total = 0.0
    weight_total = 0.0
    for key, weight in components:
        row = julian_rows.get(key)
        coach_row = coach_rows.get(key)
        value = safe_float(row.get("value") if row else None)
        standard = safe_float(coach_row.get("value") if isinstance(coach_row, dict) else None)
        if value is None or standard is None:
            continue
        weighted_total += component_score_against_standard(key, value, standard) * weight
        weight_total += weight
    if weight_total <= 0:
        return None, "良好", "review"
    score = weighted_total / weight_total
    if score >= 85:
        return score, "优秀", "good"
    if score >= 65:
        return score, "良好", "review"
    return score, "待提高", "risk"


def score_text(score: float | None) -> str:
    return "暂无" if score is None else f"{score:.0f}分"


def score_number(score: float | None) -> str:
    return "暂无" if score is None else f"{score:.0f}"


def card_status_label(label: str, klass: str) -> str:
    return {"good": "优秀", "review": "良好", "risk": "待提高"}.get(klass, label)


def display_metric_name(metric_name: str) -> str:
    return normalize_metric_name(metric_name)


def normalize_metric_name(metric_name: object) -> str:
    return str(metric_name or "").replace("（Attack Angle）", "")


LOWER_IS_BETTER_KEYS = {
    "coach_high_com_risk_index",
    "coach_rollover_forearm_roll_velocity_deg_s",
    "ready_to_contact_head_displacement_mm",
}


def component_score_against_standard(metric_key: str, value: float, standard: float) -> float:
    scale = max(abs(standard), 1.0)
    if metric_key in LOWER_IS_BETTER_KEYS:
        diff_ratio = max(0.0, (value - standard) / scale)
    else:
        diff_ratio = abs(value - standard) / scale
    if diff_ratio <= 0.12:
        return 100.0 - diff_ratio / 0.12 * 8.0
    if diff_ratio <= 0.30:
        return 92.0 - (diff_ratio - 0.12) / 0.18 * 22.0
    if diff_ratio <= 0.60:
        return 70.0 - (diff_ratio - 0.30) / 0.30 * 30.0
    return max(20.0, 40.0 - (diff_ratio - 0.60) / 0.40 * 20.0)


def peer_scores_for(
    front_metric: tuple[str, str, list[tuple[str, float]], str],
    peer_rows: list[dict[str, object]],
    coach_rows: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    _, name, components, _ = front_metric
    scores = []
    for peer_idx, row in enumerate(peer_rows):
        metric_rows = row.get("rows")
        if not isinstance(metric_rows, dict):
            continue
        weighted_total = 0.0
        weight_total = 0.0
        component_parts = []
        for key, weight in components:
            metric_row = metric_rows.get(key)
            value = safe_float(metric_row.get("value") if isinstance(metric_row, dict) else None)
            standard = safe_float(coach_rows.get(key, {}).get("value") if isinstance(coach_rows.get(key), dict) else None)
            if value is None or standard is None:
                continue
            component_score = component_score_against_standard(key, value, standard)
            weighted_total += component_score * weight
            weight_total += weight
            component_parts.append(f"{key}: {component_score:.1f}分")
        if weight_total <= 0:
            continue
        score = weighted_total / weight_total
        scores.append(
            {
                "name": row.get("name") or "peer",
                "score": score,
                "color_index": peer_idx,
                "components": "; ".join(component_parts),
            }
        )
    return scores


def nice_step(span: float) -> float:
    if span <= 0 or not math.isfinite(span):
        return 1.0
    raw = span / 5.0
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return nice * magnitude


def peer_axis_text(value: float, unit: str | None) -> str:
    if unit == "score":
        return f"{value:.0f}"
    if unit == "height_ratio":
        return f"{value * 100:.0f}%"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def peer_range_bar(
    front_metric: tuple[str, str, list[tuple[str, float]], str],
    peer_rows: list[dict[str, object]],
    coach_rows: dict[str, dict[str, str]],
    show_markers: bool = True,
    anonymize_names: bool = True,
    current_score: float | None = None,
) -> str:
    _, name, _, _ = front_metric
    peer_scores = peer_scores_for(front_metric, peer_rows, coach_rows)
    if not peer_scores:
        return """
        <div class="peer-range empty">
          <div class="peer-label">其他球员<br>表现区间</div>
          <div class="peer-empty">暂无可用区间</div>
        </div>
        """
    values = [float(item["score"]) for item in peer_scores]
    unit = "score"
    low = min(values)
    high = max(values)
    step = nice_step(high - low)
    axis_low = low
    axis_high = high
    if axis_high <= axis_low:
        axis_low -= step
        axis_high += step
    axis_span = max(axis_high - axis_low, 1.0)
    span_left = 0.0
    span_width = 100.0
    dots = []
    lanes_by_bucket: dict[int, int] = {}
    scored_names = {str(item["name"]) for item in peer_scores}

    def dot_html(item: dict[str, object], pos: float, title: str, missing: bool = False) -> str:
        bucket = round(pos / 3.5)
        lane = lanes_by_bucket.get(bucket, 0)
        lanes_by_bucket[bucket] = lane + 1
        x_offsets = [0.0, -1.1, 1.1, -2.2, 2.2, -3.3, 3.3, 4.4]
        y_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        offset_idx = min(lane, len(x_offsets) - 1)
        pos = clamp(pos + x_offsets[offset_idx], 2.0, 98.0)
        top = 50 + y_offsets[offset_idx]
        color = PEER_COLORS[int(item.get("color_index", 0)) % len(PEER_COLORS)]
        klass = "peer-dot missing" if missing else "peer-dot"
        return (
            f'<span class="{klass}" style="left:{pos:.2f}%; top:{top:.1f}%; background:{esc(color)}" '
            f'title="{esc(title)}"></span>'
        )

    if show_markers:
        for item in peer_scores:
            score = float(item["score"])
            pos = 2.0 + clamp((score - axis_low) / axis_span * 100.0) * 0.96
            peer_label = anonymous_peer_label(int(item.get("color_index", 0))) if anonymize_names else str(item["name"])
            dots.append(
                dot_html(
                    item,
                    pos,
                    f"{peer_label}: {peer_axis_text(score, unit)}分",
                )
            )

        for peer_idx, row in enumerate(peer_rows):
            name = str(row.get("name") or "peer")
            if name in scored_names:
                continue
            peer_label = anonymous_peer_label(peer_idx) if anonymize_names else name
            dots.append(
                dot_html(
                    {"name": name, "score": axis_low, "color_index": peer_idx},
                    2.0,
                    f"{peer_label}: 暂无可对照表现",
                    missing=True,
                )
            )
    elif current_score is not None and math.isfinite(current_score):
        pos = 2.0 + clamp((current_score - axis_low) / axis_span * 100.0) * 0.96
        dots.append(
            f'<span class="peer-dot current-player" style="left:{pos:.2f}%; top:50.0%" '
            f'title="该球员: {peer_axis_text(current_score, unit)}分"></span>'
        )
    klass = "peer-range" if show_markers else "peer-range no-markers"
    return f"""
        <div class="{klass}">
        <div class="peer-label">其他球员<br>表现区间</div>
        <div class="peer-min">{peer_axis_text(low, unit)}</div>
        <div class="peer-track" title="其他球员在同一训练评估标准下的表现区间">
          <span class="peer-span" style="left:{span_left:.2f}%; width:{span_width:.2f}%"></span>
          {''.join(dots)}
        </div>
        <div class="peer-max">{peer_axis_text(high, unit)}</div>
      </div>
    """


def peer_metric_values_for(metric_key: str, peer_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values = []
    for peer_idx, row in enumerate(peer_rows):
        metric_rows = row.get("rows")
        if not isinstance(metric_rows, dict):
            continue
        metric_row = metric_rows.get(metric_key)
        value = safe_float(metric_row.get("value") if isinstance(metric_row, dict) else None)
        if value is None:
            continue
        values.append(
            {
                "name": row.get("name") or "peer",
                "value": value,
                "unit": metric_row.get("unit", "") if isinstance(metric_row, dict) else "",
                "color_index": peer_idx,
            }
        )
    return values


def peer_metric_range_bar(
    metric_key: str,
    unit: str | None,
    peer_rows: list[dict[str, object]],
    show_markers: bool = True,
    anonymize_names: bool = True,
    current_value: float | None = None,
) -> str:
    peer_values = peer_metric_values_for(metric_key, peer_rows)
    if not peer_values:
        return """
        <div class="peer-range empty">
          <div class="peer-label">其他球员<br>后端区间</div>
          <div class="peer-empty">暂无可用区间</div>
        </div>
        """
    values = [float(item["value"]) for item in peer_values]
    low = min(values)
    high = max(values)
    step = nice_step(high - low)
    axis_low = low
    axis_high = high
    if axis_high <= axis_low:
        axis_low -= step
        axis_high += step
    axis_span = max(axis_high - axis_low, 1.0)
    dots = []
    lanes_by_bucket: dict[int, int] = {}
    valued_names = {str(item["name"]) for item in peer_values}

    def dot_html(item: dict[str, object], pos: float, title: str, missing: bool = False) -> str:
        bucket = round(pos / 3.5)
        lane = lanes_by_bucket.get(bucket, 0)
        lanes_by_bucket[bucket] = lane + 1
        x_offsets = [0.0, -1.1, 1.1, -2.2, 2.2, -3.3, 3.3, 4.4]
        offset_idx = min(lane, len(x_offsets) - 1)
        pos = clamp(pos + x_offsets[offset_idx], 2.0, 98.0)
        color = PEER_COLORS[int(item.get("color_index", 0)) % len(PEER_COLORS)]
        klass = "peer-dot missing" if missing else "peer-dot"
        return (
            f'<span class="{klass}" style="left:{pos:.2f}%; top:50.0%; background:{esc(color)}" '
            f'title="{esc(title)}"></span>'
        )

    if show_markers:
        for item in peer_values:
            value = float(item["value"])
            pos = 2.0 + clamp((value - axis_low) / axis_span * 100.0) * 0.96
            peer_label = anonymous_peer_label(int(item.get("color_index", 0))) if anonymize_names else str(item["name"])
            dots.append(dot_html(item, pos, f"{peer_label}: {peer_axis_text(value, unit)}"))

        for peer_idx, row in enumerate(peer_rows):
            name = str(row.get("name") or "peer")
            if name in valued_names:
                continue
            peer_label = anonymous_peer_label(peer_idx) if anonymize_names else name
            dots.append(
                dot_html(
                    {"name": name, "value": axis_low, "color_index": peer_idx},
                    2.0,
                    f"{peer_label}: 暂无可对照表现",
                    missing=True,
                )
            )
    elif current_value is not None and math.isfinite(current_value):
        pos = 2.0 + clamp((current_value - axis_low) / axis_span * 100.0) * 0.96
        dots.append(
            f'<span class="peer-dot current-player" style="left:{pos:.2f}%; top:50.0%" '
            f'title="该球员: {peer_axis_text(current_value, unit)}"></span>'
        )

    klass = "peer-range" if show_markers else "peer-range no-markers"
    return f"""
        <div class="{klass}">
        <div class="peer-label">其他球员<br>表现区间</div>
        <div class="peer-min">{peer_axis_text(low, unit)}</div>
        <div class="peer-track" title="其他球员在同一训练评估标准下的表现区间">
          <span class="peer-span" style="left:0.00%; width:100.00%"></span>
          {''.join(dots)}
        </div>
        <div class="peer-max">{peer_axis_text(high, unit)}</div>
      </div>
    """


def front_metric_card(
    front_metric: tuple[str, str, list[tuple[str, float]], str],
    julian_rows: dict[str, dict[str, str]],
    coach_rows: dict[str, dict[str, str]],
    peer_rows: list[dict[str, object]],
    show_peer_markers: bool,
    anonymize_peer_names: bool = True,
) -> str:
    _, name, components, event_key = front_metric
    score, label, klass = front_metric_score(front_metric, julian_rows, coach_rows)
    display_label = card_status_label(label, klass)
    event_row = julian_rows[event_key]
    body = FRONT_FEEDBACK.get(name, FRONT_EXPLANATIONS.get(name, ""))
    body_en = FRONT_FEEDBACK_EN.get(name, "")
    return f"""
    <article class="metric-card {klass}">
      <div class="metric-summary">
          <span class="badge {klass}">{esc(display_label)}</span>
        <div>
          <h4>{esc(name)}</h4>
        </div>
        <div class="metric-value">{esc(score_number(score))}</div>
      </div>
      {metric_illustration(name)}
      <div class="metric-detail">
        <p class="metric-detail-cn">{esc(body)}</p>
        <p class="metric-detail-en">{esc(body_en)}</p>
        {peer_range_bar(front_metric, peer_rows, coach_rows, show_peer_markers, anonymize_peer_names, score if not show_peer_markers else None)}
      </div>
    </article>
    """


def metric_card(
    metric: dict[str, str],
    coach: dict[str, str] | None,
    peer_rows: list[dict[str, object]],
    illustration_name: str,
    show_peer_markers: bool,
    anonymize_peer_names: bool = True,
) -> str:
    key = metric["metric_key"]
    label, klass = status_for(key, metric, coach)
    coach_value = fmt(coach.get("value"), coach.get("unit")) if coach else "暂无"
    body = (
        f"{EXPLANATIONS.get(key, metric.get('formula', ''))}"
        f"本次记录为 {fmt(metric.get('value'), metric.get('unit'))}；"
        f"教练示范为 {coach_value}，相差 {delta_text(metric, coach)}。"
    )
    body_en = (
        f"Player: {fmt(metric.get('value'), metric.get('unit'))}. Coach reference: {coach_value}. Gap: {delta_text(metric, coach)}. "
        f"{EXPLANATIONS_EN.get(key, '')}"
    )
    current_value = safe_float(metric.get("value"))
    return f"""
    <article class="metric-card {klass}">
      <div class="metric-summary">
          <span class="badge {klass}">{esc(label)}</span>
        <div>
          <h4>{esc(display_metric_name(metric["metric_name_zh"]))}</h4>
          <div class="metric-en">{esc(BACKEND_EN.get(key, ""))}</div>
        </div>
        <div class="metric-value">{esc(fmt(metric.get("value"), metric.get("unit")))}</div>
      </div>
      {metric_illustration(illustration_name)}
      <div class="metric-detail">
        <p class="metric-detail-cn">{esc(body)}</p>
        <p class="metric-detail-en">{esc(body_en)}</p>
        {peer_metric_range_bar(key, metric.get("unit"), peer_rows, show_peer_markers, anonymize_peer_names, current_value if not show_peer_markers else None)}
      </div>
    </article>
    """


def reconstruction_media(src: str, alt: str) -> str:
    return f'<img src="{esc(src)}" alt="{esc(alt)}" loading="lazy">'


def versioned_asset(src: str) -> str:
    path = DEFAULT_OUT.parent / src
    if not path.exists():
        return src
    return f"{src}?v={int(path.stat().st_mtime)}"


def metric_illustration(name: str) -> str:
    file_name = METRIC_ILLUSTRATIONS.get(name)
    if not file_name:
        return ""
    src = versioned_asset(f"assets/frontend_metric_illustrations_annotated_standalone/{file_name}")
    return f"""
      <figure class="metric-illustration">
        <img src="{esc(src)}" alt="{esc(name)}动作示意图" loading="lazy">
      </figure>
    """


def speed_annotation_panel(rows: dict[str, dict[str, str]], sample: str) -> str:
    media_path = versioned_asset(f"assets/vicon_reconstruction_annotated/{sample}_speed_annotated.gif")
    display_name = "球员" if sample == "julian" else "教练示范"
    return f"""
    <figure class="reconstruction-annotated">
      {reconstruction_media(media_path, f"{display_name}打击动作观察")}
      <figcaption>
        <b>{esc(display_name)} 打击动作速度与挥棒方向</b>
        <span class="caption-cn">这段画面重点看速度释放、球棒进入击球区的方式，以及手腕是否能把球棒面稳定住。</span>
        <span class="caption-en">This clip focuses on speed release, how the bat enters the hitting zone, and whether the hands keep the barrel face stable.</span>
      </figcaption>
    </figure>
    """


def kinetic_chain_panel(src: str, title: str, note: str, note_en: str) -> str:
    media_path = versioned_asset(src)
    return f"""
    <article class="visual-card kinetic-chain-card">
      <h4>{esc(title)}</h4>
      <figure class="kinetic-chain-figure">
        <img src="{esc(media_path)}" alt="{esc(title)}" loading="lazy">
      </figure>
      <p class="copy-cn">{esc(note)}</p>
      <p class="copy-en">{esc(note_en)}</p>
    </article>
    """


def event_gif_panel(
    title: str,
    julian_rows: dict[str, dict[str, str]],
    metric_key: str,
    event_slug: str,
    peer_rows: list[dict[str, object]] | None = None,
) -> str:
    julian_metric = julian_rows[metric_key]
    gif_src = versioned_asset(f"assets/vicon_reconstruction_events/julian_{event_slug}.gif")
    legend = peer_legend(peer_rows or [], embedded=True)
    notes = {
        "ready": (
            "这个模型展示的是孩子准备开始挥棒前的代表画面，方便家长观察站姿是否稳定、身体是否已经准备好发力。",
            "This model shows a representative moment before the swing begins, helping families see whether the player is balanced and ready to move.",
        ),
        "contact": (
            "这个模型展示的是孩子接近击球时的代表画面，方便家长理解身体位置、球棒方向和击球稳定性之间的关系。",
            "This model shows a representative moment near contact, helping families understand how body position, bat direction, and contact stability connect.",
        ),
    }
    note_cn, note_en = notes.get(event_slug, ("", ""))
    note_html = ""
    if note_cn or note_en:
        note_html = f"""
      <p class="copy-cn">{esc(note_cn)}</p>
      <p class="copy-en">{esc(note_en)}</p>"""
    return f"""
    <article class="visual-card event-gifs">
      <h4>{esc(title)}</h4>
      <figure class="event-gif-figure">
        <img src="{esc(gif_src)}" alt="球员{esc(title)}" loading="lazy">
        <figcaption><b>球员</b><span>代表动作片段</span></figcaption>
      </figure>
      {legend}
      {note_html}
    </article>
    """


def peer_legend(peer_rows: list[dict[str, object]], embedded: bool = False, anonymize_names: bool = True) -> str:
    if not peer_rows:
        return ""
    items = []
    for idx, row in enumerate(peer_rows):
        color = PEER_COLORS[idx % len(PEER_COLORS)]
        label = anonymous_peer_label(idx) if anonymize_names else str(row.get("name", "peer"))
        items.append(
            f'<li><span class="legend-dot" style="background:{esc(color)}"></span>{esc(label)}</li>'
        )
    tag = "div" if embedded else "aside"
    klass = "peer-legend embedded" if embedded else "peer-legend"
    return f"""
    <{tag} class="{klass}" aria-label="其他球员颜色图例">
      <h4>颜色图例</h4>
      <ul>{''.join(items)}</ul>
    </{tag}>
    """


def render(rows: list[dict[str, str]], peer_rows: list[dict[str, object]]) -> str:
    by_sample: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_name"]][row["metric_key"]] = row
    julian = by_sample["julian"]
    coach = by_sample["coach"]

    grouped: dict[str, list[str]] = defaultdict(list)
    for metric_key in BACKEND_ORDER:
        metric = julian.get(metric_key)
        if not metric:
            continue
        if metric_key in ISSUE_BACKEND_KEYS:
            module = "专项问题"
        elif metric_key.startswith("ready_"):
            module = "Ready Position"
        else:
            module = "Contact Position"
        grouped[module].append(
            metric_card(
                metric,
                coach.get(metric_key),
                peer_rows,
                BACKEND_ILLUSTRATION_NAMES.get(metric_key, metric["metric_name_zh"]),
                module == "专项问题",
                module != "专项问题",
            )
        )
    ready_cards = "".join(grouped["Ready Position"])
    contact_cards = "".join(grouped["Contact Position"])
    issue_cards = "".join(grouped["专项问题"])

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>球员打击表现报告</title>
  <style>
    :root {{
      --primary:#2563eb; --ink:#101828; --body:#344054; --mid:#667085; --mute:#98a2b3;
      --line:#d0d5dd; --canvas:#f5f7fb; --soft:#eef6ff; --card:#fff; --dusk:#101828;
      --orange:#f97316; --success:#16a34a; --red:#ef4444; --review:#e89918;
    }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--canvas); color:var(--ink); font-family:STHeiti,"PingFang SC","Microsoft YaHei",system-ui,sans-serif; line-height:1.5; letter-spacing:0; }}
    main {{ max-width:1180px; margin:auto; padding:32px 24px 72px; }}
    h1 {{ font-size:42px; line-height:52px; font-weight:500; margin:0 0 12px; }}
    h2 {{ font-size:30px; line-height:40px; font-weight:500; margin:0; }}
    h4 {{ font-size:20px; line-height:30px; margin:0; }}
    p {{ margin:0; color:var(--body); font-size:18px; overflow-wrap:anywhere; }}
    .section {{ margin-top:34px; min-width:0; }}
    .section-title {{ display:flex; align-items:center; gap:14px; margin-bottom:18px; }}
    .mark {{ width:12px; height:40px; background:var(--primary); border-radius:999px; flex:0 0 auto; }}
    .module-note {{ background:var(--soft); border:1px solid #bfdbfe; border-radius:12px; padding:16px 18px; margin-bottom:18px; }}
    .grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; }}
    .grid-2 {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; }}
    .metrics-with-media {{ display:grid; grid-template-columns:minmax(0,2fr) minmax(300px,1fr); gap:18px; align-items:start; }}
    .metrics-with-media .grid {{ grid-template-columns:1fr; }}
    .card,.metric-card,.visual-card {{ background:#fffefa; border:2px solid #d2d2d2; border-radius:24px; padding:24px; min-width:0; }}
    .card.good,.metric-card.good,
    .card.review,.metric-card.review,
    .card.risk,.metric-card.risk {{ background:#fffefa; }}
    .metric-card {{ display:grid; grid-template-columns:minmax(110px,145px) minmax(130px,165px) minmax(0,1fr); gap:18px; align-items:center; min-height:236px; border-color:#d2d2d2; border-radius:26px; background:#fffefa; }}
    .metric-summary {{ min-width:0; display:grid; align-content:center; gap:14px; }}
    .metric-en {{ color:#667085; font-size:13px; line-height:17px; font-weight:700; margin-top:0; }}
    .metric-detail {{ min-width:0; display:grid; gap:12px; }}
    .metric-detail-cn,.copy-cn,.module-note-cn,.caption-cn {{ color:#344054; font-size:15px; line-height:22px; font-weight:700; }}
    .metric-detail-en,.copy-en,.module-note-en,.caption-en {{ color:#7a8494; font-size:12px; line-height:18px; font-weight:600; }}
    .card-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }}
    .badge {{ display:inline-flex; align-items:center; justify-content:center; width:max-content; min-width:70px; border-radius:999px; padding:4px 12px; font-size:14px; line-height:20px; font-weight:700; white-space:nowrap; }}
    .badge.good {{ background:#dcfce7; color:#166534; }}
    .badge.review {{ background:#fff7ed; color:#9a3412; }}
    .badge.risk {{ background:#fef2f2; color:#b91c1c; }}
    .metric-value {{ font-size:38px; line-height:1; font-weight:800; margin:0; color:#000; overflow-wrap:anywhere; }}
    .compact-metrics {{ grid-template-columns:1fr; gap:18px; }}
    .compact-metrics .metric-card {{ padding:22px 26px; }}
    .compact-metrics .metric-card h4 {{ font-size:18px; line-height:23px; font-weight:800; }}
    .issue-with-legend {{ display:grid; grid-template-columns:minmax(0,1fr) 150px; gap:14px; align-items:start; }}
    .issue-metrics {{ grid-template-columns:1fr; }}
    .issue-metrics .metric-card {{ max-width:100%; min-height:220px; padding:22px 26px; grid-template-columns:minmax(180px,220px) minmax(118px,150px) minmax(0,1fr); gap:18px; }}
    .issue-metrics .metric-card h4 {{ font-size:18px; line-height:23px; }}
    .issue-metrics .metric-value {{ font-size:32px; line-height:1.05; white-space:nowrap; overflow-wrap:normal; word-break:keep-all; }}
    .issue-metrics .metric-detail {{ gap:10px; }}
    .issue-metrics .metric-detail-cn {{ font-size:13px; line-height:20px; }}
    .issue-metrics .metric-detail-en {{ font-size:11px; line-height:17px; }}
    .metric-illustration {{ margin:0; width:100%; aspect-ratio:1; border-radius:16px; overflow:hidden; background:transparent; }}
    .metric-illustration img {{ width:100%; height:100%; object-fit:contain; display:block; }}
    .peer-range {{ display:grid; grid-template-columns:max-content 28px minmax(90px,1fr) 28px; gap:7px; align-items:center; margin-top:0; }}
    .peer-label {{ color:#000; font-size:12px; line-height:15px; font-weight:800; }}
    .peer-min,.peer-max {{ color:#344054; font-size:13px; line-height:16px; font-weight:800; text-align:center; }}
    .peer-track {{ position:relative; height:28px; border-radius:999px; background:linear-gradient(180deg,transparent 0 11px,#eef2f7 11px 17px,transparent 17px); }}
    .peer-span {{ position:absolute; top:11px; height:6px; border-radius:999px; background:linear-gradient(90deg,#dcfce7,#bae6fd); }}
    .peer-dot {{ position:absolute; top:50%; width:10px; height:10px; border:2px solid #fff; border-radius:999px; transform:translate(-50%,-50%); box-shadow:0 0 0 1px rgba(16,24,40,.12); }}
    .peer-dot.current-player {{ width:12px; height:12px; background:#101828; box-shadow:0 0 0 2px rgba(37,99,235,.28),0 0 0 1px rgba(16,24,40,.18); }}
    .peer-range.no-markers .peer-track {{ height:18px; background:linear-gradient(180deg,transparent 0 6px,#eef2f7 6px 12px,transparent 12px); }}
    .peer-range.no-markers .peer-span {{ top:6px; }}
    .peer-dot.missing {{ opacity:.45; box-shadow:0 0 0 1px rgba(16,24,40,.22),0 0 0 4px rgba(16,24,40,.04); }}
    .peer-empty {{ color:var(--mid); font-size:16px; font-weight:700; }}
    .peer-legend {{ margin-top:12px; padding:12px 10px; border:1px solid #d0d5dd; border-radius:14px; background:#fff; }}
    .peer-legend h4 {{ font-size:14px; line-height:18px; margin:0 0 8px; }}
    .peer-legend ul {{ list-style:none; margin:0; padding:0; display:grid; gap:7px; }}
    .issue-with-legend > .peer-legend {{ position:sticky; top:18px; margin-top:0; }}
    .issue-with-legend > .peer-legend ul {{ grid-template-columns:1fr; }}
    .peer-legend li {{ display:flex; align-items:center; gap:7px; color:#344054; font-size:12px; line-height:16px; font-weight:700; }}
    .legend-dot {{ width:10px; height:10px; border-radius:999px; box-shadow:0 0 0 1px rgba(16,24,40,.12); flex:0 0 auto; }}
    .visual-card p,.metric-card p,.card p {{ margin-top:8px; }}
    .reconstruction-annotated {{ position:relative; margin:0; background:#fff; border:1px solid var(--line); border-radius:18px; overflow:hidden; }}
    .reconstruction-annotated img {{ width:100%; aspect-ratio:16/10; object-fit:contain; display:block; background:#fff; }}
    .reconstruction-annotated figcaption {{ display:grid; gap:4px; padding:12px 14px; border-top:1px solid #e4e7ec; }}
    .reconstruction-annotated figcaption b {{ color:var(--ink); font-size:15px; line-height:20px; }}
    .reconstruction-annotated figcaption span {{ display:block; }}
    .event-gifs {{ position:sticky; top:18px; }}
    .event-gif-figure {{ margin:12px 0 0; border:1px solid var(--line); border-radius:14px; overflow:hidden; background:#fff; }}
    .event-gif-figure img {{ width:100%; aspect-ratio:16/10; object-fit:contain; display:block; background:#fff; }}
    .event-gif-figure figcaption {{ display:flex; justify-content:space-between; gap:8px; padding:8px 10px; border-top:1px solid #e4e7ec; }}
    .event-gif-figure b {{ color:var(--ink); font-size:13px; }}
    .event-gif-figure span {{ color:var(--mid); font-size:12px; text-align:right; }}
    .event-comparison {{ display:grid; grid-template-columns:minmax(260px,.9fr) minmax(0,1.8fr); gap:18px; align-items:stretch; margin:0 0 18px; }}
    .event-comparison .event-gifs {{ position:static; }}
    .event-comparison .visual-card {{ display:grid; align-content:start; }}
    .event-comparison .visual-card p {{ font-size:13px; line-height:18px; }}
    .event-comparison .section-annotation {{ width:100%; margin:0; }}
    .two-column-metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
    .two-column-metrics .metric-card {{ min-height:256px; padding:20px; grid-template-columns:minmax(88px,112px) minmax(104px,132px) minmax(0,1fr); gap:12px; }}
    .two-column-metrics .metric-card h4 {{ font-size:17px; line-height:22px; }}
    .two-column-metrics .metric-value {{ font-size:34px; }}
    .two-column-metrics .metric-detail {{ gap:8px; }}
    .two-column-metrics .metric-detail-cn {{ font-size:13px; line-height:19px; }}
    .two-column-metrics .metric-detail-en {{ font-size:11px; line-height:16px; }}
    .section-annotation {{ width:calc((100% - 18px) / 1.46); margin:0 0 18px; border:1px solid var(--line); border-radius:18px; overflow:hidden; background:#fff; }}
    .section-annotation img {{ width:100%; aspect-ratio:16/9; object-fit:contain; display:block; background:#fff; }}
    .section-annotation figcaption {{ display:flex; justify-content:space-between; gap:10px; padding:10px 12px; border-top:1px solid #e4e7ec; }}
    .section-annotation b {{ color:var(--ink); font-size:14px; }}
    .section-annotation span {{ color:var(--mid); font-size:13px; text-align:right; }}
    .kinetic-chain-card {{ padding:22px; }}
    .kinetic-chain-card p {{ max-width:920px; }}
    .kinetic-chain-figure {{ margin:14px 0 0; border:1px solid var(--line); border-radius:18px; overflow:hidden; background:#fff; }}
    .kinetic-chain-figure img {{ width:100%; aspect-ratio:1600/760; object-fit:contain; display:block; background:#fff; }}
    @media (max-width:1100px) {{ .metric-card {{ grid-template-columns:minmax(100px,130px) minmax(120px,150px) minmax(0,1fr); gap:14px; }} .metric-detail-cn,.copy-cn,.module-note-cn,.caption-cn {{ font-size:14px; line-height:21px; }} .metric-detail-en,.copy-en,.module-note-en,.caption-en {{ font-size:11px; line-height:17px; }} .compact-metrics .metric-card {{ padding:20px 22px; }} }}
    @media (max-width:960px) {{ .grid-2,.metrics-with-media,.issue-with-legend,.event-comparison {{ grid-template-columns:1fr; }} .grid,.compact-metrics,.metrics-with-media .grid,.issue-metrics,.two-column-metrics {{ grid-template-columns:1fr; }} .section-annotation {{ width:100%; }} .event-gifs {{ position:static; }} h1 {{ font-size:36px; line-height:44px; }} .metric-card,.issue-metrics .metric-card {{ grid-template-columns:170px minmax(160px,210px); }} .metric-detail {{ grid-column:1 / -1; }} }}
    @media (max-width:640px) {{ main {{ padding-left:16px; padding-right:16px; }} .metric-card {{ grid-template-columns:1fr; min-height:0; gap:18px; }} .compact-metrics .metric-card {{ padding:22px 18px; }} .metric-illustration {{ max-width:260px; justify-self:center; }} .peer-range {{ grid-template-columns:1fr 32px minmax(100px,1fr) 32px; gap:8px; }} }}
  </style>
</head>
<body>
  <main>
    <section class="section" id="player-coach-batting-report">
      <div class="section-title"><span class="mark"></span><h1>球员打击表现报告</h1></div>
    </section>

    <section class="section">
      <div class="section-title"><span class="mark"></span><h2>挥棒速度与动作对照</h2></div>
      <div class="grid-2">
        <article class="visual-card">
          <h4>球员速度与挥棒方向</h4>
          {speed_annotation_panel(julian, "julian")}
        </article>
        <article class="visual-card">
          <h4>教练示范动作对照</h4>
          {speed_annotation_panel(coach, "coach")}
        </article>
      </div>
    </section>

    <section class="section">
      <div class="section-title"><span class="mark"></span><h2>准备姿态</h2></div>
      <div class="event-comparison">
        {event_gif_panel("准备姿态动作片段", julian, "ready_com_height_ratio", "ready")}
        <figure class="section-annotation">
          <img src="{esc(versioned_asset("assets/vicon_2d_geometry_annotations/ready_position_vicon_geometry_on_2d.png"))}" alt="球员准备姿态动作观察" loading="lazy">
          <figcaption><b>准备姿态动作参考</b><span>启动前代表画面</span></figcaption>
        </figure>
      </div>
      <div class="grid compact-metrics two-column-metrics">
        {ready_cards}
      </div>
    </section>

    <section class="section">
      <div class="section-title"><span class="mark"></span><h2>击球瞬间</h2></div>
      <div class="event-comparison">
        {event_gif_panel("击球瞬间动作片段", julian, "contact_bat_speed_kmh", "contact")}
        <figure class="section-annotation">
          <img src="{esc(versioned_asset("assets/vicon_2d_geometry_annotations/contact_position_vicon_geometry_on_2d.png"))}" alt="球员击球瞬间动作观察" loading="lazy">
          <figcaption><b>击球瞬间动作参考</b><span>击球附近代表画面</span></figcaption>
        </figure>
      </div>
      <div class="grid compact-metrics two-column-metrics">
        {contact_cards}
      </div>
    </section>

    <section class="section">
      <div class="section-title"><span class="mark"></span><h2>专项问题</h2></div>
      <div class="issue-with-legend">
        <div class="grid compact-metrics issue-metrics">{issue_cards}</div>
        {peer_legend(peer_rows, anonymize_names=False)}
      </div>
    </section>

  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a standalone player-vs-coach batting report section.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--peers", type=Path, default=DEFAULT_PEERS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = read_csv(args.metrics)
    peer_rows = read_peer_metrics(args.peers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows, peer_rows), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
