"""Analyze Vicon baseball swing CSV exports against SlyMask-style metrics."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/baseball_mpl_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("../Vicon_Wave_250506(1)")


@dataclass(frozen=True)
class FrameData:
    frame: int
    timestamp_sec: float
    markers: dict[str, np.ndarray]


@dataclass(frozen=True)
class TrialMetrics:
    trial: str
    sample_rate_hz: float
    prefix: str
    phase_start_frame: int
    phase_start_sec: float
    event_frame: int
    event_sec: float
    phase_end_frame: int
    phase_end_sec: float
    swing_time_sec: float
    bat_axis_angle_deg: float
    bat1_speed_kmh: float
    bat4_speed_kmh: float
    hand_speed_kmh: float
    hand_marker: str
    wrist_speed_kmh: float
    wrist_marker: str
    hip_rotation_deg: float | None
    hip_shoulder_sep_deg: float | None
    weight_transfer_pct: float | None
    lead_knee_angle_deg: float | None
    trunk_tilt_deg: float | None
    head_stability_pct: float | None
    attack_angle_deg: float
    stride_angle_deg: float | None
    stride_length_ratio: float | None
    foot_direction_deg: float | None
    elbow_bend_deg: float | None
    arm_abduction_deg: float | None
    wrist_snap_deg: float | None


@dataclass(frozen=True)
class MetricDetail:
    trial: str
    family: str
    metric: str
    value: str
    unit: str
    status: str
    method_or_reason: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials = []
    for path in sorted(input_dir.glob("*.csv")):
        if path.name.startswith("._"):
            continue
        trials.append(analyze_trial(path))

    details = [row for trial in trials for row in metric_details(trial)]
    write_visualizations(input_dir, output_dir, trials)
    write_summary_csv(output_dir / "vicon_slymask_metrics.csv", trials)
    write_detail_csv(output_dir / "vicon_slymask_metric_detail.csv", details)
    write_report(output_dir / "vicon_slymask_metrics.md", trials)
    print(output_dir / "vicon_slymask_metrics.csv")
    print(output_dir / "vicon_slymask_metric_detail.csv")
    print(output_dir / "vicon_slymask_metrics.md")


def analyze_trial(path: Path) -> TrialMetrics:
    sample_rate_hz, frames = load_vicon_csv(path)
    prefix = select_bat_prefix(frames)
    bat1 = f"{prefix}:bat1"
    bat4 = f"{prefix}:bat4"
    event_frame = event_frame_by_bat_angle(frames, bat1, bat4, target_angle_deg=-27.0)
    phase_start, phase_end = swing_phase_by_speed(frames, bat4, event_frame, threshold_ratio=0.40)
    event = frame_by_id(frames, event_frame)

    bat1_speed = marker_speed_at_frame(frames, bat1, event_frame, sample_rate_hz)
    bat4_speed = marker_speed_at_frame(frames, bat4, event_frame, sample_rate_hz)
    hand_marker, hand_speed = best_hand_speed_at_frame(frames, prefix, event_frame, sample_rate_hz)
    wrist_marker, wrist_speed = best_wrist_speed_at_frame(frames, prefix, event_frame)

    phase_frames = [item for item in frames if phase_start <= item.frame <= phase_end]
    phase_start_data = frame_by_id(frames, phase_start)
    return TrialMetrics(
        trial=path.stem,
        sample_rate_hz=sample_rate_hz,
        prefix=prefix,
        phase_start_frame=phase_start,
        phase_start_sec=frame_by_id(frames, phase_start).timestamp_sec,
        event_frame=event_frame,
        event_sec=event.timestamp_sec,
        phase_end_frame=phase_end,
        phase_end_sec=frame_by_id(frames, phase_end).timestamp_sec,
        swing_time_sec=frame_by_id(frames, phase_end).timestamp_sec - frame_by_id(frames, phase_start).timestamp_sec,
        bat_axis_angle_deg=bat_axis_angle(event.markers[bat1], event.markers[bat4]),
        bat1_speed_kmh=bat1_speed,
        bat4_speed_kmh=bat4_speed,
        hand_speed_kmh=hand_speed,
        hand_marker=hand_marker,
        wrist_speed_kmh=wrist_speed,
        wrist_marker=wrist_marker,
        hip_rotation_deg=rotation_range(phase_frames, hip_yaw),
        hip_shoulder_sep_deg=hip_shoulder_sep(event.markers, prefix),
        weight_transfer_pct=weight_transfer_proxy(phase_start_data.markers, event.markers, prefix),
        lead_knee_angle_deg=lead_knee_angle(event.markers, prefix),
        trunk_tilt_deg=trunk_tilt(event.markers, prefix),
        head_stability_pct=head_stability(phase_frames, prefix),
        attack_angle_deg=bat_axis_angle(event.markers[bat1], event.markers[bat4]),
        stride_angle_deg=stride_angle(event.markers, prefix),
        stride_length_ratio=stride_length_ratio(event.markers, prefix),
        foot_direction_deg=foot_direction(event.markers, prefix),
        elbow_bend_deg=elbow_bend(event.markers, prefix),
        arm_abduction_deg=arm_abduction(event.markers, prefix),
        wrist_snap_deg=wrist_snap_proxy(phase_start_data.markers, event.markers, prefix, hand_marker),
    )


def load_vicon_csv(path: Path) -> tuple[float, list[FrameData]]:
    rows = list(csv.reader(path.open("r", encoding="utf-8-sig", newline="")))
    sample_rate_hz = float(rows[1][0])
    marker_header = rows[2]
    marker_columns: list[tuple[str, int]] = []
    for index, value in enumerate(marker_header):
        if index >= 2 and value.strip():
            marker_columns.append((value.strip(), index))

    frames: list[FrameData] = []
    for row in rows[5:]:
        if not row or not row[0].strip():
            continue
        frame = int(float(row[0]))
        markers: dict[str, np.ndarray] = {}
        for marker_name, start_index in marker_columns:
            point = optional_point(row, start_index)
            if point is not None:
                markers[marker_name] = point
        frames.append(FrameData(frame=frame, timestamp_sec=(frame - 1) / sample_rate_hz, markers=markers))
    return sample_rate_hz, frames


def optional_point(row: list[str], start_index: int) -> np.ndarray | None:
    values = []
    for offset in range(3):
        index = start_index + offset
        if index >= len(row) or row[index].strip() == "":
            return None
        values.append(float(row[index]))
    return np.array(values, dtype=float)


def select_bat_prefix(frames: list[FrameData]) -> str:
    prefixes = sorted(
        marker.rsplit(":", 1)[0]
        for frame in frames
        for marker in frame.markers
        if marker.endswith(":bat1")
    )
    if not prefixes:
        raise ValueError("No bat marker prefix found.")
    return prefixes[0]


def frame_by_id(frames: list[FrameData], frame_id: int) -> FrameData:
    for frame in frames:
        if frame.frame == frame_id:
            return frame
    raise KeyError(frame_id)


def event_frame_by_bat_angle(frames: list[FrameData], bat1: str, bat4: str, target_angle_deg: float) -> int:
    candidates = []
    for frame in frames:
        if bat1 in frame.markers and bat4 in frame.markers:
            angle = bat_axis_angle(frame.markers[bat1], frame.markers[bat4])
            candidates.append((abs(angle - target_angle_deg), frame.frame))
    if not candidates:
        raise ValueError("No valid bat axis frames.")
    return min(candidates)[1]


def swing_phase_by_speed(
    frames: list[FrameData],
    marker_name: str,
    event_frame: int,
    threshold_ratio: float,
) -> tuple[int, int]:
    speeds = marker_speed_series(frames, marker_name)
    if not speeds:
        return event_frame, event_frame
    event_index = min(range(len(speeds)), key=lambda index: abs(speeds[index][0] - event_frame))
    peak_speed = max(speed for _, _, speed in speeds)
    threshold = threshold_ratio * peak_speed
    start = event_index
    end = event_index
    while start > 0 and speeds[start - 1][2] >= threshold:
        start -= 1
    while end + 1 < len(speeds) and speeds[end + 1][2] >= threshold:
        end += 1
    return speeds[start][0], speeds[end][0]


def marker_speed_series(frames: list[FrameData], marker_name: str) -> list[tuple[int, float, float]]:
    rows = []
    for previous, current in zip(frames, frames[1:]):
        if marker_name not in previous.markers or marker_name not in current.markers:
            continue
        dt = current.timestamp_sec - previous.timestamp_sec
        if dt <= 0:
            continue
        speed_kmh = np.linalg.norm(current.markers[marker_name] - previous.markers[marker_name]) / dt * 0.0036
        rows.append((current.frame, current.timestamp_sec, float(speed_kmh)))
    return rows


def bat_angle_series(frames: list[FrameData], bat1: str, bat4: str) -> list[tuple[int, float, float]]:
    rows = []
    for frame in frames:
        if bat1 in frame.markers and bat4 in frame.markers:
            rows.append((frame.frame, frame.timestamp_sec, bat_axis_angle(frame.markers[bat1], frame.markers[bat4])))
    return rows


def marker_speed_at_frame(
    frames: list[FrameData],
    marker_name: str,
    frame_id: int,
    sample_rate_hz: float,
) -> float:
    del sample_rate_hz
    for frame, _, speed in marker_speed_series(frames, marker_name):
        if frame == frame_id:
            return speed
    return float("nan")


def best_hand_speed_at_frame(
    frames: list[FrameData],
    prefix: str,
    frame_id: int,
    sample_rate_hz: float,
) -> tuple[str, float]:
    del sample_rate_hz
    candidates = ("LFIN", "RFIN", "LWRA", "LWRB", "RWRA", "RWRB")
    speeds = []
    for marker in candidates:
        full_name = f"{prefix}:{marker}"
        speed = marker_speed_at_frame(frames, full_name, frame_id, 0.0)
        if not math.isnan(speed):
            speeds.append((marker, speed))
    if not speeds:
        return "", float("nan")
    return max(speeds, key=lambda item: item[1])


def best_wrist_speed_at_frame(frames: list[FrameData], prefix: str, frame_id: int) -> tuple[str, float]:
    speeds = []
    for marker in ("LWRA", "LWRB", "RWRA", "RWRB"):
        full_name = f"{prefix}:{marker}"
        speed = marker_speed_at_frame(frames, full_name, frame_id, 0.0)
        if not math.isnan(speed):
            speeds.append((marker, speed))
    if not speeds:
        return "", float("nan")
    return max(speeds, key=lambda item: item[1])


def bat_axis_angle(bat1: np.ndarray, bat4: np.ndarray) -> float:
    vector = bat1 - bat4
    horizontal = math.hypot(float(vector[0]), float(vector[1]))
    return math.degrees(math.atan2(float(vector[2]), horizontal))


def hip_yaw(frame: FrameData, prefix: str) -> float | None:
    left = marker_mean(frame.markers, prefix, ("LASI", "LPSI"))
    right = marker_mean(frame.markers, prefix, ("RASI", "RPSI"))
    if left is None or right is None:
        return None
    vector = right - left
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def shoulder_yaw(markers: dict[str, np.ndarray], prefix: str) -> float | None:
    left = markers.get(f"{prefix}:LSHO")
    right = markers.get(f"{prefix}:RSHO")
    if left is None or right is None:
        return None
    vector = right - left
    return math.degrees(math.atan2(float(vector[1]), float(vector[0])))


def rotation_range(frames: list[FrameData], getter) -> float | None:
    values = [getter(frame, select_bat_prefix(frames)) for frame in frames]
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    unwrapped = np.unwrap(np.radians(valid))
    return float(np.degrees(np.max(unwrapped) - np.min(unwrapped)))


def hip_shoulder_sep(markers: dict[str, np.ndarray], prefix: str) -> float | None:
    hip = hip_yaw(FrameData(0, 0.0, markers), prefix)
    shoulder = shoulder_yaw(markers, prefix)
    if hip is None or shoulder is None:
        return None
    return abs(signed_delta(hip, shoulder))


def lead_knee_angle(markers: dict[str, np.ndarray], prefix: str) -> float | None:
    # For right-handed batting in these trials, the lead side is approximated as left.
    return joint_angle(markers, prefix, "LASI", "LKNE", "LANK")


def trunk_tilt(markers: dict[str, np.ndarray], prefix: str) -> float | None:
    shoulder_mid = marker_mean(markers, prefix, ("LSHO", "RSHO"))
    hip_mid = marker_mean(markers, prefix, ("LASI", "RASI", "LPSI", "RPSI"))
    if shoulder_mid is None or hip_mid is None:
        return None
    torso = shoulder_mid - hip_mid
    vertical = np.array((0.0, 0.0, 1.0))
    return angle_between(torso, vertical)


def head_stability(frames: list[FrameData], prefix: str) -> float | None:
    head_offsets = []
    for frame in frames:
        head = marker_mean(frame.markers, prefix, ("LFHD", "RFHD", "LBHD", "RBHD"))
        hip = marker_mean(frame.markers, prefix, ("LASI", "RASI", "LPSI", "RPSI"))
        if head is not None and hip is not None:
            head_offsets.append(head[:2] - hip[:2])
    if len(head_offsets) < 2:
        return None
    offsets = np.array(head_offsets)
    drift = float(np.max(np.linalg.norm(offsets - np.mean(offsets, axis=0), axis=1)))
    scale = float(np.linalg.norm(np.ptp(offsets, axis=0)))
    denominator = max(scale, 1.0)
    return max(0.0, min(100.0, 100.0 * (1.0 - drift / denominator)))


def weight_transfer_proxy(
    start_markers: dict[str, np.ndarray],
    event_markers: dict[str, np.ndarray],
    prefix: str,
) -> float | None:
    start_pelvis = pelvis_midpoint(start_markers, prefix)
    event_pelvis = pelvis_midpoint(event_markers, prefix)
    left_ankle = event_markers.get(f"{prefix}:LANK")
    right_ankle = event_markers.get(f"{prefix}:RANK")
    if start_pelvis is None or event_pelvis is None or left_ankle is None or right_ankle is None:
        return None
    stance_width = float(np.linalg.norm(left_ankle[:2] - right_ankle[:2]))
    if stance_width <= 1e-9:
        return None
    pelvis_shift = float(np.linalg.norm(event_pelvis[:2] - start_pelvis[:2]))
    return 100.0 * pelvis_shift / stance_width


def stride_angle(markers: dict[str, np.ndarray], prefix: str) -> float | None:
    left_ankle = markers.get(f"{prefix}:LANK")
    right_ankle = markers.get(f"{prefix}:RANK")
    left_hip = marker_mean(markers, prefix, ("LASI", "LPSI"))
    right_hip = marker_mean(markers, prefix, ("RASI", "RPSI"))
    if left_ankle is None or right_ankle is None or left_hip is None or right_hip is None:
        return None
    return planar_axis_angle(left_ankle - right_ankle, left_hip - right_hip)


def stride_length_ratio(markers: dict[str, np.ndarray], prefix: str) -> float | None:
    left_ankle = markers.get(f"{prefix}:LANK")
    right_ankle = markers.get(f"{prefix}:RANK")
    height = body_height_proxy(markers, prefix)
    if left_ankle is None or right_ankle is None or height is None or height <= 1e-9:
        return None
    return float(np.linalg.norm(left_ankle[:2] - right_ankle[:2]) / height)


def foot_direction(markers: dict[str, np.ndarray], prefix: str) -> float | None:
    left_toe = markers.get(f"{prefix}:LTOE")
    left_heel = markers.get(f"{prefix}:LHEE")
    left_ankle = markers.get(f"{prefix}:LANK")
    right_ankle = markers.get(f"{prefix}:RANK")
    if left_toe is None or left_heel is None or left_ankle is None or right_ankle is None:
        return None
    return planar_axis_angle(left_toe - left_heel, left_ankle - right_ankle)


def elbow_bend(markers: dict[str, np.ndarray], prefix: str) -> float | None:
    return joint_angle(markers, prefix, "RSHO", "RELB", "RWRA")


def arm_abduction(markers: dict[str, np.ndarray], prefix: str) -> float | None:
    shoulder_mid = marker_mean(markers, prefix, ("LSHO", "RSHO"))
    hip_mid = pelvis_midpoint(markers, prefix)
    elbow = markers.get(f"{prefix}:RELB")
    shoulder = markers.get(f"{prefix}:RSHO")
    if shoulder_mid is None or hip_mid is None or elbow is None or shoulder is None:
        return None
    torso_axis = shoulder_mid - hip_mid
    upper_arm = elbow - shoulder
    return angle_between(upper_arm, torso_axis)


def wrist_snap_proxy(
    start_markers: dict[str, np.ndarray],
    event_markers: dict[str, np.ndarray],
    prefix: str,
    hand_marker: str,
) -> float | None:
    side = hand_marker[0] if hand_marker and hand_marker[0] in ("L", "R") else "R"
    start_angle = wrist_finger_angle(start_markers, prefix, side)
    event_angle = wrist_finger_angle(event_markers, prefix, side)
    if start_angle is None or event_angle is None:
        return None
    return abs(event_angle - start_angle)


def wrist_finger_angle(markers: dict[str, np.ndarray], prefix: str, side: str) -> float | None:
    elbow = markers.get(f"{prefix}:{side}ELB")
    finger = markers.get(f"{prefix}:{side}FIN")
    wrist = marker_mean(markers, prefix, (f"{side}WRA", f"{side}WRB"))
    if elbow is None or finger is None or wrist is None:
        return None
    return angle_between(elbow - wrist, finger - wrist)


def pelvis_midpoint(markers: dict[str, np.ndarray], prefix: str) -> np.ndarray | None:
    return marker_mean(markers, prefix, ("LASI", "RASI", "LPSI", "RPSI"))


def body_height_proxy(markers: dict[str, np.ndarray], prefix: str) -> float | None:
    head = marker_mean(markers, prefix, ("LFHD", "RFHD", "LBHD", "RBHD"))
    feet = marker_mean(markers, prefix, ("LANK", "RANK", "LHEE", "RHEE", "LTOE", "RTOE"))
    if head is None or feet is None:
        return None
    return float(abs(head[2] - feet[2]))


def planar_axis_angle(a: np.ndarray, b: np.ndarray) -> float | None:
    angle = angle_between(np.array((a[0], a[1], 0.0)), np.array((b[0], b[1], 0.0)))
    if angle is None:
        return None
    return min(angle, 180.0 - angle)


def marker_mean(markers: dict[str, np.ndarray], prefix: str, names: Iterable[str]) -> np.ndarray | None:
    points = [markers[f"{prefix}:{name}"] for name in names if f"{prefix}:{name}" in markers]
    if not points:
        return None
    return np.mean(points, axis=0)


def joint_angle(markers: dict[str, np.ndarray], prefix: str, a: str, b: str, c: str) -> float | None:
    if not all(f"{prefix}:{name}" in markers for name in (a, b, c)):
        return None
    return angle_between(markers[f"{prefix}:{a}"] - markers[f"{prefix}:{b}"], markers[f"{prefix}:{c}"] - markers[f"{prefix}:{b}"])


def angle_between(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return None
    cosine = max(-1.0, min(1.0, float(np.dot(a, b) / denom)))
    return math.degrees(math.acos(cosine))


def signed_delta(a: float, b: float) -> float:
    return (b - a + 180.0) % 360.0 - 180.0


def format_optional(value: float | None, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def metric_details(trial: TrialMetrics) -> list[MetricDetail]:
    rows = [
        detail(trial, "Swing Analysis", "Estimated Bat Speed", trial.bat1_speed_kmh, "km/h", "direct", "3D speed of bat1 at selected swing event; bat4 is also reported in the summary because barrel marker identity is ambiguous."),
        detail(trial, "Swing Analysis", "Swing Speed", trial.bat1_speed_kmh, "km/h", "raw_only", "Raw bat endpoint speed is available; SlyMask-style percentile needs a reference population."),
        detail(trial, "Swing Analysis", "Hip Rotation", trial.hip_rotation_deg, "deg", "direct", "Pelvis yaw range over the selected high-speed swing window."),
        detail(trial, "Swing Analysis", "Hip-Shoulder Sep", trial.hip_shoulder_sep_deg, "deg", "direct", "Absolute yaw difference between pelvis axis and shoulder axis at the event frame."),
        detail(trial, "Swing Analysis", "Weight Transfer", trial.weight_transfer_pct, "%", "proxy", "Pelvis midpoint translation from phase start to event, normalized by ankle stance width; not a full COM model."),
        detail(trial, "Swing Analysis", "Lead Knee Angle", trial.lead_knee_angle_deg, "deg", "direct", "Left lead-knee angle from LASI-LKNE-LANK."),
        detail(trial, "Swing Analysis", "Trunk Tilt", trial.trunk_tilt_deg, "deg", "direct", "Torso midpoint vector relative to vertical Z axis."),
        unavailable(trial, "Swing Analysis", "Contact Time", "ms", "No ball-contact label, bat-ball impact event, or ball marker exists in this CSV."),
        detail(trial, "Swing Analysis", "Attack Angle", trial.attack_angle_deg, "deg", "proxy", "Bat axis angle relative to the horizontal plane at selected event; true attack angle needs a contact-frame definition."),
        detail(trial, "Swing Analysis", "Head Stability", trial.head_stability_pct, "%", "proxy", "Head midpoint drift relative to pelvis midpoint over the selected swing window."),
        detail(trial, "Motion Metrics", "Elbow Bend", trial.elbow_bend_deg, "deg", "direct_batting_context", "Right elbow angle from RSHO-RELB-RWRA; SlyMask's pitching acceleration interpretation is not directly applicable to this batting dataset."),
        detail(trial, "Motion Metrics", "Arm Abduction", trial.arm_abduction_deg, "deg", "direct_batting_context", "Right upper-arm angle relative to torso axis; pitching arm-slot interpretation is not directly applicable."),
        detail(trial, "Motion Metrics", "Trunk Lean", trial.trunk_tilt_deg, "deg", "direct_batting_context", "Same torso-vs-vertical geometry as Trunk Tilt, measured at swing event instead of pitching release."),
        detail(trial, "Motion Metrics", "Stride Angle", trial.stride_angle_deg, "deg", "proxy", "Planar angle between ankle stance line and pelvis axis at event; pitching front-foot-landing definition is unavailable."),
        detail(trial, "Motion Metrics", "Lead Knee", trial.lead_knee_angle_deg, "deg", "direct_batting_context", "Left lead-knee angle from LASI-LKNE-LANK; measured at swing event."),
        detail(trial, "Motion Metrics", "Hip-Shoulder Sep", trial.hip_shoulder_sep_deg, "deg", "direct_batting_context", "Same pelvis-shoulder yaw separation as batting metric, measured at swing event."),
        detail(trial, "Motion Metrics", "Arm Speed", trial.wrist_speed_kmh, "km/h", "raw_only", f"Fastest wrist marker at event is {trial.wrist_marker}; percentile needs a reference population."),
        detail(trial, "Motion Metrics", "Stride Length", trial.stride_length_ratio, "body heights", "proxy", "Ankle stance distance divided by head-to-foot height proxy; pitching stride event definition is unavailable."),
        detail(trial, "Motion Metrics", "Weight Transfer", trial.weight_transfer_pct, "%", "proxy", "Same pelvis-shift proxy as batting Weight Transfer; not a validated COM transfer metric."),
        detail(trial, "Motion Metrics", "Head Stability", trial.head_stability_pct, "%", "proxy", "Head drift relative to pelvis over the swing window; SlyMask stride-line definition is unavailable."),
        detail(trial, "Motion Metrics", "Foot Direction", trial.foot_direction_deg, "deg", "proxy", "Left heel-to-toe direction relative to stance line at event; home-plate target direction is unavailable."),
        detail(trial, "Motion Metrics", "Wrist Snap", trial.wrist_snap_deg, "deg", "proxy", "Change in elbow-wrist-finger angle from phase start to event using the fastest hand side."),
        detail(trial, "Motion Metrics", "Fingertip Speed", trial.hand_speed_kmh, "km/h", "raw_only", f"Fastest hand/finger marker at event is {trial.hand_marker}; percentile needs a reference population."),
    ]
    return rows


def detail(
    trial: TrialMetrics,
    family: str,
    metric: str,
    value: float | None,
    unit: str,
    status: str,
    method: str,
) -> MetricDetail:
    return MetricDetail(
        trial=trial.trial,
        family=family,
        metric=metric,
        value=format_optional(value, 3),
        unit=unit,
        status=status if format_optional(value, 3) else "unavailable",
        method_or_reason=method if format_optional(value, 3) else f"Required marker data was missing or invalid. {method}",
    )


def unavailable(trial: TrialMetrics, family: str, metric: str, unit: str, reason: str) -> MetricDetail:
    return MetricDetail(
        trial=trial.trial,
        family=family,
        metric=metric,
        value="",
        unit=unit,
        status="unavailable",
        method_or_reason=reason,
    )


def raw_data_source(row: MetricDetail) -> str:
    sources = {
        ("Swing Analysis", "Estimated Bat Speed"): "`bat1/bat4` 3D coordinates, frame timestamps",
        ("Swing Analysis", "Swing Speed"): "`bat1/bat4` 3D coordinates, frame timestamps",
        ("Swing Analysis", "Hip Rotation"): "`LASI/RASI/LPSI/RPSI` pelvis markers",
        ("Swing Analysis", "Hip-Shoulder Sep"): "`LASI/RASI/LPSI/RPSI`, `LSHO/RSHO`",
        ("Swing Analysis", "Weight Transfer"): "`LASI/RASI/LPSI/RPSI`, `LANK/RANK`",
        ("Swing Analysis", "Lead Knee Angle"): "`LASI/LKNE/LANK`",
        ("Swing Analysis", "Trunk Tilt"): "`LSHO/RSHO`, `LASI/RASI/LPSI/RPSI`, vertical `Z`",
        ("Swing Analysis", "Contact Time"): "Missing ball marker or bat-ball contact label",
        ("Swing Analysis", "Attack Angle"): "`bat1/bat4` 3D coordinates",
        ("Swing Analysis", "Head Stability"): "`LFHD/RFHD/LBHD/RBHD`, pelvis markers",
        ("Motion Metrics", "Elbow Bend"): "`RSHO/RELB/RWRA`",
        ("Motion Metrics", "Arm Abduction"): "`RSHO/RELB`, shoulder midpoint, pelvis midpoint",
        ("Motion Metrics", "Trunk Lean"): "`LSHO/RSHO`, pelvis markers, vertical `Z`",
        ("Motion Metrics", "Stride Angle"): "`LANK/RANK`, pelvis axis markers",
        ("Motion Metrics", "Lead Knee"): "`LASI/LKNE/LANK`",
        ("Motion Metrics", "Hip-Shoulder Sep"): "`LASI/RASI/LPSI/RPSI`, `LSHO/RSHO`",
        ("Motion Metrics", "Arm Speed"): "`LWRA/LWRB/RWRA/RWRB` 3D coordinates, frame timestamps",
        ("Motion Metrics", "Stride Length"): "`LANK/RANK`, head and foot height proxy markers",
        ("Motion Metrics", "Weight Transfer"): "`LASI/RASI/LPSI/RPSI`, `LANK/RANK`",
        ("Motion Metrics", "Head Stability"): "`LFHD/RFHD/LBHD/RBHD`, pelvis markers",
        ("Motion Metrics", "Foot Direction"): "`LTOE/LHEE`, `LANK/RANK`",
        ("Motion Metrics", "Wrist Snap"): "`LELB/RELB`, wrist pair, `LFIN/RFIN`",
        ("Motion Metrics", "Fingertip Speed"): "`LFIN/RFIN` 3D coordinates, frame timestamps",
    }
    return sources.get((row.family, row.metric), "")


def write_visualizations(input_dir: Path, output_dir: Path, trials: list[TrialMetrics]) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    trial_lookup = {trial.trial: trial for trial in trials}
    loaded_trials = []
    for path in sorted(input_dir.glob("*.csv")):
        if path.name.startswith("._") or path.stem not in trial_lookup:
            continue
        sample_rate_hz, frames = load_vicon_csv(path)
        del sample_rate_hz
        loaded_trials.append((trial_lookup[path.stem], frames))
    plot_bat_angle_time(figure_dir / "vicon_bat_angle_time.png", loaded_trials)
    plot_speed_time(figure_dir / "vicon_speed_time.png", loaded_trials)


def plot_bat_angle_time(path: Path, loaded_trials: list[tuple[TrialMetrics, list[FrameData]]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    for trial, frames in loaded_trials:
        bat1 = f"{trial.prefix}:bat1"
        bat4 = f"{trial.prefix}:bat4"
        series = bat_angle_series(frames, bat1, bat4)
        if not series:
            continue
        times = [time for _, time, _ in series]
        angles = [angle for _, _, angle in series]
        ax.plot(times, angles, label=trial.trial)
        ax.axvspan(trial.phase_start_sec, trial.phase_end_sec, alpha=0.10)
        ax.axvline(trial.event_sec, linestyle="--", linewidth=1)
    ax.axhline(-27.0, color="black", linestyle=":", linewidth=1, label="-27 deg reference")
    ax.set_title("Bat Axis Angle Over Time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("bat angle vs horizontal (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_speed_time(path: Path, loaded_trials: list[tuple[TrialMetrics, list[FrameData]]]) -> None:
    fig, axes = plt.subplots(len(loaded_trials), 1, figsize=(10, 4 * max(1, len(loaded_trials))), dpi=160, sharex=False)
    if len(loaded_trials) == 1:
        axes = [axes]
    for ax, (trial, frames) in zip(axes, loaded_trials):
        for marker, label, linestyle in (
            ("bat1", "bat1 speed", "-"),
            ("bat4", "bat4 speed", "--"),
            (trial.wrist_marker, f"{trial.wrist_marker} speed", ":"),
        ):
            full_name = f"{trial.prefix}:{marker}"
            series = marker_speed_series(frames, full_name)
            if not series:
                continue
            times = [time for _, time, _ in series]
            speeds = [speed for _, _, speed in series]
            ax.plot(times, speeds, linestyle=linestyle, label=label)
        ax.axvspan(trial.phase_start_sec, trial.phase_end_sec, alpha=0.10, label="swing window")
        ax.axvline(trial.event_sec, linestyle="--", linewidth=1, color="black", label="event frame")
        ax.set_title(trial.trial)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("speed (km/h)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Bat and Wrist Speed Over Time", y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_summary_csv(path: Path, trials: list[TrialMetrics]) -> None:
    fields = list(TrialMetrics.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for trial in trials:
            writer.writerow({field: getattr(trial, field) for field in fields})


def write_detail_csv(path: Path, details: list[MetricDetail]) -> None:
    fields = ["trial", "family", "metric", "value", "unit", "status", "raw_data_source", "method_or_reason"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in details:
            writer.writerow(
                {
                    "trial": row.trial,
                    "family": row.family,
                    "metric": row.metric,
                    "value": row.value,
                    "unit": row.unit,
                    "status": row.status,
                    "raw_data_source": raw_data_source(row).replace("`", ""),
                    "method_or_reason": row.method_or_reason,
                }
            )


def write_report(path: Path, trials: list[TrialMetrics]) -> None:
    lines = [
        "# Vicon Swing Metrics",
        "",
        "Input: `../Vicon_Wave_250506(1)` Vicon trajectory CSV files.",
        "",
        "Assumptions: coordinates are in millimeters; sample rate comes from row 2 (`100 Hz`); `Z` is vertical; the bat axis is `bat4 -> bat1`. Bat angle is `atan2(delta_Z, horizontal_distance)` in degrees.",
        "",
        "Swing event is selected as the frame where the bat-axis angle is closest to `-27 deg`, matching the provided reference. Swing duration is the contiguous high-speed window around that event where `bat4` speed remains above `40%` of its trial peak. This threshold gives the Coach reference trial a `0.16 s` window.",
        "",
        "## CSV Structure",
        "",
        "- Row 1 identifies the Vicon section as `Trajectories`.",
        "- Row 2 contains the sample rate, `100 Hz` for both files.",
        "- Row 3 contains marker names. Each marker then has three coordinate columns.",
        "- Row 4 contains `Frame`, `Sub Frame`, then repeated `X,Y,Z` coordinate fields.",
        "- Row 5 states coordinate units as `mm`.",
        "- Timestamps are reconstructed as `(Frame - 1) / sample_rate_hz`.",
        "- Body markers include head (`LFHD/RFHD/LBHD/RBHD`), shoulders, elbows, wrists, pelvis (`LASI/RASI/LPSI/RPSI`), knees, ankles, heels, toes, and finger markers.",
        "- Bat markers are `bat1`, `bat2`, `bat3`, `bat4`; the longest and most stable axis is `bat1-bat4` at about `293 mm`.",
        "",
        "## 1. Metric Feasibility vs Vicon Raw Data",
        "",
        "Status meanings: `direct` is a direct 3D marker geometry calculation; `raw_only` means the physical value exists but SlyMask's percentile/rating needs a reference population; `proxy` means the value is computable but the app's exact event or target definition is missing; `direct_batting_context` means the geometry is direct, but SlyMask describes the metric for pitching while these Vicon files are batting swings.",
        "",
        "The SlyMask categorical labels (`Good`, `Attention`, `Deviate`) and reliability percentages are not reproduced here because they require proprietary thresholds, a reference population, and the app's internal confidence model. This report focuses on whether the underlying physical metric can be extracted automatically from the Vicon CSV.",
        "",
        "| trial | family | metric | value | unit | status | Vicon raw data | calculation / reason |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for row in [item for trial in trials for item in metric_details(trial)]:
        lines.append(
            f"| {row.trial} | {row.family} | {row.metric} | {row.value} | {row.unit} | "
            f"{row.status} | {raw_data_source(row)} | {row.method_or_reason} |"
        )
    lines.extend(
        [
            "",
            "General formulas used in the table: joint angles use the vector dot product at the middle marker, `acos((BA dot BC) / (||BA|| ||BC||))`; marker speeds use 3D position differencing over adjacent frames; pelvis/shoulder rotations use planar yaw from left-right marker axes; normalization metrics use available body marker distances because the CSV has no explicit whole-body COM model.",
            "",
            "## 2. Specific Parameter Output and Validation",
            "",
            "Bat-head angle is computed from the selected bat axis `bat4 -> bat1`: `angle = atan2(Z_bat1 - Z_bat4, sqrt((X_bat1 - X_bat4)^2 + (Y_bat1 - Y_bat4)^2))`. The event frame is the frame whose bat-axis angle is closest to the provided `-27 deg` reference.",
            "",
            "Speed is computed by frame-to-frame 3D finite difference: `speed_km_h = ||P_t - P_(t-1)|| / delta_t * 0.0036`, where coordinates are in millimeters and `delta_t = 0.01 s` at `100 Hz`.",
            "",
            "Swing time is defined as the contiguous high-speed window around the event frame where `bat4` speed remains above `40%` of that trial's peak `bat4` speed. This operational definition is automatic and gives the Coach reference trial a `0.16 s` window.",
            "",
            "| trial | event time (s) | bat angle (deg) | bat1 speed (km/h) | bat4 speed (km/h) | best wrist marker | wrist speed (km/h) | best hand marker | hand speed (km/h) | swing start-end (s) | swing time (s) |",
            "|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|",
        ]
    )
    for trial in trials:
        lines.append(
            f"| {trial.trial} | {trial.event_sec:.3f} | {trial.bat_axis_angle_deg:.1f} | "
            f"{trial.bat1_speed_kmh:.1f} | {trial.bat4_speed_kmh:.1f} | {trial.wrist_marker} | "
            f"{trial.wrist_speed_kmh:.1f} | {trial.hand_marker} | {trial.hand_speed_kmh:.1f} | "
            f"{trial.phase_start_sec:.3f}-{trial.phase_end_sec:.3f} | {trial.swing_time_sec:.3f} |"
        )
    lines.extend(
        [
            "",
            "Reference comparison: `0506Coach_wave` matches the requested bat angle (`-27.1 deg`). Its fastest endpoint `bat1` is `131.7 km/h`; the opposite endpoint `bat4` is `87.9 km/h`, closer to the requested `95 km/h`. The best wrist marker at the same frame is `LWRA = 30.4 km/h`; the best hand/finger marker is `LFIN = 39.4 km/h`, matching the requested wrist/hand-speed reference better if the reference used a hand/finger marker rather than a strict wrist marker.",
            "",
            "## 4. Visualizations",
            "",
            "![Bat axis angle over time](figures/vicon_bat_angle_time.png)",
            "",
            "![Bat and wrist speed over time](figures/vicon_speed_time.png)",
            "",
            "## Body Metrics at Event",
            "",
            "| trial | hip rotation (deg) | hip-shoulder sep (deg) | lead knee (deg) | trunk tilt (deg) | head stability (%) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for trial in trials:
        lines.append(
            f"| {trial.trial} | {format_optional(trial.hip_rotation_deg)} | "
            f"{format_optional(trial.hip_shoulder_sep_deg)} | {format_optional(trial.lead_knee_angle_deg)} | "
            f"{format_optional(trial.trunk_tilt_deg)} | {format_optional(trial.head_stability_pct)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
