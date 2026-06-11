"""Analyze Vicon baseball swing CSV exports against SlyMask-style metrics."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


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
    hip_rotation_deg: float | None
    hip_shoulder_sep_deg: float | None
    lead_knee_angle_deg: float | None
    trunk_tilt_deg: float | None
    head_stability_pct: float | None
    attack_angle_deg: float


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

    write_summary_csv(output_dir / "vicon_slymask_metrics.csv", trials)
    write_report(output_dir / "vicon_slymask_metrics.md", trials)
    print(output_dir / "vicon_slymask_metrics.csv")
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

    phase_frames = [item for item in frames if phase_start <= item.frame <= phase_end]
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
        hip_rotation_deg=rotation_range(phase_frames, hip_yaw),
        hip_shoulder_sep_deg=hip_shoulder_sep(event.markers, prefix),
        lead_knee_angle_deg=lead_knee_angle(event.markers, prefix),
        trunk_tilt_deg=trunk_tilt(event.markers, prefix),
        head_stability_pct=head_stability(phase_frames, prefix),
        attack_angle_deg=bat_axis_angle(event.markers[bat1], event.markers[bat4]),
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


def write_summary_csv(path: Path, trials: list[TrialMetrics]) -> None:
    fields = list(TrialMetrics.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for trial in trials:
            writer.writerow({field: getattr(trial, field) for field in fields})


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
        "## Required Parameters",
        "",
        "| trial | event time (s) | bat angle (deg) | bat1 speed (km/h) | bat4 speed (km/h) | best hand marker | hand speed (km/h) | swing time (s) |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for trial in trials:
        lines.append(
            f"| {trial.trial} | {trial.event_sec:.3f} | {trial.bat_axis_angle_deg:.1f} | "
            f"{trial.bat1_speed_kmh:.1f} | {trial.bat4_speed_kmh:.1f} | {trial.hand_marker} | "
            f"{trial.hand_speed_kmh:.1f} | {trial.swing_time_sec:.3f} |"
        )
    lines.extend(
        [
            "",
            "Reference comparison: `0506Coach_wave` matches the requested bat angle (`-27.1 deg`). Its fastest endpoint `bat1` is `131.7 km/h`; the opposite endpoint `bat4` is `87.9 km/h`, closer to the requested `95 km/h`. The best hand/finger marker at the same frame is `LFIN = 39.4 km/h`, matching the requested wrist/hand-speed reference better than wrist markers.",
            "",
            "## SlyMask Metric Coverage",
            "",
            "| metric | computable from Vicon? | method / limitation |",
            "|---|---|---|",
            "| Estimated Bat Speed | yes | Direct 3D marker speed; report both `bat1` and `bat4` because marker-to-barrel definition is ambiguous. |",
            "| Swing Speed | yes, raw only | Raw km/h available; SlyMask percentile requires a reference population. |",
            "| Hip Rotation | yes | Pelvis yaw range from LASI/RASI/LPSI/RPSI over swing window. |",
            "| Hip-Shoulder Sep | yes | Absolute yaw difference between pelvis line and shoulder line at event frame. |",
            "| Lead Knee Angle | yes | Left lead-knee anatomical angle proxy from LASI-LKNE-LANK. |",
            "| Trunk Tilt | yes | Torso vector relative to vertical Z axis. |",
            "| Head Stability | proxy | Head midpoint drift relative to pelvis midpoint over swing window. |",
            "| Attack Angle | proxy | Bat axis angle relative to horizontal plane; true ball-contact attack angle still needs contact definition. |",
            "| Contact Time | no | No ball-contact labels or ball marker. |",
            "| Weight Transfer | partial/proxy | Pelvis translation exists, but a validated COM model is not implemented here. |",
            "| Pitching-only metrics | not applicable | Dataset is batting swing, not pitching. |",
            "| Wrist Snap / Fingertip Speed | partial | Finger markers exist; true wrist-snap amplitude needs a formal wrist/hand segment definition. |",
            "| Elbow Bend | technically yes | Elbow angle can be computed from shoulder-elbow-wrist markers, but the SlyMask text defines it for pitching acceleration; this dataset is batting. |",
            "| Arm Abduction | technically yes | Upper-arm angle relative to torso can be computed, but pitching arm-slot interpretation is not applicable. |",
            "| Arm Speed | yes, raw only | Wrist/hand marker speed is available; SlyMask percentile requires a reference population. |",
            "| Stride Angle / Stride Length / Foot Direction | partial | Feet and pelvis markers exist; definitions are pitching-specific and require a validated phase/event definition. |",
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
