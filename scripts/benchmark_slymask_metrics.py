"""Export SlyMask-style benchmark metrics from the local 2D/3D pipeline.

The report is intentionally explicit about capability boundaries. A metric is
marked as:
- available: directly computed from current pipeline outputs.
- proxy: approximated with a defensible substitute.
- unavailable: current inputs do not support the metric.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np

from baseball_pose.io.pose3d_csv import read_pose3d_records
from baseball_pose.pose3d.schema import Pose3DRecord


DEFAULT_CLIPS = (
    "benchmark_pitch_vertical_10",
    "benchmark_pitch_vertical_09",
    "benchmark_hit_vertical_02",
    "benchmark_hit_horizontal_06",
)
CONDITION_2D = "image_center_motion_grabcut_pose_complete_smooth"
CONDITION_3D = "image_center_motion_grabcut_pose_complete_smooth_3d_smooth"


@dataclass(frozen=True)
class MetricRow:
    clip_id: str
    action_type: str
    metric_name: str
    value: str
    unit: str
    availability: str
    source: str
    event_frame: str
    reason: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_full/benchmark_rtmpose_test")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--clips", nargs="*", default=list(DEFAULT_CLIPS))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[MetricRow] = []
    for clip_id in args.clips:
        rows.extend(_clip_metrics(data_dir, clip_id))

    csv_path = output_dir / "slymask_benchmark_metrics.csv"
    md_path = output_dir / "slymask_benchmark_metrics.md"
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def _clip_metrics(data_dir: Path, clip_id: str) -> list[MetricRow]:
    action_type = "pitching" if "pitch" in clip_id else "batting"
    pose3d_path = (
        data_dir
        / "processed"
        / "poses3d"
        / clip_id
        / f"{CONDITION_3D}.csv"
    )
    object_path = (
        data_dir
        / "processed"
        / "object_features"
        / clip_id
        / f"{CONDITION_2D}.csv"
    )
    frames = _read_pose3d_frames(pose3d_path)
    object_rows = _read_object_rows(object_path)
    if not frames:
        return [
            MetricRow(
                clip_id,
                action_type,
                "all_body_metrics",
                "",
                "",
                "unavailable",
                "3d_pose",
                "",
                f"3D pose file missing or empty: {pose3d_path}",
            )
        ]

    event_frame = _event_frame(frames, action_type, object_rows)
    landing_frame = _landing_frame(frames, event_frame, action_type)
    dominant_side = _dominant_side(frames)
    lead_side = _lead_side(frames, landing_frame)
    rows = _shared_body_metrics(clip_id, action_type, frames, event_frame, landing_frame, dominant_side, lead_side)

    if action_type == "pitching":
        rows.extend(_pitching_metrics(clip_id, frames, object_rows, event_frame, landing_frame, dominant_side, lead_side))
    else:
        rows.extend(_batting_metrics(clip_id, frames, object_rows, event_frame, landing_frame, dominant_side, lead_side))
    return rows


def _shared_body_metrics(
    clip_id: str,
    action_type: str,
    frames: dict[int, dict[str, np.ndarray]],
    event_frame: int,
    landing_frame: int,
    dominant_side: str,
    lead_side: str,
) -> list[MetricRow]:
    phase_frames = _phase_window(frames, landing_frame, event_frame)
    hip_sep = _at(frames, event_frame, _hip_shoulder_sep)
    lead_knee = _joint_angle(frames[landing_frame], f"{lead_side}_hip", f"{lead_side}_knee", f"{lead_side}_ankle")
    trunk = _at(frames, event_frame, _trunk_tilt)
    head_stability = _head_stability(phase_frames)
    return [
        _metric(clip_id, action_type, "Hip-Shoulder Sep", hip_sep, "deg", "available", "3d_pose", event_frame, "SMPL24 hip/shoulder lines projected to horizontal plane."),
        _metric(clip_id, action_type, "Lead Knee Angle", lead_knee, "deg", "available", "3d_pose", landing_frame, f"Lead side inferred as {lead_side}; value is anatomical knee angle, not flexion-only label."),
        _metric(clip_id, action_type, "Trunk Tilt", trunk, "deg", "available", "3d_pose", event_frame, "Torso vector relative to reconstructed vertical axis."),
        _unavailable(clip_id, action_type, "Weight Transfer", "Current GVHMR output is not calibrated to field/world translation; hip/root drift should not be interpreted as COM transfer."),
        _metric(clip_id, action_type, "Head Stability", head_stability, "%", "proxy", "3d_pose", event_frame, "Root-relative head drift score; no SlyMask reference scale."),
        MetricRow(clip_id, action_type, "Dominant Side", dominant_side, "", "proxy", "3d_pose", str(event_frame), "Inferred from larger hand peak speed."),
        MetricRow(clip_id, action_type, "Lead Side", lead_side, "", "proxy", "3d_pose", str(landing_frame), "Inferred from foot position along stride direction."),
    ]


def _pitching_metrics(
    clip_id: str,
    frames: dict[int, dict[str, np.ndarray]],
    object_rows: list[dict[str, str]],
    release_frame: int,
    landing_frame: int,
    side: str,
    lead_side: str,
) -> list[MetricRow]:
    elbow = _joint_angle(frames[release_frame], f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
    abduction = _arm_abduction(frames[release_frame], side)
    stride_angle = _stride_angle(frames[landing_frame])
    stride_length = _stride_length_ratio(frames, landing_frame)
    foot_direction = _foot_direction(frames[landing_frame], lead_side, _stride_direction(frames))
    wrist_snap = _wrist_snap_proxy(frames, release_frame, side)
    hand_speed = _joint_speed(frames, f"{side}_hand", release_frame)
    wrist_speed = _joint_speed(frames, f"{side}_wrist", release_frame)
    ball_speed = _object_peak(object_rows, "ball_speed_px_s")
    rows = [
        _metric(clip_id, "pitching", "Elbow Bend", elbow, "deg", "available", "3d_pose", release_frame, "Throwing side inferred by peak hand speed."),
        _metric(clip_id, "pitching", "Arm Abduction", abduction, "deg", "available", "3d_pose", release_frame, "Upper arm angle relative to torso axis."),
        _metric(clip_id, "pitching", "Stride Angle", stride_angle, "deg", "proxy", "3d_pose", landing_frame, "Angle between foot line and hip line at inferred landing frame."),
        _metric(clip_id, "pitching", "Stride Length", stride_length, "height_ratio", "proxy", "3d_pose", landing_frame, "Foot separation normalized by reconstructed body-height proxy."),
        _metric(clip_id, "pitching", "Foot Direction", foot_direction, "deg", "proxy", "3d_pose", landing_frame, "SMPL24 has foot marker but no toe; this approximates toe direction from ankle-to-foot vector."),
        _metric(clip_id, "pitching", "Wrist Snap", wrist_snap, "deg", "proxy", "3d_pose", release_frame, "Uses elbow-wrist-hand angle change; no fingertip joint is available."),
        _metric(clip_id, "pitching", "Arm Speed", wrist_speed, "3d_unit/s", "proxy", "3d_pose", release_frame, "Raw wrist speed at release proxy; percentile needs a normative database."),
        _metric(clip_id, "pitching", "Fingertip Speed", hand_speed, "3d_unit/s", "proxy", "3d_pose", release_frame, "SMPL24 hand joint is used because fingertip joints are unavailable."),
    ]
    if ball_speed is None:
        rows.append(_unavailable(clip_id, "pitching", "Ball Speed", "Ball track is missing or filtered out for this clip."))
    else:
        rows.append(_metric(clip_id, "pitching", "Ball Speed", ball_speed, "px/s", "proxy", "object_2d", release_frame, "2D ball speed without camera calibration; not physical mph/km/h."))
    return rows


def _batting_metrics(
    clip_id: str,
    frames: dict[int, dict[str, np.ndarray]],
    object_rows: list[dict[str, str]],
    contact_frame: int,
    landing_frame: int,
    side: str,
    lead_side: str,
) -> list[MetricRow]:
    del lead_side
    phase_frames = _phase_window(frames, landing_frame, contact_frame)
    hip_rotation = _rotation_range(phase_frames, _hip_yaw)
    bat_speed = _object_at_frame(object_rows, contact_frame, "bat_speed_px_s")
    bat_speed_norm = _object_at_frame(object_rows, contact_frame, "bat_speed_norm_s")
    attack_angle = _object_at_frame(object_rows, contact_frame, "bat_angle_deg")
    wrist_speed = _joint_speed(frames, f"{side}_wrist", contact_frame)
    rows = [
        _metric(clip_id, "batting", "Swing Speed", bat_speed_norm, "norm/s", "proxy" if bat_speed_norm is not None else "unavailable", "object_2d", contact_frame, "SlyMask percentile is unavailable; this is normalized 2D bat speed.") if bat_speed_norm is not None else _unavailable(clip_id, "batting", "Swing Speed", "No valid bat track after confidence/speed filtering."),
        _metric(clip_id, "batting", "Estimated Bat Speed", bat_speed, "px/s", "proxy" if bat_speed is not None else "unavailable", "object_2d", contact_frame, "No camera calibration/bat scale, so km/h cannot be recovered.") if bat_speed is not None else _unavailable(clip_id, "batting", "Estimated Bat Speed", "No valid bat track after confidence/speed filtering."),
        _metric(clip_id, "batting", "Hip Rotation", hip_rotation, "deg", "available", "3d_pose", contact_frame, "Range of pelvis yaw over the landing-to-event phase window."),
        _metric(clip_id, "batting", "Attack Angle", attack_angle, "deg", "proxy" if attack_angle is not None else "unavailable", "object_2d", contact_frame, "Image-plane bat angle at peak bat speed; not true 3D attack angle.") if attack_angle is not None else _unavailable(clip_id, "batting", "Attack Angle", "No valid bat angle at peak-speed/contact proxy frame."),
        _metric(clip_id, "batting", "Wrist/Hand Speed", wrist_speed, "3d_unit/s", "proxy", "3d_pose", contact_frame, "Useful internal body-speed proxy; SlyMask swing percentile needs a reference database."),
        _unavailable(clip_id, "batting", "Contact Time", "No ball track in batting benchmark and no bat-ball impact event detector; cannot determine physical contact duration."),
    ]
    return rows


def _read_pose3d_frames(path: Path) -> dict[int, dict[str, np.ndarray]]:
    if not path.exists():
        return {}
    records = read_pose3d_records(path)
    frames: dict[int, dict[str, np.ndarray]] = {}
    for record in records:
        frames.setdefault(record.frame_index, {})[record.joint_name] = _xyz(record)
    return frames


def _read_object_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _phase_window(
    frames: dict[int, dict[str, np.ndarray]],
    start_frame: int,
    end_frame: int,
) -> dict[int, dict[str, np.ndarray]]:
    start = min(start_frame, end_frame)
    end = max(start_frame, end_frame)
    window = {
        frame_index: points
        for frame_index, points in frames.items()
        if start <= frame_index <= end
    }
    return window or frames


def _event_frame(
    frames: dict[int, dict[str, np.ndarray]],
    action_type: str,
    object_rows: list[dict[str, str]],
) -> int:
    if action_type == "batting":
        bat_peak_frame = _object_peak_frame(object_rows, "bat_speed_norm_s")
        if bat_peak_frame in frames:
            return bat_peak_frame
    joint = "right_hand" if _dominant_side(frames) == "right" else "left_hand"
    speeds = _speeds_by_frame(frames, joint)
    if not speeds:
        return sorted(frames)[len(frames) // 2]
    ordered = sorted(speeds.items())
    if action_type == "pitching":
        candidates = ordered
    else:
        candidates = ordered[int(len(ordered) * 0.20) :]
    return max(candidates, key=lambda item: item[1])[0]


def _landing_frame(frames: dict[int, dict[str, np.ndarray]], event_frame: int, action_type: str) -> int:
    stride = _ankle_separations(frames)
    if not stride:
        return event_frame
    prior = [(frame, value) for frame, value in stride.items() if frame <= event_frame]
    if not prior:
        prior = list(stride.items())
    if action_type == "pitching":
        return max(prior, key=lambda item: item[1])[0]
    max_sep = max(value for _, value in prior)
    threshold = max_sep * 0.9
    for frame, value in sorted(prior):
        if value >= threshold:
            return frame
    return max(prior, key=lambda item: item[1])[0]


def _dominant_side(frames: dict[int, dict[str, np.ndarray]]) -> str:
    left = max(_speeds_by_frame(frames, "left_hand").values() or [0.0])
    right = max(_speeds_by_frame(frames, "right_hand").values() or [0.0])
    return "right" if right >= left else "left"


def _lead_side(frames: dict[int, dict[str, np.ndarray]], frame: int) -> str:
    direction = _stride_direction(frames)
    pts = frames[frame]
    left = _project_horizontal(pts["left_ankle"])
    right = _project_horizontal(pts["right_ankle"])
    return "left" if float(np.dot(left, direction)) >= float(np.dot(right, direction)) else "right"


def _joint_angle(frame: dict[str, np.ndarray], a: str, b: str, c: str) -> float | None:
    try:
        return _angle(frame[a] - frame[b], frame[c] - frame[b])
    except KeyError:
        return None


def _arm_abduction(frame: dict[str, np.ndarray], side: str) -> float | None:
    try:
        upper_arm = frame[f"{side}_elbow"] - frame[f"{side}_shoulder"]
        torso = frame["neck"] - frame["hip"]
    except KeyError:
        return None
    return _angle(upper_arm, torso)


def _trunk_tilt(frame: dict[str, np.ndarray]) -> float | None:
    try:
        torso = frame["neck"] - frame["hip"]
    except KeyError:
        return None
    vertical = np.array([0.0, 1.0, 0.0])
    return _angle(torso, vertical)


def _hip_shoulder_sep(frame: dict[str, np.ndarray]) -> float | None:
    hip = _segment_yaw(frame, "left_hip", "right_hip")
    shoulder = _segment_yaw(frame, "left_shoulder", "right_shoulder")
    if hip is None or shoulder is None:
        return None
    return abs(_signed_delta(hip, shoulder))


def _hip_yaw(frame: dict[str, np.ndarray]) -> float | None:
    return _segment_yaw(frame, "left_hip", "right_hip")


def _segment_yaw(frame: dict[str, np.ndarray], a: str, b: str) -> float | None:
    try:
        vec = frame[b] - frame[a]
    except KeyError:
        return None
    return math.degrees(math.atan2(float(vec[2]), float(vec[0])))


def _stride_angle(frame: dict[str, np.ndarray]) -> float | None:
    try:
        feet = _project_horizontal(frame["right_ankle"] - frame["left_ankle"])
        hips = _project_horizontal(frame["right_hip"] - frame["left_hip"])
    except KeyError:
        return None
    angle = _angle2(feet, hips)
    if angle is None:
        return None
    return min(angle, 180.0 - angle)


def _stride_length_ratio(frames: dict[int, dict[str, np.ndarray]], landing_frame: int) -> float | None:
    try:
        pts = frames[landing_frame]
        stride = float(np.linalg.norm(_project_horizontal(pts["right_ankle"] - pts["left_ankle"])))
    except KeyError:
        return None
    height = _body_height_proxy(frames)
    if height is None or height <= 0:
        return None
    return stride / height


def _weight_transfer(frames: dict[int, dict[str, np.ndarray]], event_frame: int) -> float | None:
    direction = _stride_direction(frames)
    frame_ids = sorted(frames)
    early = frame_ids[: max(1, len(frame_ids) // 10)]
    try:
        start = np.mean([_project_horizontal(frames[i]["hip"]) for i in early], axis=0)
        event = _project_horizontal(frames[event_frame]["hip"])
    except KeyError:
        return None
    stride_len = max(_ankle_separations(frames).values() or [0.0])
    if stride_len <= 1e-9:
        return None
    return max(0.0, min(100.0, float(np.dot(event - start, direction) / stride_len * 100.0)))


def _head_stability(frames: dict[int, dict[str, np.ndarray]]) -> float | None:
    direction = _stride_direction(frames)
    perpendicular = np.array([-direction[1], direction[0]])
    heads = []
    for pts in frames.values():
        if "head" in pts and "hip" in pts:
            heads.append(_project_horizontal(pts["head"] - pts["hip"]))
    if len(heads) < 2:
        return None
    projected = [float(np.dot(pt, perpendicular)) for pt in heads]
    drift = max(projected) - min(projected)
    stride_len = max(_ankle_separations(frames).values() or [0.0])
    if stride_len <= 1e-9:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - drift / stride_len)))


def _foot_direction(frame: dict[str, np.ndarray], lead_side: str, stride_direction: np.ndarray) -> float | None:
    try:
        foot = _project_horizontal(frame[f"{lead_side}_foot"] - frame[f"{lead_side}_ankle"])
    except KeyError:
        return None
    angle = _angle2(foot, stride_direction)
    if angle is None:
        return None
    return min(angle, 180.0 - angle)


def _wrist_snap_proxy(frames: dict[int, dict[str, np.ndarray]], event_frame: int, side: str) -> float | None:
    frame_ids = sorted(frames)
    if event_frame not in frame_ids:
        return None
    idx = frame_ids.index(event_frame)
    before = frame_ids[max(0, idx - 2)]
    after = frame_ids[min(len(frame_ids) - 1, idx + 2)]
    a = _joint_angle(frames[before], f"{side}_elbow", f"{side}_wrist", f"{side}_hand")
    b = _joint_angle(frames[after], f"{side}_elbow", f"{side}_wrist", f"{side}_hand")
    if a is None or b is None:
        return None
    return abs(b - a)


def _joint_speed(frames: dict[int, dict[str, np.ndarray]], joint: str, frame: int) -> float | None:
    speeds = _speeds_by_frame(frames, joint)
    return speeds.get(frame)


def _speeds_by_frame(frames: dict[int, dict[str, np.ndarray]], joint: str) -> dict[int, float]:
    result: dict[int, float] = {}
    frame_ids = sorted(frames)
    for prev, cur in zip(frame_ids, frame_ids[1:]):
        if joint not in frames[prev] or joint not in frames[cur]:
            continue
        dt = _timestamp_delta(prev, cur, frame_ids)
        result[cur] = float(np.linalg.norm(frames[cur][joint] - frames[prev][joint]) / dt)
    return result


def _timestamp_delta(prev: int, cur: int, frame_ids: list[int]) -> float:
    del prev, cur, frame_ids
    return 1.0 / 30.0


def _stride_direction(frames: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
    separations = _ankle_separations(frames)
    if not separations:
        return np.array([1.0, 0.0])
    frame = max(separations, key=separations.get)
    pts = frames[frame]
    vec = _project_horizontal(pts["right_ankle"] - pts["left_ankle"])
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-9:
        return np.array([1.0, 0.0])
    direction = vec / norm
    hip_delta = _hip_displacement_direction(frames)
    if hip_delta is not None and float(np.dot(direction, hip_delta)) < 0:
        direction = -direction
    return direction


def _hip_displacement_direction(frames: dict[int, dict[str, np.ndarray]]) -> np.ndarray | None:
    frame_ids = sorted(frames)
    window = max(1, len(frame_ids) // 10)
    early = [frames[i]["hip"] for i in frame_ids[:window] if "hip" in frames[i]]
    late = [frames[i]["hip"] for i in frame_ids[-window:] if "hip" in frames[i]]
    if not early or not late:
        return None
    delta = _project_horizontal(np.mean(late, axis=0) - np.mean(early, axis=0))
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-9:
        return None
    return delta / norm


def _ankle_separations(frames: dict[int, dict[str, np.ndarray]]) -> dict[int, float]:
    result = {}
    for frame, pts in frames.items():
        if "left_ankle" in pts and "right_ankle" in pts:
            result[frame] = float(np.linalg.norm(_project_horizontal(pts["right_ankle"] - pts["left_ankle"])))
    return result


def _body_height_proxy(frames: dict[int, dict[str, np.ndarray]]) -> float | None:
    values = []
    for pts in frames.values():
        if "head" not in pts:
            continue
        lows = [pts[name] for name in ("left_foot", "right_foot", "left_ankle", "right_ankle") if name in pts]
        if lows:
            values.append(max(float(np.linalg.norm(pts["head"] - low)) for low in lows))
    return median(values) if values else None


def _rotation_range(frames: dict[int, dict[str, np.ndarray]], getter) -> float | None:
    values = [getter(frame) for frame in frames.values()]
    values = [value for value in values if value is not None]
    if not values:
        return None
    unwrapped = np.unwrap(np.radians(values))
    return float(np.degrees(np.max(unwrapped) - np.min(unwrapped)))


def _object_peak(rows: list[dict[str, str]], field: str) -> float | None:
    values = [_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _object_peak_frame(rows: list[dict[str, str]], field: str) -> int | None:
    best: tuple[float, int] | None = None
    for row in rows:
        value = _float(row.get(field))
        if value is None:
            continue
        try:
            frame = int(row["frame_index"])
        except (KeyError, ValueError):
            continue
        if best is None or value > best[0]:
            best = (value, frame)
    return None if best is None else best[1]


def _object_at_frame(rows: list[dict[str, str]], frame: int, field: str) -> float | None:
    for row in rows:
        try:
            row_frame = int(row["frame_index"])
        except (KeyError, ValueError):
            continue
        if row_frame == frame:
            return _float(row.get(field))
    return None


def _object_at_peak(rows: list[dict[str, str]], peak_field: str, target_field: str) -> float | None:
    best: tuple[float, float] | None = None
    for row in rows:
        peak = _float(row.get(peak_field))
        target = _float(row.get(target_field))
        if peak is None or target is None:
            continue
        if best is None or peak > best[0]:
            best = (peak, target)
    return None if best is None else best[1]


def _project_horizontal(vec: np.ndarray) -> np.ndarray:
    return np.array([float(vec[0]), float(vec[2])])


def _angle(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return None
    cosine = max(-1.0, min(1.0, float(np.dot(a, b) / denom)))
    return math.degrees(math.acos(cosine))


def _angle2(a: np.ndarray, b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return None
    cosine = max(-1.0, min(1.0, float(np.dot(a, b) / denom)))
    return math.degrees(math.acos(cosine))


def _signed_delta(a: float, b: float) -> float:
    return (b - a + 180.0) % 360.0 - 180.0


def _at(frames: dict[int, dict[str, np.ndarray]], frame: int, func) -> float | None:
    return func(frames[frame])


def _xyz(record: Pose3DRecord) -> np.ndarray:
    return np.array([record.x_3d, record.y_3d, record.z_3d], dtype=float)


def _float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _metric(
    clip_id: str,
    action_type: str,
    name: str,
    value: float | None,
    unit: str,
    availability: str,
    source: str,
    event_frame: int,
    reason: str,
) -> MetricRow:
    if value is None:
        return _unavailable(clip_id, action_type, name, reason)
    return MetricRow(
        clip_id=clip_id,
        action_type=action_type,
        metric_name=name,
        value=f"{value:.3f}",
        unit=unit,
        availability=availability,
        source=source,
        event_frame=str(event_frame),
        reason=reason,
    )


def _unavailable(clip_id: str, action_type: str, name: str, reason: str) -> MetricRow:
    return MetricRow(clip_id, action_type, name, "", "", "unavailable", "none", "", reason)


def _write_csv(path: Path, rows: Iterable[MetricRow]) -> None:
    fieldnames = list(MetricRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _suspicious_output_lines(rows: list[MetricRow]) -> list[str]:
    lines: list[str] = []
    for row in rows:
        value = _float(row.value)
        if row.metric_name == "Weight Transfer" and value is not None and value in {0.0, 100.0}:
            lines.append(
                f"- {row.clip_id}: Weight Transfer is {row.value}%, likely saturated by the current COM/stride proxy."
            )
        if row.metric_name in {"Lead Knee Angle", "Stride Angle", "Stride Length", "Foot Direction"} and row.event_frame == "0":
            lines.append(
                f"- {row.clip_id}: {row.metric_name} uses frame 0 as landing frame, so this landing-phase metric is weak."
            )
        if row.metric_name == "Wrist/Hand Speed" and value is not None and value < 0.5:
            lines.append(
                f"- {row.clip_id}: Wrist/Hand Speed is {row.value} 3d_unit/s at bat normalized-speed frame, indicating event mismatch."
            )
        if row.metric_name == "Attack Angle" and value is not None and abs(value) > 45.0:
            lines.append(
                f"- {row.clip_id}: Attack Angle is {row.value} deg from image-plane bat tracking; this is not a credible true attack angle."
            )
    if not lines:
        lines.append("- No rule-based suspicious outputs were detected.")
    return lines


def _write_markdown(path: Path, rows: list[MetricRow]) -> None:
    by_clip: dict[str, list[MetricRow]] = {}
    for row in rows:
        by_clip.setdefault(row.clip_id, []).append(row)
    lines = [
        "# SlyMask Benchmark Metrics",
        "",
        "Source decision: body kinematics use GVHMR/SMPL24 3D because hip/shoulder rotation, trunk tilt, stride direction, and COM shift are angle/view dependent in 2D. Bat/ball metrics use the existing 2D object pipeline because there is no 3D bat/ball reconstruction.",
        "",
        "Capability boundary: SlyMask-style percentile and reliability scores require a proprietary or population reference distribution. This report outputs raw values or proxy values and marks unsupported outputs explicitly.",
        "",
        "## Preliminary Conclusions",
        "",
        "### Trustworthy enough for first-pass comparison",
        "",
        "- Hip-Shoulder Sep: geometric definition is clear in 3D, using projected hip and shoulder lines. It is suitable for relative comparison between clips, but still depends on GVHMR orientation stability.",
        "- Lead Knee Angle / Elbow Bend / Arm Abduction / Trunk Tilt: these are direct joint or torso angles from SMPL24. They have clear geometry and are the most defensible body metrics in the current pipeline.",
        "- Hip Rotation: usable as a 3D pelvis-yaw range, especially for within-clip or same-camera comparisons.",
        "- Stride Length: usable as a height-normalized foot-separation proxy. It is not exactly SlyMask's proprietary definition but the biomechanical meaning is clear.",
        "- Ball Speed in pitching: usable only as 2D px/s for tracking QA and relative comparison inside the same video setup. It is not a physical speed.",
        "",
        "### Usable only as proxy",
        "",
        "- Head Stability: uses root-relative head drift against an inferred stride direction. It is useful for automation experiments, but should not be treated as a validated coaching score yet.",
        "- Stride Angle / Foot Direction: the outputs are geometric, but landing-frame and toe-direction inference are approximate. SMPL24 has a foot marker, not a real toe orientation model.",
        "- Swing Speed / Estimated Bat Speed / Attack Angle: current values come from the 2D object tracker. They are useful for debugging bat tracking, but without calibration they cannot be reported as km/h or true 3D attack angle.",
        "- Wrist Snap / Fingertip Speed: SMPL24 has wrist/hand joints but no fingertip joints, so these are hand/wrist proxies only.",
        "",
        "### Clearly unreasonable or not actionable yet",
        "",
        "- SlyMask-style percentiles and reliability percentages cannot be reproduced from our pipeline alone because we do not have their reference population or reliability model.",
        "- Contact Time is unavailable for the current batting benchmark because there is no ball track and no bat-ball impact event detector.",
        "- Weight Transfer is now marked unavailable. The GVHMR global hip/root track is not a calibrated field-coordinate COM trajectory, so previous 0%/100% values were a calculation-definition problem, not a trustworthy biomechanics finding.",
        "- `benchmark_hit_horizontal_06` reports Wrist/Hand Speed near zero at the bat peak-speed frame, which means bat peak and body wrist-speed event are not aligned. That metric is not reliable for this clip without better contact/release event logic.",
        "",
        "### Motion-phase handling caveat",
        "",
        "- The current script does not perform full phase segmentation. It uses event proxies: pitching release is dominant-hand peak speed; pitching landing is the maximum ankle-separation frame before release; batting contact is normalized bat-speed peak when a bat track exists; batting landing is the first frame before the event where ankle separation reaches 90% of its pre-event maximum.",
        "- Range-style body metrics now use the landing-to-event phase window instead of the full clip, so non-batting ending motion such as the running segment in `benchmark_hit_horizontal_06` is not included in Hip Rotation or Head Stability.",
        "- That means preparation or ending frames can still leak into metrics when the clip starts late, ends late, or the object/body peak-speed proxy does not match the real biomechanical event.",
        "- Phase-dependent metrics should be upgraded with explicit phase classifiers before being used as coaching-grade outputs: front-foot landing, max external rotation/acceleration, release/contact, and follow-through.",
        "",
        "### Concrete suspicious outputs in this run",
        "",
        *_suspicious_output_lines(rows),
        "",
    ]
    for clip_id, clip_rows in by_clip.items():
        lines.append(f"## {clip_id}")
        lines.append("")
        lines.append("| metric | value | unit | status | source | frame | reason |")
        lines.append("|---|---:|---|---|---|---:|---|")
        for row in clip_rows:
            value = row.value if row.value else "N/A"
            frame = row.event_frame if row.event_frame else "N/A"
            lines.append(
                f"| {row.metric_name} | {value} | {row.unit} | {row.availability} | {row.source} | {frame} | {row.reason} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
