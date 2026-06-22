from __future__ import annotations

import argparse
import csv
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DEFAULT_INPUT = PROJECT_ROOT / "vicon_2026"
DEFAULT_METRICS = ROOT / "reports" / "vicon_2026_metrics.csv"
DEFAULT_POINTS = ROOT / "reports" / "vicon_2026_point_summary.csv"


@dataclass
class C3DTrial:
    path: Path
    labels: list[str]
    points: np.ndarray
    rate_hz: float
    units: str


def _read_record_params(data: bytes, param_block: int) -> dict[tuple[str, str], bytes]:
    pos = (param_block - 1) * 512 + 4
    groups: dict[int, str] = {}
    params: dict[tuple[str, str], bytes] = {}
    while pos < len(data):
        name_len = struct.unpack_from("b", data, pos)[0]
        if name_len == 0:
            break
        group_id = struct.unpack_from("b", data, pos + 1)[0]
        size = abs(name_len)
        name = data[pos + 2 : pos + 2 + size].decode("latin1").strip()
        offset = struct.unpack_from("<h", data, pos + 2 + size)[0]
        payload_start = pos + 2 + size + 2
        next_pos = pos + 2 + size + offset
        if group_id < 0:
            groups[-group_id] = name
        else:
            group = groups.get(group_id, str(group_id))
            params[(group, name)] = data[payload_start:next_pos]
        if offset == 0 or next_pos <= pos:
            break
        pos = next_pos
    return params


def _decode_param(payload: bytes) -> object:
    dtype = struct.unpack_from("b", payload, 0)[0]
    dim_count = payload[1]
    dims = list(payload[2 : 2 + dim_count])
    offset = 2 + dim_count
    count = 1
    for dim in dims:
        count *= max(1, dim)
    raw = payload[offset : offset + count * abs(dtype)]
    if dtype == -1:
        width = dims[0] if dims else count
        n = dims[1] if len(dims) > 1 else 1
        return [raw[i * width : (i + 1) * width].decode("latin1").strip() for i in range(n)]
    if dtype == 2:
        return struct.unpack("<" + "h" * count, raw)
    if dtype == 4:
        return struct.unpack("<" + "f" * count, raw)
    return raw


def read_c3d(path: Path) -> C3DTrial:
    data = path.read_bytes()
    param_block = data[0]
    header = data[:512]
    point_count = struct.unpack_from("<H", header, 2)[0]
    first_frame = struct.unpack_from("<H", header, 6)[0]
    last_frame = struct.unpack_from("<H", header, 8)[0]
    data_start = struct.unpack_from("<H", header, 16)[0]
    rate_hz = struct.unpack_from("<f", header, 20)[0]
    scale = struct.unpack_from("<f", header, 12)[0]
    frame_count = last_frame - first_frame + 1
    params = _read_record_params(data, param_block)
    labels = _decode_param(params[("POINT", "LABELS")])
    units = (_decode_param(params.get(("POINT", "UNITS"), b"\xff\x01\x02mm")) or ["mm"])[0]
    start = (data_start - 1) * 512
    if scale < 0:
        values = np.frombuffer(data, dtype="<f4", count=frame_count * point_count * 4, offset=start)
        points = values.reshape(frame_count, point_count, 4).astype(float)
    else:
        values = np.frombuffer(data, dtype="<i2", count=frame_count * point_count * 4, offset=start)
        points = values.reshape(frame_count, point_count, 4).astype(float)
        points[:, :, :3] *= scale
    invalid = (points[:, :, 3] < 0) | np.isclose(np.abs(points[:, :, :3]).sum(axis=2), 0)
    points[:, :, :3][invalid] = np.nan
    return C3DTrial(path=path, labels=list(labels), points=points, rate_hz=rate_hz, units=str(units))


def clean_label(label: str) -> str:
    return label.split(":", 1)[-1].strip()


def safe_nanmean(arrays: list[np.ndarray] | np.ndarray, axis: int = 0) -> np.ndarray:
    stacked = np.stack(arrays) if isinstance(arrays, list) else arrays
    valid = np.isfinite(stacked)
    counts = valid.sum(axis=axis)
    total = np.nansum(stacked, axis=axis)
    return np.divide(total, counts, out=np.full_like(total, np.nan, dtype=float), where=counts > 0)


def marker(trial: C3DTrial, *names: str) -> np.ndarray:
    clean = [clean_label(label) for label in trial.labels]
    series = []
    for name in names:
        if name in clean:
            series.append(trial.points[:, clean.index(name), :3])
    if not series:
        return np.full((trial.points.shape[0], 3), np.nan)
    return safe_nanmean(series, axis=0)


def speed_kmh(points_mm: np.ndarray, rate_hz: float) -> np.ndarray:
    diff_m = np.diff(points_mm, axis=0) / 1000.0
    speed = np.linalg.norm(diff_m, axis=1) * rate_hz * 3.6
    return np.concatenate([[np.nan], speed])


def angle_series(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    dot = np.einsum("ij,ij->i", ba, bc)
    cos_v = np.clip(dot / denom, -1.0, 1.0)
    out = np.degrees(np.arccos(cos_v))
    out[~np.isfinite(out)] = np.nan
    return out


def plane_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    vec = a - b
    return np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))


def circular_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180) % 360 - 180)


def trunk_tilt_deg(hip: np.ndarray, neck: np.ndarray) -> np.ndarray:
    vec = neck - hip
    horizontal = np.linalg.norm(vec[:, :2], axis=1)
    vertical = np.abs(vec[:, 2])
    out = np.degrees(np.arctan2(horizontal, vertical))
    out[~np.isfinite(out)] = np.nan
    return out


def finite_stat(values: np.ndarray, fn: str) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if fn == "max":
        return float(np.nanmax(finite))
    if fn == "median":
        return float(np.nanmedian(finite))
    if fn == "p95":
        return float(np.nanpercentile(finite, 95))
    return float(np.nanmean(finite))


def active_duration_sec(speed: np.ndarray, rate_hz: float) -> float:
    peak = finite_stat(speed, "max")
    if not math.isfinite(peak) or peak <= 0:
        return float("nan")
    mask = np.isfinite(speed) & (speed >= peak * 0.3)
    active = np.where(mask)[0]
    if active.size == 0:
        return float("nan")
    peak_idx = int(np.nanargmax(speed))
    start = peak_idx
    while start > 0 and mask[start - 1]:
        start -= 1
    end = peak_idx
    while end + 1 < len(mask) and mask[end + 1]:
        end += 1
    return float((end - start + 1) / rate_hz)


def infer_action(path: Path) -> str:
    return "batting" if "bat" in path.name.lower() else "pitching"


def trial_id(path: Path) -> str:
    return f"{path.parent.name}_{path.stem}".replace(" ", "_").lower()


def compute_trial_metrics(trial: C3DTrial) -> dict[str, object]:
    action = infer_action(trial.path)
    athlete = trial.path.parent.name
    lfhd = marker(trial, "LFHD")
    rfhd = marker(trial, "RFHD")
    lbhd = marker(trial, "LBHD")
    rbhd = marker(trial, "RBHD")
    head = np.nanmean(np.stack([lfhd, rfhd, lbhd, rbhd]), axis=0)
    lsho, rsho = marker(trial, "LSHO"), marker(trial, "RSHO")
    lelb, relb = marker(trial, "LELB"), marker(trial, "RELB")
    lwrist = marker(trial, "LWRA", "LWRB")
    rwrist = marker(trial, "RWRA", "RWRB")
    lhip = marker(trial, "LASI", "LPSI")
    rhip = marker(trial, "RASI", "RPSI")
    lkne, rkne = marker(trial, "LKNE"), marker(trial, "RKNE")
    lank, rank = marker(trial, "LANK"), marker(trial, "RANK")
    c7, t10, clav, strn = marker(trial, "C7"), marker(trial, "T10"), marker(trial, "CLAV"), marker(trial, "STRN")
    shoulder_mid = np.nanmean(np.stack([lsho, rsho]), axis=0)
    hip_mid = np.nanmean(np.stack([lhip, rhip]), axis=0)
    neck = np.nanmean(np.stack([c7, clav]), axis=0)
    trunk = np.nanmean(np.stack([c7, t10, clav, strn]), axis=0)
    hip_angle = plane_angle(lhip, rhip)
    shoulder_angle = plane_angle(lsho, rsho)
    hip_shoulder = circular_abs_diff(shoulder_angle, hip_angle)
    left_knee = angle_series(lhip, lkne, lank)
    right_knee = angle_series(rhip, rkne, rank)
    right_elbow = angle_series(rsho, relb, rwrist)
    left_elbow = angle_series(lsho, lelb, lwrist)
    trunk_tilt = trunk_tilt_deg(hip_mid, neck)
    hand = rwrist if finite_stat(speed_kmh(rwrist, trial.rate_hz), "max") >= finite_stat(speed_kmh(lwrist, trial.rate_hz), "max") else lwrist
    hand_speed = speed_kmh(hand, trial.rate_hz)
    trunk_speed = speed_kmh(trunk, trial.rate_hz)
    hip_speed = speed_kmh(hip_mid, trial.rate_hz)
    bat1 = marker(trial, "Bat1")
    bat5 = marker(trial, "Bat5")
    bat_mid = safe_nanmean([bat1, bat5], axis=0)
    bat_speed = speed_kmh(bat1 if np.isfinite(bat1).any() else bat_mid, trial.rate_hz)
    bat_vec_angle = np.degrees(np.arctan2((bat1 - bat5)[:, 2], np.linalg.norm((bat1 - bat5)[:, :2], axis=1)))
    frame_count, point_count = trial.points.shape[:2]
    valid_points = np.isfinite(trial.points[:, :, 0])
    valid_ratio = float(valid_points.sum() / valid_points.size) if valid_points.size else float("nan")
    return {
        "trial_id": trial_id(trial.path),
        "athlete": athlete,
        "action_type": action,
        "source_file": str(trial.path.relative_to(PROJECT_ROOT)),
        "frames": frame_count,
        "rate_hz": trial.rate_hz,
        "duration_sec": frame_count / trial.rate_hz,
        "point_count": point_count,
        "valid_point_pct": valid_ratio * 100,
        "hip_shoulder_sep_deg": finite_stat(hip_shoulder, "p95"),
        "lead_knee_angle_deg": finite_stat(left_knee if action == "pitching" else right_knee, "median"),
        "right_elbow_angle_deg": finite_stat(right_elbow, "median"),
        "left_elbow_angle_deg": finite_stat(left_elbow, "median"),
        "trunk_tilt_deg": finite_stat(trunk_tilt, "p95"),
        "hand_speed_kmh": finite_stat(hand_speed, "max"),
        "trunk_speed_kmh": finite_stat(trunk_speed, "max"),
        "hip_speed_kmh": finite_stat(hip_speed, "max"),
        "bat_speed_kmh": finite_stat(bat_speed, "max"),
        "swing_time_sec": active_duration_sec(bat_speed, trial.rate_hz),
        "bat_angle_deg": finite_stat(bat_vec_angle, "median"),
    }


def point_summary_rows(trial: C3DTrial) -> list[dict[str, object]]:
    rows = []
    clean = [clean_label(label) for label in trial.labels]
    keep = [
        "LFHD",
        "RFHD",
        "C7",
        "T10",
        "LSHO",
        "RSHO",
        "LELB",
        "RELB",
        "LWRA",
        "LWRB",
        "RWRA",
        "RWRB",
        "LASI",
        "RASI",
        "LKNE",
        "RKNE",
        "LANK",
        "RANK",
        "Bat1",
        "Bat5",
    ]
    for name in keep:
        if name not in clean:
            continue
        pts = trial.points[:, clean.index(name), :3]
        valid = np.isfinite(pts[:, 0])
        if not valid.any():
            continue
        mid = pts[valid][len(pts[valid]) // 2]
        rows.append(
            {
                "trial_id": trial_id(trial.path),
                "athlete": trial.path.parent.name,
                "action_type": infer_action(trial.path),
                "point": name,
                "valid_pct": float(valid.sum() / valid.size * 100),
                "mid_x_mm": float(mid[0]),
                "mid_y_mm": float(mid[1]),
                "mid_z_mm": float(mid[2]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report-ready metrics from vicon_2026 C3D exports.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--points-out", type=Path, default=DEFAULT_POINTS)
    args = parser.parse_args()
    c3d_paths = sorted(path for path in args.input_dir.glob("*/*.c3d") if not path.name.startswith("._"))
    metric_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    for path in c3d_paths:
        trial = read_c3d(path)
        metric_rows.append(compute_trial_metrics(trial))
        point_rows.extend(point_summary_rows(trial))
    write_csv(args.metrics_out, metric_rows)
    write_csv(args.points_out, point_rows)
    print(args.metrics_out)
    print(args.points_out)


if __name__ == "__main__":
    main()
