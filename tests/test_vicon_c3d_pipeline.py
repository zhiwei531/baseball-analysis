from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_vicon_2026_metrics import C3DTrial, all_point_rows, point_summary_rows, pose3d_rows
from render_vicon_reconstruction_images import bat1_trajectory_points, export_key_pose_obj


def _trial(tmp_path: Path) -> C3DTrial:
    path = tmp_path / "bryan" / "001 Bat 01.c3d"
    path.parent.mkdir(exist_ok=True)
    labels = [
        "Body:LFHD",
        "Body:RFHD",
        "Body:C7",
        "Body:RWRA",
        "Body:RWRB",
        "Bat:Bat1",
        "Bat:Bat2",
        "Bat:Bat3",
        "Bat:Bat4",
        "Bat:Bat5",
        "RightKneeAngle",
    ]
    points = np.zeros((3, len(labels), 4), dtype=float)
    for frame in range(points.shape[0]):
        for idx in range(len(labels)):
            points[frame, idx, :3] = (frame * 100 + idx, frame * 50 + idx, 1000 + idx)
            points[frame, idx, 3] = 1.0
    points[:, 5, 0] = [0.0, 300.0, 900.0]
    points[:, 6, 0] = [10.0, 260.0, 760.0]
    points[:, 7, 0] = [20.0, 220.0, 620.0]
    points[:, 8, 0] = [30.0, 180.0, 480.0]
    points[:, 9, 0] = [40.0, 140.0, 340.0]
    return C3DTrial(path=path, labels=labels, points=points, rate_hz=100.0, units="mm")


def _pitch_trial(tmp_path: Path) -> C3DTrial:
    trial = _trial(tmp_path)
    return C3DTrial(
        path=tmp_path / "bryan" / "001 Pitch 01.c3d",
        labels=trial.labels,
        points=trial.points,
        rate_hz=trial.rate_hz,
        units=trial.units,
    )


def test_vicon_all_point_export_contains_every_frame_and_skips_derived_angles(tmp_path):
    rows = all_point_rows(_trial(tmp_path))

    assert len(rows) == 3 * 10
    assert {row["frame_index"] for row in rows} == {0, 1, 2}
    assert "RightKneeAngle" not in {row["point"] for row in rows}
    assert rows[0]["timestamp_sec"] == 0.0
    assert rows[-1]["units"] == "mm"


def test_vicon_pose3d_export_uses_project_csv_contract(tmp_path):
    rows = pose3d_rows(_trial(tmp_path), condition_id="vicon_test")

    assert rows[0]["clip_id"] == "bryan_001_bat_01"
    assert rows[0]["condition_id"] == "vicon_test"
    assert rows[0]["joint_name"] == "LFHD"
    assert rows[0]["scale_mode"] == "vicon_c3d_mm"
    assert rows[0]["lift_backend"] == "vicon_c3d"


def test_key_pose_obj_is_exported_from_key_pose_summary(tmp_path):
    rows = point_summary_rows(_trial(tmp_path))
    out = export_key_pose_obj(rows, tmp_path / "models")

    assert out is not None
    text = out.read_text(encoding="utf-8")
    assert "o key_pose_reconstruction" in text
    assert "# key_event: 球棒峰值速度" in text
    assert "# LFHD" in text


def test_key_pose_summary_does_not_backfill_points_from_unrelated_frames(tmp_path):
    trial = _trial(tmp_path)
    trial = C3DTrial(
        path=trial.path,
        labels=trial.labels,
        points=trial.points,
        rate_hz=1.0,
        units=trial.units,
    )
    bat4_index = [label.split(":", 1)[-1] for label in trial.labels].index("Bat4")
    trial.points[:, bat4_index, :3] = np.nan
    trial.points[0, bat4_index, :3] = (999.0, 999.0, 999.0)

    rows = point_summary_rows(trial)

    assert "Bat4" not in {row["point"] for row in rows}


def test_bat1_trajectory_is_exported_only_for_batting_trials(tmp_path):
    frame_indices = np.array([0, 1, 2])

    batting = bat1_trajectory_points(_trial(tmp_path), frame_indices)
    pitching = bat1_trajectory_points(_pitch_trial(tmp_path), frame_indices)

    assert batting == [(0.0, 5.0, 1005.0), (300.0, 55.0, 1005.0), (900.0, 105.0, 1005.0)]
    assert pitching == []
