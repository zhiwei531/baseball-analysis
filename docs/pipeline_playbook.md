# Baseball Analysis Pipeline Playbook

This runbook is based on the current code and recent git history, not
`log.md`. The current project center is RTMPose for robust 2D pose and GVHMR
for 3D pose/lifting. YOLO bat/ball tracking and SlyMask/Vicon/RealSense reports
are downstream benchmark/report layers, not the core pipeline.

```text
RTMPose benchmark 2D pose
  -> pose completion and smoothing
  -> GVHMR 3D handoff/export
  -> 3D smoothing and overlays
  -> optional YOLO bat/ball tracking
  -> benchmark/report artifacts
```

Use this file before creating a new script. Most repeated work should be a
config plus `src/baseball_pose/cli.py`; standalone scripts should be reserved
for external-tool handoff or report generation.

## Environment

```bash
cd "/Volumes/T7/DKU/Course/CS 207/final-project/baseball-analysis"

export MPLCONFIGDIR=/private/tmp/baseball_mpl_cache
export XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

PY=.venv312/bin/python
```

Common flags:

- `--clip-id <id>` can be repeated.
- `--max-frames <n>` is for smoke tests.
- `--condition <id>` can be repeated for commands that read existing artifacts.

Validate before long runs:

```bash
$PY -m baseball_pose.cli --config configs/experiments/rtmpose_benchmark_baseball_1.yaml validate-config
$PY -m baseball_pose.cli --config configs/experiments/gvhmr_benchmark_baseball_1.yaml validate-config
```

## Current Main Pipeline: RTMPose 2D Benchmark

RTMPose is the current 2D pose backbone for benchmark clips. This path creates
the completed and smoothed 2D condition that GVHMR, object tracking, and metric
reports consume.

Main benchmark config:

```text
configs/experiments/rtmpose_benchmark_baseball_1.yaml
```

Benchmark clips:

```text
benchmark_pitch_vertical_10
benchmark_pitch_vertical_09
benchmark_hit_vertical_02
benchmark_hit_horizontal_06
```

Reusable RTMPose 2D sequence:

```bash
$PY -m baseball_pose.cli --config configs/experiments/rtmpose_benchmark_baseball_1.yaml run-image-proposal-roi
$PY -m baseball_pose.cli --config configs/experiments/rtmpose_benchmark_baseball_1.yaml complete-poses --condition image_center_motion_grabcut_pose
$PY -m baseball_pose.cli --config configs/experiments/rtmpose_benchmark_baseball_1.yaml smooth-poses --condition image_center_motion_grabcut_pose_complete
$PY -m baseball_pose.cli --config configs/experiments/rtmpose_benchmark_baseball_1.yaml extract-features --condition image_center_motion_grabcut_pose_complete_smooth
```

Key outputs:

```text
data_full/benchmark_rtmpose_test/processed/poses/<clip_id>/image_center_motion_grabcut_pose_complete_smooth.csv
data_full/benchmark_rtmpose_test/processed/features/<clip_id>/image_center_motion_grabcut_pose_complete_smooth.csv
```

## Current 3D Pipeline: GVHMR Benchmark Handoff

Recent 3D commits added external video-HMR support, GVHMR CSV export, preserved
SMPL joints, 3D smoothing, and 3D overlay rendering.

Config:

```text
configs/experiments/gvhmr_benchmark_baseball_1.yaml
```

Build a GVHMR input video from sampled frames when rotation metadata is unsafe:

```bash
$PY scripts/build_gvhmr_input_from_frames.py \
  --frames-csv data_full/benchmark_rtmpose_test/interim/frames/benchmark_pitch_vertical_09/image_center_motion_grabcut_pose.csv \
  --output output/gvhmr_inputs/benchmark_pitch_vertical_09.mp4 \
  --fps 30
```

Convert external GVHMR results into the project 3D CSV contract:

```bash
$PY scripts/export_gvhmr_joints.py \
  --gvhmr-root external/GVHMR \
  --result path/to/hmr4d_results.pt \
  --clip-id benchmark_pitch_vertical_09 \
  --output data_full/benchmark_rtmpose_test/external_pose3d/gvhmr/benchmark_pitch_vertical_09.csv \
  --face-z
```

Then run configured 3D stages:

```bash
$PY -m baseball_pose.cli --config configs/experiments/gvhmr_benchmark_baseball_1.yaml plan-3d --condition image_center_motion_grabcut_pose_complete_smooth
$PY -m baseball_pose.cli --config configs/experiments/gvhmr_benchmark_baseball_1.yaml lift-pose-3d --condition image_center_motion_grabcut_pose_complete_smooth
$PY -m baseball_pose.cli --config configs/experiments/gvhmr_benchmark_baseball_1.yaml smooth-pose-3d --condition image_center_motion_grabcut_pose_complete_smooth_3d
$PY -m baseball_pose.cli --config configs/experiments/gvhmr_benchmark_baseball_1.yaml render-overlays-3d --condition image_center_motion_grabcut_pose_complete_smooth_3d_smooth
```

Key outputs:

```text
data_full/benchmark_rtmpose_test/external_pose3d/gvhmr/<clip_id>.csv
data_full/benchmark_rtmpose_test/processed/poses3d/<clip_id>/image_center_motion_grabcut_pose_complete_smooth_3d.csv
data_full/benchmark_rtmpose_test/processed/poses3d/<clip_id>/image_center_motion_grabcut_pose_complete_smooth_3d_smooth.csv
outputs_full/benchmark_rtmpose_test/overlays3d/<clip_id>__image_center_motion_grabcut_pose_complete_smooth_3d_smooth.mp4
```

## Optional Downstream: YOLO Object Tracking

YOLO is useful for bat/ball benchmark and report signals after the RTMPose 2D
artifacts exist. It should not be treated as the current pose pipeline center.

Full object stage on the RTMPose benchmark condition:

```bash
$PY -m baseball_pose.cli --config configs/experiments/rtmpose_benchmark_baseball_1.yaml detect-objects --condition image_center_motion_grabcut_pose_complete_smooth
$PY -m baseball_pose.cli --config configs/experiments/rtmpose_benchmark_baseball_1.yaml extract-object-features --condition image_center_motion_grabcut_pose_complete_smooth
$PY -m baseball_pose.cli --config configs/experiments/rtmpose_benchmark_baseball_1.yaml render-object-overlays --condition image_center_motion_grabcut_pose_complete_smooth
```

Quick detector sample without rerunning pose:

```bash
$PY scripts/run_yolo_object_sample.py --frames 12
$PY scripts/run_yolo_object_sample.py --clip-id benchmark_hit_vertical_02 --detect both --frames 48
$PY scripts/run_yolo_object_sample.py --clip-id benchmark_pitch_vertical_09 --detect ball --frames 48
```

Recent YOLO behavior includes pose-prior-independent tracking, short-gap
interpolation, isolated ball filtering, large-jump filtering, small-ball
extension, and bat smoothing that preserves fast swing/contact windows.

Object outputs:

```text
data_full/benchmark_rtmpose_test/processed/objects/<clip_id>/image_center_motion_grabcut_pose_complete_smooth.csv
data_full/benchmark_rtmpose_test/processed/object_features/<clip_id>/image_center_motion_grabcut_pose_complete_smooth.csv
outputs_full/benchmark_rtmpose_test/object_overlays/<clip_id>__image_center_motion_grabcut_pose_complete_smooth.mp4
```

### YOLO Object Sample Iteration Workflow

Use the sample runner for detector iterations when pose artifacts already
exist. It is intentionally pose-independent and should be used to prove object
tracking is separate from human 2D/3D pose.

Use MPS on the user's Mac:

```bash
PYTHONPATH=src \
MPLCONFIGDIR=/private/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache \
YOLO_CONFIG_DIR=/private/tmp/ultralytics_config \
.venv_gvhmr310/bin/python scripts/run_yolo_object_sample.py \
  --clip-id benchmark_pitch_vertical_09 \
  --frames 53 \
  --detect ball
```

Expected first line for a correct Mac GPU run:

```text
YOLO device: mps
```

Useful sample commands:

```bash
# Pitching ball only
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/baseball_mpl_cache XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache YOLO_CONFIG_DIR=/private/tmp/ultralytics_config \
.venv_gvhmr310/bin/python scripts/run_yolo_object_sample.py \
  --clip-id benchmark_pitch_vertical_09 --frames 53 --detect ball

# Batting bat only
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/baseball_mpl_cache XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache YOLO_CONFIG_DIR=/private/tmp/ultralytics_config \
.venv_gvhmr310/bin/python scripts/run_yolo_object_sample.py \
  --clip-id benchmark_hit_vertical_02 --frames 220 --detect bat

# Batting bat and ball on benchmark
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/baseball_mpl_cache XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache YOLO_CONFIG_DIR=/private/tmp/ultralytics_config \
.venv_gvhmr310/bin/python scripts/run_yolo_object_sample.py \
  --clip-id benchmark_hit_horizontal_06 --frames 220 --detect both

# Batting bat and ball on Suzhou arbitrary clip
PYTHONPATH=src MPLCONFIGDIR=/private/tmp/baseball_mpl_cache XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache YOLO_CONFIG_DIR=/private/tmp/ultralytics_config \
.venv_gvhmr310/bin/python scripts/run_yolo_object_sample.py \
  --source-data-dir data_full/suzhou_all_2d \
  --clip-id suzhou_img_8159 \
  --frames 355 \
  --detect both
```

Sample outputs:

```text
data_full/benchmark_yolo_object_sample/processed/objects/<clip_id>/yolo_object_sample.csv
outputs_full/benchmark_yolo_object_sample/object_overlays/<clip_id>__yolo_object_sample.mp4
outputs_full/benchmark_yolo_object_sample/object_overlays/frames/<clip_id>/yolo_object_sample/
```

Quick CSV inspection:

```bash
.venv312/bin/python - <<'PY'
import csv, collections
from pathlib import Path
p = Path("data_full/benchmark_yolo_object_sample/processed/objects/<clip_id>/yolo_object_sample.csv")
rows = list(csv.DictReader(p.open()))
print("rows", len(rows))
print("by_object", dict(collections.Counter(r["object_name"] for r in rows)))
print("by_source", dict(collections.Counter(r["source"] for r in rows)))
for r in rows:
    print(r["frame_index"], r["object_name"], r["x"], r["y"], r["confidence"], r["source"])
PY
```

Do not treat counts as success. Always verify:

- Ball path is temporally and spatially continuous. Watch for jumps such as
  center -> far right -> far left.
- Interpolation did not bridge unrelated false positives.
- Bat line forms a plausible swing arc during contact and stays stable outside
  the swing window.
- `source` values identify whether a point came from YOLO, temporal
  interpolation, or small-ball extension.

### Object Tracker Debugging Recipes

Create a contact sheet from rendered overlay frames:

```bash
.venv312/bin/python - <<'PY'
from pathlib import Path
import cv2, numpy as np

clip_id = "<clip_id>"
base = Path("outputs_full/benchmark_yolo_object_sample/object_overlays/frames") / clip_id / "yolo_object_sample"
idxs = [0, 8, 16, 24, 32, 40, 48]
imgs = []
for idx in idxs:
    p = base / f"{clip_id}__yolo_object_sample__frame_{idx:06d}.png"
    im = cv2.imread(str(p))
    if im is None:
        continue
    im = cv2.resize(im, (320, 180))
    cv2.putText(im, f"frame {idx}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(im, f"frame {idx}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    imgs.append(im)
while imgs and len(imgs) % 4:
    imgs.append(np.zeros_like(imgs[0]))
canvas = np.vstack([np.hstack(imgs[i:i+4]) for i in range(0, len(imgs), 4)])
out = Path("/private/tmp") / f"{clip_id}_object_sheet.png"
cv2.imwrite(str(out), canvas)
print(out)
PY
```

Compute bat jitter diagnostics:

```bash
.venv312/bin/python - <<'PY'
import csv, math
from pathlib import Path
p = Path("data_full/benchmark_yolo_object_sample/processed/objects/<clip_id>/yolo_object_sample.csv")
rows = [r for r in csv.DictReader(p.open()) if r["object_name"] == "bat"]
rows.sort(key=lambda r: int(r["frame_index"]))
barrel, handle = [], []
for r in rows:
    w, h = float(r["width"]), float(r["height"])
    barrel.append((float(r["x"]) * w, float(r["y"]) * h))
    handle.append((float(r["x2"]) * w, float(r["y2"]) * h))
def mean_second(points):
    vals = [math.hypot(c[0]-2*b[0]+a[0], c[1]-2*b[1]+a[1]) for a,b,c in zip(points, points[1:], points[2:])]
    return sum(vals) / len(vals), max(vals)
print("barrel_second_mean_max", tuple(round(x, 2) for x in mean_second(barrel)))
print("handle_second_mean_max", tuple(round(x, 2) for x in mean_second(handle)))
PY
```

Print contact/swing angles:

```bash
.venv312/bin/python - <<'PY'
import csv, math
from pathlib import Path
p = Path("data_full/benchmark_yolo_object_sample/processed/objects/<clip_id>/yolo_object_sample.csv")
rows = [r for r in csv.DictReader(p.open()) if r["object_name"] == "bat"]
rows.sort(key=lambda r: int(r["frame_index"]))
prev = None
for r in rows:
    f = int(r["frame_index"])
    if f < 290 or f > 320:
        continue
    w, h = float(r["width"]), float(r["height"])
    barrel = (float(r["x"]) * w, float(r["y"]) * h)
    handle = (float(r["x2"]) * w, float(r["y2"]) * h)
    angle = math.degrees(math.atan2(barrel[1]-handle[1], barrel[0]-handle[0]))
    length = math.hypot(barrel[0]-handle[0], barrel[1]-handle[1])
    delta = ""
    if prev is not None:
        d = angle - prev
        while d > 180: d -= 360
        while d < -180: d += 360
        delta = f" d_angle={d:.1f}"
    print(f, f"angle={angle:.1f}", f"len={length:.1f}", delta)
    prev = angle
PY
```

### Known Object-Tracking Corrections

Object tracker vs pose tracker:

- Keep object tracker and human skeleton tracker independent.
- YOLO object tracking should not be used for human 2D or 3D pose.
- The sample runner passes no pose CSV by design. This is expected.

MPS vs CPU:

- If `.venv_gvhmr310` reports `YOLO device: mps`, the run is using Metal.
- If the sandbox reports MPS unavailable, rerun the YOLO sample with approval
  outside the sandbox. Do not silently accept CPU for benchmark iterations.

Ball tracking:

- COCO YOLO `sports ball` is weak on small, fast baseballs. It can detect only
  a few frames and can also hallucinate white uniforms, gloves, hats, chalk
  lines, and fence highlights.
- Use large-jump filtering to split teleporting detections before interpolation.
- Use short-track filtering to drop isolated false positives.
- Small-ball detector is an assistant, not a standalone detector. It should
  extend YOLO-seeded tracks only near predicted positions.
- On `benchmark_pitch_vertical_09`, pure YOLO left a conservative seed segment
  around frames 8-11. Small-ball extension has been used conservatively to add
  later frames, but visual review is still required.

Bat tracking:

- YOLO bat boxes should be treated as ROI proposals. The tracker refines the
  bat with line fitting inside the ROI.
- A globally smoothed bat can lag behind real contact motion and stop looking
  like a swing arc.
- Current bat smoothing uses swing windows: outside the window it strongly
  smooths jitter; inside the window it fine-tunes with baseball-bat priors:
  smoothed center, continuous/unwrapped angle, constrained bat length, and
  bounded angle step.
- For `suzhou_img_8159`, ball was observed around frames 298-311. Bat contact
  motion must preserve the fast angle sweep rather than flattening it.

## SlyMask-Style Benchmark Metrics

Recent commits refined phase selection and clarified trust boundaries for
SlyMask-like metrics. Use the existing script:

```bash
$PY scripts/benchmark_slymask_metrics.py \
  --data-dir data_full/benchmark_rtmpose_test \
  --output-dir reports
```

Outputs:

```text
reports/slymask_benchmark_metrics.csv
reports/slymask_benchmark_metrics.md
```

Interpretation rule: the script deliberately marks unsupported SlyMask metrics
as `proxy` or `unavailable`; do not turn those into definitive biomechanics
claims without new calibrated inputs or a reference population.

## Vicon / RealSense / Deck Reporting

Recent commits from 2026-06-11 added Vicon metric analysis, RealSense D435 lab
comparison, and the SlyMask/Vicon/RealSense benchmark presentation deck.

Vicon metrics:

```bash
$PY scripts/analyze_vicon_wave_metrics.py \
  --input-dir "../Vicon_Wave_250506(1)" \
  --output-dir reports
```

Outputs:

```text
reports/vicon_slymask_metrics.csv
reports/vicon_slymask_metric_detail.csv
reports/vicon_slymask_metrics.md
reports/figures/vicon_bat_angle_time.png
reports/figures/vicon_speed_time.png
reports/figures/vicon_csv_structure.png
```

Vicon XLSX import path for feature compatibility tests:

```bash
$PY scripts/import_vicon_xlsx.py \
  --xlsx path/to/trial.xlsx \
  --clip-id vicon_trial \
  --condition-id vicon_xy_projected \
  --out data/processed/poses/vicon_trial/vicon_xy_projected.csv
```

Vicon 2026 C3D exports for the current Chinese benchmark HTML report:

```bash
.venv312/bin/python scripts/run_vicon_c3d_pipeline.py --input-dir ../vicon_2026
```

Inputs:

```text
../vicon_2026/bryan/001 Cal 04 Bat 05.c3d
../vicon_2026/bryan/001 Cal 04 Pitch 05.c3d
../vicon_2026/green/006 Bat 04.c3d
../vicon_2026/green/006 Pitch 09.c3d
```

Notes:

- Ignore `../vicon_2026/*/._*.c3d`; those are macOS resource-fork files.
- The script has a small built-in C3D parser for `POINT:LABELS`, frame rate,
  point scale/data start, and float point frames. It does not require `ezc3d`.
- It outputs trial-level optical reference metrics, an all-frame point CSV,
  a project-style 3D pose CSV, and a key-pose point-summary table used by the
  reconstruction renderer to build body/bat 3D PNG screenshots, GIFs, and OBJ
  key-pose model files.
- `reports/vicon_2026_points_all.csv` contains every reconstruction point for
  every frame of every trial, not only the key-action window.
- `reports/vicon_2026_pose3d.csv` mirrors the project 3D CSV contract
  (`clip_id`, `frame_index`, `joint_name`, `x_3d`, `y_3d`, `z_3d`, ...), with
  Vicon marker names used as joint names and millimeter coordinates preserved.
- The point-summary table is not a global point average. The script first
  extracts a key action position, then averages a short local window around
  that frame: pitching uses the dominant hand-speed peak; batting uses the
  bat-speed peak. Use `key_event`, `key_frame_index`, and `key_time_sec` to
  trace every reconstructed point figure.
- `sample_name` is the direct folder name under `../vicon_2026/`; do not invent
  display names for Vicon samples.
- Run `scripts/render_vicon_reconstruction_images.py` before building the HTML.
  The one-command pipeline runs it automatically. It generates key-frame PNGs,
  key-action-window animated GIF/MP4/AVI files, and OBJ model files under
  `reports/assets/vicon_reconstruction_models/`. The report embeds the GIF
  files when available instead of drawing C3D reconstruction figures inline
  with SVG. MP4 is retained for compatibility; MJPG AVI is the color-accurate
  video export when white backgrounds must match the PNG.
- Pitching C3D files include full-body AI/model points; batting C3D files
  include body markers plus `Bat:Bat1` through `Bat:Bat5`.
- Current 3D reconstruction style is fixed for professional report visuals:
  white background, light gray grid, red body connections, blue body markers,
  green bat, and gray dashed bat-head trajectory. Marker names are not drawn
  on body or bat points, and the compact legend only lists the bat and bat-head
  trajectory.
- Animation coordinate axes are fixed once per key-action window. Do not
  autoscale axes per frame. PNG and video exports share figure size/DPI,
  camera view, and coordinate-limit logic; Y-axis centering should keep foot
  markers near the visual Y center without changing Z centering.

Current benchmark HTML report:

```bash
.venv312/bin/python scripts/run_vicon_c3d_pipeline.py
python3 scripts/build_benchmark_report_html.py
```

Report inputs:

```text
reports/slymask_benchmark_metrics.csv
output/data/benchmark_pitch_vertical_09_motion_metrics_full.csv
output/data/benchmark_pitch_vertical_09_vs_pitch_horizontal_coach_metrics.csv
reports/vicon_2026_metrics.csv
reports/vicon_2026_point_summary.csv
reports/vicon_2026_points_all.csv
reports/vicon_2026_pose3d.csv
reports/vicon_2026_key_pose_models.csv
reports/assets/vicon_reconstruction/*.png
reports/assets/vicon_reconstruction/*.gif
reports/assets/vicon_reconstruction/*.mp4
reports/assets/vicon_reconstruction/*.avi
reports/assets/vicon_reconstruction_models/*.obj
data_full/benchmark_rtmpose_test/processed/poses3d/*/image_center_motion_grabcut_pose_complete_smooth_3d_smooth.csv
data_full/coach_pose3d/gvhmr/pitch_horizontal_coach.csv
outputs/manual-20260611-slymask/presentations/slymask-benchmark-deck/assets/*.png
```

Report output:

```text
report.html
```

Design constraints for this report are in `DESIGN.md`: Chinese-only user-facing
copy, compact graph x-axes, no graph text/curve overlap, real graphs whenever
CSV/C3D data can compute them, and no user-facing units such as `px/s` or
`3d_unit/s`.

RealSense report:

```bash
$PY scripts/build_realsense_task3_report.py
```

Outputs:

```text
reports/realsense_d435_lab_comparison.md
reports/realsense_slymask_metric_comparison.csv
reports/figures/realsense_3cam_lab_layout.png
reports/figures/realsense_depth_pointcloud_example.png
```

Presentation deck artifacts live under:

```text
outputs_full/presentations/baseball_slymask_vicon_realsense_benchmark.pptx
outputs_full/presentations/baseball_slymask_vicon_realsense_benchmark_contact_sheet.png
```

## Chinese Pitching Report Template

This script is currently untracked in the worktree but present locally. It
builds a Chinese PDF report template around benchmark pitching artifacts.

```bash
$PY scripts/build_pitch_report_template.py
```

Output convention:

```text
output/pdf/benchmark_pitch_vertical_09_biomech_report_template_zh.pdf
```

Because it is not tracked yet, treat it as local work until it is intentionally
added to git.

## Legacy / Reusable 2D Full-Video Pipelines

These are still useful, but recent commits show they are no longer the newest
active pipeline.

Local four-video config:

```text
configs/experiments/full_video.yaml
```

Final 2D condition:

```text
image_center_motion_grabcut_pose_smooth
```

Reusable sequence:

```bash
$PY -m baseball_pose.cli --config configs/experiments/full_video.yaml run-image-proposal-roi
$PY -m baseball_pose.cli --config configs/experiments/full_video.yaml smooth-poses --condition image_center_motion_grabcut_pose
$PY -m baseball_pose.cli --config configs/experiments/full_video.yaml extract-features --condition image_center_motion_grabcut_pose_smooth
$PY -m baseball_pose.cli --config configs/experiments/full_video.yaml make-figures --condition image_center_motion_grabcut_pose_smooth
$PY -m baseball_pose.cli --config configs/experiments/full_video.yaml render-overlays --condition image_center_motion_grabcut_pose_smooth
```

Suzhou full 2D wrapper:

```bash
scripts/run_suzhou_full_2d.sh
```

Regenerate Suzhou final artifacts after raw pose CSVs already exist:

```bash
scripts/finalize_suzhou_smooth_outputs.sh
```

Suzhou configs:

```text
configs/experiments/full_video_suzhou.yaml
configs/experiments/mediapipe_heavy_suzhou_test.yaml
configs/experiments/rtmpose_suzhou_test.yaml
configs/experiments/gvhmr_suzhou_test.yaml
```

## Generic Report CLI

For feature CSV based report packages:

```bash
$PY -m baseball_pose.cli --config configs/experiments/full_video.yaml build-report-summary --condition image_center_motion_grabcut_pose_smooth
$PY -m baseball_pose.cli --config configs/experiments/full_video.yaml build-report-prompt --condition image_center_motion_grabcut_pose_smooth
$PY -m baseball_pose.cli --config configs/experiments/full_video.yaml generate-llm-report --condition image_center_motion_grabcut_pose_smooth
```

PDF from an existing LLM report package:

```bash
$PY scripts/build_pdf_report.py \
  --project-root "$PWD" \
  --clip-id batting_1 \
  --condition image_center_motion_grabcut_pose_smooth \
  --out output/pdf/batting_1_report.pdf
```

## Script Inventory

| Script | Current role |
| --- | --- |
| `scripts/run_yolo_object_sample.py` | Latest YOLO bat/ball detector sample runner; useful for small fast iterations. |
| `scripts/benchmark_slymask_metrics.py` | SlyMask-style benchmark metrics with explicit proxy/unavailable statuses. |
| `scripts/analyze_vicon_wave_metrics.py` | Vicon swing metric analysis and report figures. |
| `scripts/build_vicon_2026_metrics.py` | C3D parser and Vicon 2026 report metrics/point-summary builder. |
| `scripts/build_benchmark_report_html.py` | Current Chinese benchmark HTML report builder using SlyMask, GVHMR/RTMPose, and Vicon 2026 data. |
| `scripts/build_realsense_task3_report.py` | RealSense D435 comparison report and figures. |
| `scripts/build_pitch_report_template.py` | Local Chinese pitching report template; currently untracked. |
| `scripts/export_gvhmr_joints.py` | External GVHMR `.pt` to project 3D CSV handoff. |
| `scripts/build_gvhmr_input_from_frames.py` | GVHMR input video creation from sampled project frames. |
| `scripts/import_vicon_xlsx.py` | Vicon trajectory XLSX to `baseball_pose` pose CSV. |
| `scripts/run_suzhou_full_2d.sh` | Older but reusable Suzhou full 2D batch loop. |
| `scripts/finalize_suzhou_smooth_outputs.sh` | Regenerate Suzhou smooth/features/figures/overlays after raw pose CSVs exist. |
| `scripts/build_pdf_report.py` | PDF generation from report prompt/LLM artifacts. |
| `scripts/make_figures.py`, `scripts/prepare_clips.py`, `scripts/run_baseline.py`, `scripts/run_experiment.py`, `scripts/summarize_results.py` | Thin wrappers around `baseball_pose.cli`; prefer direct CLI commands. |

## When to Add a New Script

Add a script only when one of these is true:

- It integrates an external tool that cannot be expressed by config and CLI
  options, such as GVHMR export.
- It generates a standalone report or deck artifact.
- It is a stable batch wrapper used repeatedly across many clips.

When adding a script, update this playbook with the command, inputs, outputs,
and whether the script is a reusable pipeline stage or a one-off report helper.
