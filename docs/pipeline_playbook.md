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
- Window defaults are action-specific: batting uses `0.6s` before / `0.4s`
  after the bat-speed peak; pitching uses `1.4s` before / `0.4s` after the
  hand-speed peak to include the front-leg lift before the stride down phase.

### Julian/Coach Batting Metrics Artifacts

The Julian/Coach batting section is a standalone report slice, not the main
`report.html`. It exists to test a Vicon + bat-rigid-body metrics dashboard
using Julian as the main athlete and Coach as the comparison source.

Main inputs:

```text
reports/vicon_2026_julian_coach/vicon_2026_points_all.csv
reports/vicon_2026_julian_coach/vicon_2026_point_summary.csv
../vicon_2026/julian/007-julian Cal 04 Bat 05.c3d
../vicon_2026/coach/008-coach Cal 03 Bat 02.c3d
```

Build order:

```bash
.venv312/bin/python scripts/build_batting_dashboard_metrics.py \
  --points reports/vicon_2026_julian_coach/vicon_2026_points_all.csv \
  --out reports/vicon_2026_julian_coach/batting_dashboard_metrics.csv \
  --wide-out reports/vicon_2026_julian_coach/batting_dashboard_metrics_wide.csv \
  --ready-valid-start-frame 770

MPLCONFIGDIR=/private/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache \
.venv312/bin/python scripts/build_julian_coach_event_gifs.py \
  --metrics reports/vicon_2026_julian_coach/batting_dashboard_metrics.csv \
  --out-dir reports/vicon_2026_julian_coach/assets/vicon_reconstruction_events

MPLCONFIGDIR=/private/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache \
.venv312/bin/python scripts/build_julian_coach_annotated_speed_gifs.py \
  --metrics reports/vicon_2026_julian_coach/batting_dashboard_metrics.csv \
  --points reports/vicon_2026_julian_coach/vicon_2026_point_summary.csv \
  --out-dir reports/vicon_2026_julian_coach/assets/vicon_reconstruction_annotated

.venv312/bin/python scripts/build_julian_coach_metrics_section.py \
  --metrics reports/vicon_2026_julian_coach/batting_dashboard_metrics.csv \
  --out reports/vicon_2026_julian_coach/julian_coach_metrics_section.html

node scripts/build_batting_metrics_xlsx.mjs

.venv312/bin/python ../srs_2d_video_report_package_20260702_194156/render_vicon_geometry_metrics_on_2d.py
```

Outputs:

```text
reports/vicon_2026_julian_coach/batting_dashboard_metrics.csv
reports/vicon_2026_julian_coach/batting_dashboard_metrics_wide.csv
reports/vicon_2026_julian_coach/julian_coach_metrics_section.html
reports/vicon_2026_julian_coach/assets/vicon_reconstruction_events/{julian,coach}_{ready,contact}.gif
reports/vicon_2026_julian_coach/assets/vicon_reconstruction_annotated/{julian,coach}_speed_annotated.gif
reports/vicon_2026_julian_coach/assets/vicon_2d_geometry_annotations/{ready_position,contact_position}_vicon_geometry_on_2d.png
../srs_2d_video_report_package_20260702_194156/outputs/julian_bat_2d_vicon_alignment/vicon_geometry_metric_annotations/vicon_geometry_metrics_on_2d_events.mp4
outputs/batting_metrics_excel/007-julian Cal 04 Bat 05_batting_report_metrics.xlsx
```

Large frame-level exports (`vicon_2026_points_all.csv`,
`vicon_2026_pose3d.csv`), report zip packages, and `output/` export folders are
local/generated artifacts and are ignored by Git. The repository keeps the
summary CSVs, standalone HTML, key PNG/GIF visual assets, scripts, and build
documentation needed to regenerate them from C3D inputs.

Script roles:

- `scripts/build_batting_dashboard_metrics.py` reads the all-frame Vicon point
  CSV and computes event-based batting metrics. It also writes a wide CSV for
  quick spreadsheet-style inspection.
- `scripts/build_julian_coach_event_gifs.py` renders short Ready and Contact
  GIFs around the event center frame. The HTML currently embeds only Julian's
  Ready/Contact GIFs in the corresponding sections.
- `scripts/build_julian_coach_annotated_speed_gifs.py` renders whole
  reconstruction-window GIFs with left-side metric cards baked into the image.
  These GIF cards show only frame-by-frame instantaneous values, not fixed
  event summary values.
- `scripts/build_julian_coach_metrics_section.py` builds the standalone
  metrics-only HTML section using Julian as the main data source and Coach as
  the comparison source.
- `scripts/build_batting_metrics_xlsx.mjs` is the
  current Excel export helper for the Julian batting metrics. It mirrors the
  three-sheet pitching metrics workbook format: `报告指标`, `事件定位`, and `说明`.
- `../srs_2d_video_report_package_20260702_194156/render_vicon_geometry_metrics_on_2d.py`
  renders Ready/Contact stills and a short preview MP4 from the aligned 2D
  skeleton overlay. It uses the 2D video only as a visual scaffold; displayed
  geometry values are Vicon values from the metrics CSV/Excel.

#### Event Detection Rules

Do not compute these metrics over the full trial or a fixed wall-clock window.
The C3D includes pre-swing walking/waving and post-swing bat drop artifacts, so
the metrics pipeline first isolates the actual batting action:

1. Compute Bat1 frame speed from 3D point differences:
   `speed_kmh = norm(diff(Bat1_xyz_mm) / 1000) * rate_hz * 3.6`.
2. Smooth Bat1 speed with a small moving window.
3. Find the swing peak from the smoothed Bat1 speed.
4. Find the continuous high-speed segment around that peak, then expand by
   approximately `0.15 s` on each side. This is the detected swing segment.
5. Ready Position is a continuous low-speed raised-bat block before the
   detected swing. Current default is 5 frames. For the Julian/Coach Vicon
   extract, pass `--ready-valid-start-frame 770` so pre-action content before
   the actual batting setup cannot be selected.
6. Contact Position is the lowest Bat1_Z event inside the detected swing
   segment. Current default is 5 frames. There is no ball marker, so this is a
   contact proxy rather than true ball contact.
7. The 3D annotated speed GIF ignores Ready/Contact event aggregation and uses
   the reconstruction action window around the bat-speed peak, defaulting to
   `0.6 s` before and `0.4 s` after the peak. Each rendered GIF frame recomputes
   instantaneous Bat1 speed, Attack Angle, and forearm roll speed for that
   frame.

#### 2D Geometry Annotation Event Mapping

The aligned 2D overlay video has a playback FPS of `29.4802`, but the slow
motion capture mapping uses `240 fps`. Do not map by video playback frame rate.
Current mapping:

```text
video_frame = event_frame + (vicon_time - event_time) * 240
event_frame = 184
event_time = Vicon frame 854 / Vicon capture rate
```

Sanity checks:

```text
Vicon frame 844 -> video frame 160
Vicon frame 854 -> video frame 184
Vicon frame 870 -> video frame 222
```

Ready Position currently uses Vicon frames `770;771;772;773;774` with event
center `772`, enforced by `--ready-valid-start-frame 770`. The strict mapped
Ready frame is outside the useful 2D overlay window, so the annotation renderer
uses a stable early-video visual frame as the background while still labeling
the value source as Vicon frame 772. Contact Position uses Vicon frames
`849;850;851;852;853`; the renderer uses Vicon frame `853` / video frame `182`
because that overlay frame has a better skeleton-to-body fit than the nominal
center frame.

The current role mapping assumes a right-handed hitter:

| Annotation | Vicon value source | 2D visual guide |
| --- | --- | --- |
| Ready rear hip flexion | `ready_rear_hip_flexion_deg` at frame 772 | shoulder mid -> right hip -> right knee |
| Ready rear knee flexion | `ready_rear_knee_flexion_deg` at frame 772 | right hip -> right knee -> right ankle |
| Ready hip-shoulder separation | `ready_hip_shoulder_separation_deg` at frame 772 | shoulder line and pelvis line |
| Contact pelvis rotation | `contact_pelvis_rotation_open_deg` at frame 853 | hip line plus rotation arrow |
| Contact torso rotation | `contact_torso_rotation_open_deg` at frame 853 | shoulder line plus rotation arrow |
| Contact front knee flexion | `contact_front_knee_flexion_deg` at frame 853 | left hip -> left knee -> left ankle |

Never swap front/rear leg roles when drawing these guides. The 2D overlay may
have imperfect depth ordering; role assignment comes from batting handedness
and the Vicon metric definition, not from whichever 2D limb appears closer in a
single frame.

#### Batting Metric Definitions

The CSV stores one row per metric per sample and includes `event_name`,
`event_rule`, `event_frame`, `event_frames`, `points_used`, `formula`,
`components_json`, and `notes`. User-facing HTML currently shows 16 metrics;
`coach_hitting_zone_stability_score` may still be present in CSV output but is
hidden from the current dashboard.

Ready Position metrics:

| Metric key | Display metric | Event aggregation | Formula / points |
| --- | --- | --- | --- |
| `ready_com_height_ratio` | 重心高度 | mean over Ready frames | `mean(COM_Z_ready) / height_proxy`; COM uses `CentreOfMass` if present, else `0.6 * hip_mid + 0.4 * trunk_mid`; height uses head markers minus foot markers. |
| `ready_rear_hip_flexion_deg` | 后髋屈曲角 | mean over Ready frames | `180 - angle(shoulder_mid, rear_hip, rear_knee)`; right-handed assumption: rear side is right. |
| `ready_rear_knee_flexion_deg` | 后膝屈曲角 | mean over Ready frames | `180 - angle(rear_hip, rear_knee, rear_ankle)` using right hip/knee/ankle-heel-toe proxy. |
| `ready_hip_shoulder_separation_deg` | 髋肩分离角 | mean over Ready frames | `abs(wrap_to_180(torso_rotation_xy - pelvis_rotation_xy))` using `LSHO/RSHO` and `LASI/RASI/LPSI/RPSI`. |
| `ready_bat_tilt_deg` | 球棒倾角 | mean over Ready frames | `atan2(abs((Bat1 - Bat5)_Z), norm((Bat1 - Bat5)_XY))`; 0 deg is parallel to ground, 90 deg is vertical. |
| `ready_hand_height_ratio` | 握棒手高度 | mean over Ready frames | `mean(grip_hand_center_Z_ready) / height_proxy`; grip hand center is the mean of left/right wrist centers. |

Contact Position metrics:

| Metric key | Display metric | Event aggregation | Formula / points |
| --- | --- | --- | --- |
| `contact_bat_speed_kmh` | 球棒速度 | mean over Contact frames | `mean(norm(diff(Bat1_xyz) / dt)) * 3.6 / 1000`; Contact is the lowest Bat1_Z event inside the detected swing segment, not the speed peak. |
| `contact_attack_angle_deg` | 挥棒路径角（Attack Angle） | mean over Contact frames | `atan2(Bat1_velocity_Z, norm(Bat1_velocity_XY))`; negative means the bat head velocity points downward. |
| `contact_pelvis_rotation_open_deg` | 骨盆旋转角 | mean over Contact frames | `abs(wrap_to_180(pelvis_rotation_xy_contact - mean(pelvis_rotation_xy_ready)))`; display value is direction-normalized opening magnitude. Signed raw event value is stored in `components_json`. |
| `contact_torso_rotation_open_deg` | 躯干旋转角 | mean over Contact frames | `abs(wrap_to_180(torso_rotation_xy_contact - mean(torso_rotation_xy_ready)))`; display value is direction-normalized opening magnitude. Signed raw event value is stored in `components_json`. |
| `contact_front_knee_flexion_deg` | 前膝屈曲角 | mean over Contact frames | `180 - angle(front_hip, front_knee, front_ankle)`; right-handed assumption: front side is left. |
| `ready_to_contact_head_displacement_mm` | 头部位移 | event-to-event distance | `norm(mean(head_center_contact) - mean(head_center_ready))` using `LFHD/RFHD/LBHD/RBHD`. |

Coach Flag metrics:

| Metric key | Display metric | Event aggregation | Formula / points |
| --- | --- | --- | --- |
| `coach_high_com_risk_index` | 重心偏高指数 | Ready composite | `100 * mean(clip((COM_height_ratio - 0.48) / 0.14), clip((35 - rear_hip_flexion) / 35), clip((35 - rear_knee_flexion) / 35))`; higher means taller COM plus straighter rear hip/knee. |
| `coach_rear_elbow_height_diff_mm` | 后肘高度差（掉肘） | mean over Ready frames | `mean(RELB_Z - RSHO_Z)`; negative means rear elbow is below rear shoulder. |
| `coach_bat_loading_angle_to_catcher_deg` | 球棒加载角（引棒不足） | mean over Ready frames | `angle(project_xy(Bat5 - Bat1), catcher_direction)`; catcher direction is inferred as `-project_xy(mean(Bat1_velocity_contact))`. |
| `coach_rollover_forearm_roll_velocity_deg_s` | 手腕翻转角速度（翻腕） | peak over Contact frames | `max(abs(d/dt signed_angle_about_axis(wrist_marker_axis, elbow_to_wrist_axis, global_Z_reference)))` using `RELB/RWRA/RWRB`; this is a forearm pronation proxy. |

Hidden/research metric:

| Metric key | Display metric | Event aggregation | Formula / points |
| --- | --- | --- | --- |
| `coach_hitting_zone_stability_score` | 击球区稳定性 | high-speed hitting zone composite | Uses frames where Bat1 speed is at least 90% of swing-segment peak. Score combines high-speed path length, Attack Angle standard deviation, and barrel-path curvature. Currently not shown in the Julian metrics section or Excel main table. |

#### Visual QA Rules

- Ready/Contact section media should show only Julian event GIFs.
- Ready/Contact 2D geometry PNGs should sit directly below the section title,
  above the metric grid, with width close to the left three-card metric area.
  Do not place these screenshots in the right-side media card.
- For 2D geometry annotations, the value labels are Vicon 3D metrics. The 2D
  skeleton lines are visual guides only and must not be described as the
  measurement source.
- Joint angles should not use angle arcs in the current design. Draw thin limb
  guide lines, keep the dashed complementary extension lines at the actual
  angle center, then draw a thin leader line from the joint/angle center to a
  small, light angle value.
- Rotation metrics use a standard single elliptical arc arrow, not an S-curve.
  The arc should be horizontally flattened, reduced enough to avoid covering
  the athlete, shifted so the arrow head touches or nearly touches the body
  outline, and trimmed so the tail removes roughly the final 20% while the
  existing direction and radius remain unchanged.
- If the mapped Contact frame has a visibly broken 2D skeleton, select another
  frame from the Contact event window with a better skeleton fit while keeping
  Vicon values from the selected Vicon event frame.
- Speed annotations should be baked into the 3D GIF, placed on the left side,
  with readable font size and no overlap with the skeleton, bat, or trajectory.
- Speed annotation cards show only instantaneous values for each frame. Do not
  add the fixed event summary values back into those cards unless the report
  design changes.
- Verify representative frames visually after changing camera limits, card
  size, font size, or action-window sampling.

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
| `scripts/build_batting_dashboard_metrics.py` | Julian/Coach Vicon batting Ready/Contact event detector and dashboard metrics CSV builder. |
| `scripts/build_julian_coach_event_gifs.py` | Short Ready/Contact event GIF renderer for the Julian/Coach metrics section. |
| `scripts/build_julian_coach_annotated_speed_gifs.py` | Whole-window Vicon 3D GIF renderer with baked-in frame-by-frame speed, Attack Angle, and forearm roll annotations. |
| `scripts/build_julian_coach_metrics_section.py` | Standalone Julian-centered, Coach-comparison batting metrics HTML section builder. |
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
