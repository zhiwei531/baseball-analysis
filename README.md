# baseball-analysis

Reproducible 2D baseball pose-analysis pipeline for generating readable full-video baseball posture analysis from local batting and pitching clips.

The current repo is still 2D-first, but a staged 3D extension plan now exists
in [docs/3d_extension_plan.md](docs/3d_extension_plan.md). The intended
expansion path is `smoothed 2D pose CSV -> temporal 3D lifting -> 3D features
-> readable report visuals`, not direct world-coordinate reconstruction.

## Current Scope

The current best full-video pipeline follows this path:

```text
raw video
  -> frame sampling
  -> skeleton-free image proposal ROI/mask
  -> MediaPipe Pose on the masked crop
  -> remap keypoints to full-frame coordinates
  -> temporal smoothing
  -> posture feature CSVs
  -> posture/wrist figures and overlay videos
```

The image proposal is intentionally project-specific. It uses center priors, contrast enhancement, frame difference, MOG2 foreground, GrabCut, connected-component scoring, optical-flow center tracking, and temporal mask stabilization. For the current videos, clip-specific hardcoded overrides are enabled when a shared prior is too strict:

```text
batting_1: shared prior plus right-only lower-body widening
batting_2: shared prior plus modest symmetric lower-body widening
pitching_1: wider action envelope plus wider lower-body envelope
pitching_2: left-shifted start prior plus wider lower-body envelope
```

The current report condition is `image_center_motion_grabcut_pose_smooth`. Earlier baseline, pose-prior, center-prior, and body-mask conditions remain in the repo as ablation history and fallback comparisons, but they are not the final report path.

The first experiment uses four local raw videos:

- `raw/batting-1.mov`
- `raw/batting-2.mov`
- `raw/pitching-1.mov`
- `raw/pitching-2.mov`

Raw videos are intentionally ignored by git because they are large binary inputs. Their paths are tracked in `data/metadata/clips.csv`.

## Structure

```text
configs/              YAML configs for runs and ablations
data/metadata/        clip metadata, difficulty labels, manual ROI placeholders
src/baseball_pose/    reusable pipeline package
scripts/              thin command-line wrappers
tests/                contract and utility tests
```

## Reusable Pipeline Playbook

Before creating another script, check
[docs/pipeline_playbook.md](docs/pipeline_playbook.md). Most repeated work is
already covered by `src/baseball_pose/cli.py` plus an experiment config. For
future agent sessions, [AGENTS.md](AGENTS.md) records the same rule and the
current high-frequency pipeline map.

## Output Location Rule

All generated project artifacts must stay inside the T7 project workspace:

```text
/Volumes/T7/DKU/Course/CS 207/final-project/baseball-analysis
```

This includes Vicon/C3D CSVs, report previews, PNG/GIF reconstructions, OBJ
models, HTML/PDF/PPTX exports, and any user-facing inspection outputs. Do not
write deliverables to `/private/tmp`, `/tmp`, the user home directory, or other
local machine paths. Cache-only environment variables such as
`MPLCONFIGDIR=/private/tmp/baseball_mpl_cache` and
`XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache` are allowed only for library
cache files, not for result artifacts.

For quick previews, use project-local paths such as:

```text
reports/previews/<task_name>/
reports/assets/vicon_reconstruction/
reports/assets/vicon_reconstruction_models/
```

The current project focus is RTMPose 2D plus GVHMR 3D:

```text
RTMPose: run-image-proposal-roi
  -> complete-poses
  -> smooth-poses
  -> extract-features

GVHMR: build/export external 3D
  -> lift-pose-3d
  -> smooth-pose-3d
  -> render-overlays-3d
```

Use `configs/experiments/rtmpose_benchmark_baseball_1.yaml` for the 2D backbone
and `configs/experiments/gvhmr_benchmark_baseball_1.yaml` for the 3D handoff.
YOLO bat/ball tracking is documented as a downstream benchmark/report layer.
The older local four-video and Suzhou 2D flows are still documented in the
playbook, but they are no longer the active center of the project.

## Benchmark HTML Report

The current Chinese benchmark report is Vicon-first. Raw body metrics, curves,
source tables, and reconstruction GIFs are generated from `../vicon_2026`
C3D files. The current main athlete is `bryan`; `green` is used only as the
coach-section comparison. The older benchmark CV/GVHMR video metrics are no
longer mixed into the report body. Existing coach data is still used temporarily
as a pitch reference line.

For the complete report build guide, including data selection, C3D metric
calculation, reconstruction GIF generation, and known limitations, see
[REPORT_README.md](REPORT_README.md).

The HTML/PDF/PPTX export workflow is documented in the same guide. Current PDF
export preserves the natural `report.html` layout by slicing the rendered HTML
vertically into A4 pages; it should not repack individual cards into a new grid.
Detailed layout rules for the report and exported artifacts live in
[DESIGN.md](DESIGN.md).

Entry points:

```bash
.venv312/bin/python scripts/run_vicon_c3d_pipeline.py --input-dir ../vicon_2026
.venv312/bin/python scripts/build_benchmark_report_html.py
npm run export:report
```

Julian/Coach batting metrics standalone artifacts use a narrower Vicon C3D
workflow centered on `reports/vicon_2026_julian_coach/`. It reuses the C3D
all-points output, then computes event-based Ready/Contact batting metrics,
renders Ready/Contact GIFs, renders frame-by-frame velocity annotated 3D GIFs,
renders Vicon-valued geometry annotations on the aligned 2D skeleton video,
builds a metrics-only HTML section, and can export a pitching-template-style
Excel workbook:

```bash
.venv312/bin/python scripts/build_batting_dashboard_metrics.py \
  --points reports/vicon_2026_julian_coach/vicon_2026_points_all.csv \
  --out reports/vicon_2026_julian_coach/batting_dashboard_metrics.csv \
  --wide-out reports/vicon_2026_julian_coach/batting_dashboard_metrics_wide.csv \
  --ready-valid-start-frame 770

MPLCONFIGDIR=/private/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache \
.venv312/bin/python scripts/build_julian_coach_event_gifs.py \
  --metrics reports/vicon_2026_julian_coach/batting_dashboard_metrics.csv

MPLCONFIGDIR=/private/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache \
.venv312/bin/python scripts/build_julian_coach_annotated_speed_gifs.py \
  --metrics reports/vicon_2026_julian_coach/batting_dashboard_metrics.csv \
  --points reports/vicon_2026_julian_coach/vicon_2026_point_summary.csv

.venv312/bin/python scripts/build_julian_coach_metrics_section.py \
  --metrics reports/vicon_2026_julian_coach/batting_dashboard_metrics.csv \
  --out reports/vicon_2026_julian_coach/julian_coach_metrics_section.html

node scripts/build_batting_metrics_xlsx.mjs
```

After the aligned 2D overlay video has been generated in the sibling
`srs_2d_video_report_package_20260702_194156` package, rebuild the 2D geometry
annotation stills with:

```bash
.venv312/bin/python ../srs_2d_video_report_package_20260702_194156/render_vicon_geometry_metrics_on_2d.py
```

This writes Ready/Contact PNGs and an event preview MP4 under:

```text
../srs_2d_video_report_package_20260702_194156/outputs/julian_bat_2d_vicon_alignment/vicon_geometry_metric_annotations/
reports/vicon_2026_julian_coach/assets/vicon_2d_geometry_annotations/
```

The batting metrics are not trial-wide averages. The script first detects the
actual swing segment from Bat1 speed, chooses Ready Position from low-speed
raised-bat frames before the swing, and chooses Contact Position from the
lowest Bat1_Z frames inside the detected swing. See
[docs/pipeline_playbook.md](docs/pipeline_playbook.md) for the current metric
definitions, formulas, and output contract.

The frame-level Vicon exports `vicon_2026_points_all.csv` and
`vicon_2026_pose3d.csv`, report zip files, and `output/` export folders are
local/generated artifacts and are ignored by Git because they are large or
duplicative. Regenerate them from the C3D inputs when rebuilding the report.

For the 2D geometry annotation images, all displayed values must come from the
Vicon metrics CSV/Excel; the 2D skeleton is only a visual scaffold. The current
alignment uses `video_frame = event_frame + (vicon_time - event_time) * 240`,
with Vicon bat-speed peak frame 854 aligned to video frame 184. Ready uses
Vicon frame 772 metrics on an early-video fallback visual frame because the
strict mapped Ready frame is outside the visible overlay window. Contact uses
Vicon frame 853 on video frame 182, selected because the skeleton fit is better
than the nominal center frame. In the HTML section, these PNGs sit directly
below the Ready/Contact section titles and are sized roughly to the left
three-card metric grid; the right side remains the Julian event GIF.

Inputs:

```text
../vicon_2026/*/*.c3d
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
data_full/coach_pose3d/gvhmr/pitch_horizontal_coach.csv
output/data/benchmark_pitch_vertical_09_vs_pitch_horizontal_coach_metrics.csv
```

Generated report artifacts:

```text
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
report.html
```

`run_vicon_c3d_pipeline.py` includes a lightweight C3D reader for the local
Vicon exports. It ignores macOS `._*.c3d` resource-fork files, extracts marker
labels, frame rate, 3D marker points, bat markers, and full-body model points
when present, then writes report-ready optical reference metrics, all-frame
marker CSVs, a project-style pose3d CSV, key-pose summaries, key-frame PNGs,
key-action-window GIF/MP4/AVI videos, and OBJ key-pose models. Do not use
global trial-wide point averages for reconstruction figures: pitching uses the
hand-speed peak frame, and batting uses the bat-speed peak frame, with a short
local window around that event. Videos use fixed coordinate limits so the grid
and axes do not scale per frame; only the athlete, bat, and trajectory move.
Batting videos use `0.6s` before / `0.4s` after the bat-speed peak; pitching
videos use `1.4s` before / `0.4s` after the hand-speed peak so the front-leg
lift before stride down is included.
Vicon sample names are taken directly from the `../vicon_2026/{sample}/`
folder name. MP4 is retained for compatibility; MJPG AVI is the
color-accurate video artifact when the white background must match the PNG.

Report design rules live in [DESIGN.md](DESIGN.md). Keep the final HTML Chinese,
use compact graph x-axes, avoid text/curve overlap, and convert user-facing
speed values into common units such as kilometers per hour.

Maintenance rule: when a conversation changes source code, make a focused git
commit for those code changes before finishing the turn. Stage only files that
belong to the current request, and do not mix unrelated dirty outputs into the
commit. See [AGENTS.md](AGENTS.md) for the working rule future agents should
follow.

For the Suzhou batch, still prefer the existing wrapper:

```bash
scripts/run_suzhou_full_2d.sh
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

On the external-drive macOS setup, MediaPipe Tasks currently works under Python 3.12 but fails under Python 3.13 with an OpenGL service initialization error. Use the Python 3.12 venv for pose runs:

```bash
/opt/anaconda3/bin/python3.12 -m venv .venv312
.venv312/bin/python -m pip install --timeout 120 -e ".[dev]"
```

MediaPipe Tasks also needs a local pose model file. The default configuration uses
the Heavy model:

```bash
mkdir -p models
curl -L \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task \
  -o models/pose_landmarker_heavy.task
```

The Lite model can still be used for faster comparison runs by overriding
`pose.model_asset_path`:

```bash
curl -L \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task \
  -o models/pose_landmarker_lite.task
```

The `.task` file is ignored by git.

Optional RTMPose runs use RTMLib and ONNXRuntime rather than the full
MMPose/MMCV stack:

```bash
.venv312/bin/python -m pip install -e ".[rtmpose]"
```

The current RTMPose test config expects a local ONNX model at
`models/rtmpose-x-coco17.onnx`.

## Initial Checks

```bash
python -m compileall src
python -m pytest
baseball-pose validate-config
baseball-pose plan
baseball-pose run-motion-preview --clip-id batting_1 --max-frames 60
baseball-pose run-baseline --clip-id batting_1 --max-frames 30
baseball-pose run-auto-roi --clip-id batting_1 --max-frames 30
baseball-pose run-pose-prior-roi --clip-id batting_1 --max-frames 30
```

For MediaPipe baseline on this machine:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/default.yaml run-baseline --clip-id batting_1 --max-frames 30
```

Full-video runs use a separate config and separate output roots so the 30-frame experiment outputs are preserved:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml run-baseline

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml run-pose-prior-roi

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml run-center-prior-roi

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml run-body-prior-mask-roi

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml smooth-poses

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml extract-features

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml make-figures

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml render-overlays

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml render-body-mask-debug --condition body_prior_mask_roi

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml render-image-proposal-debug

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml run-image-proposal-roi

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml smooth-poses --condition image_center_motion_grabcut_pose
```

Full-video outputs are written under:

```text
data_full/
outputs_full/
```

Full-video feature and figure outputs are written to:

```text
data_full/processed/poses/<clip_id>/<condition_id>_smooth.csv
data_full/processed/features/<clip_id>/<condition_id>.csv
outputs_full/figures/<clip_id>__wrist_trajectories.png
outputs_full/figures/<clip_id>__posture_analysis.png
outputs_full/overlays/<clip_id>__<condition_id>_smooth.mp4
outputs_full/body_mask_debug/<clip_id>__body_prior_mask_roi__proposal_overlay.mp4
outputs_full/body_mask_debug/<clip_id>__body_prior_mask_roi__masked_frame.mp4
outputs_full/image_proposal_debug/<clip_id>__image_center_motion_grabcut__proposal_overlay.mp4
outputs_full/image_proposal_debug/<clip_id>__image_center_motion_grabcut__masked_frame.mp4
data_full/processed/poses/<clip_id>/image_center_motion_grabcut_pose.csv
data_full/processed/poses/<clip_id>/image_center_motion_grabcut_pose_smooth.csv
outputs_full/overlays/<clip_id>__image_center_motion_grabcut_pose_smooth.mp4
```

The current optimization path prioritizes the best readable output over baseline comparison:

```text
image_center_motion_grabcut_pose CSV
  -> torso-continuity gate
  -> short-gap interpolation
  -> median filter
  -> Savitzky-Golay smoothing
  -> moving-average refinement
  -> smoothed feature CSVs, figures, and overlay videos
```

The center-prior and body-prior-mask ROI conditions are retained as ablation history. The center-prior ROI assumes the athlete is centered in the raw videos; the body-prior mask uses an earlier smoothed skeleton to black out pixels outside the subject. These were useful stepping stones, but the current final path avoids using a potentially wrong skeleton as the proposal source.

For debugging proposals without trusting any skeleton, `render-image-proposal-debug` writes the pure OpenCV proposal videos. This proposal is wired into `run-image-proposal-roi`; the current best full-video outputs for all four clips are `image_center_motion_grabcut_pose_smooth`. The image proposal supports clip-specific hardcoded overrides under `conditions.image_center_motion_grabcut_pose.roi.clip_overrides`; `pitching_1` uses a wider action envelope for the pitching arm and leg-lift window, while `pitching_2` starts with a left-shifted, wider prior because the pitcher is not centered at the beginning of the clip. Pitching clips use `lower_body_width_ratio` to widen the lower-body envelope separately from the upper body so extended legs and feet are not cropped. Batting clips can also override the lower-body envelope: `batting_1` widens only the right lower side to preserve the right shin/foot without admitting more left-lower background people, while `batting_2` uses a modest symmetric lower-body widening.

The feature CSV includes report-oriented 2D posture proxies that can be computed from the current skeleton-only data:

- joint angles for elbows, shoulders, and knees,
- pelvis and shoulder line rotation,
- hip-shoulder separation,
- pelvis and trunk rotation velocity,
- approximate center-of-mass path from torso and lower-body landmarks,
- knee extension from the first valid frame and knee angular velocity,
- wrist speed and a max-wrist hand-speed proxy.

These are intentionally limited to 2D pose-derived quantities. Ball, bat, release, impact, MER, and SFC-specific metrics require event labels or extra object tracking before they should be reported as exact biomechanics.

## Planned 3D Extension

The repo now includes minimal 3D scaffolding so the architecture can grow
without breaking the current 2D workflow:

```text
src/baseball_pose/pose3d/
src/baseball_pose/features3d/
src/baseball_pose/visualization3d/
src/baseball_pose/pipeline/pose3d.py
```

Two new CLI commands expose the planned handoff point:

```bash
baseball-pose plan-3d --condition image_center_motion_grabcut_pose_smooth
baseball-pose lift-pose-3d --condition image_center_motion_grabcut_pose_smooth
```

At this stage, `plan-3d` prints the intended 2D-to-3D artifact flow and
`lift-pose-3d` runs a first real backend based on MediaPipe world landmarks for
supported source conditions. This is a relative 3D skeleton path, not calibrated
world-coordinate biomechanics.

Motion preview outputs are written to:

```text
data/interim/frames/<clip_id>/motion_preview/
data/interim/frames/<clip_id>/motion_preview.csv
outputs/motion_preview/<clip_id>__motion_preview.mp4
outputs/motion_preview/frames/<clip_id>/
```

Baseline pose outputs are written to:

```text
data/interim/frames/<clip_id>/baseline_raw/
data/interim/frames/<clip_id>/baseline_raw.csv
data/processed/poses/<clip_id>/baseline_raw.csv
outputs/overlays/<clip_id>__baseline_raw.mp4
outputs/overlays/frames/<clip_id>/baseline_raw/
```

Auto ROI pose outputs are written to:

```text
data/interim/frames/<clip_id>/auto_roi_raw/
data/interim/frames/<clip_id>/auto_roi_raw.csv
data/interim/rois/<clip_id>/auto_roi_raw.csv
data/processed/poses/<clip_id>/auto_roi_raw.csv
outputs/roi_debug/<clip_id>__auto_roi_raw.mp4
outputs/roi_debug/frames/<clip_id>/
outputs/overlays/<clip_id>__auto_roi_raw.mp4
outputs/overlays/frames/<clip_id>/auto_roi_raw/
```

Pose-prior ROI uses `baseline_raw` pose CSV files to estimate a tighter athlete crop:

```text
data/interim/frames/<clip_id>/auto_roi_pose_prior/
data/interim/frames/<clip_id>/auto_roi_pose_prior.csv
data/interim/rois/<clip_id>/auto_roi_pose_prior.csv
data/processed/poses/<clip_id>/auto_roi_pose_prior.csv
outputs/roi_debug/<clip_id>__auto_roi_pose_prior.mp4
outputs/overlays/<clip_id>__auto_roi_pose_prior.mp4
```

## Conditions

Current final report condition:

- `image_center_motion_grabcut_pose_smooth`

Important ablation and historical conditions:

- `baseline_raw`
- `baseline_raw_smooth`
- `auto_roi_pose_prior`
- `auto_roi_pose_prior_smooth`
- `center_prior_roi`
- `center_prior_roi_smooth`
- `body_prior_mask_roi`
- `body_prior_mask_roi_smooth`
- `image_center_motion_grabcut_pose`

Implementation should keep all outputs tagged by `clip_id` and `condition_id` so processed and baseline results remain comparable.
