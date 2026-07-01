# Agent Notes

This repository already has reusable pipeline entry points. Before adding a new
script, check [docs/pipeline_playbook.md](docs/pipeline_playbook.md) and
`src/baseball_pose/cli.py`.

## Research Product Goal

The research goal is to generate professional motion-capture-grade baseball
biomechanics assessment reports and action-improvement plans for domestic
baseball clubs. Treat this as the product direction when making technical,
reporting, visualization, and LLM-prompt decisions.

Primary audience and language:

- Reports must be Chinese-first because the collaboration target is domestic
  baseball clubs in China.
- English terms are acceptable for technical precision, but any English content
  shown to coaches, parents, or athletes needs a corresponding Chinese note.
- Avoid unsupported medical or biomechanics certainty. Label proxy metrics,
  limitations, and data quality clearly in Chinese.

Report product tiers:

- Basic report: readable visual summary, key gap metrics, major risk/technique
  findings, and concise improvement recommendations.
- Professional report: coach-vs-athlete quantitative comparison, phase-specific
  biomechanics metrics, curves, overlays, kinetic-chain timing, LLM-generated
  training plan, and injury-prevention notes.
- Longitudinal archive: annual growth profile with progress curves, repeated
  metric trends, and action-evolution videos across sessions.

Every report should answer three questions:

- Where is the child different from the coach? Show intuitive visuals such as
  skeleton overlays, side-by-side frames, 3D views, joint/segment curves, and
  phase-aligned comparisons.
- How much is the gap? Use measurable values: speed, angle, distance in cm when
  calibrated, normalized distance when not calibrated, timing difference,
  frame-by-frame deviation, percentages, and confidence/availability labels.
- How should the athlete improve? Use LLM-assisted interpretation to produce
  concrete drill suggestions, training priorities, cue language, and
  injury-prevention guidance grounded in the available metrics.

Priority quantitative outputs include:

- Skeleton-overlay videos and 3D skeleton overlays.
- Release/swing speed, trunk and pelvis rotation, hip-shoulder separation, arm
  slot, elbow/knee/shoulder angles, stride length, stride direction, trunk tilt,
  head stability, and center-of-mass/center-of-pelvis trajectory.
- Bat tracking, ball tracking, bat/ball speed proxies, bat angle curves, joint
  angle curves, kinetic-chain sequence, and phase timing.
- Coach-vs-athlete frame-aligned deviations, percentage differences, and
  automatically generated comparison tables.
- LLM-generated targeted training plan, technique explanation, and
  injury-prevention recommendations.
- Annual growth archive: progress curves plus action-evolution video artifacts.

## Default Rule

Use the CLI first:

```bash
MPLCONFIGDIR=/private/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config <config.yaml> <command>
```

Only add a new script when the task is not expressible as an existing CLI
command plus config. If a new script is unavoidable, document it in
`docs/pipeline_playbook.md` in the same change.

## Workspace Output Rule

All project deliverables, previews, reports, CSVs, PNG/GIF images, OBJ models,
and regenerated Vicon/C3D artifacts must be written under the T7 project
workspace:

```text
/Volumes/T7/DKU/Course/CS 207/final-project/baseball-analysis
```

Do not write preview outputs or handoff artifacts to `/private/tmp`, `/tmp`, the
user home directory, or other local machine paths. Temporary system/cache paths
such as `MPLCONFIGDIR=/private/tmp/baseball_mpl_cache` and
`XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache` are acceptable only for cache
noise and must not be treated as result locations. For quick Vicon/C3D checks,
use a project-local output directory such as:

```text
reports/previews/<task_name>/
reports/assets/vicon_reconstruction/
reports/assets/vicon_reconstruction_models/
```

## Vicon C3D 3D Reconstruction Memory

Use `scripts/run_vicon_c3d_pipeline.py` for the current Vicon 2026 C3D flow.
It wraps metric extraction, all-frame marker export, project-style pose3d CSV
export, key-pose summary export, PNG/GIF/MP4/AVI rendering, and OBJ key-pose
model export.

Rendering conventions from the latest review:

- Key-frame PNGs and videos must use the same figure size, DPI, view, and fixed
  coordinate limits. Videos must not autoscale per frame; only the athlete,
  bat, and trajectory should move.
- Batting key-action videos use the default `0.6s` before / `0.4s` after
  window. Pitching videos use a longer `1.4s` before / `0.4s` after window so
  the pre-pitch front-leg lift is included before the hand-speed peak.
- Keep the visual style calm and report-oriented: white background, light gray
  grid, red body connections, blue body markers, green bat markers/structure,
  and gray dashed bat-head trajectory.
- Do not write marker names on body or bat points. Keep the legend compact and
  limited to bat and bat-head trajectory.
- Treat "points" comments as applying to all markers, including `Bat1-Bat5`,
  not only human body markers.
- For batting, render the bat as a rigid structure connecting `Bat1-Bat5` with
  outer/complete connections; the dashed trajectory is the bat-head (`Bat1`)
  trail.
- Y-axis display recentering should keep the feet near the visual Y center.
  Do not recenter Z just to move the feet; the user specifically asked for Y
  centering.
- MP4 is kept for compatibility, but OpenCV MP4 can tint white backgrounds
  slightly. The MJPG AVI output is the color-accurate video artifact.

## Git Tracking Discipline

When a conversation changes source code, commit those code changes before the
turn is finished so the work is traceable. Keep commits scoped to the task:

- Review `git status --short` before staging.
- Stage only files changed for the current request; do not include unrelated
  dirty files or generated outputs unless the user explicitly asks for them.
- Run the narrowest relevant test or validation command before committing when
  feasible, and mention any skipped validation in the final response.
- Use a concise commit message that names the pipeline or subsystem, for
  example `Improve RTMPose benchmark smoothing` or `Document GVHMR handoff`.
- If a required code change cannot be committed because tests fail, the repo is
  blocked by unrelated worktree changes, or git credentials are unavailable,
  report that explicitly instead of silently leaving code uncommitted.

## High-Frequency Pipelines

- Current project center is RTMPose 2D plus GVHMR 3D, not the older MediaPipe
  full-video path.
- RTMPose benchmark 2D:
  `run-image-proposal-roi -> complete-poses -> smooth-poses -> extract-features`
  with `configs/experiments/rtmpose_benchmark_baseball_1.yaml`.
- GVHMR benchmark 3D:
  `build_gvhmr_input_from_frames.py -> external GVHMR run ->
  export_gvhmr_joints.py -> lift-pose-3d -> smooth-pose-3d ->
  render-overlays-3d` with `configs/experiments/gvhmr_benchmark_baseball_1.yaml`.
- YOLO bat/ball tracking is downstream of the benchmark pose artifacts:
  `detect-objects -> extract-object-features -> render-object-overlays`, or
  `scripts/run_yolo_object_sample.py` for quick detector iterations.
- Use `scripts/benchmark_slymask_metrics.py`,
  `scripts/analyze_vicon_wave_metrics.py`, and
  `scripts/build_realsense_task3_report.py` for the SlyMask/Vicon/RealSense
  comparison artifacts.
- The local four-video and Suzhou 2D pipelines are still documented, but they
  are legacy/reusable paths rather than the latest commit-history focus.

## Object Tracker Operating Memory

Object tracking is a separate subsystem from human 2D pose and 3D pose. Do not
conflate them:

- YOLO object tracking is responsible for baseball equipment only: bat and
  ball.
- RTMPose/GVHMR remain responsible for human skeletons. YOLO must not be used
  as a replacement for 2D or 3D human pose.
- The quick object sample runner intentionally passes no pose CSV. This proves
  bat/ball tracking can run independently from pose.
- Pose priors are optional and disabled by default for the recent YOLO object
  work. Keep `use_pose_priors=False` unless the task explicitly asks for
  pose-guided equipment tracking.

Mac execution rule:

- The user's Mac has Metal GPU. For YOLO object runs, prefer
  `.venv_gvhmr310` and confirm `YOLO device: mps`.
- The sandbox may report MPS unavailable even when the outer environment can
  use it. If a YOLO benchmark/sample must use Metal, run the sample command
  with approval outside the sandbox instead of silently falling back to CPU.
- Keep the cache variables set to avoid permission noise:
  `MPLCONFIGDIR=/private/tmp/baseball_mpl_cache`,
  `XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache`, and
  `YOLO_CONFIG_DIR=/private/tmp/ultralytics_config`.

Reusable object sample command shape:

```bash
PYTHONPATH=src \
MPLCONFIGDIR=/private/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/private/tmp/baseball_xdg_cache \
YOLO_CONFIG_DIR=/private/tmp/ultralytics_config \
.venv_gvhmr310/bin/python scripts/run_yolo_object_sample.py \
  --clip-id <clip_id> --frames <n> --detect ball|bat|both
```

Important benchmark examples:

- `benchmark_pitch_vertical_09 --detect ball --frames 53`
- `benchmark_pitch_vertical_10 --detect ball`
- `benchmark_hit_vertical_02 --detect bat`
- `benchmark_hit_horizontal_06 --detect both`
- Suzhou arbitrary clip:
  `--source-data-dir data_full/suzhou_all_2d --clip-id suzhou_img_8159 --frames 355 --detect both`

Do not trust record counts alone. Always inspect the object CSV and at least a
contact sheet or overlay frames when tuning. For ball tracking, list
`frame_index`, normalized `x/y`, confidence, and source. For bat tracking,
inspect angle, length, frame-to-frame jumps, and second differences.

Recent object-tracker corrections to preserve:

- `benchmark_pitch_vertical_09`: pure COCO YOLO `sports ball` produced obvious
  false positives and wrong interpolation. Large-jump filtering now removes
  track segments that teleport across the frame. Conservative small-ball
  assistance can extend YOLO seed tracks, but it should not start tracks from
  the full frame by itself.
- Small-ball detector behavior is intentionally seeded by YOLO and prediction
  constrained. If it is allowed to search the full image, it will grab white
  uniforms, gloves, hats, chalk lines, and fence highlights.
- Bat tracking should not be globally over-smoothed during a swing. The current
  strategy is: strong smoothing outside a swing window, but within the swing
  window use bat prior fine tuning: continuous/unwrapped angle, smoothed center,
  constrained bat length, and bounded angle step.
- For batting clips such as `suzhou_img_8159`, ball detection can be valid only
  over a short high-confidence span. Do not mistake absence outside that span
  for a pose problem.
- For `suzhou_img_8159`, the stable ball span observed after YOLO plus
  filtering is around frames 298-311; bat smoothing must preserve fast contact
  motion rather than turning the swing into a delayed line.

Common output paths for quick samples:

```text
data_full/benchmark_yolo_object_sample/processed/objects/<clip_id>/yolo_object_sample.csv
outputs_full/benchmark_yolo_object_sample/object_overlays/<clip_id>__yolo_object_sample.mp4
outputs_full/benchmark_yolo_object_sample/object_overlays/frames/<clip_id>/yolo_object_sample/
```

Recent object-tracker commit sequence worth knowing:

- `3f64ec9 Add optional YOLO equipment tracking`
- `697a341 Add YOLO object sample runner`
- `ede4324 Refine YOLO bat tracking with ROI line fitting`
- `e62e1d7 Support paired YOLO bat and ball samples`
- `0168a69 Allow YOLO sample runner on arbitrary clips`
- `dd92e15 Filter isolated YOLO ball detections`
- `acc671f Reacquire YOLO ball tracks for batting samples`
- `f5c7e69 Smooth YOLO bat object tracks`
- `a9261b4 Strengthen bat track smoothing`
- `61e7041 Preserve fast bat motion near contact`
- `a4316cc Adapt bat smoothing to swing speed`
- `065247d Use swing windows for bat smoothing`
- `b43c640 Filter large jumps in YOLO ball tracks`
- `e6c5cdb Add small ball detector to extend YOLO tracks`

## Script Hygiene

- `scripts/make_figures.py`, `scripts/prepare_clips.py`,
  `scripts/run_baseline.py`, `scripts/run_experiment.py`, and
  `scripts/summarize_results.py` are thin wrappers around `baseball_pose.cli`.
  Prefer direct CLI commands in documentation and automation.
- Keep generated outputs under the configured `data_dir` and `output_dir`.
- Keep raw videos, model weights, external GVHMR outputs, and rendered reports
  out of git unless explicitly needed as small documentation artifacts.
