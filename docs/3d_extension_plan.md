# 3D Extension Plan

## Goal

Extend the current repo from a readable 2D baseball pose-analysis pipeline into
a staged `2D -> 3D` system without breaking the current full-video workflow.

The near-term target is **monocular relative 3D skeleton reconstruction**, not
absolute real-world coordinates. In practice, that means recovering a pelvis-
centered 3D joint sequence from the existing cleaned 2D skeletons. This is the
right next step for the repo because the current system already solves the
hardest preconditions for 3D lifting:

- stable frame sampling,
- clip-aware ROI selection,
- postprocessed 2D keypoints,
- action-window selection,
- report-oriented feature extraction.

Trying to jump directly to world-coordinate 3D would force camera calibration,
scale recovery, or multi-view capture into a repo that is currently built around
single-video local analysis. The design below keeps the current strengths and
adds 3D incrementally.

## Existing 2D Boundaries

The current best path is:

```text
raw video
  -> frame sampling
  -> image proposal ROI/mask
  -> 2D pose backend
  -> remap to full frame
  -> temporal smoothing / filtering
  -> 2D feature CSV
  -> figures / overlays / report
```

The important architectural observation is that the repo already has a strong
boundary after smoothing. The smoothed 2D pose CSV is the right handoff point
for 3D. That means the first 3D integration should be:

```text
smoothed 2D pose CSV
  -> temporal 3D lifting
  -> 3D pose CSV
  -> 3D feature CSV
  -> 3D figures / report fields
```

This keeps the current preprocessing and reporting work intact instead of
replacing it.

## Recommended Scope

### Phase 1: Relative 3D Skeleton

Target output:

- one 3D joint sequence per clip,
- pelvis-centered coordinates,
- normalized or pseudo-metric scale,
- confidence or quality flags inherited from the 2D inputs.

This phase should avoid claims such as:

- exact bat speed,
- exact release position,
- exact world-coordinate stride length,
- medical-grade joint loading.

### Phase 2: 3D Features

Once a stable relative 3D skeleton exists, extend the current metrics toward:

- trunk tilt in 3D,
- pelvis rotation and shoulder rotation with less perspective distortion,
- hip-shoulder separation in 3D,
- stride-knee lift height,
- hand path depth change,
- left-right asymmetry in 3D,
- action-window 3D summaries for report generation.

### Phase 3: Better Calibration / Event Semantics

Only after Phases 1 and 2 are stable should the repo consider:

- weak-perspective scale refinement,
- optional camera calibration,
- optional multi-view fusion,
- event-conditioned baseball metrics with stronger timing semantics.

## Repo-Level Design

### New Package Areas

Add four 3D-focused areas that mirror the current 2D layout:

```text
src/baseball_pose/pose3d/
src/baseball_pose/features3d/
src/baseball_pose/pipeline/run_3d.py
src/baseball_pose/visualization3d/
```

Suggested responsibilities:

- `pose3d/`
  - common 3D schema,
  - lifting backend interface,
  - backend adapters such as VideoPose3D-style or MotionBERT-style lifting.
- `features3d/`
  - baseball-facing 3D metrics derived from a 3D joint sequence.
- `pipeline/run_3d.py`
  - orchestration from smoothed 2D CSV to 3D outputs.
- `visualization3d/`
  - simple static 3D renders,
  - orthographic multi-view panels,
  - optional reprojected 2D + 3D comparison visuals.

### Minimal New Pipeline Stages

The current `PipelineStage` enum should grow by adding explicit 3D stages after
2D postprocessing:

```text
postprocess_pose
  -> estimate_pose_3d
  -> extract_features_3d
  -> write_visualizations_3d
```

That allows the repo to support mixed 2D-only and 2D+3D runs without making 3D
mandatory for all experiments.

## Data Model Proposal

### 2D Input Contract to 3D

The 3D module should consume the existing smoothed pose CSV rather than raw
frame detections. Required fields should include:

- `frame_index`
- `timestamp_sec`
- `joint_name`
- `x`
- `y`
- joint confidence/visibility score

The 3D lifting stage should build per-frame joint vectors from the canonical
joint set already used in `pose/schema.py`.

### Proposed 3D Pose Row

Each 3D record should minimally contain:

- `clip_id`
- `condition_id`
- `frame_index`
- `timestamp_sec`
- `joint_name`
- `x_3d`
- `y_3d`
- `z_3d`
- `scale_mode`
- `lift_backend`
- `input_quality_score`

This mirrors the current 2D pose CSV design and keeps downstream joins simple.

### Proposed 3D Feature Row

The 3D feature CSV should stay action-window-friendly and report-friendly. Early
fields should include:

- `pelvis_rotation_3d_deg`
- `shoulder_rotation_3d_deg`
- `hip_shoulder_separation_3d_deg`
- `trunk_tilt_3d_deg`
- `lead_knee_lift_3d`
- `hand_depth_excursion`
- `center_of_mass_z_proxy`

These fields should be added alongside, not mixed into, the current 2D feature
CSV at the first integration step. A combined report layer can merge them later.

## Backend Strategy

### Current Recommendation

Use a video-HMR-first backend strategy:

1. **GVHMR or WHAM-style video-HMR as the preferred 3D source**
2. **the existing cleaned 2D pose as a trust/gating prior**
3. **MediaPipe world landmarks or pure 2D-to-3D lifting only as baselines**

This is a better fit for the current Suzhou and benchmark findings. The main
failure mode is not missing code around 3D; it is unreliable distal 2D joints
during high-speed throwing and occlusion. A pure 2D-to-3D lifter would inherit
and amplify those errors. A video-HMR backend can use image/video evidence,
temporal context, and a body prior before the project applies its own 2D
reprojection-style quality gate.

The repo-level contract is intentionally narrow:

```text
external video-HMR output
  -> normalized project 3D joint CSV
  -> 2D prior gating
  -> 3D smoothing / feature extraction / overlays
```

The current external adapter accepts CSV rows with:

```text
frame_index,joint_name,x_3d,y_3d,z_3d
```

Optional CSV fields are:

```text
timestamp_sec,confidence,score,input_quality_score,scale_mode,lift_backend
```

NPZ import is also supported for arrays named `joints_3d`, `pred_joints_3d`,
`world_joints`, or `joints`, with shape `[frames, joints, 3]`. NPZ inputs need
either a `joint_names` array or `pose3d.external_joint_names` in config.

The preferred Suzhou entrypoint is:

```bash
python -m baseball_pose.cli \
  --config configs/experiments/gvhmr_suzhou_test.yaml \
  lift-pose-3d \
  --condition image_center_motion_grabcut_pose_complete_smooth
```

Expected external result location:

```text
data_full/suzhou_rtmpose_halpe26_test/external_pose3d/gvhmr/{clip_id}.csv
```

For one Suzhou clip, the intended handoff is:

```bash
# Run from the GVHMR environment.
cd external/GVHMR
python tools/demo/demo.py \
  --video ../../../baseball-dataset-suzhou/IMG_8084.MOV \
  --output_root ../../outputs_full/gvhmr_suzhou_raw \
  -s

cd ../..
python scripts/export_gvhmr_joints.py \
  --gvhmr-root external/GVHMR \
  --result outputs_full/gvhmr_suzhou_raw/IMG_8084/hmr4d_results.pt \
  --output data_full/suzhou_rtmpose_halpe26_test/external_pose3d/gvhmr/suzhou_img_8084.csv \
  --clip-id suzhou_img_8084 \
  --face-z

python -m baseball_pose.cli \
  --config configs/experiments/gvhmr_suzhou_test.yaml \
  lift-pose-3d \
  --clip-id suzhou_img_8084 \
  --condition image_center_motion_grabcut_pose_complete_smooth
```

### Baseline Backends

MediaPipe world landmarks remain useful as a lightweight baseline. MotionBERT
or other 2D-to-3D lifters can be added later, but they should not be the main
robustness path for pitching because they depend too heavily on already-noisy
wrists and elbows.

## CLI Evolution

The current CLI already groups the pipeline into readable commands. Add 3D in
the same style:

```text
lift-pose-3d
extract-features-3d
make-figures-3d
```

Optional later commands:

```text
render-overlays-3d
summarize-pose-stability-3d
```

The recommended first run path is:

```text
run-image-proposal-roi
  -> smooth-poses
  -> lift-pose-3d
  -> extract-features-3d
  -> make-figures-3d
```

## Config Evolution

Add a top-level `pose3d` section rather than mixing 3D settings into the
existing `pose` section. Suggested fields:

```yaml
pose3d:
  enabled: false
  backend: gvhmr
  input_condition_suffix: _smooth
  root_joint: pelvis_center
  normalize_scale: true
  min_valid_joints: 8
  external_result_path: "{data_dir}/external_pose3d/{backend}/{clip_id}.csv"
```

This keeps 2D config readable and lets 3D remain opt-in.

Also add optional experiment-level 3D defaults, for example:

```yaml
experiments:
  default_3d_conditions:
    - image_center_motion_grabcut_pose_smooth_3d
```

## Visualization Plan

The current report charts are family-facing and should remain readable. The 3D
expansion should not replace them with technical mesh plots. Instead, add a few
simple visuals:

- three-view skeleton snapshot: front / side / top,
- 2D overlay next to 3D skeleton panel,
- 3D trunk-rotation and separation summary cards,
- depth-excursion bar for hand path or center-of-mass proxy.

These should remain understandable to coaches and parents, not only to research
audiences.

## Reporting Impact

The LLM layer should not be rewritten for 3D. Instead, extend the prompt
payload:

- keep current 2D metrics,
- add selected 3D metrics,
- clearly label which values are 3D relative pose measures,
- avoid overstating them as calibrated physical ground truth.

That means the existing report pipeline remains the right outer shell; only the
metric inventory becomes richer.

## Risks

### Risk 1: Weak 2D In, Bad 3D Out

The biggest technical risk is still distal-joint quality. If the 2D wrists or
ankles drift, the 3D lifting stage will amplify those errors. The repo should
continue treating conservative 2D filtering as a requirement, not as optional
cleanup.

### Risk 2: False Precision

A relative 3D skeleton can easily be oversold. The repo must clearly separate:

- relative 3D movement structure,
- calibrated absolute biomechanics.

### Risk 3: Domain Mismatch

Many 3D lifting backends are trained on general human-motion datasets rather
than baseball-specific clips. Expect a validation phase before trusting 3D
outputs for report claims.

## Recommended Implementation Order

1. Keep the current 2D pipeline as the automatic prior and quality gate.
2. Add the external video-HMR import backend and Suzhou config.
3. Run GVHMR/WHAM outside this repo and export joint CSV/NPZ files.
4. Import those files with `lift-pose-3d`, then run 3D smoothing and overlays.
5. Reject or mark joints whose 3D output conflicts with trusted 2D evidence.
6. Add a small set of 3D baseball metrics.
7. Validate batting and pitching clips separately before broadening to all inputs.

## Success Definition

The 3D extension should be considered successful only when the repo can do all
of the following on the current best batting pipeline:

- produce a 3D joint CSV from a smoothed 2D CSV,
- compute at least a small stable set of 3D baseball metrics,
- visualize those metrics in a readable report-friendly way,
- pass 3D metrics into the current report generation workflow without breaking
  existing 2D outputs.

That is the correct expansion path for this repo: preserve the 2D strengths,
add 3D as a new stage, and keep the final outputs readable rather than chasing
prematurely complex biomechanics claims.
