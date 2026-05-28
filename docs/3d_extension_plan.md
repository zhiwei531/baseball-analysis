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

### Short-Term Recommendation

Use a staged backend strategy:

1. **MediaPipe world landmarks now**
2. **temporal 3D lifting later**

The first real backend in the repo can be MediaPipe world landmarks because it
is immediately compatible with the current 2D pose stack and gives a genuine
relative 3D skeleton output. The longer-term target should still be a temporal
3D lifting backend that consumes cleaned 2D joints rather than a full
human-mesh-recovery model. That remains the best fit for the repo because:

- the repo already produces smoothed 2D sequences,
- the project is single-view,
- baseball actions are fast and benefit from sequence context,
- the current reporting layer expects joint-level metrics more than dense mesh
  outputs.

The longer-term lifting backend should expose a narrow interface:

```python
lift_sequence(frames_2d) -> frames_3d
```

### Why Not Start with HMR

A mesh-recovery or pseudo-depth-heavy approach would add:

- much heavier dependencies,
- less transparent failure analysis,
- a larger domain gap between current 2D metrics and report outputs,
- harder debugging when distal joints are already weak.

For this repo, sequence-based 3D lifting is the better first research step.

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
  backend: temporal_lifter_stub
  input_condition_suffix: _smooth
  root_joint: pelvis_center
  normalize_scale: true
  min_valid_joints: 8
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

1. Add 3D path helpers, config hooks, and a 3D schema.
2. Add a stub lifting backend and CLI skeleton without changing current 2D runs.
3. Add a CSV writer/reader for 3D pose records.
4. Add the first temporal 3D lifting backend behind the stub interface.
5. Add a small set of 3D baseball metrics.
6. Add simple 3D figures and extend the report payload.
7. Validate on the current action-window clips before broadening to all inputs.

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
