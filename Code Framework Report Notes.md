# Code Framework Report Notes

This document is a report-writing map for the `baseball-analysis` codebase. It summarizes the repository structure, implemented pipeline, data contracts, algorithmic techniques, experiment conditions, outputs, and reference targets. It is intended to support the final LaTeX manuscript and slide scripts, not to replace the manuscript.

## 1. Project Summary

The project implements a reproducible 2D baseball pose-analysis pipeline for local batting and pitching videos. The final report path is:

```text
raw baseball video
  -> frame sampling and resizing
  -> skeleton-free image proposal ROI/mask
  -> MediaPipe PoseLandmarker on masked full-frame input
  -> canonical 13-joint pose CSV
  -> temporal gating, interpolation, median filtering, Savitzky-Golay smoothing
  -> posture feature extraction
  -> posture plots, wrist trajectory plots, and overlay videos
```

The current best condition is:

```text
image_center_motion_grabcut_pose_smooth
```

Earlier baseline and ROI conditions remain in the codebase as ablations and development history:

- `baseline_raw`: raw full-frame MediaPipe pose.
- `auto_roi_raw`: fixed clip-level ROI estimated from motion and edges.
- `auto_roi_pose_prior`: ROI estimated from a first-pass pose skeleton.
- `center_prior_roi`: fixed center crop based on the recording setup.
- `body_prior_mask_roi`: skeleton-shaped mask from a prior pose pass.
- `image_center_motion_grabcut_pose`: skeleton-free image proposal plus MediaPipe.
- `*_smooth`: temporally filtered pose outputs.

## 2. Repository Structure

```text
baseball-analysis/
  pyproject.toml
  requirements.txt
  README.md
  log.md
  configs/
    default.yaml
    experiments/
      full_video.yaml
      minimal_mvp.yaml
      ablation_roi_clahe.yaml
      optional_rtmpose.yaml
  data/
    metadata/
      clips.csv
      difficulty_labels.csv
      manual_rois.csv
    interim/
      frames/<clip_id>/<condition_id>/
      rois/<clip_id>/<condition_id>.csv
    processed/
      poses/<clip_id>/<condition_id>.csv
      features/<clip_id>/<condition_id>.csv
      metrics/*.csv
  data_full/
    interim/
    processed/
      poses/
      features/
      metrics/
  outputs/
    overlays/
    roi_debug/
    motion_preview/
  outputs_full/
    overlays/
    figures/
    roi_debug/
    body_mask_debug/
    image_proposal_debug/
  raw/
    batting-1.mov
    batting-2.mov
    pitching-1.mov
    pitching-2.mov
  scripts/
    run_experiment.py
    run_baseline.py
    make_figures.py
    prepare_clips.py
    summarize_results.py
  src/baseball_pose/
    cli.py
    config.py
    io/
    pose/
    preprocessing/
    postprocess/
    features/
    evaluation/
    visualization/
    pipeline/
  tests/
```

Important note for submission: `raw/`, `data_full/`, and `outputs_full/` are large. The local size observed during inspection was about `281M` for `raw/`, `17G` for `data_full/`, and `52G` for `outputs_full/`. For final submission, the large raw/generated data should probably be uploaded separately, with a downloadable GitHub or cloud link in the report/package.

## 3. Main Entry Points

### CLI

The user-facing command is registered in `pyproject.toml`:

```text
baseball-pose = "baseball_pose.cli:main"
```

Main CLI file:

```text
src/baseball_pose/cli.py
```

Implemented commands:

- `validate-config`: load YAML config and print configured clips/conditions.
- `plan`: print pipeline stages.
- `run-baseline`: raw full-frame MediaPipe pose.
- `run-motion-preview`: Lucas-Kanade optical-flow preview without pose.
- `run-auto-roi`: motion/edge fixed ROI experiment.
- `run-pose-prior-roi`: pose-derived ROI experiment.
- `run-center-prior-roi`: fixed center-prior ROI experiment.
- `run-body-prior-mask-roi`: skeleton-shaped masking experiment.
- `run-image-proposal-roi`: final skeleton-free image proposal path.
- `smooth-poses`: temporal smoothing from pose CSV files.
- `extract-features`: frame-level feature extraction from pose CSV files.
- `make-figures`: report-oriented Matplotlib figures.
- `render-overlays`: pose overlay videos from stored pose CSVs.
- `render-body-mask-debug`: debug videos for the body-prior mask.
- `render-image-proposal-debug`: debug videos for the image proposal mask.
- `summarize-roi-ablation`: metric table for comparing ROI variants.

### Full-Video Reproducibility Commands

The main full-video config is:

```text
configs/experiments/full_video.yaml
```

It redirects generated files to:

```text
data_full/
outputs_full/
```

Typical final pipeline commands:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli \
  --config configs/experiments/full_video.yaml run-image-proposal-roi

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli \
  --config configs/experiments/full_video.yaml smooth-poses \
  --condition image_center_motion_grabcut_pose

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli \
  --config configs/experiments/full_video.yaml extract-features \
  --condition image_center_motion_grabcut_pose_smooth

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli \
  --config configs/experiments/full_video.yaml make-figures \
  --condition image_center_motion_grabcut_pose_smooth

MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli \
  --config configs/experiments/full_video.yaml render-overlays \
  --condition image_center_motion_grabcut_pose_smooth
```

## 4. Dataset and Metadata

Clip metadata is stored in:

```text
data/metadata/clips.csv
```

Current clips:

| clip_id | source_path | action_type | target fps | tags |
|---|---|---:|---:|---|
| `batting_1` | `raw/batting-1.mov` | batting | 30 | slow motion, low motion blur |
| `batting_2` | `raw/batting-2.mov` | batting | 30 | slow motion, low motion blur |
| `pitching_1` | `raw/pitching-1.mov` | pitching | 30 | slow motion, low motion blur |
| `pitching_2` | `raw/pitching-2.mov` | pitching | 30 | slow motion, low motion blur |

The project intentionally uses local manually collected baseball clips rather than a large public dataset. This makes the report a reproducible case study: the code focuses on robust processing under a specific recording setup with nearby background people and non-laboratory conditions.

## 5. Core Data Contracts

### Frame Records

Implemented in:

```text
src/baseball_pose/io/video.py
src/baseball_pose/io/frame_csv.py
```

Frame records contain:

- `clip_id`
- `frame_index`
- `timestamp_sec`
- `frame_path`
- `condition_id`
- `width`
- `height`

Sampling uses OpenCV `VideoCapture`, keeps every `round(source_fps / target_fps)` frame, resizes by longest side when configured, and writes PNG frames.

### Pose Records

Implemented in:

```text
src/baseball_pose/pose/schema.py
src/baseball_pose/io/pose_csv.py
```

Each CSV row is one joint in one frame:

- `clip_id`
- `condition_id`
- `frame_index`
- `timestamp_sec`
- `joint_name`
- `x`, `y`
- `visibility`
- `confidence`
- `backend`
- `inference_time_ms`

Coordinates are normalized to full-frame coordinates after ROI/crop remapping.

Canonical 13-joint subset:

```text
nose
left_shoulder, right_shoulder
left_elbow, right_elbow
left_wrist, right_wrist
left_hip, right_hip
left_knee, right_knee
left_ankle, right_ankle
```

### Feature Records

Implemented in:

```text
src/baseball_pose/features/extraction.py
src/baseball_pose/io/feature_csv.py
```

Each CSV row is one frame-level feature vector. Main features:

- elbow, shoulder, and knee joint angles;
- trunk tilt;
- pelvis line angle and shoulder line angle;
- hip-shoulder separation;
- pelvis and trunk angular velocity;
- approximate center-of-mass path from torso/lower-body landmarks;
- left/right knee extension relative to first valid frame;
- knee angular velocities;
- left/right wrist position and speed;
- `hand_speed_proxy`, computed as the maximum available wrist speed.

Important manuscript caveat: these are 2D image-plane pose proxies, not true 3D biomechanics. The report should avoid claiming exact ball velocity, bat speed, maximum external rotation, stride-foot contact timing, or impact/release metrics unless additional event labels or object tracking are added.

## 6. Pose Estimation Backend

Implemented in:

```text
src/baseball_pose/pose/mediapipe_pose.py
```

The project uses MediaPipe Tasks `PoseLandmarker`, not the older `mp.solutions.pose` API. The wrapper:

- loads `models/pose_landmarker_lite.task`;
- forces `BaseOptions.Delegate.CPU`;
- uses `vision.RunningMode.IMAGE`;
- requests `num_poses=1`;
- maps MediaPipe's 33 output landmarks to the canonical 13-joint schema;
- records per-frame inference time.

Important reproducibility note from `log.md`: MediaPipe failed under Python 3.13 on this macOS setup because of an OpenGL service initialization error. The successful environment used Python 3.12 with:

```text
.venv312/
mediapipe>=0.10
opencv-python>=4.8
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
pyyaml>=6.0
```

This should be mentioned in the reproducibility section.

## 7. Preprocessing and ROI Techniques

### 7.1 Baseline

Condition:

```text
baseline_raw
```

File path:

```text
src/baseball_pose/pipeline/run_clip.py
```

Method:

```text
sample frames -> MediaPipe on full frame -> pose CSV -> overlay video
```

Why it matters for the report:

- It gives a simple baseline.
- It exposes the main problem: MediaPipe can switch to nearby background people or miss limbs when the subject is small/partially occluded.

### 7.2 Motion and Edge Auto ROI

Condition:

```text
auto_roi_raw
```

Implemented in:

```text
src/baseball_pose/preprocessing/roi.py
```

Algorithm:

```text
for early frames:
  grayscale conversion
  Gaussian blur
  absolute frame difference
  thresholded motion mask
  edge mask
  combined motion-edge mask
  connected-component / contour candidate boxes
aggregate boxes over frames
expand and clamp a fixed clip-level ROI
crop frame
run pose on crop
remap crop-normalized coordinates to full-frame normalized coordinates
```

Report interpretation:

- This is a classical image-processing ROI baseline.
- A fixed clip-level ROI avoids per-frame crop jitter, which would contaminate trajectory smoothness.

### 7.3 Pose-Prior ROI

Condition:

```text
auto_roi_pose_prior
```

Implemented in:

```text
src/baseball_pose/preprocessing/roi.py
```

Algorithm:

```text
read baseline pose CSV
keep confident joints
build per-frame bounding boxes when enough joints exist
use robust percentiles for left/top/right/bottom
pad box horizontally and vertically
crop frames and rerun MediaPipe
remap coordinates to full frame
```

Report interpretation:

- This can improve crop focus if the first pass is mostly correct.
- It can fail if the first pass locks onto the wrong person, so it is not the final condition.

### 7.4 Center Prior ROI

Condition:

```text
center_prior_roi
```

Implemented in:

```text
src/baseball_pose/preprocessing/roi.py
```

Algorithm:

```text
use configured center_x, center_y, width_ratio, height_ratio
crop fixed center action area
run MediaPipe
remap to full frame
```

Report interpretation:

- This uses recording setup knowledge: athlete usually appears near the center.
- It is simple and stable, but cannot adapt to off-center pitcher starts or extended limbs.

### 7.5 Body-Prior Mask ROI

Condition:

```text
body_prior_mask_roi
```

Implemented in:

```text
src/baseball_pose/preprocessing/body_mask.py
```

Algorithm:

```text
read prior smoothed pose
collect confident joints
build padded ROI around skeleton
draw skeleton-shaped binary mask:
  limbs as thick lines
  torso as filled polygon
  joints as circles
blur mask edges
black out pixels outside the body mask
run MediaPipe on masked crop
```

Report interpretation:

- This suppresses background people and clutter.
- It depends on prior skeleton quality, so errors in the prior can be reinforced.
- It is useful as an ablation but not the final method.

### 7.6 Final Image Proposal: Center + Motion + MOG2 + GrabCut

Condition:

```text
image_center_motion_grabcut_pose
```

Implemented in:

```text
src/baseball_pose/preprocessing/image_proposal.py
src/baseball_pose/preprocessing/image_proposal_config.py
src/baseball_pose/pipeline/run_clip.py
src/baseball_pose/pipeline/image_proposal_debug.py
```

This is the final skeleton-free subject proposal. It intentionally avoids using a pose skeleton as the proposal source.

Main algorithm:

```text
input frame
  -> optional downscale for proposal processing
  -> LAB color conversion
  -> CLAHE on lightness channel
  -> unsharp enhancement
  -> center-band prior mask
  -> frame-difference motion mask
  -> MOG2 foreground mask
  -> GrabCut initialized by center/motion/foreground/previous-mask evidence
  -> combine with center support
  -> morphological open/close/dilate cleanup
  -> connected-component scoring
  -> vertical body-core selection
  -> body-envelope constraint
  -> GrabCut refinement
  -> temporal mask stabilization
  -> full-frame black canvas containing only proposal pixels
  -> MediaPipe PoseLandmarker
```

Key implementation details:

- `ImageProposalTracker` tracks subject center and proposal width across frames.
- Lucas-Kanade optical flow is used internally to track feature points in the subject core when available.
- Shi-Tomasi features are selected with `cv2.goodFeaturesToTrack`.
- The tracker constrains maximum per-frame center and width updates.
- Large sudden mask area changes are rejected.
- Warmup frames can accumulate center/width samples before tracker updates.
- The proposal mask is temporally stabilized with the previous mask.
- Clip-specific lower-body envelopes prevent knees/feet from being cropped while limiting background people.

Clip-specific overrides in `configs/default.yaml`:

- `batting_1`: shared prior plus right-only lower-body widening.
- `batting_2`: modest symmetric lower-body widening.
- `pitching_1`: wider action envelope and wider lower-body envelope.
- `pitching_2`: left-shifted start prior and wider lower-body envelope.

Report interpretation:

- This method combines classical computer vision with a learned pose estimator.
- It is a practical hybrid: image processing controls what MediaPipe sees; MediaPipe estimates joints.
- It directly supports course topics: contrast enhancement, frame differencing, foreground segmentation, morphology, connected components, optical flow, and temporal filtering.

## 8. Temporal Postprocessing

Implemented in:

```text
src/baseball_pose/postprocess/smoothing.py
src/baseball_pose/pipeline/postprocess.py
```

Default full pipeline:

```text
raw pose CSV
  -> torso-continuity gate
  -> per-joint jump outlier removal
  -> short-gap linear interpolation
  -> median filtering
  -> Savitzky-Golay smoothing
  -> moving-average refinement
  -> smoothed pose CSV
```

Key configuration from `configs/default.yaml`:

```yaml
postprocess:
  confidence_threshold: 0.5
  interpolate_max_gap_frames: 12
  smoothing:
    method: savgol
    window_length: 15
    polyorder: 2
    median_window_length: 5
    refine_window_length: 9
    jump_threshold_multiplier: 4.0
    torso_gate_enabled: true
    torso_jump_threshold_multiplier: 6.0
    min_torso_jump_distance: 0.06
```

Why this matters:

- Low-confidence landmarks become missing values.
- Torso continuity detects wrong-person switches by checking jumps in hip/shoulder center.
- Interpolation repairs short gaps but preserves longer missing segments.
- Median filtering removes isolated spikes.
- Savitzky-Golay smoothing preserves trajectory shape better than a simple moving average alone.
- Moving-average refinement gives a final visually readable overlay.

## 9. Feature Engineering

Implemented in:

```text
src/baseball_pose/features/angles.py
src/baseball_pose/features/extraction.py
src/baseball_pose/features/normalization.py
src/baseball_pose/features/events.py
src/baseball_pose/features/trajectories.py
```

Primary report-ready feature groups:

1. Upper-limb joint posture:
   - left/right elbow angle;
   - left/right shoulder angle.

2. Lower-body posture:
   - left/right knee angle;
   - knee extension from the first valid frame;
   - knee angular velocity.

3. Rotation-chain proxies:
   - pelvis segment angle from hip line;
   - shoulder segment angle from shoulder line;
   - hip-shoulder separation;
   - pelvis and trunk rotation velocities.

4. Motion path:
   - approximate center-of-mass path;
   - left/right wrist trajectory;
   - left/right wrist speed;
   - hand-speed proxy.

Potential report language:

> We use 2D keypoint-derived posture proxies rather than clinical 3D kinematic measurements. The quantities are suitable for comparing trajectory readability and relative motion trends across preprocessing conditions, but not for diagnosing injury risk or measuring exact baseball biomechanics.

## 10. Evaluation Metrics

Implemented in:

```text
src/baseball_pose/evaluation/completeness.py
src/baseball_pose/evaluation/jitter.py
src/baseball_pose/evaluation/smoothness.py
src/baseball_pose/evaluation/runtime.py
src/baseball_pose/evaluation/roi_ablation.py
```

Metric groups:

- `keypoint_completeness`: fraction of required joints present above confidence threshold.
- `runtime_ms_per_frame`: mean unique-frame inference time.
- `missing_rate`: missing fraction for selected joints.
- `temporal_jitter`: mean frame-to-frame joint displacement.
- `trajectory_smoothness`: mean second-difference magnitude.

Report-ready metric table:

```text
data_full/processed/metrics/roi_ablation_summary.csv
```

Observed final-condition summary:

| clip_id | condition | completeness all | runtime ms/frame | left wrist jitter | right wrist jitter |
|---|---|---:|---:|---:|---:|
| batting_1 | image proposal raw | 0.911 | 15.54 | 0.0380 | 0.0154 |
| batting_1 | image proposal smooth | 0.937 | 15.54 | 0.00352 | 0.00167 |
| batting_2 | image proposal raw | 0.694 | 14.06 | 0.0247 | 0.0268 |
| batting_2 | image proposal smooth | 0.875 | 14.06 | 0.00528 | 0.00507 |
| pitching_1 | image proposal raw | 0.998 | 16.93 | 0.0217 | 0.0194 |
| pitching_1 | image proposal smooth | 1.000 | 16.93 | 0.00344 | 0.00468 |
| pitching_2 | image proposal raw | 0.972 | 15.66 | 0.0146 | 0.0161 |
| pitching_2 | image proposal smooth | 1.000 | 15.66 | 0.00411 | 0.00451 |

Interpretation:

- Smoothing sharply reduces wrist jitter and second-difference smoothness metrics.
- Batting clips are harder than pitching clips in this setup, especially `batting_2`, where raw completeness is lower.
- Runtime is dominated by pose inference and remains around 14-17 ms/frame for the final image-proposal condition.

## 11. Visualization Outputs

Implemented in:

```text
src/baseball_pose/visualization/overlays.py
src/baseball_pose/visualization/plots.py
src/baseball_pose/visualization/motion_preview.py
src/baseball_pose/visualization/diagnostics.py
src/baseball_pose/pipeline/figures.py
src/baseball_pose/pipeline/overlays.py
```

Main report figures:

```text
outputs_full/figures/<clip_id>__wrist_trajectories.png
outputs_full/figures/<clip_id>__posture_analysis.png
```

Main qualitative videos:

```text
outputs_full/overlays/<clip_id>__image_center_motion_grabcut_pose_smooth.mp4
outputs_full/image_proposal_debug/<clip_id>__image_center_motion_grabcut__proposal_overlay.mp4
outputs_full/image_proposal_debug/<clip_id>__image_center_motion_grabcut__masked_frame.mp4
```

Figure content:

- Wrist trajectory plot: left/right wrist x/y coordinates over time.
- Posture analysis plot: pelvis/shoulder rotation, rotation velocity, approximate COM path, knee extension, hand-speed proxy.
- Overlay video: skeleton and wrist tracks drawn on original video frames.
- Proposal debug videos: show proposal mask and the masked input sent to MediaPipe.

## 12. Current Full-Video Output Counts

Line counts include one CSV header row.

| clip_id | smoothed pose CSV lines | feature CSV lines | approximate frames |
|---|---:|---:|---:|
| batting_1 | 12,962 | 998 | 997 |
| batting_2 | 9,855 | 759 | 758 |
| pitching_1 | 8,113 | 625 | 624 |
| pitching_2 | 5,513 | 425 | 424 |

Because each pose frame has 13 canonical joints, the approximate frame count is:

```text
(pose_csv_lines - 1) / 13
```

## 13. Testing and Code Quality

Tests are in:

```text
tests/
```

Coverage areas:

- config loading and YAML extension behavior;
- metadata loading;
- ROI box expansion/clamping/remapping;
- image proposal behavior;
- body mask behavior;
- smoothing behavior;
- feature extraction;
- metrics and ROI ablation summaries.

Useful validation commands:

```bash
python -m compileall src tests
python -m pytest
baseball-pose validate-config
baseball-pose plan
```

## 14. Suggested Manuscript Structure

### Abstract

State the problem: readable 2D pose analysis from local baseball clips with clutter/background-person issues. Summarize the hybrid solution and main result: image proposal plus temporal smoothing produces complete, smoother trajectories and report-ready posture features.

### Introduction

Topics to cover:

- baseball batting/pitching are fast full-body motions;
- 2D video pose estimation is accessible but unreliable under clutter, occlusion, and off-center athletes;
- the project goal is not clinical biomechanics but a reproducible computer-vision pipeline for readable posture analysis;
- contributions:
  - configurable video-to-pose pipeline;
  - skeleton-free image proposal combining center priors, motion, foreground segmentation, GrabCut, morphology, and optical flow;
  - temporal smoothing and posture feature extraction;
  - qualitative videos and quantitative trajectory metrics.

### Related Work

Group references by:

- human pose estimation and MediaPipe/BlazePose;
- classical foreground segmentation and GrabCut;
- optical flow and feature tracking;
- temporal smoothing;
- baseball pitching/batting biomechanics and kinetic-chain concepts.

### Methods

Recommended subsections:

1. Dataset and metadata.
2. Pipeline overview.
3. Pose schema and MediaPipe backend.
4. ROI/proposal methods.
5. Temporal smoothing.
6. Feature extraction.
7. Metrics and visualization.

### Experiments

Compare at least:

- `baseline_raw`;
- one or two ROI ablations;
- `image_center_motion_grabcut_pose`;
- `image_center_motion_grabcut_pose_smooth`.

Use both:

- quantitative metrics from `roi_ablation_summary.csv`;
- qualitative overlay/debug frame examples.

### Results

Key points:

- final smoothing reduces wrist jitter by roughly an order of magnitude in all clips;
- pitching clips achieved near-complete smoothed keypoint coverage;
- batting clips are more difficult, but smoothing and proposal masking improve readability;
- final overlay videos are the strongest qualitative evidence.

### Discussion

Important limitations:

- 2D normalized coordinates are not metric 3D kinematics;
- no ball/bat/release/impact detector;
- clip-specific priors improve current videos but may reduce generalization;
- MediaPipe version and Python version affect reproducibility;
- no manual ground-truth joint labels, so metrics evaluate completeness/smoothness rather than absolute anatomical accuracy.

### Conclusion

Summarize the hybrid computer-vision approach and future work: event detection, ball/bat tracking, 3D pose or multi-view capture, manually labeled validation frames, and automatic adaptation of clip-specific priors.

## 15. Slide Deck Outline With Script Notes

Suggested 12-slide structure:

1. Title and Problem
   - Script: We analyze local batting and pitching videos and convert them into readable 2D posture trajectories.

2. Motivation
   - Script: Off-the-shelf pose estimation is useful but unstable when the athlete is small, off-center, or surrounded by other people.

3. Dataset
   - Script: Four local clips: two batting and two pitching, sampled at 30 fps, stored through metadata records for reproducibility.

4. Pipeline Overview
   - Script: Show the full flow from raw video to frame sampling, image proposal, MediaPipe pose, smoothing, features, figures, and overlays.

5. Baseline Failure Mode
   - Script: Raw MediaPipe is the baseline. It can switch subjects or miss lower-body landmarks, motivating ROI/proposal methods.

6. ROI Ablations
   - Script: Explain motion-edge ROI, pose-prior ROI, center prior, and body-prior mask as development steps.

7. Final Image Proposal
   - Script: The final method is skeleton-free: it uses contrast enhancement, motion, MOG2, GrabCut, morphology, center priors, and optical-flow tracking.

8. Pose and Data Schema
   - Script: MediaPipe outputs are mapped to 13 canonical joints, stored as long-form CSV, and remapped to full-frame coordinates after preprocessing.

9. Temporal Smoothing
   - Script: Torso gating removes wrong-person jumps; interpolation repairs short gaps; median/Savitzky-Golay/moving average smoothing improves trajectory readability.

10. Feature Extraction
    - Script: We compute 2D posture proxies: elbow/knee/shoulder angles, pelvis and shoulder rotation, hip-shoulder separation, COM path, wrist speed.

11. Results
    - Script: Show metric table and figures. Emphasize reduced wrist jitter and improved completeness in the smoothed image-proposal condition.

12. Limitations and Future Work
    - Script: This is not 3D clinical biomechanics. Future work needs event labels, ball/bat tracking, ground-truth annotations, and less clip-specific tuning.

## 16. Reference Targets to Verify Before Final Bibliography

The final report requirement penalizes fake references. Use the following as verified reference targets and still double-check all BibTeX fields before the final LaTeX submission.

Computer vision and pose estimation:

1. MediaPipe Pose Landmarker documentation, Google AI for Developers. Use for the actual Tasks API and model asset behavior.
2. Bazarevsky et al., "BlazePose: On-device Real-time Body Pose tracking", 2020. Use for MediaPipe/BlazePose background.
3. Lugaresi et al., "MediaPipe: A Framework for Building Perception Pipelines", 2019. Use for MediaPipe framework background.
4. Bradski, "The OpenCV Library", Dr. Dobb's Journal, 2000. Use for OpenCV.
5. Rother, Kolmogorov, and Blake, "GrabCut: Interactive Foreground Extraction using Iterated Graph Cuts", ACM Transactions on Graphics, 2004, DOI `10.1145/1015706.1015720`.
6. Zivkovic, "Improved Adaptive Gaussian Mixture Model for Background Subtraction", ICPR, 2004, DOI `10.1109/ICPR.2004.1333992`.
7. Zivkovic and van der Heijden, "Efficient adaptive density estimation per image pixel for the task of background subtraction", Pattern Recognition Letters, 2006, DOI `10.1016/j.patrec.2005.11.005`.
8. Lucas and Kanade, "An Iterative Image Registration Technique with an Application to Stereo Vision", IJCAI, 1981.
9. Shi and Tomasi, "Good Features to Track", CVPR, 1994.
10. Savitzky and Golay, "Smoothing and Differentiation of Data by Simplified Least Squares Procedures", Analytical Chemistry, 1964, DOI `10.1021/ac60214a047`.
11. SciPy 1.0 paper: Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python", Nature Methods, 2020.
12. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, 2007.
13. Harris et al., "Array programming with NumPy", Nature, 2020.
14. McKinney, "Data Structures for Statistical Computing in Python", SciPy conference, 2010.

Baseball biomechanics and interpretation:

15. Diffendaffer et al., "The Clinician's Guide to Baseball Pitching Biomechanics", Sports Health, 2023, DOI `10.1177/19417381221078537`.
16. Bullock et al., "Baseball pitching biomechanics in relation to pain, injury, and surgery: A systematic review", Journal of Science and Medicine in Sport, 2021, DOI `10.1016/j.jsams.2020.06.015`.
17. Orishimo et al., "Role of Pelvis and Trunk Biomechanics in Generating Ball Velocity in Baseball Pitching", Journal of Strength and Conditioning Research, 2023, DOI `10.1519/JSC.0000000000004314`.
18. "Pitching mechanics and performance of adult baseball pitchers: A systematic review and meta-analysis for normative data", Journal of Science and Medicine in Sport, 2023, DOI `10.1016/j.jsams.2022.11.004`.
19. "Investigation of optimal lower body movement in presence of the constrained pelvis rotation in baseball batting", Journal of Biomechanics, 2022, DOI `10.1016/j.jbiomech.2022.111219`.

OpenCV documentation pages can be cited sparingly for implementation details:

- `cv::BackgroundSubtractorMOG2`
- `cv.grabCut`
- `cv.calcOpticalFlowPyrLK`
- `cv.goodFeaturesToTrack`
- `cv.createCLAHE`

## 17. Report Claims That Are Safe vs. Unsafe

Safe claims:

- The code implements a configurable 2D video pose-analysis pipeline.
- The final proposal method uses image-processing evidence before pose estimation.
- Smoothing reduces trajectory jitter metrics.
- Feature extraction produces 2D posture proxies from normalized keypoints.
- The current evaluation measures completeness, smoothness, jitter, and runtime, not ground-truth anatomical accuracy.

Unsafe claims unless more data are added:

- Exact baseball swing speed, bat speed, or ball velocity.
- Clinical injury-risk assessment.
- Accurate 3D pelvis/trunk rotation.
- Release point, impact moment, maximum external rotation, or stride-foot contact timing.
- Generalization to arbitrary videos without clip-specific tuning.

## 18. Implementation Phrases for the Report

Reusable wording:

> The pipeline stores intermediate artifacts after each stage, including sampled frames, ROI/debug videos, pose CSV files, smoothed pose CSV files, feature CSV files, and final figures. This design makes failures inspectable and allows each preprocessing condition to be rerun independently.

> Instead of trusting a first-pass skeleton to locate the player, the final proposal method builds a foreground mask from image evidence. This reduces the risk of reinforcing wrong-person detections from a failed pose estimate.

> All pose coordinates are remapped to the original full-frame normalized coordinate system, so downstream feature extraction and visualization use the same coordinate convention across preprocessing conditions.

> The biomechanics quantities are intentionally described as 2D proxies because the project uses a single monocular camera and does not estimate depth or calibrate real-world distances.

