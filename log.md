# Iteration Log

This log records implementation iterations, failures, revisions, and report-relevant decisions. It is intended to support the final project report and iteration summaries.

## Iteration 1: Planning Document

Date: 2026-04-25

Commit: not in `baseball-analysis`; generated outside the repo as `Code Framework Planning.md`.

Goal:

- Convert the project plan into an implementation-oriented framework.
- Define repository structure, module boundaries, data contracts, experiment conditions, and phased tasks.

Work completed:

- Created a code framework planning document based on `Project Plan.md`.
- Proposed structure for configs, data, source modules, scripts, tests, notebooks, and outputs.
- Defined expected records for clips, frames, pose landmarks, motion features, and metrics.
- Defined MVP conditions: `baseline_raw`, `roi_clahe`, and `roi_clahe_smooth`.
- Established implementation phases from scaffolding through visualization and optional extensions.

Key decisions:

- Start with MediaPipe baseline before ROI/CLAHE or RTMPose.
- Keep raw videos separate from generated outputs.
- Track outputs by both `clip_id` and `condition_id`.
- Treat deblurring and RTMPose as stretch goals, not MVP blockers.

Revision notes:

- No code was written in this iteration.
- The planning document became the basis for the initial repo scaffold.

## Iteration 2: Initial Code Framework Scaffold

Date: 2026-04-25

Commit: `e340c70 Scaffold baseball pose analysis framework`

Goal:

- Start implementing the planning document as a real Python project structure.
- Avoid video processing until the package boundaries and metadata conventions were clear.

Work completed:

- Added `pyproject.toml`, `requirements.txt`, and package layout under `src/baseball_pose/`.
- Added config files under `configs/`.
- Added metadata templates under `data/metadata/`.
- Registered four local raw videos in `data/metadata/clips.csv`:
  - `batting_1`
  - `batting_2`
  - `pitching_1`
  - `pitching_2`
- Added placeholders for manual ROI and difficulty labels.
- Added initial modules for video I/O, pose backends, preprocessing, post-processing, features, evaluation, visualization, and pipeline orchestration.
- Added initial tests for config, metadata, angle computation, and basic metrics.
- Added `.gitignore` so raw videos and generated outputs are not committed.

Validation:

- `python3 -m compileall src tests` passed.
- `PYTHONPATH=src python3 -m baseball_pose.cli --config configs/default.yaml validate-config` passed.
- `PYTHONPATH=src python3 -m baseball_pose.cli --config configs/default.yaml plan` passed.

Failure / issue:

- The base environment did not have `pytest`, `PyYAML`, OpenCV, MediaPipe, NumPy, Pandas, Matplotlib, or SciPy installed.
- Direct CLI execution without `PYTHONPATH` failed before editable install because `src` was not on the import path.

Revision:

- Added a lightweight YAML fallback parser so config validation can work before `PyYAML` is installed.
- Added module-level `if __name__ == "__main__": main()` to support `python -m baseball_pose.cli`.

Report relevance:

- This iteration establishes reproducibility structure rather than model performance.
- The metadata-first approach makes later experiment comparisons traceable.

## Iteration 3: Baseline Video and Pose Pipeline

Date: 2026-04-25

Commit: `36ec6c6 Implement baseline video pose pipeline`

Goal:

- Implement the first executable baseline path for raw videos.
- Generate actual frame records, pose records, and overlay visualization once dependencies are available.

Work completed:

- Implemented frame sampling in `src/baseball_pose/io/video.py` using OpenCV.
- Added CSV serializers for sampled frame records and pose records.
- Implemented output path helpers for frames, pose CSV, overlay frames, and overlay video.
- Added a MediaPipe pose estimator wrapper.
- Added skeleton overlay drawing and wrist-track drawing.
- Added `run-baseline` CLI command.
- Added `video.max_frames_per_clip` to keep early runs bounded.

Expected outputs:

- Sampled frames:
  - `data/interim/frames/<clip_id>/baseline_raw/`
- Frame manifest:
  - `data/interim/frames/<clip_id>/baseline_raw.csv`
- Pose landmarks:
  - `data/processed/poses/<clip_id>/baseline_raw.csv`
- Overlay frames:
  - `outputs/overlays/frames/<clip_id>/baseline_raw/`
- Overlay video:
  - `outputs/overlays/<clip_id>__baseline_raw.mp4`

Validation:

- `python3 -m compileall src tests` passed.
- Config validation and pipeline plan commands passed.

Failure / issue:

- The code could not yet be run end-to-end because runtime dependencies were not installed.

Revision:

- Created `.venv` and installed dependencies after network approval.
- This enabled `pytest` and runtime checks in the next iteration.

Report relevance:

- This iteration defines the baseline experimental condition: raw video to MediaPipe pose to overlay visualization.
- It also defines the first track visualization: left and right wrist trajectories drawn over the pose overlay.

## Iteration 4: Dependency Installation and MediaPipe API Mismatch

Date: 2026-04-25

Commit: `b59f602 Adapt MediaPipe backend to task model`

Goal:

- Run the baseline pipeline on `batting_1` for a short 30-frame test.

Work completed:

- Created `.venv`.
- Installed project dependencies with `pip install -e ".[dev]"`.
- Downloaded the MediaPipe pose landmarker model into:
  - `models/pose_landmarker_lite.task`
- Updated README with model download instructions and expected baseline output paths.
- Updated `.gitignore` so `.task` model files are not committed.

Validation:

- `.venv/bin/python -m pytest` passed: 5 tests.
- Config validation passed.

Failure:

- First runtime attempt failed because MediaPipe `0.10.33` no longer exposes the older `mp.solutions.pose` API used in the initial wrapper.
- Error:
  - `AttributeError: module 'mediapipe' has no attribute 'solutions'`

Cause:

- The installed MediaPipe package exposes the newer `mediapipe.tasks` API.
- The older `mp.solutions.pose.Pose` path is not available in this environment.

Revision:

- Rewrote the MediaPipe backend to use `mediapipe.tasks.python.vision.PoseLandmarker`.
- Added explicit model path config:
  - `pose.model_asset_path: models/pose_landmarker_lite.task`
- Mapped PoseLandmarker output to the existing canonical pose schema.

Report relevance:

- This is an implementation constraint worth reporting: MediaPipe API version affects reproducibility.
- The project should document exact dependency versions and model asset requirements.

## Iteration 5: MediaPipe Runtime GPU/OpenGL Failure

Date: 2026-04-25 to 2026-04-26

Commit: `1facce2 Force MediaPipe pose inference onto CPU`

Goal:

- Resolve MediaPipe runtime failure and produce the first overlay video.

Work completed:

- Added local cache directories for Matplotlib and fontconfig-related cache behavior:
  - `.cache/matplotlib`
  - `.cache/fontconfig`
- Set `MPLCONFIGDIR` and `XDG_CACHE_HOME` inside the MediaPipe backend.
- Updated MediaPipe `BaseOptions` to request `BaseOptions.Delegate.CPU`.

Validation:

- `.venv/bin/python -m compileall src/baseball_pose/pose/mediapipe_pose.py` passed.
- `.venv/bin/python -m pytest` passed: 5 tests.

Failure:

- Running:
  - `.venv/bin/python -m baseball_pose.cli --config configs/default.yaml run-baseline --clip-id batting_1 --max-frames 30`
- still failed during PoseLandmarker initialization.

Runtime error:

```text
RuntimeError: Service "kGpuService", required by node ... ImageToTensorCalculator,
was not provided and cannot be created: Could not create an NSOpenGLPixelFormat
```

Observed warnings:

```text
failed to create pixel format; trying without acceleration
Fontconfig error: No writable cache directories
```

Likely cause:

- On this macOS environment, MediaPipe Tasks still attempts to initialize an OpenGL/GPU-related service internally even when CPU delegate is requested.
- The issue may be tied to the installed MediaPipe wheel, Python 3.13, or this machine's headless/sandboxed OpenGL access.

Revision attempted:

- Checked available Python versions.
- Found:
  - system `python3`: Python 3.13.1
  - `/opt/anaconda3/bin/python3.12`: Python 3.12
- Created `.venv312` using Python 3.12 to test whether a different wheel/runtime avoids the OpenGL failure.

Additional failure:

- Installing dependencies into `.venv312` timed out while downloading packages.
- Error:
  - `ReadTimeoutError: HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out.`

Current status:

- Code for baseline frame sampling, pose CSV writing, and overlay visualization exists.
- The first 30-frame run created a frame manifest and sampled frames before failing at pose estimator initialization.
- No successful pose CSV or overlay video has been produced yet.

Next revision options:

1. Retry Python 3.12 dependency installation with a longer timeout:
   - `.venv312/bin/python -m pip install --timeout 120 -e ".[dev]"`
2. Pin MediaPipe to a version known to provide `mp.solutions.pose`, if compatible with available Python:
   - likely easier on Python 3.10 or 3.11 than Python 3.13
3. Add a fallback pose backend using OpenCV DNN / MMPose / a non-MediaPipe runtime if MediaPipe remains blocked.
4. Keep frame sampling and visualization code, but decouple pose inference so the pipeline can proceed with saved pose CSV from another tool if needed.

Report relevance:

- This is a useful failure case for the final report: reproducibility depends not only on algorithm choice but also runtime compatibility, model assets, and platform graphics backends.
- The implementation should document both the intended MediaPipe path and the fallback strategy if the runtime is unavailable.

## Iteration 6: Add Iteration Log

Date: 2026-04-26

Commit: `20b53ac Add iteration log`

Goal:

- Add a persistent log for iteration report and final project report writing.

Work completed:

- Created `log.md`.
- Recorded completed work, failures, diagnoses, revisions, and next steps from previous iterations.
- Added `.venv312/` and `*.egg-info/` to `.gitignore` because they are local build artifacts.

Next steps:

- Commit this log.
- Retry runtime execution through Python 3.12 or choose a fallback pose backend if MediaPipe Tasks remains blocked.

## Iteration 7: External Drive Git Cleanup

Date: 2026-04-26

Commit: `faa70b2 Ignore external drive metadata files`

Goal:

- Restore clean git tracking after moving the project to external storage.

Failure:

- `git status` reported many untracked `._*` AppleDouble files.
- Git also printed:

```text
error: non-monotonic index .git/objects/pack/._pack-...idx
```

Cause:

- macOS generated AppleDouble resource-fork files on the external drive.
- Some of these files were created inside `.git/objects/pack/`, so git attempted to read them as pack indexes.

Revision:

- Removed generated `._*` metadata files.
- Added ignore rules for external-drive and macOS metadata:
  - `._*`
  - `.Spotlight-V100/`
  - `.Trashes/`
  - `.fseventsd/`
  - `.TemporaryItems/`
- Pushed the cleanup commit to `origin/main`.

Current status:

- `main` tracks `origin/main`.
- Local `HEAD` matches `origin/main`.
- The working tree is clean after metadata cleanup.

Report relevance:

- External storage can affect reproducibility and git hygiene through filesystem metadata files.
- The repository now explicitly ignores those files.

## Iteration 8: Motion Preview Fallback Visualization

Date: 2026-04-26

Commit: pending

Goal:

- Produce a first working video-processing visualization without depending on the currently blocked MediaPipe runtime.

Work completed:

- Added an OpenCV Lucas-Kanade optical-flow preview path.
- Added CLI command:
  - `baseball-pose run-motion-preview --clip-id <clip_id> --max-frames <n>`
- Added output paths for motion preview frames and videos.

Expected outputs:

- `data/interim/frames/<clip_id>/motion_preview/`
- `data/interim/frames/<clip_id>/motion_preview.csv`
- `outputs/motion_preview/frames/<clip_id>/`
- `outputs/motion_preview/<clip_id>__motion_preview.mp4`

Interpretation:

- This preview is not pose estimation.
- It verifies that video decoding, frame sampling, track visualization, and video writing work end to end.
- Pose visualization will still use MediaPipe or another pose backend once the runtime issue is resolved.

## Iteration 9: Restore MediaPipe on Python 3.12

Date: 2026-04-26

Commit: `8f15f25 Fix baseline clip result return`

Goal:

- Continue the MediaPipe baseline path after the Python 3.13 OpenGL failure.
- Verify whether Python 3.12 can run MediaPipe Tasks successfully on the same machine.

Work completed:

- Installed project dependencies into `.venv312` with Python 3.12.
- Removed AppleDouble `._*` files generated inside the virtual environment because they broke Matplotlib style loading.
- Re-ran MediaPipe baseline with cache paths set to `/tmp`:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/default.yaml run-baseline --clip-id batting_1 --max-frames 5
```

Failure found:

- The first 5-frame run reached CLI output but returned `None` because the baseline result-return code had been accidentally placed after the motion-preview return path.

Revision:

- Fixed `run_baseline_clip()` so it writes frame records, pose records, overlay frames, overlay video, and returns `ClipRunResult`.

Validation:

- `pytest` passed: 5 tests.
- AST parse passed for `src/` and `tests/`.
- A 5-frame MediaPipe baseline succeeded:
  - `data/interim/frames/batting_1/baseline_raw.csv`
  - `data/processed/poses/batting_1/baseline_raw.csv`
  - `outputs/overlays/batting_1__baseline_raw.mp4`
  - 5 frames
  - 65 pose records

Current interpretation:

- MediaPipe Tasks is viable on this machine when run through Python 3.12.
- Python 3.13 remains unsuitable for this MediaPipe runtime because it triggers the earlier OpenGL service failure.

Next steps:

- Run baseline pose extraction on all four clips for a small bounded frame count.
- Inspect overlay quality and decide whether manual ROI should be added before expanding frame counts.

## Iteration 10: Four-Clip MediaPipe Baseline Run

Date: 2026-04-26

Commit: pending

Goal:

- Generate the first successful MediaPipe pose outputs and overlay videos for all four raw clips.

Command:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/default.yaml run-baseline --max-frames 30
```

Outputs generated:

- `data/interim/frames/batting_1/baseline_raw.csv`
- `data/interim/frames/batting_2/baseline_raw.csv`
- `data/interim/frames/pitching_1/baseline_raw.csv`
- `data/interim/frames/pitching_2/baseline_raw.csv`
- `data/processed/poses/batting_1/baseline_raw.csv`
- `data/processed/poses/batting_2/baseline_raw.csv`
- `data/processed/poses/pitching_1/baseline_raw.csv`
- `data/processed/poses/pitching_2/baseline_raw.csv`
- `outputs/overlays/batting_1__baseline_raw.mp4`
- `outputs/overlays/batting_2__baseline_raw.mp4`
- `outputs/overlays/pitching_1__baseline_raw.mp4`
- `outputs/overlays/pitching_2__baseline_raw.mp4`

Validation:

- Each clip processed 30 frames.
- Each pose CSV contains 390 landmark rows:
  - 30 frames x 13 canonical joints.
- 120 overlay frames were generated in total.
- Overlay video files were successfully written:
  - `batting_1__baseline_raw.mp4`: about 702 KB
  - `batting_2__baseline_raw.mp4`: about 627 KB
  - `pitching_1__baseline_raw.mp4`: about 440 KB
  - `pitching_2__baseline_raw.mp4`: about 478 KB

Observations:

- MediaPipe initializes successfully under Python 3.12.
- Runtime logs still mention GL context creation, but the run completes and TensorFlow Lite uses the CPU delegate.
- The output is now true pose-overlay visualization, unlike the earlier optical-flow motion preview.

Next steps:

- Review the four overlay videos visually.
- Add manual ROI support before expanding frame count if overlays show background or wrong-person detections.
- Start computing basic completeness and runtime summaries from the generated pose CSV files.

## Iteration 11: Plan Automatic ROI Module

Date: 2026-04-26

Commit: pending

Goal:

- Update project documentation to include an automatic ROI crop module based on traditional image processing.
- Address observed wrong-person switches in MediaPipe baseline overlays.

Planned method:

```text
sampled frames
  -> grayscale frame difference
  -> motion mask
  -> Canny / Sobel edge mask
  -> combined motion-edge ROI score
  -> contour / connected-component candidate boxes
  -> clip-level fixed ROI aggregation
  -> expanded crop
  -> MediaPipe pose
  -> keypoint remapping to original frame coordinates
```

Why this matters:

- Current MediaPipe baseline uses one pose but can switch to a different person across frames.
- Automatic ROI can reduce wrong-person switches without relying on a learned detector.
- The method directly matches course topics such as edge detection, feature extraction, connected components, and image-processing based preprocessing.

Design decision:

- Use a clip-level fixed ROI first instead of a per-frame ROI.
- Per-frame ROI may introduce crop jitter and distort downstream trajectory smoothness metrics.
- Manual ROI remains a fallback and upper-bound comparison when automatic ROI fails.

Planned conditions:

- `auto_roi_raw`
- `auto_roi_clahe`
- optional manual ROI comparison on failure clips

Expected diagnostics:

- selected ROI CSV per clip,
- mask / contour debug frames,
- ROI debug video,
- side-by-side `baseline_raw` vs `auto_roi_raw` overlay comparison.

## Iteration 12: Implement and Run `auto_roi_raw`

Date: 2026-04-26

Commits:

- `2b9e68f Add automatic ROI proposal helpers`
- `2e195b2 Add auto ROI pipeline command`

Goal:

- Implement the first automatic ROI crop based on traditional image processing.
- Preserve separate outputs for report comparison without overwriting `baseline_raw` or `motion_preview`.

Implementation:

- Added `RoiBox` helpers for expansion, clamping, crop extraction, and coordinate remapping.
- Added automatic ROI proposal:
  - grayscale frame difference,
  - Otsu-thresholded motion mask,
  - Canny edge mask,
  - motion-edge combined mask,
  - contour candidate boxes,
  - clip-level fixed ROI aggregation,
  - expanded ROI crop.
- Added coordinate remapping so MediaPipe keypoints estimated on cropped ROI are converted back to full-frame normalized coordinates.
- Added ROI debug videos with selected boxes drawn on original frames.
- Added CLI command:

```bash
.venv312/bin/python -m baseball_pose.cli --config configs/default.yaml run-auto-roi --max-frames 30
```

Outputs generated:

- `data/interim/frames/<clip_id>/auto_roi_raw.csv`
- `data/interim/rois/<clip_id>/auto_roi_raw.csv`
- `data/processed/poses/<clip_id>/auto_roi_raw.csv`
- `outputs/roi_debug/<clip_id>__auto_roi_raw.mp4`
- `outputs/roi_debug/frames/<clip_id>/`
- `outputs/overlays/<clip_id>__auto_roi_raw.mp4`
- `outputs/overlays/frames/<clip_id>/auto_roi_raw/`

Validation:

- `pytest` passed: 7 tests.
- `auto_roi_raw` ran on all four clips for 30 frames each.
- Each clip produced 390 pose rows:
  - 30 frames x 13 canonical joints.
- Each clip produced 30 ROI debug frames and 30 overlay frames.

Selected ROI boxes:

```text
batting_1:  x=102.675, y=0.0,     w=1177.325, h=714.0
batting_2:  x=143.775, y=182.575, w=1136.225, h=531.425
pitching_1: x=0.0,     y=0.0,     w=1280.0,   h=714.0
pitching_2: x=0.0,     y=0.0,     w=1280.0,   h=714.0
```

Observations:

- The first version gives useful crops for the batting clips.
- The pitching clips currently fall back to full-frame ROI, likely because the motion-edge candidate aggregation captures too much background or whole-scene motion.
- This is not a failure of the pipeline; it is a useful failure case for the automatic ROI module.

Next revision ideas:

- Add a maximum ROI area threshold before aggregation.
- Prefer vertically centered human-scale boxes instead of unioning too many candidates.
- Use median candidate box rather than full union for clips with many candidates.
- Add an optional action-region prior per clip type.
- Compare overlay videos to decide whether `auto_roi_raw` actually reduces wrong-person switches on batting clips.

## Iteration 13: Pose-Prior ROI for Tighter Subject Crops

Date: 2026-04-26

Commits:

- `b3b683d Add pose-prior ROI estimation`
- `3fe5930 Add pose-prior ROI pipeline command`

Goal:

- Improve automatic ROI after observing that `auto_roi_raw` kept too much background and cropped the head in `batting_2`.
- Use baseline pose landmarks as a subject prior while preserving a separate output condition.

Implementation:

- Added `auto_roi_pose_prior`.
- Reads `data/processed/poses/<clip_id>/baseline_raw.csv`.
- Filters confident baseline landmarks.
- Builds per-frame pose boxes.
- Aggregates boxes with percentile bounds instead of full min/max union.
- Adds asymmetric safety padding:
  - larger top padding to avoid head cropping,
  - horizontal padding for arms and bat motion,
  - bottom padding for lower-body context.
- Reruns MediaPipe on cropped frames and remaps keypoints back to full-frame normalized coordinates.

Command:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache \
XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/default.yaml run-pose-prior-roi --max-frames 30
```

Outputs generated:

- `data/interim/frames/<clip_id>/auto_roi_pose_prior.csv`
- `data/interim/rois/<clip_id>/auto_roi_pose_prior.csv`
- `data/processed/poses/<clip_id>/auto_roi_pose_prior.csv`
- `outputs/roi_debug/<clip_id>__auto_roi_pose_prior.mp4`
- `outputs/overlays/<clip_id>__auto_roi_pose_prior.mp4`

Validation:

- `pytest` passed: 8 tests.
- All four clips ran for 30 frames.
- Each clip produced 390 pose rows.

Selected ROI boxes:

```text
batting_1:  x=0.0,     y=0.0, w=1050.49, h=714.0
batting_2:  x=487.41,  y=0.0, w=468.16,  h=714.0
pitching_1: x=406.29,  y=0.0, w=207.48,  h=714.0
pitching_2: x=339.94,  y=0.0, w=269.63,  h=714.0
```

Observations:

- `batting_2` no longer has the head-cropping problem from `auto_roi_raw`.
- Pitching clips now have much tighter horizontal crops instead of full-frame ROI.
- The first pose-prior implementation is intentionally conservative vertically and keeps full frame height for all four clips.

Next revision ideas:

- Add vertical crop constraints once overlay review confirms the head, feet, and throwing arm are consistently inside the ROI.
- Compare `baseline_raw`, `auto_roi_raw`, and `auto_roi_pose_prior` visually in a report figure.
- Compute keypoint completeness and jitter across the three conditions.

## Iteration 14: ROI Ablation Metrics

Date: 2026-04-26

Commits:

- `2dbce97 Add ROI ablation metrics`
- `27f9339 Add ROI ablation summary table`

Goal:

- Quantify the effect of ROI conditions before adding more preprocessing.
- Compare `baseline_raw`, `auto_roi_raw`, and `auto_roi_pose_prior`.

Command:

```bash
.venv312/bin/python -m baseball_pose.cli --config configs/default.yaml summarize-roi-ablation
```

Outputs:

- `data/processed/metrics/roi_ablation.csv`
- `data/processed/metrics/roi_ablation_summary.csv`

Metrics:

- keypoint completeness for all joints,
- keypoint completeness for upper body,
- wrist missing rate,
- wrist temporal jitter,
- wrist trajectory smoothness,
- mean MediaPipe inference time per frame.

Validation:

- `pytest` passed: 9 tests.
- Metric table contains 120 long-form rows.
- Summary table contains 12 rows:
  - 4 clips x 3 conditions.

Initial findings:

- Completeness is 1.0 for the tested 30-frame subset across all three conditions, so this subset is too easy for completeness to distinguish conditions.
- ROI conditions generally reduce wrist jitter and smoothness values compared with baseline.
- `pitching_1` has identical baseline and `auto_roi_raw` pose metrics because `auto_roi_raw` fell back to a full-frame ROI.
- `auto_roi_pose_prior` reduces runtime for pitching clips by using a tighter horizontal crop.

Example summary:

```text
batting_1 left_wrist jitter:
baseline_raw          0.0891
auto_roi_raw          0.0142
auto_roi_pose_prior   0.0189

pitching_1 runtime ms/frame:
baseline_raw          63.35
auto_roi_raw          71.60
auto_roi_pose_prior   31.25
```

Interpretation:

- ROI is helping most clearly on trajectory stability and runtime.
- Completeness alone is not enough for this dataset slice.
- The report should combine metrics with overlay examples because lower jitter can also indicate over-stabilization or wrong-person lock-in.

Next steps:

- Generate report-ready visual comparisons for selected frames.
- Add wrist trajectory plots across conditions.
- Then implement angle features and smoothing.

## Iteration 15: Full-Video Baseline and Pose-Prior ROI Runs

Date: 2026-04-26

Commit:

- `6beeff7 Add full-video experiment config`

Goal:

- Expand from short 30-frame checks to full-length video processing.
- Preserve the previous short-run outputs by writing full-video artifacts to separate directories.
- Generate complete overlay and ROI-debug videos for report review.

Config:

- `configs/experiments/full_video.yaml`
- `project.data_dir: data_full`
- `project.output_dir: outputs_full`
- `video.max_frames_per_clip: null`
- Conditions run: `baseline_raw`, `auto_roi_pose_prior`

Commands:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml run-baseline

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml run-pose-prior-roi

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml summarize-roi-ablation
```

Outputs:

- `data_full/interim/frames/<clip_id>/baseline_raw.csv`
- `data_full/interim/frames/<clip_id>/auto_roi_pose_prior.csv`
- `data_full/interim/rois/<clip_id>/auto_roi_pose_prior.csv`
- `data_full/processed/poses/<clip_id>/baseline_raw.csv`
- `data_full/processed/poses/<clip_id>/auto_roi_pose_prior.csv`
- `outputs_full/overlays/<clip_id>__baseline_raw.mp4`
- `outputs_full/overlays/<clip_id>__auto_roi_pose_prior.mp4`
- `outputs_full/roi_debug/<clip_id>__auto_roi_pose_prior.mp4`
- `data_full/processed/metrics/roi_ablation.csv`
- `data_full/processed/metrics/roi_ablation_summary.csv`

Frame and pose record counts:

| clip_id | frames | baseline pose records | pose-prior ROI pose records |
| --- | ---: | ---: | ---: |
| batting_1 | 997 | 12961 | 12961 |
| batting_2 | 758 | 9854 | 9854 |
| pitching_1 | 624 | 8112 | 8112 |
| pitching_2 | 424 | 5512 | 5512 |

Pose-prior ROI boxes:

| clip_id | x | y | width | height | candidate_count |
| --- | ---: | ---: | ---: | ---: | ---: |
| batting_1 | 371.40 | 0.00 | 742.48 | 714.00 | 997 |
| batting_2 | 446.40 | 0.00 | 595.05 | 714.00 | 677 |
| pitching_1 | 184.72 | 0.00 | 750.86 | 714.00 | 615 |
| pitching_2 | 188.73 | 0.00 | 1007.41 | 714.00 | 424 |

Metrics summary:

- `batting_1`: pose-prior ROI preserved completeness at 1.0 and reduced wrist jitter relative to baseline.
- `batting_2`: pose-prior ROI reduced jitter but completeness dropped from about 0.893 to 0.765, so this clip needs visual inspection before treating ROI as an improvement.
- `pitching_1`: pose-prior ROI reduced runtime from about 37.06 ms/frame to 29.59 ms/frame and reduced wrist jitter, with a small completeness drop.
- `pitching_2`: completeness stayed at 1.0; ROI and baseline runtime were similar, with slightly lower jitter for pose-prior ROI.

Failures or limitations:

- Full-video `auto_roi_raw` was not rerun in this iteration; the ablation long table records `missing_pose_file` rows for that condition.
- `batting_2` remains the main weak case for ROI because the subject crop can lose useful body context.
- The current pose-prior crop is horizontal only, so the vertical extent remains full height.

Disk usage:

- `data_full`: about 7.6 GB.
- `outputs_full`: about 12 GB.

Next revision ideas:

- Review full-video overlays for wrong-person tracking and `batting_2` body coverage.
- Add report-ready still-frame comparisons from `outputs_full`.
- Continue with trajectory/angle feature extraction using full-video pose CSVs.

## Iteration 16: Full-Video Motion Features and Trajectory Figures

Date: 2026-04-26

Goal:

- Add a real feature extraction stage after full-video pose estimation.
- Generate report-friendly trajectory plots from the full-video feature CSVs.
- Keep generated data and figures outside git while tracking the reusable code and instructions.

Implementation:

- Added frame-level motion feature extraction from long-form pose landmarks.
- Added feature CSV writing under `data_full/processed/features/<clip_id>/<condition_id>.csv`.
- Added report figure generation under `outputs_full/figures/`.
- Added CLI commands:
  - `extract-features`
  - `make-figures`

Extracted features:

- left and right elbow angle,
- left and right shoulder angle,
- left and right knee angle,
- trunk tilt,
- left and right wrist x/y trajectory,
- left and right wrist speed.

Commands:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml extract-features

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml make-figures
```

Feature row counts:

| clip_id | condition_id | feature rows |
| --- | --- | ---: |
| batting_1 | baseline_raw | 997 |
| batting_1 | auto_roi_pose_prior | 997 |
| batting_2 | baseline_raw | 758 |
| batting_2 | auto_roi_pose_prior | 758 |
| pitching_1 | baseline_raw | 624 |
| pitching_1 | auto_roi_pose_prior | 624 |
| pitching_2 | baseline_raw | 424 |
| pitching_2 | auto_roi_pose_prior | 424 |

Report figures:

- `outputs_full/figures/batting_1__wrist_trajectories.png`
- `outputs_full/figures/batting_2__wrist_trajectories.png`
- `outputs_full/figures/pitching_1__wrist_trajectories.png`
- `outputs_full/figures/pitching_2__wrist_trajectories.png`

Validation:

- Added a unit test for angle extraction and wrist speed.
- Initial feature extraction attempt failed under sandboxed write permissions on the T7 drive; reran with approved elevated access and completed successfully.

Limitations:

- The trajectory figures currently compare wrist paths only; angle-time plots are still needed for a stronger report narrative.
- Wrist trajectories can still include wrong-person tracks if MediaPipe locks onto another person in the frame.

Next revision ideas:

- Add angle-over-time plots for elbow and shoulder angles.
- Add selected overlay still frames for report figures.
- Add a compact per-clip feature summary table with min/max/range for key angles and wrist speed.

## Iteration 17: Temporal Smoothing for Full-Video Trajectories

Date: 2026-04-26

Goal:

- Fix the overly noisy wrist trajectory plots from raw MediaPipe coordinates.
- Add a real temporal smoothing stage instead of the previous placeholder function.
- Generate smoothed pose CSVs, smoothed features, updated trajectory figures, and metrics.

Implementation:

- Implemented `smooth_pose_records` using:
  - low-confidence coordinate masking,
  - isolated jump outlier rejection,
  - short-gap linear interpolation,
  - Savitzky-Golay filtering on valid trajectory segments.
- Added CLI command:
  - `smooth-poses`
- Added full-video smooth conditions:
  - `baseline_raw_smooth`
  - `auto_roi_pose_prior_smooth`
- Updated `summarize-roi-ablation` to follow configured conditions, so smooth metrics are included.

Commands:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml smooth-poses

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml extract-features

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml make-figures

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml summarize-roi-ablation
```

Outputs:

- `data_full/processed/poses/<clip_id>/baseline_raw_smooth.csv`
- `data_full/processed/poses/<clip_id>/auto_roi_pose_prior_smooth.csv`
- `data_full/processed/features/<clip_id>/baseline_raw_smooth.csv`
- `data_full/processed/features/<clip_id>/auto_roi_pose_prior_smooth.csv`
- Updated `outputs_full/figures/<clip_id>__wrist_trajectories.png`
- Updated `data_full/processed/metrics/roi_ablation_summary.csv`

Metric changes:

- `batting_1` auto-ROI left wrist jitter: `0.0110 -> 0.0068`.
- `batting_1` auto-ROI left wrist smoothness: `0.0189 -> 0.0056`.
- `batting_2` baseline left wrist jitter: `0.0323 -> 0.0174`.
- `batting_2` auto-ROI left wrist jitter: `0.0224 -> 0.0137`.
- `pitching_1` auto-ROI left wrist jitter: `0.0168 -> 0.0086`.
- `pitching_2` auto-ROI left wrist smoothness: `0.0136 -> 0.0045`.

Validation:

- Added unit tests for isolated jump rejection and low-confidence gap interpolation.
- Smoothed pose record counts match the original pose record counts for all four clips.

Limitations:

- Temporal smoothing reduces jitter but does not fully solve long wrong-person tracking segments.
- If the detector follows the wrong person for many consecutive frames, smoothing can make that wrong track look cleaner.
- The report should present smoothing as temporal denoising, not as identity tracking.

Next revision ideas:

- Add visual still-frame comparisons for raw vs smoothed overlays.
- Add a simple identity-consistency gate using torso center continuity.
- Add angle-over-time plots for the smoothed conditions.

## Iteration 18: Posture Analysis Features and Figures

Date: 2026-05-01

Goal:

- Expand the feature CSV beyond wrist trajectories into report-ready posture analysis.
- Select metrics that are meaningful for pitching and batting while remaining honest about the current 2D skeleton-only inputs.
- Add visualization for rotation chain, hip-shoulder separation, COM path, knee extension, and hand-speed proxy.

Implemented feature additions:

- pelvis rotation from the left-hip to right-hip line,
- shoulder/trunk rotation from the left-shoulder to right-shoulder line,
- hip-shoulder separation as the signed shoulder-vs-pelvis angle,
- pelvis and trunk rotation velocity from frame-to-frame signed angle deltas,
- approximate center of mass as the average of available shoulder, hip, knee, and ankle landmarks,
- left/right knee extension from the first valid frame,
- left/right knee angular velocity,
- hand-speed proxy as the larger visible wrist speed per frame.

Implemented visualization additions:

- `outputs_full/figures/<clip_id>__posture_analysis.png`
- Four-panel layout:
  - pelvis / shoulder rotation and hip-shoulder separation,
  - pelvis and trunk rotation velocity,
  - approximate COM path,
  - knee extension and hand-speed proxy.

Design decision:

- Did not implement exact stride length, SFC/MER/BR/IMP windows, shoulder external rotation, bat-tip velocity, attack angle, contact location, or swing plane yet. Those require event annotations, 3D joint orientation, ball/bat tracking, or a calibrated target axis. Reporting them from the current 2D body-only pose CSV would be misleading.

Validation:

- Unit tests cover the new angle helpers and posture proxy extraction.
- `python -m compileall src tests` passed after removing macOS `._*` metadata files.
- `.venv312/bin/python -m pytest` passed: 15 tests.
- `.venv312/bin/ruff check` passed on the modified feature, CSV, figure, plot, and feature-test files.
- Regenerated full-video feature CSVs for all four clips and four configured conditions.
- Generated both wrist trajectory and posture analysis figures for all four clips.
- Existing feature extraction and figure commands remain the same:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml extract-features

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml make-figures
```

## Iteration 19: Stronger Pose Stabilization for Cleaner Report Output

Date: 2026-05-01

Goal:

- Reduce visible skeleton jitter and unreadable trajectory figures.
- Prioritize the best stabilized condition because the baseline comparison is not central to the final report.

Implementation:

- Added a torso-continuity gate before per-joint smoothing.
  - The gate tracks the hip/shoulder torso center.
  - Frames that jump too far from the previous accepted torso center are masked before interpolation.
  - This targets short wrong-person locks and full-skeleton swaps, which per-joint filters alone cannot fix.
- Added a median filter stage before Savitzky-Golay smoothing.
  - This reduces alternating frame-to-frame landmark wiggle.
- Strengthened default smoothing parameters:
  - longer interpolation gap,
  - larger Savitzky-Golay window,
  - median refinement,
  - moving-average refinement,
  - lower jump threshold.
- Updated default report figure selection so figures prefer smoothed ROI conditions when available.
- Added `render-overlays`, which redraws overlay videos from existing smoothed pose CSVs instead of using only the raw inference-time overlays.

Commands:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml smooth-poses --condition auto_roi_pose_prior

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml extract-features

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml make-figures

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml render-overlays --condition auto_roi_pose_prior_smooth
```

Validation:

- `python -m pytest tests/test_smoothing.py` passed.
- `python -m compileall src/baseball_pose/postprocess src/baseball_pose/pipeline tests/test_smoothing.py` passed after removing macOS `._*` metadata files.
- Ruff passed on the modified smoothing, postprocess, figures, overlays, CLI, and smoothing-test files.
- Full test suite passed: 17 tests.
- Regenerated `auto_roi_pose_prior_smooth` pose CSVs, feature CSVs, report figures, smoothed overlay videos, and ROI ablation metrics.

Metric changes after stronger stabilization:

| clip_id | metric | auto_roi_pose_prior | auto_roi_pose_prior_smooth |
| --- | --- | ---: | ---: |
| batting_1 | left wrist jitter | 0.0110 | 0.0028 |
| batting_1 | left wrist smoothness | 0.0189 | 0.0014 |
| batting_2 | keypoint completeness | 0.7652 | 0.8650 |
| batting_2 | left wrist jitter | 0.0224 | 0.0049 |
| pitching_1 | left wrist jitter | 0.0168 | 0.0035 |
| pitching_1 | left wrist smoothness | 0.0276 | 0.0007 |
| pitching_2 | left wrist jitter | 0.0080 | 0.0037 |
| pitching_2 | left wrist smoothness | 0.0136 | 0.0006 |

Remaining limitation:

- This is still post-processing of a single detected skeleton. If MediaPipe misses the athlete for a long continuous segment or consistently tracks another person, smoothing can hide short gaps but cannot recover the correct body without better person selection or manual/automatic ROI tightening.

## Iteration 20: Center-Prior Subject Recognition

Date: 2026-05-01

Goal:

- Reduce wrong-person selection from spectators and nearby players.
- Use the project-specific prior that the true athlete is centered in every raw video.

Implementation:

- Added a hard center-prior ROI condition:
  - condition: `center_prior_roi`
  - smoothed condition: `center_prior_roi_smooth`
  - default crop: center x/y at 0.5, width ratio 0.62, full height.
- Added CLI command:
  - `run-center-prior-roi`
- Added config entries for the new center-prior conditions.
- Updated report figure and overlay selection so `center_prior_roi_smooth` is preferred when available.
- Fixed the previously incomplete `run_auto_roi_clip` path while adding the new center-prior path.

Commands run:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml run-center-prior-roi

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml smooth-poses --condition center_prior_roi

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml extract-features

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml make-figures

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml render-overlays --condition center_prior_roi_smooth
```

Outputs:

- `data_full/processed/poses/<clip_id>/center_prior_roi.csv`
- `data_full/processed/poses/<clip_id>/center_prior_roi_smooth.csv`
- `data_full/processed/features/<clip_id>/center_prior_roi_smooth.csv`
- `outputs_full/roi_debug/<clip_id>__center_prior_roi.mp4`
- `outputs_full/overlays/<clip_id>__center_prior_roi_smooth.mp4`
- Updated default report figures under `outputs_full/figures/`.

Metric notes:

| clip_id | metric | auto_roi_pose_prior_smooth | center_prior_roi_smooth |
| --- | --- | ---: | ---: |
| batting_1 | completeness | 0.9829 | 0.9829 |
| batting_1 | left wrist jitter | 0.0028 | 0.0029 |
| batting_2 | completeness | 0.8650 | 0.9591 |
| batting_2 | left wrist jitter | 0.0049 | 0.0079 |
| pitching_1 | completeness | 1.0000 | 1.0000 |
| pitching_1 | left wrist jitter | 0.0035 | 0.0034 |
| pitching_2 | completeness | 1.0000 | 0.9991 |
| pitching_2 | left wrist jitter | 0.0037 | 0.0034 |

Interpretation:

- Center-prior ROI is meant to improve subject selection, not necessarily minimize every smoothness metric.
- The biggest quantitative gain is `batting_2` completeness, which increased from 0.8650 to 0.9591.
- Qualitatively, the central crop should reduce the probability that MediaPipe locks onto side/background people because they are outside the inference image.

Validation:

- Full test suite passed: 18 tests.
- `python -m compileall src tests` passed after removing macOS `._*` metadata files.
- Ruff passed on the modified ROI, pipeline, CLI, and ROI test files.

## Iteration 21: Body-Prior Irregular Mask Inference

Date: 2026-05-03

Goal:

- Reduce remaining wrong-person switches when a nearby player or spectator is still inside the centered rectangular crop.
- Use an irregular subject proposal instead of another rectangle.

Implementation:

- Added `body_prior_mask_roi`, which uses `center_prior_roi_smooth` as the prior body track.
- For each frame, the pipeline:
  - reads the prior subject skeleton,
  - builds a dynamic body ROI around prior landmarks,
  - draws a skeleton-shaped mask from torso, limb connections, and joint circles,
  - blacks out pixels outside that irregular mask,
  - runs MediaPipe on the masked crop,
  - remaps landmarks back to full-frame coordinates.
- Added smoothed condition:
  - `body_prior_mask_roi_smooth`
- Added CLI command:
  - `run-body-prior-mask-roi`
- Updated default figure and overlay selection to prefer `body_prior_mask_roi_smooth`.

Commands run:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml run-body-prior-mask-roi

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml smooth-poses --condition body_prior_mask_roi

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml extract-features

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml make-figures

MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml render-overlays --condition body_prior_mask_roi_smooth
```

Outputs:

- `data_full/processed/poses/<clip_id>/body_prior_mask_roi.csv`
- `data_full/processed/poses/<clip_id>/body_prior_mask_roi_smooth.csv`
- `data_full/processed/features/<clip_id>/body_prior_mask_roi_smooth.csv`
- `outputs_full/overlays/<clip_id>__body_prior_mask_roi_smooth.mp4`
- Updated default report figures under `outputs_full/figures/`.

Metric notes:

| clip_id | metric | center_prior_roi_smooth | body_prior_mask_roi_smooth |
| --- | --- | ---: | ---: |
| batting_1 | left wrist jitter | 0.0029 | 0.0019 |
| batting_1 | runtime ms/frame | 27.60 | 15.74 |
| batting_2 | completeness | 0.9591 | 0.9550 |
| batting_2 | left wrist jitter | 0.0079 | 0.0067 |
| pitching_1 | right wrist jitter | 0.0044 | 0.0039 |
| pitching_1 | runtime ms/frame | 27.50 | 16.22 |
| pitching_2 | completeness | 0.9991 | 1.0000 |
| pitching_2 | runtime ms/frame | 27.70 | 15.57 |

Interpretation:

- The body-prior mask is more aggressive than the centered rectangle. It can sacrifice a small amount of completeness when prior limbs are too tight, but it removes more non-subject pixels before inference.
- The main value is qualitative subject isolation: nearby bodies inside the center crop are mostly blacked out unless they overlap the prior subject skeleton region.
- Runtime also improved because the dynamic crop is tighter than the full centered crop.

Validation:

- Full test suite passed: 20 tests.
- `python -m compileall src tests` passed after removing macOS `._*` metadata files.
- Ruff passed on the modified body-mask, pipeline, CLI, and body-mask-test files.

## Iteration 22: Body-Mask Intermediate Debug Outputs

Date: 2026-05-03

Goal:

- Export the intermediate irregular proposal results for reporting and inspection.
- Make the body-prior mask visible before final MediaPipe inference.

Implementation:

- Added `render-body-mask-debug`.
- Added two intermediate debug videos per clip:
  - `proposal_overlay`: original frame with the dynamic ROI rectangle and green irregular body mask overlaid.
  - `masked_frame`: full-frame black canvas containing only the pixels retained by the body-prior mask.
- Added output path helpers under `outputs_full/body_mask_debug/`.

Command:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml render-body-mask-debug --condition body_prior_mask_roi
```

Outputs:

- `outputs_full/body_mask_debug/batting_1__body_prior_mask_roi__proposal_overlay.mp4`
- `outputs_full/body_mask_debug/batting_1__body_prior_mask_roi__masked_frame.mp4`
- `outputs_full/body_mask_debug/batting_2__body_prior_mask_roi__proposal_overlay.mp4`
- `outputs_full/body_mask_debug/batting_2__body_prior_mask_roi__masked_frame.mp4`
- `outputs_full/body_mask_debug/pitching_1__body_prior_mask_roi__proposal_overlay.mp4`
- `outputs_full/body_mask_debug/pitching_1__body_prior_mask_roi__masked_frame.mp4`
- `outputs_full/body_mask_debug/pitching_2__body_prior_mask_roi__proposal_overlay.mp4`
- `outputs_full/body_mask_debug/pitching_2__body_prior_mask_roi__masked_frame.mp4`

Validation:

- `python -m pytest tests/test_body_mask.py` passed.
- Targeted `compileall` passed.
- Ruff passed on modified body-mask, debug pipeline, path, CLI, and body-mask-test files.

## Iteration 23: Skeleton-Free Image Proposal Debug for Batting 1

Date: 2026-05-03

Goal:

- Avoid using a potentially wrong skeleton as the proposal source.
- Test a pure image-processing ROI/mask proposal on `batting_1` only for faster iteration.

Implementation:

- Added `render-image-proposal-debug`.
- Added an image-processing proposal that combines:
  - center prior,
  - previous-frame difference,
  - MOG2 foreground extraction,
  - GrabCut initialized from image evidence,
  - connected-component scoring by center distance, area, and vertical extent.
- Added downsampled processing and `--max-frames` support so debugging is bounded and usable.

Command:

```bash
MPLCONFIGDIR=/tmp/baseball_mpl_cache XDG_CACHE_HOME=/tmp/baseball_xdg_cache \
.venv312/bin/python -m baseball_pose.cli --config configs/experiments/full_video.yaml render-image-proposal-debug --clip-id batting_1 --max-frames 180
```

Outputs:

- `outputs_full/image_proposal_debug/batting_1__image_center_motion_grabcut__proposal_overlay.mp4`
- `outputs_full/image_proposal_debug/batting_1__image_center_motion_grabcut__masked_frame.mp4`

Validation:

- `python -m pytest tests/test_image_proposal.py` passed.
- Targeted `compileall` passed.
- Ruff passed on image-proposal, image-proposal-debug, CLI, path, and image-proposal-test files.

Note:

- This is intentionally not wired into MediaPipe inference yet. The next decision should be based on visual inspection of the proposal overlay and masked-frame videos.

Revision:

- The first image proposal version flickered heavily because it segmented each frame mostly independently.
- Revised the proposal to:
  - apply LAB CLAHE and unsharp masking before segmentation,
  - lower the motion threshold after contrast enhancement,
  - seed GrabCut from the previous accepted mask,
  - score connected components by overlap with the previous mask,
  - blend the current mask with the previous mask and constrain it to current support.
- Regenerated the `batting_1` 180-frame image proposal debug videos after the stability update.
