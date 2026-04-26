# baseball-analysis

Reproducible 2D baseball pose-analysis pipeline for comparing raw video pose estimation against lightweight ROI and image-processing conditions.

## Current Scope

The MVP follows this pipeline:

```text
raw video -> frame sampling -> optional ROI -> CLAHE -> MediaPipe Pose
  -> temporal post-processing -> motion features -> evaluation -> visualization
```

Current next ROI direction:

```text
sampled frames
  -> frame-difference motion mask
  -> Canny/Sobel edge mask
  -> connected components / contour boxes
  -> clip-level fixed auto ROI
  -> MediaPipe Pose on cropped ROI
  -> remap keypoints to full-frame coordinates
```

The planned condition name is `auto_roi_raw`, followed by `auto_roi_clahe` after the ROI-only effect is inspected. Manual ROI remains a fallback and comparison point when automatic ROI follows another person or background structure.

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

MediaPipe Tasks also needs a local pose model file:

```bash
mkdir -p models
curl -L \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task \
  -o models/pose_landmarker_lite.task
```

The `.task` file is ignored by git.

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

## MVP Conditions

- `baseline_raw`
- `roi_clahe`
- `roi_clahe_smooth`

Implementation should keep all outputs tagged by `clip_id` and `condition_id` so processed and baseline results remain comparable.
