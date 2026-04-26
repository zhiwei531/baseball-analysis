# baseball-analysis

Reproducible 2D baseball pose-analysis pipeline for comparing raw video pose estimation against lightweight ROI and image-processing conditions.

## Current Scope

The MVP follows this pipeline:

```text
raw video -> frame sampling -> optional ROI -> CLAHE -> MediaPipe Pose
  -> temporal post-processing -> motion features -> evaluation -> visualization
```

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

## MVP Conditions

- `baseline_raw`
- `roi_clahe`
- `roi_clahe_smooth`

Implementation should keep all outputs tagged by `clip_id` and `condition_id` so processed and baseline results remain comparable.
