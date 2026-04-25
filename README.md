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

## Initial Checks

```bash
python -m compileall src
python -m pytest
baseball-pose validate-config
baseball-pose plan
```

## MVP Conditions

- `baseline_raw`
- `roi_clahe`
- `roi_clahe_smooth`

Implementation should keep all outputs tagged by `clip_id` and `condition_id` so processed and baseline results remain comparable.
