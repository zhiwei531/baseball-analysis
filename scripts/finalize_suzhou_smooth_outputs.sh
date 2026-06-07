#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv312/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT/configs/experiments/full_video_suzhou.yaml}"
CONDITION_RAW="image_center_motion_grabcut_pose"
CONDITION_SMOOTH="image_center_motion_grabcut_pose_smooth"
CLIPS_CSV="$ROOT/data/metadata/clips_suzhou_full.csv"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/baseball_mpl_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp/baseball_xdg_cache}"

mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

CLIP_IDS=()
while IFS= read -r clip_id; do
  CLIP_IDS+=("$clip_id")
done < <(
  "$PYTHON_BIN" - "$CLIPS_CSV" <<'PY'
import csv
import sys
from pathlib import Path

clips_csv = Path(sys.argv[1]).resolve()
with clips_csv.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        print(row["clip_id"])
PY
)

"$PYTHON_BIN" -m baseball_pose.cli --config "$CONFIG_PATH" validate-config

for clip_id in "${CLIP_IDS[@]}"; do
  "$PYTHON_BIN" -m baseball_pose.cli --config "$CONFIG_PATH" smooth-poses --clip-id "$clip_id" --condition "$CONDITION_RAW"
  "$PYTHON_BIN" -m baseball_pose.cli --config "$CONFIG_PATH" extract-features --clip-id "$clip_id" --condition "$CONDITION_SMOOTH"
  "$PYTHON_BIN" -m baseball_pose.cli --config "$CONFIG_PATH" make-figures --clip-id "$clip_id" --condition "$CONDITION_SMOOTH"
  "$PYTHON_BIN" -m baseball_pose.cli --config "$CONFIG_PATH" render-overlays --clip-id "$clip_id" --condition "$CONDITION_SMOOTH"
done
