"""Command-line entry points for the baseball pose pipeline."""

from __future__ import annotations

import argparse

from baseball_pose.config import load_config
from baseball_pose.io.metadata import load_clips
from baseball_pose.pipeline.run_experiment import run_baseline_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baseball-pose")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "command",
        choices=["validate-config", "plan", "run-baseline"],
        help="Command to run.",
    )
    parser.add_argument(
        "--clip-id",
        action="append",
        help="Clip id to process. Repeat to process multiple clips. Defaults to all configured clips.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Override config video.max_frames_per_clip for quick runs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.command == "validate-config":
        print(f"Loaded config: {config.path}")
        print(f"Configured clips: {len(config.clip_ids)}")
        print(f"Configured conditions: {', '.join(config.condition_ids)}")
        return

    if args.command == "plan":
        print("Pipeline stages:")
        for stage in config.pipeline_stages:
            print(f"- {stage}")
        return

    if args.command == "run-baseline":
        clips = load_clips(config.clips_file)
        clip_filter = set(args.clip_id) if args.clip_id else set(config.clip_ids)
        results = run_baseline_experiment(
            clips,
            config,
            clip_ids=clip_filter,
            max_frames=args.max_frames,
        )
        for result in results:
            print(
                f"{result.clip_id}/{result.condition_id}: "
                f"{result.frame_count} frames, {result.pose_record_count} pose records"
            )
            print(f"  frames: {result.frames_csv}")
            print(f"  poses: {result.poses_csv}")
            print(f"  overlay: {result.overlay_video}")
        return


if __name__ == "__main__":
    main()
