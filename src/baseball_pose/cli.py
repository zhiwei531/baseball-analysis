"""Command-line entry points for the baseball pose pipeline."""

from __future__ import annotations

import argparse

from baseball_pose.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baseball-pose")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "command",
        choices=["validate-config", "plan"],
        help="Command to run.",
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


if __name__ == "__main__":
    main()
