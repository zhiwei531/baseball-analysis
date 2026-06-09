"""Build a GVHMR input video from project-sampled frames.

Using the frame manifest avoids relying on phone-video rotation metadata, which
can be dropped when GVHMR rewrites MOV files into MP4.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from baseball_pose.io.frame_csv import read_frame_records
from baseball_pose.io.video import write_video_from_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    frames = read_frame_records(args.frames_csv)
    write_video_from_frames([frame.frame_path for frame in frames], args.output, fps=args.fps)
    print(f"wrote {len(frames)} frames: {args.output}")


if __name__ == "__main__":
    main()
