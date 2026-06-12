"""Generate a small YOLO-only equipment tracking sample.

The sample intentionally passes no pose CSV to the equipment tracker, so the
output proves bat/ball tracking can run independently from 2D/3D human pose.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from baseball_pose.equipment.detection import EquipmentTrackingConfig, _resolve_yolo_device, detect_equipment_tracks
from baseball_pose.io.frame_csv import read_frame_records, write_frame_records
from baseball_pose.io.object_csv import write_object_tracks
from baseball_pose.io.video import read_frame, write_video_from_frames
from baseball_pose.visualization.equipment import draw_equipment_overlay


DEFAULT_CLIPS = ("benchmark_hit_vertical_02", "benchmark_pitch_vertical_10")
SUPPORTED_CLIPS = (
    "benchmark_hit_vertical_02",
    "benchmark_hit_horizontal_06",
    "benchmark_pitch_vertical_10",
    "benchmark_pitch_vertical_09",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pose-independent YOLO object tracking sample.")
    parser.add_argument("--frames", type=int, default=12, help="Number of frames per clip to process.")
    parser.add_argument("--data-dir", default="data_full/benchmark_yolo_object_sample")
    parser.add_argument("--output-dir", default="outputs_full/benchmark_yolo_object_sample")
    parser.add_argument("--source-data-dir", default="data_full/benchmark_rtmpose_test")
    parser.add_argument("--source-condition", default="image_center_motion_grabcut_pose")
    parser.add_argument("--condition", default="yolo_object_sample")
    parser.add_argument("--model", default="external/GVHMR/inputs/checkpoints/yolo/yolov8x.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.18)
    parser.add_argument(
        "--detect",
        choices=("auto", "bat", "ball", "both"),
        default="auto",
        help="Objects to detect. auto uses bat for hitting clips and ball for pitching clips.",
    )
    parser.add_argument(
        "--clip-id",
        action="append",
        help=f"Clip id to process. Known examples: {', '.join(SUPPORTED_CLIPS)}.",
    )
    args = parser.parse_args()

    clips = args.clip_id if args.clip_id else list(DEFAULT_CLIPS)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    resolved_device = _resolve_yolo_device(args.device)
    print(f"YOLO device: {resolved_device}")

    for clip_id in clips:
        is_batting = "hit" in clip_id
        uses_batting_objects = is_batting or args.detect in {"bat", "both"}
        source_frames = (
            Path(args.source_data_dir)
            / "interim"
            / "frames"
            / clip_id
            / f"{args.source_condition}.csv"
        )
        frames = read_frame_records(source_frames)[: args.frames]
        frame_manifest = data_dir / "interim" / "frames" / clip_id / f"{args.condition}.csv"
        write_frame_records(frame_manifest, frames)

        config = EquipmentTrackingConfig(
            detector_backend="yolo",
            yolo_model_path=args.model,
            yolo_confidence=args.conf,
            yolo_image_size=args.imgsz,
            yolo_device=args.device,
            detect_bat=_should_detect_bat(args.detect, is_batting),
            detect_ball=_should_detect_ball(args.detect, is_batting),
            use_pose_priors=False,
            interpolate_max_gap_frames=3,
            bat_smoothing_window_frames=5,
            ball_min_track_length_frames=3 if uses_batting_objects else 1,
            ball_track_max_gap_frames=2,
            ball_max_y_ratio=1.0 if uses_batting_objects and _should_detect_ball(args.detect, is_batting) else 0.66,
            ball_max_below_anchor_ratio=1.0 if uses_batting_objects and _should_detect_ball(args.detect, is_batting) else 0.20,
        )
        records = detect_equipment_tracks(
            frames_csv=frame_manifest,
            clip_id=clip_id,
            condition_id=args.condition,
            pose_csv=None,
            config=config,
        )
        object_path = data_dir / "processed" / "objects" / clip_id / f"{args.condition}.csv"
        write_object_tracks(object_path, records)

        overlay_dir = output_dir / "object_overlays" / "frames" / clip_id / args.condition
        overlay_dir.mkdir(parents=True, exist_ok=True)
        records_by_frame = defaultdict(list)
        for record in records:
            records_by_frame[record.frame_index].append(record)
        overlay_paths = []
        tracks: dict[str, list[tuple[int, int]]] = {"bat": [], "ball": []}
        for frame in frames:
            image = read_frame(frame.frame_path)
            frame_records = records_by_frame.get(frame.frame_index, [])
            _update_tracks(tracks, frame_records, frame.width, frame.height)
            overlay = draw_equipment_overlay(image, frame_records, tracks=tracks)
            overlay_path = overlay_dir / f"{clip_id}__{args.condition}__frame_{frame.frame_index:06d}.png"
            _write_image(overlay_path, overlay)
            overlay_paths.append(overlay_path)
        video_path = output_dir / "object_overlays" / f"{clip_id}__{args.condition}.mp4"
        write_video_from_frames(overlay_paths, video_path, fps=30)

        counts: dict[str, int] = {}
        sources: dict[str, int] = {}
        for record in records:
            counts[record.object_name] = counts.get(record.object_name, 0) + 1
            sources[record.source] = sources.get(record.source, 0) + 1
        print(f"{clip_id}: {len(records)} records {counts} {sources}")
        print(f"  objects: {object_path}")
        print(f"  overlay: {video_path}")


def _should_detect_bat(mode: str, is_batting: bool) -> bool:
    return mode in {"bat", "both"} or (mode == "auto" and is_batting)


def _should_detect_ball(mode: str, is_batting: bool) -> bool:
    return mode in {"ball", "both"} or (mode == "auto" and not is_batting)


def _update_tracks(
    tracks: dict[str, list[tuple[int, int]]],
    records,
    width: int | None,
    height: int | None,
) -> None:
    if width is None or height is None:
        return
    for record in records:
        if record.x is None or record.y is None:
            continue
        tracks.setdefault(record.object_name, []).append((int(record.x * width), int(record.y * height)))


def _write_image(path: Path, image) -> None:
    cv2 = _require_cv2()
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write image: {path}")


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("OpenCV is required for YOLO object sample rendering.") from exc
    return cv2


if __name__ == "__main__":
    main()
