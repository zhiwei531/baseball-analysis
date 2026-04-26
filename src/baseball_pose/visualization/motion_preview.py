"""OpenCV motion-track preview visualization."""

from __future__ import annotations

from pathlib import Path

from baseball_pose.io.video import FrameRecord, read_frame, write_video_from_frames


def create_motion_preview(
    frames: list[FrameRecord],
    output_dir: str | Path,
    output_video_path: str | Path,
    fps: float,
    max_corners: int = 80,
    max_track_length: int = 40,
    redetect_interval: int = 15,
) -> list[Path]:
    """Create a lightweight point-track visualization from sampled frames.

    This is not a pose-estimation result. It is a video-processing sanity check
    that shows whether frame sampling, output writing, and motion visualization
    are working before pose inference is stable.
    """

    cv2 = _require_cv2()
    if not frames:
        raise ValueError("frames cannot be empty.")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tracks: list[list[tuple[float, float]]] = []
    previous_gray = None
    output_paths: list[Path] = []

    for index, frame_record in enumerate(frames):
        frame = read_frame(frame_record.frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if previous_gray is not None and tracks:
            previous_points = _track_endpoints(tracks)
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                previous_gray,
                gray,
                previous_points,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            tracks = _advance_tracks(tracks, next_points, status, max_track_length)

        if index == 0 or index % redetect_interval == 0 or len(tracks) < max_corners // 3:
            tracks.extend(_detect_new_tracks(gray, tracks, max_corners - len(tracks)))

        preview = _draw_tracks(frame, tracks, frame_record)
        output_path = target_dir / frame_record.frame_path.name
        if not cv2.imwrite(str(output_path), preview):
            raise RuntimeError(f"Could not write motion preview frame: {output_path}")
        output_paths.append(output_path)
        previous_gray = gray

    write_video_from_frames(output_paths, output_video_path, fps=fps)
    return output_paths


def _track_endpoints(tracks: list[list[tuple[float, float]]]):
    cv2 = _require_cv2()
    import numpy as np

    points = [track[-1] for track in tracks]
    return np.asarray(points, dtype="float32").reshape(-1, 1, 2)


def _advance_tracks(
    tracks: list[list[tuple[float, float]]],
    next_points,
    status,
    max_track_length: int,
) -> list[list[tuple[float, float]]]:
    updated: list[list[tuple[float, float]]] = []
    for track, point, is_valid in zip(tracks, next_points.reshape(-1, 2), status.reshape(-1)):
        if not is_valid:
            continue
        x, y = float(point[0]), float(point[1])
        next_track = [*track, (x, y)][-max_track_length:]
        updated.append(next_track)
    return updated


def _detect_new_tracks(
    gray,
    existing_tracks: list[list[tuple[float, float]]],
    count: int,
) -> list[list[tuple[float, float]]]:
    cv2 = _require_cv2()
    if count <= 0:
        return []

    mask = _feature_mask(gray, existing_tracks)
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=count,
        qualityLevel=0.01,
        minDistance=12,
        blockSize=7,
        mask=mask,
    )
    if points is None:
        return []
    return [[(float(point[0][0]), float(point[0][1]))] for point in points]


def _feature_mask(gray, existing_tracks: list[list[tuple[float, float]]]):
    cv2 = _require_cv2()
    import numpy as np

    mask = np.full(gray.shape, 255, dtype="uint8")
    for track in existing_tracks:
        x, y = track[-1]
        cv2.circle(mask, (int(x), int(y)), 10, 0, -1)
    return mask


def _draw_tracks(frame, tracks: list[list[tuple[float, float]]], frame_record: FrameRecord):
    cv2 = _require_cv2()
    output = frame.copy()

    for track in tracks:
        if len(track) < 2:
            x, y = track[-1]
            cv2.circle(output, (int(x), int(y)), 3, (80, 220, 120), -1, cv2.LINE_AA)
            continue
        for previous, current in zip(track, track[1:]):
            cv2.line(
                output,
                (int(previous[0]), int(previous[1])),
                (int(current[0]), int(current[1])),
                (255, 120, 40),
                2,
                cv2.LINE_AA,
            )
        x, y = track[-1]
        cv2.circle(output, (int(x), int(y)), 3, (60, 230, 255), -1, cv2.LINE_AA)

    label = f"{frame_record.clip_id} frame={frame_record.frame_index:04d} t={frame_record.timestamp_sec:.2f}s tracks={len(tracks)}"
    cv2.rectangle(output, (12, 12), (12 + min(760, 12 * len(label)), 44), (0, 0, 0), -1)
    cv2.putText(output, label, (22, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return output


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for motion preview visualization. Install dependencies first."
        ) from exc

    return cv2
