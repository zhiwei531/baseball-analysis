"""Planning and placeholder execution for 3D lifting stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from math import nan

from baseball_pose.config import RuntimeConfig, resolve_pose3d_config
from baseball_pose.io.frame_csv import read_frame_records
from baseball_pose.io.metadata import ClipMetadata
from baseball_pose.io.paths import feature3d_path, frame_manifest_path, pose3d_path, pose_path
from baseball_pose.io.pose_csv import read_pose_records
from baseball_pose.io.pose3d_csv import write_pose3d_records
from baseball_pose.io.video import FrameRecord, read_frame
from baseball_pose.pose.quality import threshold_for_joint
from baseball_pose.pose3d.external_video_hmr import read_external_video_hmr_records
from baseball_pose.pose3d.mediapipe_world import MediaPipeWorldPoseEstimator
from baseball_pose.pose.mediapipe_pose import _require_cv2
from baseball_pose.pose.schema import pose_score
from baseball_pose.preprocessing.image_proposal import (
    ImageProposalTracker,
    apply_image_proposal_mask,
    create_center_motion_grabcut_proposal,
)
from baseball_pose.preprocessing.image_proposal_config import image_proposal_roi_config
from baseball_pose.preprocessing.roi import crop_to_roi


@dataclass(frozen=True)
class Pose3DPlan:
    clip_id: str
    input_condition_id: str
    source_condition_id: str
    output_condition_id: str
    input_pose_path: Path
    input_frames_path: Path
    output_pose3d_path: Path
    output_feature3d_path: Path
    backend: str
    root_joint: str


def build_pose3d_plan(
    clip: ClipMetadata,
    config: RuntimeConfig,
    *,
    input_condition_id: str,
    output_condition_id: str | None = None,
) -> Pose3DPlan:
    pose3d_config = resolve_pose3d_config(config.raw, clip.clip_id)
    source_condition_id = _source_condition_for_3d(input_condition_id)
    derived_output_condition = output_condition_id or f"{input_condition_id}_3d"
    return Pose3DPlan(
        clip_id=clip.clip_id,
        input_condition_id=input_condition_id,
        source_condition_id=source_condition_id,
        output_condition_id=derived_output_condition,
        input_pose_path=pose_path(config.data_dir, clip.clip_id, input_condition_id),
        input_frames_path=frame_manifest_path(config.data_dir, clip.clip_id, source_condition_id),
        output_pose3d_path=pose3d_path(config.data_dir, clip.clip_id, derived_output_condition),
        output_feature3d_path=feature3d_path(config.data_dir, clip.clip_id, derived_output_condition),
        backend=str(pose3d_config.get("backend", "mediapipe_world")),
        root_joint=str(pose3d_config.get("root_joint", "pelvis_center")),
    )


def lift_pose_sequence(plan: Pose3DPlan, clip: ClipMetadata, config: RuntimeConfig) -> int:
    """Execute a real 3D backend for supported source conditions."""

    pose3d_config = resolve_pose3d_config(config.raw, clip.clip_id)
    if plan.backend in {"external_video_hmr", "gvhmr", "wham"}:
        records = _lift_external_video_hmr(plan, clip, config, pose3d_config)
        if plan.input_pose_path.exists():
            records = _gate_pose3d_with_2d_prior(
                records,
                pose2d_path=plan.input_pose_path,
                pose3d_config=pose3d_config,
                hard_gate=bool(pose3d_config.get("gate_with_2d_prior", False)),
            )
        write_pose3d_records(plan.output_pose3d_path, records)
        return len(records)

    if plan.backend != "mediapipe_world":
        raise NotImplementedError(f"Unsupported configured 3D backend: {plan.backend}")
    if plan.source_condition_id not in {"baseline_raw", "image_center_motion_grabcut_pose"}:
        raise NotImplementedError(
            "The first real 3D backend currently supports only baseline_raw and "
            "image_center_motion_grabcut_pose as source conditions."
        )
    if not plan.input_frames_path.exists():
        raise FileNotFoundError(
            f"3D lifting requires existing sampled frames for {plan.source_condition_id}: "
            f"{plan.input_frames_path}"
        )

    frames = read_frame_records(plan.input_frames_path)
    estimator = MediaPipeWorldPoseEstimator(
        model_asset_path=config.raw["pose"].get(
            "model_asset_path",
            "models/pose_landmarker_lite.task",
        ),
        min_detection_confidence=float(config.raw["pose"].get("min_detection_confidence", 0.5)),
        min_tracking_confidence=float(config.raw["pose"].get("min_tracking_confidence", 0.5)),
    )
    try:
        if plan.source_condition_id == "baseline_raw":
            records = _lift_baseline_frames(frames, estimator, plan.output_condition_id)
        else:
            records = _lift_image_proposal_frames(frames, clip, config, estimator, plan.output_condition_id)
    finally:
        estimator.close()

    if plan.input_pose_path.exists():
        records = _gate_pose3d_with_2d_prior(
            records,
            pose2d_path=plan.input_pose_path,
            pose3d_config=pose3d_config,
            hard_gate=bool(pose3d_config.get("gate_with_2d_prior", True)),
        )

    write_pose3d_records(plan.output_pose3d_path, records)
    return len(records)


def _lift_external_video_hmr(
    plan: Pose3DPlan,
    clip: ClipMetadata,
    config: RuntimeConfig,
    pose3d_config: dict[str, object],
):
    frames = _read_external_3d_timeline(plan)
    result_path = _external_video_hmr_result_path(
        pose3d_config,
        config=config,
        clip_id=clip.clip_id,
        input_condition_id=plan.input_condition_id,
        source_condition_id=plan.source_condition_id,
        output_condition_id=plan.output_condition_id,
        backend=plan.backend,
    )
    return read_external_video_hmr_records(
        result_path,
        clip_id=clip.clip_id,
        condition_id=plan.output_condition_id,
        frames=frames,
        backend_name=plan.backend,
        scale_mode=str(pose3d_config.get("external_scale_mode", "external_video_hmr")),
        joint_names=_string_list(pose3d_config.get("external_joint_names")),
        joint_name_map=_string_map(pose3d_config.get("external_joint_name_map")),
    )


def _read_external_3d_timeline(plan: Pose3DPlan) -> list[FrameRecord]:
    if plan.input_frames_path.exists():
        return read_frame_records(plan.input_frames_path)
    if not plan.input_pose_path.exists():
        raise FileNotFoundError(
            f"External 3D import requires either sampled frames or input 2D poses: "
            f"{plan.input_frames_path} / {plan.input_pose_path}"
        )
    records_by_frame = {}
    for record in read_pose_records(plan.input_pose_path):
        records_by_frame.setdefault(record.frame_index, record.timestamp_sec)
    return [
        FrameRecord(
            clip_id=plan.clip_id,
            condition_id=plan.source_condition_id,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            frame_path=Path(""),
            width=None,
            height=None,
        )
        for frame_index, timestamp_sec in sorted(records_by_frame.items())
    ]


def _external_video_hmr_result_path(
    pose3d_config: dict[str, object],
    *,
    config: RuntimeConfig,
    clip_id: str,
    input_condition_id: str,
    source_condition_id: str,
    output_condition_id: str,
    backend: str,
) -> Path:
    template = str(
        pose3d_config.get(
            "external_result_path",
            "{data_dir}/external_pose3d/{backend}/{clip_id}.csv",
        )
    )
    return Path(
        template.format(
            data_dir=config.data_dir,
            output_dir=config.output_dir,
            clip_id=clip_id,
            input_condition_id=input_condition_id,
            source_condition_id=source_condition_id,
            output_condition_id=output_condition_id,
            backend=backend,
        )
    )


def _string_list(value: object) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("pose3d.external_joint_names must be a list.")
    return [str(item) for item in value]


def _string_map(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("pose3d.external_joint_name_map must be a mapping.")
    return {str(key): str(mapped) for key, mapped in value.items()}


def lift_pose_sequence_placeholder(plan: Pose3DPlan) -> None:
    """Backward-compatible alias retained for the initial 3D planning commit."""

    raise NotImplementedError(
        "3D lifting placeholder has been superseded by lift_pose_sequence(). Planned input:"
        f" {plan.input_frames_path} -> {plan.output_pose3d_path}"
    )


def _source_condition_for_3d(input_condition_id: str) -> str:
    if input_condition_id.endswith("_smooth"):
        return input_condition_id[: -len("_smooth")]
    return input_condition_id


def _lift_baseline_frames(frames, estimator: MediaPipeWorldPoseEstimator, output_condition_id: str):
    all_records = []
    for frame in frames:
        image = read_frame(frame.frame_path)
        all_records.extend(estimator.estimate_frame_3d(image, frame, output_condition_id))
    return all_records


def _lift_image_proposal_frames(
    frames,
    clip: ClipMetadata,
    config: RuntimeConfig,
    estimator: MediaPipeWorldPoseEstimator,
    output_condition_id: str,
):
    cv2 = _require_cv2()
    roi_config = image_proposal_roi_config(config.raw, clip.clip_id, "image_center_motion_grabcut_pose")
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=80,
        varThreshold=24,
        detectShadows=False,
    )
    previous_image = None
    previous_mask = None
    initial_center_x = float(roi_config.get("center_x", 0.5))
    initial_width_ratio = float(roi_config.get("center_width_ratio", 0.62))
    tracker = ImageProposalTracker(
        initial_center_x=initial_center_x,
        initial_width_ratio=initial_width_ratio,
        center_x=initial_center_x,
        center_width_ratio=initial_width_ratio,
        max_offset=float(roi_config.get("tracker_max_offset", 0.12)),
        max_center_step=float(roi_config.get("tracker_max_center_step", 0.015)),
        max_width_step=float(roi_config.get("tracker_max_width_step", 0.025)),
        center_smoothing=float(roi_config.get("tracker_center_smoothing", 0.55)),
        width_smoothing=float(roi_config.get("tracker_width_smoothing", 0.45)),
        min_width_ratio=float(roi_config.get("tracker_min_width_ratio", 0.56)),
        max_width_ratio=float(roi_config.get("tracker_max_width_ratio", 0.72)),
        warmup_frames=int(roi_config.get("tracker_warmup_frames", 90)),
    )
    all_records = []
    for frame in frames:
        image = read_frame(frame.frame_path)
        proposal = create_center_motion_grabcut_proposal(
            image=image,
            previous_image=previous_image,
            previous_mask=previous_mask,
            background_subtractor=background_subtractor,
            center_x=tracker.center_x,
            center_width_ratio=tracker.center_width_ratio,
            min_area_ratio=float(roi_config.get("min_area_ratio", 0.006)),
            grabcut_iterations=int(roi_config.get("grabcut_iterations", 1)),
            processing_scale=float(roi_config.get("processing_scale", 0.45)),
            vertical_body_width_ratio=float(roi_config.get("vertical_body_width_ratio", 0.22)),
            lower_body_width_ratio=(
                None
                if roi_config.get("lower_body_width_ratio") is None
                else float(roi_config["lower_body_width_ratio"])
            ),
            lower_body_left_width_ratio=(
                None
                if roi_config.get("lower_body_left_width_ratio") is None
                else float(roi_config["lower_body_left_width_ratio"])
            ),
            lower_body_right_width_ratio=(
                None
                if roi_config.get("lower_body_right_width_ratio") is None
                else float(roi_config["lower_body_right_width_ratio"])
            ),
        )
        masked_image = apply_image_proposal_mask(image, proposal)
        crop = crop_to_roi(masked_image, proposal.roi)
        all_records.extend(estimator.estimate_frame_3d(crop, frame, output_condition_id))
        tracker.update(proposal, image=image)
        previous_image = image
        previous_mask = proposal.mask
    return all_records


def _gate_pose3d_with_2d_prior(
    records,
    *,
    pose2d_path: Path,
    pose3d_config: dict[str, object],
    hard_gate: bool = True,
):
    """Use the cleaned 2D pipeline as a trust prior for 3D joints."""

    threshold_config = pose3d_config.get("confidence_thresholds", {})
    default_threshold = float(pose3d_config.get("confidence_threshold", 0.5))
    pose2d_records = read_pose_records(pose2d_path)
    pose2d_by_key = {
        (record.frame_index, record.joint_name): record
        for record in pose2d_records
    }
    gated = []
    for record in records:
        prior = pose2d_by_key.get((record.frame_index, record.joint_name))
        if prior is None:
            gated.append(record)
            continue
        threshold = threshold_for_joint(record.joint_name, default_threshold, threshold_config if isinstance(threshold_config, dict) else {})
        prior_score = pose_score(prior)
        should_reject = prior.x is None or prior.y is None or (prior_score is not None and prior_score < threshold)
        if hard_gate and should_reject:
            gated.append(
                record.__class__(
                    clip_id=record.clip_id,
                    condition_id=record.condition_id,
                    frame_index=record.frame_index,
                    timestamp_sec=record.timestamp_sec,
                    joint_name=record.joint_name,
                    x_3d=nan,
                    y_3d=nan,
                    z_3d=nan,
                    scale_mode=record.scale_mode,
                    lift_backend=record.lift_backend,
                    input_quality_score=prior_score,
                )
            )
            continue
        merged_score = record.input_quality_score
        if prior_score is not None and merged_score is not None:
            merged_score = min(prior_score, merged_score)
        elif prior_score is not None:
            merged_score = prior_score
        gated.append(
            record.__class__(
                clip_id=record.clip_id,
                condition_id=record.condition_id,
                frame_index=record.frame_index,
                timestamp_sec=record.timestamp_sec,
                joint_name=record.joint_name,
                x_3d=record.x_3d,
                y_3d=record.y_3d,
                z_3d=record.z_3d,
                scale_mode=record.scale_mode,
                lift_backend=record.lift_backend,
                input_quality_score=merged_score,
            )
        )
    return gated
