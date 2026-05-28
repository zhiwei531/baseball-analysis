"""Command-line entry points for the baseball pose pipeline."""

from __future__ import annotations

import argparse

from baseball_pose.config import load_config
from baseball_pose.evaluation.roi_ablation import summarize_roi_ablation
from baseball_pose.io.metadata import load_clips
from baseball_pose.pipeline.body_mask_debug import render_body_mask_debug_videos
from baseball_pose.pipeline.run_experiment import (
    run_auto_roi_experiment,
    run_baseline_experiment,
    run_body_prior_mask_roi_experiment,
    run_center_prior_roi_experiment,
    run_image_proposal_roi_experiment,
    run_motion_preview_experiment,
    run_pose_prior_roi_experiment,
)
from baseball_pose.pipeline.features import extract_feature_files
from baseball_pose.pipeline.figures import make_report_figures
from baseball_pose.pipeline.image_proposal_debug import render_image_proposal_debug_videos
from baseball_pose.pipeline.overlays import render_pose_overlays
from baseball_pose.pipeline.overlays3d import render_pose3d_overlays
from baseball_pose.pipeline.pose3d import build_pose3d_plan, lift_pose_sequence
from baseball_pose.pipeline.postprocess import smooth_pose_files
from baseball_pose.pipeline.postprocess3d import smooth_pose3d_files
from baseball_pose.pipeline.report_llm import generate_llm_reports
from baseball_pose.pipeline.report_prompt import build_report_prompts
from baseball_pose.pipeline.report_summary import build_report_summaries
from baseball_pose.pipeline.stability import summarize_pose_stability


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="baseball-pose")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "command",
        choices=[
            "validate-config",
            "plan",
            "run-baseline",
            "run-motion-preview",
            "run-auto-roi",
            "run-pose-prior-roi",
            "run-center-prior-roi",
            "run-body-prior-mask-roi",
            "run-image-proposal-roi",
            "smooth-poses",
            "smooth-pose-3d",
            "extract-features",
            "make-figures",
            "render-overlays",
            "render-overlays-3d",
            "render-body-mask-debug",
            "render-image-proposal-debug",
            "summarize-roi-ablation",
            "summarize-pose-stability",
            "build-report-summary",
            "build-report-prompt",
            "generate-llm-report",
            "plan-3d",
            "lift-pose-3d",
        ],
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
    parser.add_argument(
        "--condition",
        action="append",
        help="Condition id for commands that read existing pose or feature files.",
    )
    parser.add_argument(
        "--athlete-group",
        default=None,
        help="Optional cohort alias such as youth_pitcher, high_school_pitcher, or professional_pitcher.",
    )
    parser.add_argument(
        "--reference-file",
        default="data/metadata/llm_biomechanics_reference.yaml",
        help="Path to the YAML reference file used for report summary generation.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4.1",
        help="Model name for the OpenAI-compatible chat completions endpoint.",
    )
    parser.add_argument(
        "--llm-base-url",
        default="https://api.openai.com/v1",
        help="Base URL for an OpenAI-compatible API endpoint.",
    )
    parser.add_argument(
        "--llm-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the API key.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for report generation.",
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

    if args.command == "plan-3d":
        clips = load_clips(config.clips_file)
        clip_filter = set(args.clip_id) if args.clip_id else set(config.clip_ids)
        input_conditions = args.condition or config.condition_ids
        for clip in clips:
            if clip.clip_id not in clip_filter:
                continue
            for condition_id in input_conditions:
                plan = build_pose3d_plan(
                    clip,
                    config,
                    input_condition_id=condition_id,
                )
                print(f"{plan.clip_id}: {plan.input_condition_id} -> {plan.output_condition_id}")
                print(f"  source condition: {plan.source_condition_id}")
                print(f"  backend: {plan.backend}")
                print(f"  root joint: {plan.root_joint}")
                print(f"  input pose: {plan.input_pose_path}")
                print(f"  input frames: {plan.input_frames_path}")
                print(f"  output pose3d: {plan.output_pose3d_path}")
                print(f"  output feature3d: {plan.output_feature3d_path}")
        return

    if args.command == "lift-pose-3d":
        clips = load_clips(config.clips_file)
        clip_filter = set(args.clip_id) if args.clip_id else set(config.clip_ids)
        input_conditions = args.condition or config.condition_ids
        for clip in clips:
            if clip.clip_id not in clip_filter:
                continue
            for condition_id in input_conditions:
                plan = build_pose3d_plan(
                    clip,
                    config,
                    input_condition_id=condition_id,
                )
                print(f"Planned 3D lift: {plan.clip_id} {plan.input_condition_id} -> {plan.output_condition_id}")
                record_count = lift_pose_sequence(plan, clip, config)
                print(f"  source condition: {plan.source_condition_id}")
                print(f"  output pose3d: {plan.output_pose3d_path}")
                print(f"  wrote {record_count} 3D joint records")
        return

    if args.command == "smooth-pose-3d":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        condition_ids = args.condition or config.pose3d_condition_ids
        if not condition_ids:
            raise ValueError(
                "smooth-pose-3d requires at least one --condition or experiments.default_3d_conditions."
            )
        results = smooth_pose3d_files(clip_ids, config, condition_ids)
        for result in results:
            print(f"{result.clip_id}/{result.condition_id}: {result.pose_record_count} 3D records")
            print(f"  source: {result.source_condition_id}")
            print(f"  pose3d: {result.pose3d_csv}")
        return

    if args.command == "render-overlays-3d":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        condition_ids = args.condition or config.pose3d_condition_ids
        if not condition_ids:
            raise ValueError(
                "render-overlays-3d requires at least one --condition or experiments.default_3d_conditions."
            )
        results = render_pose3d_overlays(clip_ids, config, condition_ids)
        for result in results:
            print(f"{result.clip_id}/{result.condition_id}: {result.frame_count} frames")
            print(f"  overlay3d: {result.overlay_video}")
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

    if args.command == "run-motion-preview":
        clips = load_clips(config.clips_file)
        clip_filter = set(args.clip_id) if args.clip_id else set(config.clip_ids)
        results = run_motion_preview_experiment(
            clips,
            config,
            clip_ids=clip_filter,
            max_frames=args.max_frames,
        )
        for result in results:
            print(f"{result.clip_id}/{result.condition_id}: {result.frame_count} frames")
            print(f"  frames: {result.frames_csv}")
            print(f"  preview: {result.preview_video}")
        return

    if args.command == "run-auto-roi":
        clips = load_clips(config.clips_file)
        clip_filter = set(args.clip_id) if args.clip_id else set(config.clip_ids)
        results = run_auto_roi_experiment(
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
            print(f"  roi: {result.roi_csv}")
            print(f"  roi debug: {result.roi_debug_video}")
            print(f"  poses: {result.poses_csv}")
            print(f"  overlay: {result.overlay_video}")
        return

    if args.command == "run-pose-prior-roi":
        clips = load_clips(config.clips_file)
        clip_filter = set(args.clip_id) if args.clip_id else set(config.clip_ids)
        results = run_pose_prior_roi_experiment(
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
            print(f"  roi: {result.roi_csv}")
            print(f"  roi debug: {result.roi_debug_video}")
            print(f"  poses: {result.poses_csv}")
            print(f"  overlay: {result.overlay_video}")
        return

    if args.command == "run-center-prior-roi":
        clips = load_clips(config.clips_file)
        clip_filter = set(args.clip_id) if args.clip_id else set(config.clip_ids)
        results = run_center_prior_roi_experiment(
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
            print(f"  roi: {result.roi_csv}")
            print(f"  roi debug: {result.roi_debug_video}")
            print(f"  poses: {result.poses_csv}")
            print(f"  overlay: {result.overlay_video}")
        return

    if args.command == "run-body-prior-mask-roi":
        clips = load_clips(config.clips_file)
        clip_filter = set(args.clip_id) if args.clip_id else set(config.clip_ids)
        results = run_body_prior_mask_roi_experiment(
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
            print(f"  roi: {result.roi_csv}")
            print(f"  roi debug: {result.roi_debug_video}")
            print(f"  poses: {result.poses_csv}")
            print(f"  overlay: {result.overlay_video}")
        return

    if args.command == "run-image-proposal-roi":
        clips = load_clips(config.clips_file)
        clip_filter = set(args.clip_id) if args.clip_id else {"batting_1"}
        results = run_image_proposal_roi_experiment(
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

    if args.command == "smooth-poses":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        results = smooth_pose_files(
            clip_ids=clip_ids,
            config=config,
            source_conditions=args.condition,
        )
        for result in results:
            print(
                f"{result.clip_id}/{result.source_condition_id} -> {result.condition_id}: "
                f"{result.pose_record_count} pose records"
            )
            print(f"  poses: {result.pose_csv}")
        return

    if args.command == "extract-features":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        results = extract_feature_files(clip_ids=clip_ids, config=config, conditions=args.condition)
        for result in results:
            print(
                f"{result.clip_id}/{result.condition_id}: "
                f"{result.frame_count} feature rows"
            )
            print(f"  features: {result.feature_csv}")
        return

    if args.command == "make-figures":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        results = make_report_figures(clip_ids=clip_ids, config=config, conditions=args.condition)
        for result in results:
            print(
                f"{result.clip_id}: wrote figure from "
                f"{result.condition_count} conditions"
            )
            print(f"  figure: {result.figure_path}")
        return

    if args.command == "render-overlays":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        results = render_pose_overlays(clip_ids=clip_ids, config=config, conditions=args.condition)
        for result in results:
            print(
                f"{result.clip_id}/{result.condition_id}: "
                f"{result.frame_count} overlay frames"
            )
            print(f"  overlay: {result.overlay_video}")
        return

    if args.command == "render-body-mask-debug":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        results = render_body_mask_debug_videos(
            clip_ids=clip_ids,
            config=config,
            conditions=args.condition,
        )
        for result in results:
            print(
                f"{result.clip_id}/{result.condition_id}: "
                f"{result.frame_count} body-mask debug frames"
            )
            print(f"  proposal: {result.proposal_video}")
            print(f"  masked: {result.masked_video}")
        return

    if args.command == "render-image-proposal-debug":
        clip_ids = args.clip_id if args.clip_id else ["batting_1"]
        results = render_image_proposal_debug_videos(
            clip_ids=clip_ids,
            config=config,
            max_frames=args.max_frames,
        )
        for result in results:
            print(
                f"{result.clip_id}/{result.condition_id}: "
                f"{result.frame_count} image-proposal debug frames"
            )
            print(f"  proposal: {result.proposal_video}")
            print(f"  masked: {result.masked_video}")
        return

    if args.command == "summarize-roi-ablation":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        rows = summarize_roi_ablation(
            clip_ids=clip_ids,
            data_dir=config.data_dir,
            conditions=tuple(args.condition if args.condition else config.condition_ids),
            confidence_threshold=float(config.raw["postprocess"].get("confidence_threshold", 0.5)),
        )
        output_path = config.data_dir / "processed" / "metrics" / "roi_ablation.csv"
        print(f"Wrote {len(rows)} ROI ablation metric rows: {output_path}")
        print(f"Wrote ROI ablation summary table: {output_path.with_name('roi_ablation_summary.csv')}")
        return

    if args.command == "summarize-pose-stability":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        results = summarize_pose_stability(
            clip_ids=clip_ids,
            config=config,
            conditions=args.condition,
        )
        for result in results:
            print(f"{result.clip_id}: wrote pose stability summaries")
            print(f"  joints: {result.joint_csv}")
            print(f"  limbs: {result.limb_csv}")
        return

    if args.command == "build-report-summary":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        results = build_report_summaries(
            clip_ids=clip_ids,
            config=config,
            conditions=args.condition,
            athlete_group=args.athlete_group,
            reference_path=args.reference_file,
        )
        for result in results:
            print(
                f"{result.clip_id}/{result.condition_id}: "
                f"{result.frame_count} feature rows -> "
                f"{result.available_metric_count} available, "
                f"{result.partial_metric_count} partial, "
                f"{result.unavailable_metric_count} unavailable"
            )
            print(f"  summary: {result.summary_path}")
        return

    if args.command == "build-report-prompt":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        results = build_report_prompts(
            clip_ids=clip_ids,
            config=config,
            conditions=args.condition,
        )
        for result in results:
            print(f"{result.clip_id}/{result.condition_id}: wrote prompt package")
            print(f"  payload: {result.payload_path}")
            print(f"  prompt: {result.prompt_path}")
            print(f"  draft: {result.draft_report_path}")
        return

    if args.command == "generate-llm-report":
        clip_ids = args.clip_id if args.clip_id else config.clip_ids
        results = generate_llm_reports(
            clip_ids=clip_ids,
            config=config,
            conditions=args.condition,
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key_env=args.llm_api_key_env,
            temperature=float(args.llm_temperature),
        )
        for result in results:
            print(f"{result.clip_id}/{result.condition_id}: wrote LLM report with {result.model}")
            print(f"  report: {result.report_path}")
            print(f"  request: {result.request_path}")
            print(f"  response: {result.response_path}")
        return


if __name__ == "__main__":
    main()
