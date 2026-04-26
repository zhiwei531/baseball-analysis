from baseball_pose.evaluation.roi_ablation import summarize_roi_ablation
from baseball_pose.io.pose_csv import write_pose_records
from baseball_pose.pose.schema import PoseRecord


def test_summarize_roi_ablation_writes_metrics(tmp_path):
    pose_dir = tmp_path / "processed" / "poses" / "clip"
    records = [
        PoseRecord("clip", "baseline_raw", 0, 0.0, "left_wrist", 0.0, 0.0, 1.0, 1.0, "test", 2.0),
        PoseRecord("clip", "baseline_raw", 0, 0.0, "right_wrist", 1.0, 0.0, 1.0, 1.0, "test", 2.0),
        PoseRecord("clip", "baseline_raw", 1, 0.1, "left_wrist", 0.1, 0.0, 1.0, 1.0, "test", 4.0),
        PoseRecord("clip", "baseline_raw", 1, 0.1, "right_wrist", 1.1, 0.0, 1.0, 1.0, "test", 4.0),
    ]
    write_pose_records(pose_dir / "baseline_raw.csv", records)

    rows = summarize_roi_ablation(
        ["clip"],
        tmp_path,
        conditions=("baseline_raw",),
        output_path=tmp_path / "metrics.csv",
    )

    assert rows
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "roi_ablation_summary.csv").exists()
    assert any(row.metric_name == "temporal_jitter" for row in rows)
